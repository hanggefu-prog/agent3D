#!/usr/bin/env python3
"""Small persistent SSH DAG scheduler. Exit-code success is not physical validation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

ACTIVE = {"dispatching", "queued", "running", "unknown"}
TERMINAL = {"success", "failed", "blocked"}
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
RESOURCE_KEYS = ("cpus", "memory_mb", "disk_mb", "gpu_memory_mb")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def lock(path):
    stream = Path(path).open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def positive(value, name, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < (0 if zero else 1):
        raise ValueError("Invalid numeric budget: " + name)


def validate(config):
    if not SAFE.fullmatch(config["cluster_id"]):
        raise ValueError("Invalid cluster_id")
    hosts = {h["id"]: h for h in config["hosts"]}
    jobs = {j["id"]: j for j in config["jobs"]}
    if not hosts or not jobs or len(hosts) != len(config["hosts"]) or len(jobs) != len(config["jobs"]):
        raise ValueError("Empty or duplicate host/job IDs")
    for host in hosts.values():
        if not SAFE.fullmatch(host["id"]) or not host["ssh"] or host["ssh"].startswith("-"):
            raise ValueError("Invalid host identity")
        for key in ("root", "python", "script"):
            if not Path(host[key]).is_absolute():
                raise ValueError("Host paths must be absolute: " + key)
        for key in ("slots", "cpus", "memory_mb", "disk_mb"):
            positive(host[key], key)
        if type(host["slots"]) is not int:
            raise ValueError("slots must be an integer")
        slots = host.get("slot_ids", [f"{n + 1:02d}" for n in range(host["slots"])])
        if len(slots) != host["slots"] or len(set(slots)) != len(slots) or not all(SAFE.fullmatch(s) for s in slots):
            raise ValueError("Invalid slot_ids")
        if len({str(g["index"]) for g in host.get("gpus", [])}) != len(host.get("gpus", [])):
            raise ValueError("Duplicate GPU index")
        for gpu in host.get("gpus", []):
            if not str(gpu["index"]).isdigit():
                raise ValueError("GPU index must be numeric")
            positive(gpu["memory_mb"], "GPU memory")
        positive(host.get("speed", 1), "speed")
    for job in jobs.values():
        if not SAFE.fullmatch(job["id"]):
            raise ValueError("Invalid job ID")
        if not set(job.get("deps", [])) <= jobs.keys() or job["id"] in job.get("deps", []):
            raise ValueError("Unknown or self dependency")
        eligible = job.get("hosts", list(hosts))
        if not eligible or not set(eligible) <= hosts.keys():
            raise ValueError("Invalid eligible hosts")
        positive(job["estimated_seconds"], "estimated_seconds")
        for key in RESOURCE_KEYS:
            positive(job["resources"][key], key, zero=key in {"gpu_memory_mb", "disk_mb"})
        if not isinstance(job["command"], list) or not job["command"] or not job["command"][0] or not all(isinstance(x, str) and "\0" not in x for x in job["command"]):
            raise ValueError("command must be a nonempty argv array")
        for name in eligible:
            host = hosts[name]
            allocation = {"host": name, "slot": host.get("slot_ids", ["01"])[0],
                          "gpu": str(host.get("gpus", [{"index": "none"}])[0]["index"]) if host.get("gpus") else "none",
                          "resources": job["resources"]}
            try:
                make_spec(config, job, host, allocation)
            except (KeyError, ValueError, AttributeError) as exc:
                raise ValueError("Invalid command/cwd template for " + job["id"] + " on " + name) from exc
    ranks(jobs)  # Cycle check before any side effect.
    return hosts, jobs


def ranks(jobs):
    result, visiting = {}, set()
    def visit(name):
        if name in visiting:
            raise ValueError("Dependency cycle")
        if name not in result:
            visiting.add(name)
            downstream = [visit(k) for k, j in jobs.items() if name in j.get("deps", [])]
            result[name] = jobs[name]["estimated_seconds"] + max(downstream, default=0)
            visiting.remove(name)
        return result[name]
    for name in jobs:
        visit(name)
    return result


def ready(jobs, state):
    rank = ranks(jobs)
    return sorted((k for k, j in jobs.items() if state[k]["status"] == "pending"
                   and all(state[d]["status"] == "success" for d in j.get("deps", []))),
                  key=lambda k: (-rank[k], k))


def block_dependencies(jobs, state):
    changed = True
    while changed:
        changed = False
        for name, job in jobs.items():
            bad = [d for d in job.get("deps", []) if state[d]["status"] in {"failed", "blocked"}]
            if state[name]["status"] == "pending" and bad:
                state[name] = {"status": "blocked", "reason": "Failed dependencies", "dependencies": bad}
                changed = True


def allocations(job, host, reservations, live=None):
    """Budget reservations plus conservative live headroom; busy GPUs are allowed."""
    used = [r for r in reservations if r["host"] == host["id"]]
    req = job["resources"]
    if len(used) >= host["slots"]:
        return []
    if any(sum(r["resources"][key] for r in used) + req[key] > host[key]
           for key in ("cpus", "memory_mb", "disk_mb")):
        return []
    if live is not None:
        if time.time() - live["sampled_at"] > 60:
            return []
        # Existing allocations can still be queued or grow to their reserved peak.
        if any(live[field] - sum(r["resources"][key] for r in used) < req[key]
               for key, field in (("cpus", "cpu_idle_cores"), ("memory_mb", "memory_available_mb"), ("disk_mb", "disk_free_mb"))):
            return []
    slots = host.get("slot_ids", [f"{n + 1:02d}" for n in range(host["slots"])])
    slot = next(s for s in slots if s not in {r["slot"] for r in used})
    if req["gpu_memory_mb"] == 0:
        return [{"host": host["id"], "slot": slot, "gpu": "none", "resources": req}]
    result = []
    for gpu in host.get("gpus", []):
        index = str(gpu["index"])
        reserved = sum(r["resources"]["gpu_memory_mb"] for r in used if r["gpu"] == index)
        capacity = gpu["memory_mb"] - reserved
        # Subtract reservations from observed free memory too: queued starts may not yet allocate.
        if live is not None:
            capacity = min(capacity, live["gpu_free_mb"].get(index, 0) - reserved)
        if capacity >= req["gpu_memory_mb"]:
            result.append({"host": host["id"], "slot": slot, "gpu": index, "resources": req})
    return result


def choose(job, hosts, reservations, live_by_host=None, affinity=None):
    eligible = job.get("hosts", list(hosts))
    bound = (affinity or {}).get(job.get("affinity_group"))
    choices = []
    for name in eligible:
        if bound and name != bound or live_by_host is not None and name not in live_by_host:
            continue
        for allocation in allocations(job, hosts[name], reservations, (live_by_host or {}).get(name)):
            duration = job["estimated_seconds"] / hosts[name].get("speed", 1)
            gpu = allocation["gpu"]
            gpu_jobs = [r for r in reservations if r["host"] == name and r["gpu"] == gpu]
            gpu_memory = sum(r["resources"]["gpu_memory_mb"] for r in gpu_jobs)
            # Preserve host preference, but spread this batch over its admissible GPUs.
            # This is a tie-breaker, not an exclusive-GPU admission requirement.
            gpu_index = int(gpu) if gpu != "none" else -1
            choices.append((duration, name, gpu_memory, len(gpu_jobs), gpu_index, allocation))
    return min(choices, key=lambda item: item[:-1])[-1] if choices else None


def plan(config):
    hosts, jobs = validate(config)
    state = {k: {"status": "pending"} for k in jobs}
    active, output, affinity, now = [], [], {}, 0
    while any(s["status"] != "success" for s in state.values()):
        for name in ready(jobs, state):
            a = choose(jobs[name], hosts, active, affinity=affinity)
            if a is None:
                continue
            end = now + jobs[name]["estimated_seconds"] / hosts[a["host"]].get("speed", 1)
            row = {**a, "id": name, "estimated_start": now, "estimated_finish": end}
            spec = make_spec(config, jobs[name], hosts[a["host"]], a)
            row.update(command=spec["command"], cwd=spec["cwd"])
            active.append(row)
            output.append(row)
            state[name] = {"status": "running"}
            if jobs[name].get("affinity_group"):
                affinity[jobs[name]["affinity_group"]] = a["host"]
        if not active:
            raise ValueError("No static resource-feasible schedule (check affinity and budgets)")
        now = min(a["estimated_finish"] for a in active)
        for a in active:
            if a["estimated_finish"] == now:
                state[a["id"]] = {"status": "success"}
        active = [a for a in active if a["estimated_finish"] > now]
    return {"policy": "Longest remaining critical path, earliest admissible finish; heuristic, not optimal",
            "estimated_makespan_seconds": now, "live_admission_not_checked": True, "jobs": output}


def probe(root):
    def cpu():
        values = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        return sum(values), values[3]  # iowait is not idle CPU capacity.
    before = cpu()
    time.sleep(0.5)
    after = cpu()
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    gpus = {}
    command = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    if shutil.which(command[0]):
        result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=True)
        gpus = {i.strip(): float(m.strip()) for i, m in (line.split(",") for line in result.stdout.splitlines())}
    return {"sampled_at": time.time(), "cpu_idle_cores": (os.cpu_count() or 1) * (after[1] - before[1]) / max(1, after[0] - before[0]),
            "memory_available_mb": int(mem["MemAvailable"].split()[0]) / 1024,
            "disk_free_mb": shutil.disk_usage(root).free / 2**20, "gpu_free_mb": gpus,
            "scheduler_sha256": code_hash()}


def session(spec):
    return "codesign-" + digest(spec)[:24]


def inspect(spec):
    directory = Path(spec["job_dir"])
    saved = directory / "spec.json"
    if not saved.exists():
        return {"status": "unknown", "reason": "Remote dispatch not recorded; no automatic retry"}
    if read(saved) != spec:
        raise ValueError("Remote immutable job specification differs")
    receipt = directory / "result.json"
    if receipt.exists():
        value = read(receipt)
        if value["spec_sha256"] != digest(spec) or value["status"] not in {"success", "failed"}:
            raise ValueError("Invalid remote result identity/status")
        return value
    running = subprocess.run(["tmux", "has-session", "-t", "=" + session(spec)], capture_output=True).returncode == 0
    return {"status": ("running" if (directory / "started.json").exists() else "queued") if running else "unknown",
            "reason": "No final receipt; session observed" if running else "No session or receipt; manual investigation required"}


def dispatch(spec):
    directory = Path(spec["job_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory / "dispatch.lock"):
        if (directory / "spec.json").exists():
            return inspect(spec)  # Never launch an existing specification again.
        if spec["scheduler_sha256"] != code_hash():
            raise ValueError("Remote scheduler source differs")
        write(directory / "spec.json", spec)
        argv = [spec["python"], spec["script"], "--worker", str(directory / "spec.json")]
        subprocess.run(["tmux", "new-session", "-d", "-s", session(spec), shlex.join(argv)], check=True, capture_output=True)
        return {"status": "queued", "spec_sha256": digest(spec)}


def worker(path):
    spec = read(path)
    directory = Path(spec["job_dir"])
    with lock(directory / "worker.lock"):
        if (directory / "started.json").exists() or (directory / "result.json").exists():
            return 0  # At-most-once, even after an incomplete worker died.
        write(directory / "started.json", {"pid": os.getpid(), "time": time.time(), "spec_sha256": digest(spec)})
        started, error, returncode = time.time(), None, 1
        try:
            if spec["scheduler_sha256"] != code_hash():
                raise ValueError("Worker source differs from the admitted specification")
            # Recheck live headroom immediately before the foreground command.
            live = probe(spec["root"])
            r = spec["resources"]
            if live["cpu_idle_cores"] < r["cpus"] or live["memory_available_mb"] < r["memory_mb"] or live["disk_free_mb"] < r["disk_mb"]:
                raise RuntimeError("Host headroom changed after admission; command not started")
            if r["gpu_memory_mb"] and live["gpu_free_mb"].get(spec["gpu"], 0) < r["gpu_memory_mb"]:
                raise RuntimeError("GPU headroom changed after admission; command not started")
            with (directory / "console.log").open("ab") as stream:
                returncode = subprocess.run(spec["command"], cwd=spec["cwd"], stdout=stream, stderr=subprocess.STDOUT).returncode
        except Exception as exc:
            error = str(exc)
        write(directory / "result.json", {"status": "success" if returncode == 0 else "failed", "exit_code": returncode,
              "error": error, "spec_sha256": digest(spec), "started_at": started, "finished_at": time.time(),
              "wall_seconds": time.time() - started, "meaning": "Command exit status only; physical gates belong to the experiment wrapper"})
        return returncode


def rpc(host, action, payload):
    argv = [host["python"], host["script"], action]
    if action == "--probe":
        argv.append(host["root"])
    command = argv if host["ssh"] == "local" else [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2", host["ssh"], shlex.join(argv)]
    result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, timeout=45, check=True)
    return json.loads(result.stdout)


def make_spec(config, job, host, allocation):
    values = {**host, **allocation, **job["resources"], "job_id": job["id"], "cluster_id": config["cluster_id"]}
    values["job_dir"] = str(Path(host["root"]) / config["cluster_id"] / "jobs" / job["id"])
    command = [word.format_map(values) for word in job["command"]]
    return {k: values[k] for k in ("root", "python", "script", "job_dir", "gpu", "slot", "resources")} | {
        "command": command, "cwd": job.get("cwd", "{job_dir}").format_map(values),
        "scheduler_sha256": code_hash(), "cluster_sha256": digest(config), "job_id": job["id"]}


def reconcile(record, observation):
    if observation.get("status") not in {"queued", "running", "unknown", "success", "failed"}:
        raise ValueError("Unknown remote state")
    if observation["status"] in {"success", "failed"} and "spec_sha256" not in observation:
        raise ValueError("Final receipt lacks admitted job identity")
    if "spec_sha256" in observation and observation["spec_sha256"] != digest(record["spec"]):
        raise ValueError("Receipt does not match admitted job")
    return {**record, "status": observation["status"], "observation": observation}


def run(config, directory, once=False, poll_seconds=30):
    hosts, jobs = validate(config)
    plan(config)  # Reject permanently impossible resource plans before dispatch.
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory / "scheduler.lock"):
        manifest = {"config": config, "scheduler_sha256": code_hash()}
        if (directory / "manifest.json").exists() and read(directory / "manifest.json") != manifest:
            raise ValueError("Immutable cluster manifest differs")
        if not (directory / "manifest.json").exists():
            write(directory / "manifest.json", manifest)
        state = read(directory / "state.json") if (directory / "state.json").exists() else {k: {"status": "pending"} for k in jobs}
        def save(event, **details):
            write(directory / "state.json", state)
            with (directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": time.time(), "event": event, **details}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        while True:
            for name, record in list(state.items()):
                if record["status"] in ACTIVE:
                    try:
                        observation = rpc(hosts[record["host"]], "--inspect", record["spec"])
                        state[name] = reconcile(record, observation)
                    except Exception as exc:
                        state[name] = {**record, "status": "unknown", "reason": str(exc)}
                    if state[name] != record:
                        save("reconciled", job=name, status=state[name]["status"])
            block_dependencies(jobs, state)
            save("cycle")
            if all(row["status"] in TERMINAL for row in state.values()):
                return 0 if all(row["status"] == "success" for row in state.values()) else 1
            def sample(host):
                try:
                    value = rpc(host, "--probe", {})
                    if value["scheduler_sha256"] != code_hash():
                        raise ValueError("Remote scheduler code mismatch")
                    return host["id"], value
                except Exception as exc:
                    return host["id"], {"error": str(exc)}
            with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
                observations = dict(pool.map(sample, hosts.values()))
            save("admission_samples", hosts=observations)
            live = {k: v for k, v in observations.items() if "error" not in v}
            reservations = [row for row in state.values() if row["status"] in ACTIVE]
            affinity = {jobs[k]["affinity_group"]: row["host"] for k, row in state.items()
                        if jobs[k].get("affinity_group") and "host" in row}
            for name in ready(jobs, state):
                allocation = choose(jobs[name], hosts, reservations, live, affinity)
                if allocation is None:
                    continue
                spec = make_spec(config, jobs[name], hosts[allocation["host"]], allocation)
                state[name] = {**allocation, "status": "dispatching", "spec": spec, "admitted_at": time.time()}
                save("dispatch_intent", job=name, host=allocation["host"], slot=allocation["slot"])
                try:
                    state[name] = reconcile(state[name], rpc(hosts[allocation["host"]], "--dispatch", spec))
                except Exception as exc:
                    state[name].update(status="unknown", reason=str(exc))
                reservations.append(state[name])
                if jobs[name].get("affinity_group"):
                    affinity[jobs[name]["affinity_group"]] = allocation["host"]
                save("dispatched", job=name, status=state[name]["status"])
            if once:
                return 0
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true")
    group.add_argument("--plan", action="store_true")
    group.add_argument("--probe", type=Path)
    group.add_argument("--dispatch", action="store_true")
    group.add_argument("--inspect", action="store_true")
    group.add_argument("--worker", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.probe or args.dispatch or args.inspect:
        value = probe(args.probe) if args.probe else (dispatch if args.dispatch else inspect)(json.load(sys.stdin))
    else:
        if not args.config:
            parser.error("--config is required")
        config = read(args.config)
        if args.run:
            if not args.state_dir or not 1 <= args.poll_seconds <= 60:
                parser.error("--run requires --state-dir and poll interval 1..60 seconds")
            return run(config, args.state_dir, args.once, args.poll_seconds)
        value = plan(config)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
