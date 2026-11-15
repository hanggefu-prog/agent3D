"""Run an explicit, isolated preliminary development queue, never physical finals.

Each task gets an independent source clone and a fresh Codex session. Native
qualification, sealing, final evaluation and the independent judge are separate
steps. A failed client is recorded and is not automatically retried.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from .model import cpu_container_command, develop_source
from .protocol import digest, now, sha, write_new
from .submission import clone_source, inventory, validate_scope
from .tasks import TASK_IDS, load_task, read_json


def writable_paths(task_id):
    if task_id in {"clock_compensation", "multi_hbt_fanout"}:
        return ("engineering",)
    if task_id == "cross_die_cts":
        return ("open3dbench/backend/three_d_cts.py", "open3dbench/backend/cts_helpers",
                "OpenROAD-3D/openroad/src/cts", "engineering/tests")
    raise ValueError("Unknown task")


def private_command(task, subject, **kwargs):
    # The whole source is RO. Overlay only contribution areas; exact task scope
    # is additionally checked by a host-side byte inventory after the turn.
    command = cpu_container_command(subject=subject, subject_read_only=True, **kwargs)
    options = []
    for relative in writable_paths(task.task_id):
        path = (subject / relative).resolve(strict=True)
        path.relative_to(subject.resolve())
        if (subject / relative).is_symlink() or "," in str(path):
            raise ValueError("Unsafe editable mount")
        options += ["--mount", f"type=bind,src={path},dst=/workspace/{relative},readonly=false"]
    # All recursive Git metadata remains RO, even if nested within an overlay.
    for git in sorted(subject.rglob(".git")):
        if git == subject / ".git":
            continue
        if not git.is_dir() or git.is_symlink():
            raise ValueError("Private submodule requires independent Git metadata")
        options += ["--mount", f"type=bind,src={git},dst=/workspace/{git.relative_to(subject)},readonly"]
    # Insert before the image/command, never as arguments to 'sleep'.
    return command[:-3] + options + command[-3:]


def admission(parent):
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(info["MemAvailable"].split()[0]) < 44 * 1024 * 1024:
        raise ValueError("Insufficient host memory for an 8 CPU / 32 GiB development slot")
    if shutil.disk_usage(parent).free < 35 * 1024**3:
        raise ValueError("Insufficient disk for an independent development checkout")


def verify_release(path):
    release = read_json(path)
    audit = release["runtime"]["equivalence"]
    if sha(audit["path"]) != audit["sha256"]:
        raise ValueError("Changed runtime qualification")
    image = release["runtime"]["actual_image"]
    actual = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    if actual != image:
        raise ValueError("Pinned image is not installed")
    source = Path(release["development_source"]).resolve(strict=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if release.get("kind") == "engineering-source-release-v2":
        if head != release["source_revision"] or digest(inventory(source)) != release["source_sha256"]:
            raise ValueError("Explicit engineering source binding changed")
        from grt_bench.builder import verify_runtime
        runtime = read_json(audit["path"])
        if runtime["image"] != image:
            raise ValueError("Release runtime receipt differs")
        verify_runtime(runtime)
        return source, image
    if head != "5d5dd87f1f2fff8b5d8503138bc99140f0fffbd3":
        raise ValueError("Unexpected physical base revision")
    return source, image


def run_one(task_id, *, source, image, root, reference, model, binary):
    task = load_task(task_id)
    task_root = root / task_id
    task_root.mkdir(exist_ok=False)
    write_new(task_root / "started.json", dict(task=task_id, phase="preliminary_source_development",
        started_at=now(), pid=os.getpid(), model=model, candidate_finals=0, submission_sealed=False))
    container = "o3db-agent-engineering-" + uuid.uuid4().hex[:16]
    container_started = False
    result = dict(task=task_id, model=model, candidate_finals=0, submission_sealed=False,
                  physical_qualification=False)
    try:
        admission(root)
        subject = task_root / "source"
        revision = clone_source(source, subject)
        # Empty directories add no baseline source bytes and provide mountpoints.
        (subject / "AgenticBench").mkdir(exist_ok=True)
        for relative in writable_paths(task_id):
            if not (subject / relative).exists():
                (subject / relative).mkdir(parents=True)
        control, scratch, results = (task_root / n for n in ("control", "scratch", "results"))
        for path in (control, scratch, results):
            path.mkdir()
        for path in sorted(task.root.iterdir()):
            if path.is_file():
                shutil.copyfile(path, control / path.name)
        before = inventory(subject)
        write_new(task_root / "baseline-inventory.json", before)
        write_new(control / "environment.json", dict(phase="preliminary_source_development",
            flow_revision=revision, image=image, seed=task.data["seed"], cpus=8, memory_gb=32,
            reference_root="/inputs/reference", reference_note="Historical frozen artifacts; not complete task qualification",
            candidate_finals=0, physical_qualification=False))
        framework = Path(__file__).resolve().parents[1]
        command = private_command(task, subject, name=container, image=image,
            control=control, scratch=scratch, results=results, benchmark=control,
            reference=reference, protected=(), cpus=8, memory_gb=32, trusted_framework=framework)
        write_new(task_root / "container.json", dict(name=container, command=command, started_at=now()))
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        container_started = True
        # A real mount check before the paid call. Opening read/write without
        # writing is harmless if a mount is accidentally unprotected.
        check = """from pathlib import Path
for p in ('/workspace/AGENTS.md', '/task/task.json', '/workspace/.git/config', '/workspace/AgenticBench/engineering_bench/metrics.py'):
    try:
        f = open(p, 'r+'); f.close()
    except PermissionError:
        continue
    except OSError as e:
        if e.errno == 30: continue
        raise
    raise RuntimeError('Protected file is writable: ' + p)
p=Path('/workspace/engineering/tests/.mount-smoke')
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text('private scope check'); p.unlink()
print('private-source-mounts-verified')
"""
        # T2/T3 engineering/tests is inside the permitted engineering overlay.
        test = subprocess.run(["docker", "exec", container, "python3", "-c", check],
                              capture_output=True, text=True, timeout=60)
        (task_root / "mount-check.log").write_text(test.stdout + test.stderr)
        if test.returncode:
            raise ValueError("Private mount preflight failed before model call")
        write_new(task_root / "model-started.json", dict(started_at=now(), model=model,
            task=task_id, container=container, candidate_finals=0))
        turn = develop_source(task, model=model, output=task_root / "development",
            binary=binary, control=control, container=container,
            terminal_script=framework / "global_bench/container_terminal.py", python=sys.executable)
        # Stop every process belonging to our container before reading the diff.
        subprocess.run(["docker", "stop", "--time", "10", container], check=True,
                       capture_output=True, text=True, timeout=30)
        container_started = False
        after = inventory(subject)
        changes = validate_scope(task, before, after)
        write_new(task_root / "draft-inventory.json", after)
        result.update(status="source_draft_ready_pending_integration" if turn["success"] else "developer_failed",
            changed_paths=changes, source_sha256=digest(after), thread_id=turn["thread_id"],
            developer_success=turn["success"], terminal_tool_calls=turn["terminal_tool_calls"],
            duration_seconds=turn["duration_seconds"])
    except Exception as exc:
        result.update(status="blocked", error=type(exc).__name__ + ": " + str(exc))
    finally:
        if container_started:
            subprocess.run(["docker", "stop", "--time", "10", container],
                           capture_output=True, text=True, timeout=30, check=False)
    result["completed_at"] = now()
    write_new(task_root / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("release", "output", "reference"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--tasks", choices=TASK_IDS, nargs="+", required=True)
    p.add_argument("--model", default="gpt-5.6-sol")
    p.add_argument("--binary", default="codex")
    args = p.parse_args()
    if len(args.tasks) != len(set(args.tasks)):
        p.error("One development turn per task; duplicates are not allowed")
    source, image = verify_release(args.release.resolve(strict=True))
    reference = args.reference.resolve(strict=True)
    root = args.output.absolute()
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "queue.json", dict(tasks=args.tasks, model=args.model, started_at=now(),
        release_sha256=sha(args.release), phase="preliminary_source_development", candidate_finals=0))
    rows = [run_one(task, source=source, image=image, root=root, reference=reference,
                    model=args.model, binary=args.binary) for task in args.tasks]
    write_new(root / "result.json", dict(tasks=rows, completed_at=now(), candidate_finals=0))
    return 0 if all(r["status"] == "source_draft_ready_pending_integration" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
