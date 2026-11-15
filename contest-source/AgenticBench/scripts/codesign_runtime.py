"""Resource-limited, credential-free Docker evaluations with durable attempt records."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from agentic3d.knowledge import digest, write_json

CASES = ("swerv_wrapper", "bp_fe", "bp", "ariane133")

PROFILES = {"tp_gnn_placement_sta": "tp_gnn_placement_two_pass",
            "tp_gnn_placement_cts_thermal": "tp_gnn_placement_cts_thermal",
            "tp_gnn_placement_thermal": "tp_gnn_placement_thermal",
            "tp_gnn_sta_inner": "tp_gnn_partition_proxy",
            "placement_sta_inner": "placement_proxy"}


def validate_min_gpu_free_mb(value):
    if type(value) is not int or value < 1:
        raise ValueError("min_gpu_free_mb must be a positive integer (MiB)")
    return value


class Runtime:
    min_data_free_gb = 40  # Preserve the historical direct-runtime contract.

    def __init__(self, source, batch, image, gpus, cpus=32, memory_gb=96, share_gpu=False,
                 min_gpu_free_mb=16000, min_data_free_gb=40,
                 audit_framework_revision=None, computation_source_revision=None,
                 reference_baseline=None, framework_sha256=None):
        if type(share_gpu) is not bool:
            raise ValueError("share_gpu must be a JSON boolean")
        self.share_gpu = share_gpu
        self.min_gpu_free_mb = validate_min_gpu_free_mb(min_gpu_free_mb)
        if type(min_data_free_gb) is not int or min_data_free_gb < 1:
            raise ValueError("min_data_free_gb must be a positive integer (GiB)")
        self.min_data_free_gb = min_data_free_gb
        self.source, self.batch = Path(source).resolve(), Path(batch).resolve()
        self.framework = Path(__file__).resolve().parents[1]
        self.audit_framework_identity = None
        self.latest_framework_identity = None
        self.reference_baseline = reference_baseline
        if reference_baseline == "reference_20261004":
            if audit_framework_revision is not None or computation_source_revision is not None:
                raise ValueError("Latest family uses explicit framework SHA, not historical audit-split pins")
            from scripts.workspace_framework_identity import validate_latest_framework
            self.latest_framework_identity = validate_latest_framework(self.source, self.framework,
                expected_sha256=framework_sha256)
            from local_bench.reference_latest import validate_reference_image
            validate_reference_image(image)
            if (self.batch == self.framework or self.batch in self.framework.parents
                    or self.framework in self.batch.parents or self.batch == self.source or self.source in self.batch.parents
                    or self.batch in self.source.parents):
                raise ValueError("Latest source/control plane and writable batch must be disjoint")
        elif framework_sha256 is not None:
            raise ValueError("Framework SHA pin belongs only to the explicit latest family")
        elif self.framework != self.source / "AgenticBench":
            from scripts.workspace_framework_identity import validate_audit_framework
            self.audit_framework_identity = validate_audit_framework(source, self.framework,
                source_revision=computation_source_revision, framework_revision=audit_framework_revision)
            audit_root = self.framework.parent
            if (self.batch == audit_root or self.batch in audit_root.parents or audit_root in self.batch.parents
                    or self.batch == self.source or self.batch in self.source.parents or self.source in self.batch.parents):
                raise ValueError("Audit/computation checkouts and writable batch must be disjoint")
        elif audit_framework_revision is not None or computation_source_revision is not None:
            raise ValueError("Explicit audit/source pins require separate audit and computation checkouts")
        self.image, self.gpus = image, [int(x) for x in gpus.split(",")]
        self.cpus, self.memory_gb = cpus, memory_gb
        if not image.startswith("sha256:") or len(image) != 71:
            raise ValueError("pin the Docker image by its full sha256 ID")
        if not self.gpus or len(set(self.gpus)) != len(self.gpus) or any(g not in range(8) for g in self.gpus):
            raise ValueError("invalid approved GPU list")
        if not 1 <= cpus <= 32 or not 32 <= memory_gb <= 128:
            raise ValueError("unsafe per-evaluation resource limit")
        self.source_commit = subprocess.check_output(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(self.source), "status", "--porcelain", "--untracked-files=no"], text=True):
            raise ValueError("source must be a clean, separately committed frozen snapshot")
        self.batch.mkdir(parents=True, exist_ok=True)
        (self.batch / "locks").mkdir(exist_ok=True)

    def profile(self, case):
        if case not in CASES:
            raise ValueError("case outside this batch")
        if getattr(self, "reference_baseline", None) == "reference_20261004":
            from local_bench.reference_latest import profile_path
            return profile_path(case)
        if getattr(self, "reference_baseline", None) == "reference_20260923":
            from local_bench.reference_baseline import profile_path
            profile = profile_path(case)
            if getattr(self, "audit_framework_identity", None) is None:
                return profile
            # The real input validator resolves profile IDs in this framework.
            # Audit/source recipe bytes must match, but the returned path is
            # audit-local; computation identity remains self.source_commit.
            from local_bench.reference_baseline import validate_audit_profile
            return validate_audit_profile(case, self.audit_framework_identity)
        return self.framework / "local_bench/benchmark_inputs" / (case + "_lol_ng45_tp_gnn_sta")

    def contained(self, path):
        path = Path(path).resolve()
        path.relative_to(self.batch)
        return path

    @contextlib.contextmanager
    def slot(self):
        """Shared by ALL objectives/methods in this server batch; never lock foreign GPUs."""
        while True:
            for gpu in self.gpus:
                lock = (self.batch / "locks" / f"gpu-{gpu}.lock").open("a")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    continue
                try:
                    mem = {line.split(":", 1)[0]: int(line.split()[1]) for line in
                           Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")}
                    free = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}",
                               "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                    processes = subprocess.check_output(
                        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                         "--format=csv,noheader,nounits"], text=True).strip()
                    # Shared mode admits existing device users, but keeps all
                    # capacity checks and the batch-local evaluation lock.
                    ready = (mem["MemAvailable"] >= (self.memory_gb + 16) * 1024**2
                             and (self.share_gpu or not processes) and free >= self.min_gpu_free_mb
                             and shutil.disk_usage("/").free >= 10 * 1024**3
                             and shutil.disk_usage(self.batch).free >= self.min_data_free_gb * 1024**3)
                    if ready:
                        yield gpu
                        return
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    lock.close()
            time.sleep(30)  # detached scheduler, not an interactive assistant wait

    def docker(self, command, gpu, scratch=None):
        # Writable container layers live on the host root disk, not the large
        # results volume. Use an isolated, persistent scratch directory per call.
        if scratch is None:
            parent = self.batch / "scratch"
            parent.mkdir(exist_ok=True)
            scratch = Path(tempfile.mkdtemp(prefix="call-", dir=parent))
        scratch = self.contained(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        reference_mounts = []
        reference_environment = []
        link_frozen = "1"
        if getattr(self, "reference_baseline", None) == "reference_20261004":
            from scripts.workspace_framework_identity import revalidate_runtime_framework
            revalidate_runtime_framework(self)
            link_frozen = "0"
            reference_environment = ["-e", "OMP_NUM_THREADS=4", "-e", "MKL_NUM_THREADS=4",
                                     "-e", "OPEN3DBENCH_IMAGE_ID=" + self.image,
                                     "-e", "OPEN3DBENCH_IMAGE=" + self.image]
            reference_mounts = ["-v", f"{self.framework}:{self.framework}:ro"]
            for path in getattr(self, "latest_reference_mounts", ()):
                path = Path(path).resolve(strict=True)
                if any(path == p or path in p.parents or p in path.parents for p in (self.source, self.framework, self.batch, scratch)):
                    raise ValueError("Latest reference evidence overlaps execution/control state")
                if any(c in str(path) for c in (":", "\n", "\r")):
                    raise ValueError("Unsafe latest evidence mount")
                reference_mounts += ["-v", f"{path}:{path}:ro"]
        if getattr(self, "reference_baseline", None) == "reference_20260923":
            # Adoption rebases metadata in the new run, never in the archive.
            link_frozen = "0"
            reference_environment = ["-e", "OMP_NUM_THREADS=4", "-e", "MKL_NUM_THREADS=4"]
            from local_bench.reference_baseline import validate_reference_image, identity, sha256
            validate_reference_image(self.image)
            reference_root = Path(self.reference_root).resolve(strict=True)
            index = reference_root / "reference.json"
            if sha256(index) != identity()["reference_json_sha256"]:
                raise ValueError("Frozen archive changed before container launch")
            original = Path(json.loads(index.read_text())["original_root"])
            for target in dict.fromkeys((reference_root, original)):
                if not target.is_absolute() or any(target == p or target in p.parents or p in target.parents
                    for p in (self.source, self.batch, scratch)):
                    raise ValueError("Frozen archive mount overlaps writable execution state")
                reference_mounts += ["-v", f"{reference_root}:{target}:ro"]
        return ["docker", "run", "--rm", "--network", "none", "--gpus", f"device={gpu}",
                "--cpus", str(self.cpus), "--cpu-shares", "256", "--memory", f"{self.memory_gb}g",
                "--user", f"{os.getuid()}:{os.getgid()}", "--shm-size", "8g",
                "-v", f"{self.batch}:{self.batch}", "-v", f"{self.source}:{self.source}:ro",
                *(["-v", f"{self.framework.parent}:{self.framework.parent}:ro"]
                  if getattr(self, "audit_framework_identity", None) else []),
                "-v", f"{scratch}:/tmp", "-e", "TMPDIR=/tmp",
                *reference_mounts,
                *reference_environment,
                "-e", "MPLCONFIGDIR=/tmp/matplotlib", "-e", "XDG_CACHE_HOME=/tmp/cache",
                "-w", str(self.framework), "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", "OPEN3DBENCH_PYTHON=/usr/bin/python3", "-e", "OPEN3D_LINK_FROZEN_UPSTREAM=" + link_frozen,
                "-e", "OPEN3D_LOCAL_DREAMPLACE_INSTALL=/opt/dreamplace",
                "-e", "PLACE3D_DREAMPLACE_INSTALL=/opt/dreamplace",
                "-e", f"PYTHONPATH={self.framework}:{self.source}:/opt/dreamplace:/opt/dreamplace/dreamplace",
                self.image, *command]

    def execute(self, command, output, log, contract):
        # A refused new call must not inherit dispatch evidence from a prior one.
        self.last_execution = None
        contract = {**contract, "share_gpu": self.share_gpu,
                    "min_gpu_free_mb": self.min_gpu_free_mb,
                    "min_data_free_gb": self.min_data_free_gb}
        output, log = self.contained(output), self.contained(log)
        ledger = output.parent / (output.name + ".attempt.json")
        events = output.parent / (output.name + ".attempt.events.jsonl")
        if ledger.exists() or events.exists() or output.exists():
            raise RuntimeError("incomplete attempt requires inspection; will not silently retry: " + str(output))
        log.parent.mkdir(parents=True, exist_ok=True)
        queued_clock = time.monotonic()
        queued_at = time.time()
        attempt = {"schema_version": 2, "state": "queued", "contract": contract,
                   "queued_at": queued_at, "queued_at_utc": datetime.fromtimestamp(
                       queued_at, timezone.utc).isoformat(),
                   "admission_wait_seconds": None, "preparation_wall_seconds": None,
                   "evaluator_wall_seconds": None, "total_wall_seconds": None,
                   "evaluator_dispatch_attempted": False, "evaluator_process_started": False,
                   "timing_scope": "Preparation includes docker command construction and any image-build callback; "
                                   "evaluator dispatch is not evidence of native physical execution."}
        # Exclusive creation is also a durable no-retry boundary. Later phase
        # updates preserve this record rather than dropping earlier timestamps.
        with ledger.open("x") as stream:
            json.dump(attempt, stream)
            stream.write("\n")
        self.last_execution = {**attempt, "ledger": str(ledger), "phase_events": str(events)}
        with events.open("x") as event_stream:
            def record(event, **values):
                at = time.time()
                attempt.update(values)
                write_json(ledger, attempt)
                self.last_execution = {**attempt, "ledger": str(ledger), "phase_events": str(events)}
                event_stream.write(json.dumps({"event": event, "at": at,
                    "at_utc": datetime.fromtimestamp(at, timezone.utc).isoformat(),
                    "elapsed_seconds": time.monotonic() - queued_clock, **values},
                    sort_keys=True, allow_nan=False) + "\n")
                event_stream.flush()

            record("queued")
            print(json.dumps({"event": "physical-queued", "attempt": str(ledger),
                              "approved_gpus": self.gpus, "share_gpu": self.share_gpu,
                              "min_gpu_free_mb": self.min_gpu_free_mb}), flush=True)
            phase, phase_clock = "admission", queued_clock
            try:
                with self.slot() as gpu:
                    admitted_clock, admitted_at = time.monotonic(), time.time()
                    scratch = output.parent / (output.name + ".scratch")
                    record("admitted", state="running", gpu=gpu, scratch=str(scratch),
                           started_at=admitted_at, admitted_at=admitted_at,
                           admitted_at_utc=datetime.fromtimestamp(admitted_at, timezone.utc).isoformat(),
                           admission_wait_seconds=admitted_clock - queued_clock)
                    phase, phase_clock = "preparation", time.monotonic()
                    record("preparation-started", preparation_started_at=time.time())
                    # LocalSmokeRuntime.docker may synchronously compile/package
                    # an image. It must finish before evaluator dispatch is counted.
                    argv = self.docker(command, gpu, scratch)
                    record("preparation-completed", preparation_finished_at=time.time(),
                           preparation_wall_seconds=time.monotonic() - phase_clock)
                    with log.open("w") as stream:
                        phase, phase_clock = "evaluator", time.monotonic()
                        record("evaluator-dispatch", evaluator_started_at=time.time(),
                               evaluator_dispatch_attempted=True, evaluator_process_started=None)
                        proc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
                    record("evaluator-exited", evaluator_finished_at=time.time(),
                           evaluator_wall_seconds=time.monotonic() - phase_clock,
                           evaluator_process_started=True, exit_code=proc.returncode)
                record("exited", state="exited", finished_at=time.time(),
                       total_wall_seconds=time.monotonic() - queued_clock)
            except Exception as error:
                timing_key = {"admission": "admission_wait_seconds", "preparation": "preparation_wall_seconds",
                              "evaluator": "evaluator_wall_seconds"}[phase]
                record("infrastructure-failed", state="infrastructure_failed", failed_phase=phase,
                       finished_at=time.time(), error=f"{type(error).__name__}: {error}",
                       total_wall_seconds=time.monotonic() - queued_clock,
                       **{timing_key: time.monotonic() - phase_clock})
                raise
        return proc.returncode

    def prepare(self, case, bundle):
        bundle = self.contained(bundle)
        if getattr(self, "reference_baseline", None) == "reference_20261004":
            from local_bench.reference_latest import validate_reference_bundle, validate_execution_proof, sha256
            selected = getattr(self, "reference_run", None)
            if selected is None:
                root = getattr(self, "reference_root", None)
                if root is None:
                    raise ValueError("Latest input requires an explicit completed reference; no fresh fallback")
                selected = Path(root) / case / "run"
            selected = Path(selected).resolve(strict=True)
            if (bundle / "benchmark_input.json").exists():
                validate_reference_bundle(bundle, case, fixed_partition=True)
                manifest = json.loads((bundle / "benchmark_input.json").read_text())
                receipt = json.loads(Path(manifest["latest_reference_receipt"]["path"]).read_text())
                if Path(receipt["source_run"]).resolve(strict=True) != selected:
                    raise ValueError("Existing latest bundle belongs to a different selected reference")
                return
            import yaml
            config = yaml.safe_load((selected / "config.yaml").read_text())
            proof = getattr(self, "reference_execution_proofs", {}).get(case)
            if not isinstance(proof, dict) or set(proof) != {"path", "sha256"}:
                raise ValueError("Latest preparation requires the independently collected execution proof")
            completion = json.loads((selected / "reference-result.json").read_text())
            execution = validate_execution_proof(proof["path"], proof["sha256"], case, selected,
                                                 completion.get("actual_docker_image_id"))
            mounts = [selected, *(Path(config[k]["config"]).resolve(strict=True)
                                 for k in ("partition", "placement")), Path(proof["path"]).resolve(strict=True)]
            mounts.extend(Path(row["path"]).resolve(strict=True) for row in execution["bindings"].values())
            command = ["/usr/bin/python3", str(self.framework / "scripts/adopt_reference_latest.py"),
                       "--reference-run", str(selected), "--case", case,
                       "--source", str(self.source), "--output", str(bundle),
                       "--execution-proof", str(Path(proof["path"]).resolve(strict=True)),
                       "--execution-proof-sha256", proof["sha256"]]
            equivalent = getattr(self, "reference_runtime_proofs", {}).get(case)
            if equivalent is not None:
                if not isinstance(equivalent, dict) or set(equivalent) != {"path", "sha256"}:
                    raise ValueError("Malformed latest runtime equivalence proof")
                path = Path(equivalent["path"]).resolve(strict=True)
                if sha256(path) != equivalent["sha256"]:
                    raise ValueError("Latest runtime equivalence proof changed")
                mounts.append(path)
                command += ["--runtime-proof", str(path), "--runtime-proof-sha256", equivalent["sha256"]]
            elif completion.get("actual_docker_image_id") != self.image:
                raise ValueError("Latest noncanonical Docker ID requires independent runtime equivalence proof")
            self.latest_reference_mounts = tuple(p for p in dict.fromkeys(mounts)
                if not any(q != p and q in p.parents for q in mounts))
            code = self.execute(command, bundle, bundle.parent / "prepare.log",
                {"kind": "latest-reference-input-adoption", "case": case, "source_commit": self.source_commit,
                 "image": self.image, "native_eda_dispatches": 0, "model_calls": 0})
            if code:
                raise RuntimeError("Latest input adoption failed; inspect preserved attempt, no automatic retry")
            validate_reference_bundle(bundle, case, fixed_partition=True)
            return
        writable_source = bundle.parent / "input-source"
        command = ["/usr/bin/python3", "-m", "local_bench", "prepare-input", str(self.profile(case)),
                   "--open3dbench-root", str(writable_source), "--output-dir", str(bundle),
                   "--python", "/usr/bin/python3", "--dreamplace-install", "/opt/dreamplace", "--timeout", "21600"]
        if getattr(self, "reference_baseline", None) == "reference_20260923":
            from local_bench.reference_baseline import validate_reference_root, validate_reference_image
            validate_reference_image(self.image)
            if not getattr(self, "reference_root", None):
                raise ValueError("Reference-aligned preparation requires --reference-root; no fresh input fallback")
            validate_reference_root(self.reference_root, case)
            command = ["/usr/bin/python3", str(self.framework / "scripts/adopt_reference_20260923.py"),
                       "--reference-root", str(self.reference_root), "--case", case,
                       "--source", str(writable_source), "--output", str(bundle)]
        if (bundle / "benchmark_input.json").exists():
            if getattr(self, "reference_baseline", None) == "reference_20260923":
                from local_bench.reference_baseline import validate_reference_bundle
                validate_reference_bundle(bundle, case, fixed_partition=True)
            # validate-input rechecks all frozen hashes without starting physical evaluations.
            check = subprocess.run(self.docker(["/usr/bin/python3", "-m", "local_bench", "validate-input", str(bundle)], self.gpus[0]))
            if check.returncode:
                raise RuntimeError("frozen input validation failed")
            return
        # Synthesis writes versions.txt under the flow directory. Each case gets
        # a private RW clone while the authoritative source remains read-only.
        bootstrap = (
            "import os,sys; from pathlib import Path; "
            "from local_bench.workspace import isolated_clone; "
            "isolated_clone(Path(sys.argv[1]), Path(sys.argv[2])); "
            "os.execv(sys.argv[3], sys.argv[3:])"
        )
        command = ["/usr/bin/python3", "-c", bootstrap, str(self.source), str(writable_source), *command]
        code = self.execute(command, bundle, bundle.parent / "prepare.log",
                            {"kind": "prepare", "case": case, "source_commit": self.source_commit, "image": self.image})
        if code or not (bundle / "benchmark_input.json").exists():
            raise RuntimeError("input preparation failed; see " + str(bundle.parent / "prepare.log"))

    def evaluate(self, task, candidate, root, input_kind, input_path, reference, case, log):
        if getattr(self, "reference_baseline", None) == "reference_20261004" and task in {
                "tp_gnn_placement_sta", "tp_gnn_placement_cts_thermal", "tp_gnn_placement_thermal"}:
            raise ValueError("Latest global evaluation uses the explicit workspace timing task, not a legacy endpoint wrapper")
        if task not in PROFILES or input_kind not in ("bundle", "run"):
            raise ValueError("unregistered physical evaluation")
        root, input_path = self.contained(root), self.contained(input_path)
        candidate = self.contained(candidate)
        if reference is not None:
            reference = self.contained(reference)
        content = json.loads((candidate / "candidate.json").read_text())
        contract = {"task": task, "candidate_sha256": digest(content), "input_kind": input_kind,
                    "input": str(input_path), "reference": str(reference), "case": case,
                    "source_commit": self.source_commit, "image": self.image,
                    "share_gpu": self.share_gpu,
                    "min_gpu_free_mb": self.min_gpu_free_mb,
                    "base_config_sha256": digest((self.profile(case) / "config.yaml").read_text())}
        result_file = root / "result/evaluation_result.json"
        saved = root / "evaluation_contract.json"
        if saved.exists() and json.loads(saved.read_text()) != contract:
            raise ValueError("cached evaluation contract changed")
        if result_file.exists():
            if not saved.exists():
                raise ValueError("result without attempt contract")
            return json.loads(result_file.read_text())
        root.mkdir(parents=True, exist_ok=True)
        write_json(saved, contract)
        write_json(root / "candidate/candidate.json", content)
        command = ["/usr/bin/python3", "-m", "local_bench.profile_cli", "evaluate", PROFILES[task],
                   "--candidate", str(root / "candidate"), "--open3dbench-root", str(self.source),
                   "--output-dir", str(root / "result"), "--base-config", str(self.profile(case) / "config.yaml"),
                   "--timeout", "86400" if task == "tp_gnn_placement_thermal" else "21600",
                   "--discard-workspace", "--discard-runtime",
                   "--benchmark-input" if input_kind == "bundle" else "--baseline-run", str(input_path)]
        if reference is not None:
            command += ["--reference-run", str(reference)]
        if task in {"tp_gnn_placement_sta", "tp_gnn_placement_cts_thermal"}:
            command = ["env", "OPEN3D_OPT_PASSES=2", *command]
        self.execute(command, root / "result", log, contract)
        if not result_file.exists():
            raise RuntimeError("attempt consumed but evaluator returned no result; inspect ledger/log, no automatic retry")
        return json.loads(result_file.read_text())
