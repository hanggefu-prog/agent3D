"""Start the task-2 Agent container and host-owned, Docker-backed evaluator."""
from __future__ import annotations

import argparse
from decimal import Decimal
import getpass
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

from grt_bench.queue import Queue
from grt_bench.common import write
from local_bench.matrix import qualify_matrix
from local_bench.matrix import LocalEvaluationMatrix
from local_bench.model import load_task
from local_bench.matrix_docker import DockerMatrixTransport

from .agent_bridge import AgentBridge
from .agent_container import command, run_agent
from .agent_workspace import agent_digest, stage_agent
from .budget import BudgetLedger
from .controller import TASK_IDS, Task2Controller
from .runtime_images import resolve, install, gpu_preflight
from .provider_preflight import check_provider


def _runtime(path, *, work_root=None, private_queue=False):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or
            set(value) - {"image_archive", "retain_physical_files"} != {"queue", "host", "image", "capacity"}):
        raise ValueError("Runtime needs queue, host, pinned image and host capacity")
    if type(value.get("retain_physical_files", False)) is not bool:
        raise ValueError("retain_physical_files must be a boolean")
    if not isinstance(value["image"], str) or not value["image"].startswith("sha256:"):
        raise ValueError("Both containers require the same pinned image digest")
    if "image_archive" in value:
        archive = path.parent / value["image_archive"]
        if archive.is_symlink() or not archive.is_file():
            raise ValueError("Organizer Docker archive is missing")
        archive = archive.resolve(strict=True)
        probe = subprocess.run(["docker", "image", "inspect", value["image"]],
                               capture_output=True, check=False)
        if probe.returncode:
            subprocess.run(["docker", "load", "-i", str(archive)], check=True)
    resolve(value["image"])
    queue = Path(value["queue"])
    if not queue.is_absolute():
        queue = (path.parent / queue).resolve()
    if private_queue:
        queue = Path(work_root) / "control/queue.sqlite"
    value["queue"] = str(queue)
    install(value["image"], work_root or queue.parent.parent)
    Queue(value["queue"]).register_host(value["host"], value["capacity"])
    return DockerMatrixTransport(queue=value["queue"], host=value["host"], image=value["image"], admission_timeout=None), value


def _port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _gateway_ready(process, url, token):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("Model gateway exited before Agent execution")
        request = urllib.request.Request(url[:-3] + "/health",
                                         headers={"Authorization": "Bearer " + token})
        try:
            with opener.open(request, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("Model gateway did not become ready")


def _provider_key(args, source, framework, root):
    key = os.environ.pop("DEEPSEEK_API_KEY", None)
    key_file = args.api_key_file or os.environ.pop("DEEPSEEK_API_KEY_FILE", None)
    if key_file:
        path = Path(key_file).expanduser().resolve(strict=True)
        protected = (source, framework, root, Path(args.data_root).resolve())
        if any(path == base or base in path.parents for base in protected):
            raise ValueError("Keep the API key file outside source, data and Agent work directories")
        key = path.read_text().strip()
    if not key and sys.stdin.isatty():
        key = getpass.getpass("DeepSeek API key: ")
    if not key or len(key) > 512 or any(c.isspace() for c in key):
        raise ValueError("A DeepSeek API key is required for the metered Agent session")
    return key


def run(args):
    framework = Path(__file__).resolve().parents[1]
    source = Path(args.source).resolve(strict=True)
    root = Path(args.work_root).resolve()
    mode = "smoke" if args.command == "smoke" else "formal"
    if mode == "formal" and (root / "DIAGNOSTIC_ONLY.json").exists():
        raise ValueError("Diagnostic work cannot be used for formal evaluation")
    if mode == "smoke" and root.exists():
        raise ValueError("Smoke mode requires a new work directory")
    root.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        write(root / "DIAGNOSTIC_ONLY.json", {"evaluation_mode": mode, "formal": False})
    controller = gateway = None
    key = None
    summary = {"baseline": args.baseline, "evaluation_mode": mode, "work_root": str(root)}
    try:
        transport, runtime = _runtime(args.runtime, work_root=root, private_queue=mode == "smoke")
        gpu_preflight(runtime["image"], root, runtime["capacity"]["gpus"])
        ledger = BudgetLedger(root / "control/budget.sqlite", evaluation_limit=10,
            model_cost_limit_rmb=Decimal("15"), task_id=args.baseline,
            allowed_model="deepseek-flash", evaluation_mode=mode)
        token = secrets.token_urlsafe(32)
        url = "http://127.0.0.1:" + str(_port()) + "/v1"
        codex = shutil.which("codex")
        if not codex:
            local_codex = Path.home() / ".local/bin/codex"
            if local_codex.is_file() and os.access(str(local_codex), os.X_OK):
                codex = str(local_codex)
        retain = getattr(args, "retain_physical_files", None)
        if retain is None:
            retain = runtime.get("retain_physical_files", False)
        summary["retain_physical_files"] = retain
        controller = Task2Controller(baseline=args.baseline, matrix_path=args.matrix,
            source_root=source, work_root=root, ledger=ledger, transport=transport,
            gateway_url=url, gateway_token=token, codex_binary=codex,
            retain_physical_files=retain)
        summary.update(matrix_cells=len(controller.matrix.cells), image=runtime["image"],
                       endpoint="post-opt-pre-cts")
        if args.check:
            summary["editable_functions"] = (["compute_binary_partition_loss"] if args.baseline == "binary_loss"
                                             else ["compute_tp_gnn_loss", "decode_tp_gnn_embeddings"])
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        key = _provider_key(args, source, framework, root)
        check_provider(key, root)
        if mode == "smoke":
            from .smoke import stage_agent as stage_smoke_agent
            agent_dir = stage_smoke_agent(controller)
        else:
            agent_dir = stage_agent(framework / "agent" / args.baseline,
                                    root / "participant-agent" / args.baseline)
        scratch = root / "scratch" / secrets.token_hex(8)
        scratch.mkdir(parents=True, exist_ok=False)
        workspaces = root / "agent-workspaces"
        workspaces.mkdir(exist_ok=True)
        name = "o3db-contest-agent-" + secrets.token_hex(8)
        controller.container_name = name
        train = {case: next(Path(cell["baseline_run"]) for cell in controller.matrix.cells
                            if cell["case"] == case) for case in controller.matrix.cases}
        base_env = dict(os.environ)
        for variable in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE", "OPENAI_API_KEY",
                         "CODEX_ACCESS_TOKEN", "CONTEST_GATEWAY_TOKEN"):
            base_env.pop(variable, None)
        base_env["PYTHONPATH"] = str(framework)
        gateway_env = {**base_env, "CONTEST_GATEWAY_TOKEN": token, "DEEPSEEK_API_KEY": key}
        write(root / "control/session-start.json", {**summary, "status": "starting"}, replace=True)
        with (root / "control/model-gateway.log").open("a") as log:
            gateway = subprocess.Popen([sys.executable, "-m", "contest2.model_gateway",
                "--ledger", str(ledger.path), "--task", args.baseline, "--evaluation-mode", mode,
                "--port", url.rsplit(":", 1)[1].split("/", 1)[0]],
                env=gateway_env, stdout=log, stderr=log)
            _gateway_ready(gateway, url, token)
            with AgentBridge(controller) as bridge:
                startup = command(name=name, image=runtime["image"], agent_dir=agent_dir,
                    workspaces=workspaces, scratch=scratch, bridge_dir=bridge.directory,
                    framework=framework, source=source, training_inputs=train,
                    evaluations=controller.reading_directory,
                    cpus=args.agent_cpu, memory_gb=args.agent_memory_gb, gpu=args.agent_gpu)
                candidate = run_agent(startup_command=startup, name=name, scratch=scratch)
        passed = {row["candidate_id"] for row in ledger.snapshot()["evaluations"]
                  if row["status"] == "succeeded"}
        if candidate not in passed:
            raise ValueError("Agent returned a candidate without a complete verified matrix")
        if mode == "smoke":
            from .smoke import verify_loop
            summary["smoke"] = verify_loop(controller, agent_dir, candidate)
        summary.update(status="succeeded", candidate_id=candidate, budget=ledger.snapshot())
    except BaseException as error:
        write(root / "session-result.json", {**summary, "status": "failed",
              "error_type": type(error).__name__, "message": str(error)[:2000]},
              replace=(root / "session-result.json").exists())
        raise
    finally:
        if gateway is not None:
            gateway.terminate()
            try:
                gateway.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait()
        if controller is not None:
            controller.close()
        key = None
    summary["services_stopped"] = True
    write(root / "session-result.json", summary, replace=(root / "session-result.json").exists())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def validate(args):
    """One public or private pass after freezing a completely scored program."""
    framework = Path(__file__).resolve().parents[1]
    transport, runtime = _runtime(args.runtime)
    root = Path(args.work_root).resolve(strict=True)
    if (root / "DIAGNOSTIC_ONLY.json").exists():
        raise ValueError("Diagnostic work cannot enter formal validation")
    ledger = BudgetLedger(root / "control/budget.sqlite", evaluation_limit=10,
                          model_cost_limit_rmb=Decimal("15"), task_id=args.baseline,
                          allowed_model="deepseek-flash")
    if (not isinstance(args.candidate_id, str) or len(args.candidate_id) != 64 or
            any(char not in "0123456789abcdef" for char in args.candidate_id)):
        raise ValueError("Validation needs a frozen candidate ID")
    candidate = root / "candidates" / args.candidate_id
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("Training-qualified candidate is missing")
    if args.candidate_id not in {row["candidate_id"] for row in ledger.snapshot()["evaluations"]
                                 if row["status"] == "succeeded"}:
        raise ValueError("Candidate has no complete scored training matrix")
    agent = root / "participant-agent" / args.baseline
    frozen_agent_digest = agent_digest(agent)
    task = framework / "local_tasks" / TASK_IDS[args.baseline]
    matrix = LocalEvaluationMatrix(args.matrix, load_task(task))
    public = ["swerv_wrapper-ng45", "bp_fe-asap7"]
    if args.command == "validate":
        from .halo_policy import validate_matrix_halos
        validate_matrix_halos(matrix)
        from .sampling import validate_public_matrix
        validate_public_matrix(matrix, args.baseline, "validation")
        from .scoring import validate_scoring_references
        validate_scoring_references(matrix, args.baseline)
        valid_cases = matrix.cases == public
    else:
        valid_cases = (len(matrix.cases) == 2 and
                       not set(matrix.cases) & set(("bp_fe-ng45", "bp-ng45", "ariane133-ng45", *public)))
    if (not valid_cases or len(matrix.cells) != 6 or
            [seed["partition"] for seed in matrix.seeds] != [1001, 1002, 1003]):
        raise ValueError("Validation matrix case or seed contract differs")
    from .runtime_recovery import recover_provider
    recover_provider(ledger)
    # Validate the actual capsule before changing the irreversible phase state.
    from local_bench.model import load_candidate
    from local_tasks.tp_gnn_local_task_definitions import validate_candidate
    validate_candidate(load_candidate(candidate), TASK_IDS[args.baseline])
    ledger.freeze(args.candidate_id, frozen_agent_digest)
    from .storage import TrainingStorageTransport
    retain = getattr(args, "retain_physical_files", None)
    if retain is None:
        retain = runtime.get("retain_physical_files", False)
    result = matrix.evaluate(candidate, args.source, root / args.command / args.candidate_id,
        transport=TrainingStorageTransport(transport, root / args.command, retain_physical_files=retain),
        workers=transport.workers, timeout_seconds=None)
    print(json.dumps({"candidate_id": args.candidate_id, "result": result.to_dict()},
                     indent=2, ensure_ascii=False))
    from .scoring import complete_score
    return 0 if complete_score(result.metrics, matrix) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "smoke", "qualify", "validate", "final"))
    parser.add_argument("--baseline", choices=sorted(TASK_IDS), required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "hidden"))
    parser.add_argument("--input-index", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--api-key-file", type=Path,
                        help="Host-only DeepSeek key file, outside source/data/work directories")
    parser.add_argument("--agent-cpu", type=int, default=4)
    parser.add_argument("--agent-memory-gb", type=int, default=32)
    parser.add_argument("--agent-gpu")
    parser.add_argument("--retain-physical-files", action=argparse.BooleanOptionalAction,
                        default=None, help="保留物理中间文件，默认按 runtime.json 配置或清理")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Deprecated compatibility option; native execution has no time limit")
    args = parser.parse_args(argv)
    if args.command in {"check", "run", "smoke", "validate", "final"}:
        source = Path(args.source or Path(__file__).resolve().parents[2]).resolve(strict=True)
        data = Path(args.data_root or source.parent / "contest-data" / "task2")
        args.source = source
        args.data_root = data
        args.runtime = args.runtime or data / "runtime.json"
        args.matrix = args.matrix or data / args.baseline / (
            "validation/qualified_matrix.json" if args.command == "validate"
            else "train/qualified_matrix.json" if args.command in {"check", "run", "smoke"}
            else "hidden/qualified_matrix.json")
        args.work_root = args.work_root or (source.parent / "contest-work" / "smoke" /
            (args.baseline + "-" + secrets.token_hex(8)) if args.command == "smoke"
            else source.parent / "contest-work" / "task2" / args.baseline)
    if args.command == "prepare":
        if not all((args.split, args.input_index, args.work_root)):
            parser.error("prepare requires --split, --input-index and --work-root")
        from .prepare_matrix import render
        print(render(baseline=args.baseline, split=args.split,
                     input_index=args.input_index, output=args.work_root))
        return 0
    if not all((args.matrix, args.runtime, args.source, args.work_root)):
        parser.error("check, run and qualify require matrix, runtime, source and work-root")
    if args.command in {"validate", "final"}:
        if not args.candidate_id:
            parser.error("validate/final requires --candidate-id")
        return validate(args)
    if args.command == "qualify":
        transport, _ = _runtime(args.runtime)
        framework = Path(__file__).resolve().parents[1]
        from .scoring import PUBLIC_SPLITS
        from .sampling import validate_public_matrix
        candidate_matrix = LocalEvaluationMatrix(args.matrix,
            load_task(framework / "local_tasks" / TASK_IDS[args.baseline]), references_required=False)
        for split, cases in PUBLIC_SPLITS.items():
            if candidate_matrix.cases == list(cases):
                validate_public_matrix(candidate_matrix, args.baseline, split)
        output = args.work_root / "qualification" / args.baseline
        result = qualify_matrix(args.matrix, framework / "local_tasks" / TASK_IDS[args.baseline],
            args.source, output, transport=transport, workers=transport.workers, timeout_seconds=None)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if result.metrics.get("runs_successfully") == 1 else 1
    args.check = args.command == "check"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
