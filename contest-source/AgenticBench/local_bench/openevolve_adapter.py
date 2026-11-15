from __future__ import annotations

import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import fcntl

from .benchmark_input import resolve_benchmark_run
from .evaluation import evaluate_task
from .model import EvaluationResult, load_candidate, load_task
from .program_bridge import materialize_program
from .verifier import verify_candidate


def _materialize_candidate(
    task_dir: Path,
    program_path: Path,
    evaluation_dir: Path,
    open3dbench_root: Optional[Path] = None,
) -> Path:
    spec = load_task(task_dir)
    if program_path.name == "candidate.json":
        candidate = load_candidate(program_path.parent)
    else:
        candidate = materialize_program(
            spec,
            program_path.read_text(encoding="utf-8"),
            open3dbench_root,
        )
    verified = verify_candidate(spec, candidate, open3dbench_root)["candidate"]
    candidate_dir = evaluation_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "candidate.json").write_text(
        json.dumps(verified, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return candidate_dir


def _cached_result(path: Path) -> Optional[EvaluationResult]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
        return None
    return EvaluationResult(metrics=data["metrics"], artifacts=data.get("artifacts", {}))


def _context_digest(
    task_dir: Path,
    program_path: Path,
    open3dbench_root: Path,
    base_config: Optional[Path] = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(program_path.read_bytes())
    for path in sorted(task_dir.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            name = path.relative_to(task_dir).as_posix()
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    harness_root = Path(__file__).resolve().parent
    for path in sorted(harness_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    spec = load_task(task_dir)
    for relative in sorted(spec.allowed_source_paths):
        source = open3dbench_root / relative
        if source.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(source.read_bytes())
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(open3dbench_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = ""
    context = {
        "open3dbench_root": str(open3dbench_root),
        "open3dbench_head": head,
        "baseline_run": os.environ.get("OPEN3DBENCH_BASELINE_RUN"),
        "benchmark_input": os.environ.get("OPENAGENTIC3D_BENCHMARK_INPUT"),
        "reference_run": os.environ.get("OPEN3DBENCH_REFERENCE_RUN"),
        "timeout": os.environ.get("OPEN3D_LOCAL_EVAL_TIMEOUT", "21600"),
        "base_config": (
            str(base_config)
            if base_config is not None
            else os.environ.get("OPENAGENTIC3D_BASE_CONFIG")
        ),
    }
    benchmark_input_value = context["benchmark_input"]
    if benchmark_input_value:
        benchmark_input_path = Path(str(benchmark_input_value)).expanduser().resolve()
        benchmark_manifest = (
            benchmark_input_path / "benchmark_input.json"
            if benchmark_input_path.is_dir()
            else benchmark_input_path
        )
        if not benchmark_manifest.is_file():
            raise FileNotFoundError(
                "OPENAGENTIC3D_BENCHMARK_INPUT: {}".format(benchmark_manifest)
            )
        digest.update(benchmark_manifest.read_bytes())
    base_config_value = (
        str(base_config)
        if base_config is not None
        else os.environ.get("OPENAGENTIC3D_BASE_CONFIG")
    )
    if base_config_value:
        path = Path(base_config_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("OPENAGENTIC3D_BASE_CONFIG: {}".format(path))
        digest.update(path.read_bytes())
    digest.update(json.dumps(context, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _json_environment(name: str, expected_type):
    raw = os.environ.get(name)
    if not raw:
        return expected_type()
    value = json.loads(raw)
    if not isinstance(value, expected_type):
        raise ValueError("{} must contain JSON {}".format(name, expected_type.__name__))
    return value


@contextmanager
def _gpu_evaluation_lock() -> Iterator[None]:
    """Optionally serialize real evaluations assigned to one physical GPU.

    Independent OpenEvolve processes may generate candidates concurrently, but
    two DREAMPlace containers must not share one GPU.  A launch harness opts in
    by setting ``OPENAGENTIC3D_EVAL_GPU_LOCK_DIR``; ordinary single-run use is
    unchanged.  POSIX advisory locks are released automatically if a worker is
    interrupted or exits unexpectedly.
    """

    lock_root = os.environ.get("OPENAGENTIC3D_EVAL_GPU_LOCK_DIR")
    gpu = os.environ.get("OPENAGENTIC3D_EVAL_DOCKER_GPU")
    if not lock_root or not gpu:
        yield
        return
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.," for character in gpu):
        raise ValueError("OPENAGENTIC3D_EVAL_DOCKER_GPU contains unsafe characters")
    root = Path(lock_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "gpu-{}.lock".format(gpu)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _docker_evaluation_command(
    task_dir: Path,
    candidate_path: Path,
    open3dbench_root: Path,
    result_dir: Path,
    base_config: Optional[Path] = None,
) -> List[str]:
    """Build the opt-in outer-Python to EDA-runtime evaluation command."""

    image = os.environ.get("OPENAGENTIC3D_EVAL_DOCKER_IMAGE")
    if not image:
        raise RuntimeError("OPENAGENTIC3D_EVAL_DOCKER_IMAGE is not set")
    mounts = _json_environment("OPENAGENTIC3D_EVAL_DOCKER_MOUNTS_JSON", list)
    container_env = _json_environment("OPENAGENTIC3D_EVAL_DOCKER_ENV_JSON", dict)
    if any(not isinstance(item, str) or not item for item in mounts):
        raise ValueError("OPENAGENTIC3D_EVAL_DOCKER_MOUNTS_JSON must contain strings")
    if any(
        not isinstance(key, str) or not isinstance(value, (str, int, float, bool))
        for key, value in container_env.items()
    ):
        raise ValueError("OPENAGENTIC3D_EVAL_DOCKER_ENV_JSON must contain scalar values")

    command = ["docker", "run", "--rm"]
    docker_user = os.environ.get("OPENAGENTIC3D_EVAL_DOCKER_USER")
    if docker_user:
        command.extend(["--user", docker_user])
    gpu = os.environ.get("OPENAGENTIC3D_EVAL_DOCKER_GPU")
    if gpu:
        command.extend(["--gpus", "device={}".format(gpu)])
    for mount in mounts:
        command.extend(["-v", mount])
    for key, value in sorted(container_env.items()):
        command.extend(["-e", "{}={}".format(key, value)])
    workdir = os.environ.get("OPENAGENTIC3D_EVAL_DOCKER_WORKDIR")
    if workdir:
        command.extend(["-w", workdir])

    command.extend(
        [
            image,
            os.environ.get("OPENAGENTIC3D_EVAL_DOCKER_PYTHON", "python3"),
            "-m",
            "local_bench",
            "evaluate",
            str(task_dir),
            "--candidate",
            str(candidate_path),
            "--open3dbench-root",
            str(open3dbench_root),
            "--output-dir",
            str(result_dir),
            "--timeout",
            os.environ.get("OPEN3D_LOCAL_EVAL_TIMEOUT", "21600"),
        ]
    )
    benchmark_input = os.environ.get("OPENAGENTIC3D_BENCHMARK_INPUT")
    baseline_run = os.environ.get("OPEN3DBENCH_BASELINE_RUN")
    reference_run = os.environ.get("OPEN3DBENCH_REFERENCE_RUN")
    if benchmark_input:
        command.extend(["--benchmark-input", benchmark_input])
    elif baseline_run:
        command.extend(["--baseline-run", baseline_run])
    if reference_run:
        command.extend(["--reference-run", reference_run])
    base_config_value = (
        str(base_config)
        if base_config is not None
        else os.environ.get("OPENAGENTIC3D_BASE_CONFIG")
    )
    if base_config_value:
        command.extend(["--base-config", base_config_value])
    if os.environ.get("OPENAGENTIC3D_EVAL_COMPACT_ARTIFACTS", "0") == "1":
        command.append("--compact-artifacts")
    elif os.environ.get("OPENAGENTIC3D_EVAL_KEEP_WORKSPACE", "0") != "1":
        command.append("--discard-workspace")
    return command


def _to_openevolve_result(result: EvaluationResult, task_id: Optional[str] = None):
    from .optimization_feedback import TASK_IDS, optimization_result
    if task_id in TASK_IDS:
        # Project only at the model boundary, after the full receipt is saved.
        result = optimization_result(result, task_id=task_id)
    try:
        from openevolve.evaluation_result import EvaluationResult as OpenEvolveResult
    except ImportError:
        return result.to_dict()
    artifacts = {
        key: value if isinstance(value, (str, bytes)) else json.dumps(value, ensure_ascii=False)
        for key, value in result.artifacts.items()
    }
    return OpenEvolveResult(metrics=result.metrics, artifacts=artifacts)


def _evaluate_via_docker(
    task_dir: Path,
    candidate_path: Path,
    open3dbench_root: Path,
    output_root: Path,
    base_config: Optional[Path] = None,
):
    digest = _context_digest(task_dir, candidate_path, open3dbench_root, base_config)
    evaluation_dir = output_root / task_dir.name / digest
    result_dir = evaluation_dir / "result"
    result = _cached_result(result_dir / "evaluation_result.json")
    if result is None:
        result_dir.mkdir(parents=True, exist_ok=True)
        try:
            # OpenEvolve writes each candidate program to a host-side temporary
            # file.  That path is intentionally not mounted into the EDA
            # container.  Parse and verify the program in the outer process,
            # then pass the materialized candidate directory through the
            # explicitly mounted evaluation root.
            candidate_dir = _materialize_candidate(
                task_dir, candidate_path, evaluation_dir, open3dbench_root
            )
            command = _docker_evaluation_command(
                task_dir, candidate_dir, open3dbench_root, result_dir, base_config
            )
            timeout = int(os.environ.get("OPEN3D_LOCAL_EVAL_TIMEOUT", "21600")) + 300
            with _gpu_evaluation_lock():
                process = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=timeout,
                )
            result = _cached_result(result_dir / "evaluation_result.json")
            if result is None:
                result = EvaluationResult(
                    metrics={"combined_score": 0.0, "runs_successfully": 0.0},
                    artifacts={
                        "error_type": "DockerEvaluationError",
                        "error_message": "container evaluator exited with code {}".format(
                            process.returncode
                        ),
                        "container_output_tail": process.stdout[-12000:],
                        "container_command": command,
                    },
                )
        except Exception as error:
            result = EvaluationResult(
                metrics={"combined_score": 0.0, "runs_successfully": 0.0},
                artifacts={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "cascade": [
                        {"name": "program_materialization", "status": "failed"}
                    ],
                },
            )
        if not (result_dir / "evaluation_result.json").is_file():
            (result_dir / "evaluation_result.json").write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return _to_openevolve_result(result, load_task(task_dir).task_id)


def evaluate_from_task_dir(
    task_dir_value: str,
    program_path: str,
    base_config_value: Optional[str] = None,
):
    """Bridge OpenEvolve's evaluate(program_path) API to the directory task API."""

    task_dir = Path(task_dir_value).expanduser().resolve()
    candidate_path = Path(program_path).resolve()
    root_value = os.environ.get("OPEN3DBENCH_ROOT")
    output_root = os.environ.get("OPEN3D_LOCAL_EVAL_ROOT")
    if not root_value or not output_root:
        raise RuntimeError("OPEN3DBENCH_ROOT and OPEN3D_LOCAL_EVAL_ROOT must be set")
    if not candidate_path.is_file():
        raise ValueError("OpenEvolve must pass a candidate program or candidate.json file")
    open3dbench_root = Path(root_value).expanduser().resolve()
    base_config = (
        Path(base_config_value).expanduser().resolve()
        if base_config_value
        else Path(os.environ["OPENAGENTIC3D_BASE_CONFIG"]).expanduser().resolve()
        if os.environ.get("OPENAGENTIC3D_BASE_CONFIG")
        else None
    )
    if os.environ.get("OPENAGENTIC3D_EVAL_DOCKER_IMAGE"):
        return _evaluate_via_docker(
            task_dir,
            candidate_path,
            open3dbench_root,
            Path(output_root).expanduser().resolve(),
            base_config,
        )
    digest = _context_digest(task_dir, candidate_path, open3dbench_root, base_config)
    evaluation_dir = Path(output_root) / task_dir.name / digest
    result_dir = evaluation_dir / "result"
    result = _cached_result(result_dir / "evaluation_result.json")
    if result is None:
        try:
            candidate_dir = _materialize_candidate(
                task_dir, candidate_path, evaluation_dir, open3dbench_root
            )
            benchmark_input = os.environ.get("OPENAGENTIC3D_BENCHMARK_INPUT")
            baseline_run = (
                resolve_benchmark_run(
                    Path(benchmark_input), task_id=load_task(task_dir).task_id
                )
                if benchmark_input
                else Path(os.environ["OPEN3DBENCH_BASELINE_RUN"])
                if os.environ.get("OPEN3DBENCH_BASELINE_RUN")
                else None
            )
            result = evaluate_task(
                task_dir,
                candidate_dir,
                open3dbench_root,
                result_dir,
                baseline_run=baseline_run,
                reference_run=Path(os.environ["OPEN3DBENCH_REFERENCE_RUN"]) if os.environ.get("OPEN3DBENCH_REFERENCE_RUN") else None,
                base_config=base_config,
                timeout_seconds=int(os.environ.get("OPEN3D_LOCAL_EVAL_TIMEOUT", "21600")),
            )
        except Exception as error:
            result = EvaluationResult(
                metrics={"combined_score": 0.0, "runs_successfully": 0.0},
                artifacts={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "cascade": [{"name": "program_materialization", "status": "failed"}],
                },
            )
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "evaluation_result.json").write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return _to_openevolve_result(result, load_task(task_dir).task_id)


def evaluate_from_task_file(task_evaluator_file: str, program_path: str):
    return evaluate_from_task_dir(str(Path(task_evaluator_file).resolve().parent), program_path)
