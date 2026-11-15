#!/usr/bin/env python3
"""Export a task and run native OpenEvolve with release-container evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK))

from local_bench.benchmark_input import validate_benchmark_input
from local_bench.cli import main as task_main
from local_bench.model import load_task, resolve_task_path
from scripts.configure_openevolve_deepseek import main as configure_deepseek


def below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--benchmark-input", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--open3dbench-root", type=Path, default=FRAMEWORK.parent)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--image", default=os.environ.get("IMAGE", "shiyunqi/open3dbench-v2.0:release"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--gpu", default="0", help="GPU index, or none for CPU routing")
    parser.add_argument("--model", help="Override the configured DeepSeek model")
    parser.add_argument("--max-tokens", type=int, help="Override the model output budget (including reasoning tokens)")
    parser.add_argument("--llm-timeout", type=int, help="Override the model request timeout in seconds")
    parser.add_argument("--plan-only", action="store_true", help="Export and inspect without Docker or model calls")
    args = parser.parse_args(argv)
    source = args.open3dbench_root.expanduser().resolve()
    data_root = (args.data_root or Path(os.environ.get("AGENTICBENCH_DATA_ROOT", str(source / "user_runs")))).expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    frozen = args.benchmark_input.expanduser().resolve()
    reference = args.reference_run.expanduser().resolve()
    spec = load_task(resolve_task_path(FRAMEWORK / "local_tasks", args.task))
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.llm_timeout is not None and args.llm_timeout < 1:
        parser.error("--llm-timeout must be finite and positive")
    for label, path in (("input", frozen), ("reference", reference), ("output", output)):
        if not below(path, data_root):
            parser.error("{} must be below --data-root: {}".format(label, data_root))
    if output.exists():
        parser.error("output exists; choose a fresh output directory")
    if not reference.is_dir():
        parser.error("reference run directory is missing")
    if not args.plan_only:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            parser.error("set DEEPSEEK_API_KEY in the current environment")
        if shutil.which("openevolve-run") is None:
            parser.error("install OpenEvolve into the controller environment")
    validate_benchmark_input(frozen, task_id=spec.task_id)
    if not args.plan_only:
        status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
        if status.strip():
            parser.error("commit source changes before evaluating an isolated checkout")
    # The evaluator containers all mount data_root at the same absolute path,
    # so this run-owned ccache persists across candidate containers only.
    output.mkdir(parents=True, exist_ok=False)
    compiler_cache = output / "openroad-ccache"
    compiler_cache.mkdir(mode=0o700)
    bundle = output / "bundle"
    task_main([
        "export-openevolve", args.task, "--benchmark-profile", args.profile,
        "--open3dbench-root", str(source), "--output-dir", str(bundle),
    ])
    saved_argv = sys.argv
    try:
        sys.argv = ["configure_openevolve_deepseek.py", str(bundle / "config.yaml")]
        configure_deepseek()
    finally:
        sys.argv = saved_argv
    config = yaml.safe_load((bundle / "config.yaml").read_text())
    config["max_iterations"] = args.iterations
    if args.model:
        config["llm"]["models"] = [{"name": args.model, "weight": 1.0}]
        config["llm"]["evaluator_models"] = [{"name": args.model, "weight": 1.0}]
    if args.max_tokens is not None:
        config["llm"]["max_tokens"] = args.max_tokens
    if args.llm_timeout is not None:
        config["llm"]["timeout"] = args.llm_timeout
    (bundle / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(FRAMEWORK) + os.pathsep + env.get("PYTHONPATH", "")
    container_env = {
        "HOME": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "{}:{}:/opt/dreamplace:/opt/dreamplace/dreamplace".format(FRAMEWORK, source),
        "OPEN3D_LOCAL_DREAMPLACE_INSTALL": "/opt/dreamplace",
        "PLACE3D_DREAMPLACE_INSTALL": "/opt/dreamplace",
        "OPEN3D_LOCAL_OPENROAD_BUILDER": str(FRAMEWORK / "scripts/build_openroad_candidate.sh"),
        "CCACHE_DIR": str(compiler_cache),
        "CCACHE_MAXSIZE": "8G",
        "CCACHE_COMPILERCHECK": "content",
        "OPEN3DBENCH_IMAGE": args.image,
    }
    if not args.plan_only:
        container_env["OPEN3DBENCH_IMAGE_ID"] = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
        ).strip()
    if env.get("USE_LOCAL_OPENROAD") == "1":
        for variable, suffix in (
            ("OPENROAD_EXE", "bin/openroad"), ("OPENSTA_EXE", "src/sta/sta")
        ):
            binary = source / "build/openroad-3d/cmake" / suffix
            if not binary.is_file():
                parser.error("missing local build: {}".format(binary))
            container_env[variable] = str(binary)
    # A linked Git worktree stores its administrative gitdir outside the
    # checkout. Evaluator containers need that read-only metadata to clone an
    # isolated candidate workspace; mount the common .git directory alongside
    # the source tree when applicable.
    evaluation_mounts = ["{}:{}:ro".format(source, source), "{}:{}".format(data_root, data_root)]
    if (source / ".git").is_file():
        for option in ("--git-dir", "--git-common-dir"):
            # Git < 2.31 has no --path-format; a relative answer is relative to -C.
            directory = (source / subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", option],
                text=True,
            ).strip()).resolve(strict=True)
            mount = "{}:{}:ro".format(directory, directory)
            if not below(directory, source) and mount not in evaluation_mounts:
                evaluation_mounts.append(mount)
    env.update({
        "OPEN3DBENCH_ROOT": str(source),
        "OPENAGENTIC3D_BENCHMARK_INPUT": str(frozen),
        "OPEN3DBENCH_REFERENCE_RUN": str(reference),
        "OPEN3D_LOCAL_EVAL_ROOT": str(output / "evaluations"),
        "OPENAGENTIC3D_EVAL_DOCKER_IMAGE": args.image,
        "OPENAGENTIC3D_EVAL_DOCKER_GPU": "" if args.gpu == "none" else args.gpu,
        # Match the host user that owns the mounted worktree and cache. This
        # prevents Git's dubious-ownership check inside the evaluator image.
        "OPENAGENTIC3D_EVAL_DOCKER_USER": "{}:{}".format(os.getuid(), os.getgid()),
        "OPENAGENTIC3D_EVAL_DOCKER_WORKDIR": str(FRAMEWORK),
        "OPENAGENTIC3D_EVAL_DOCKER_MOUNTS_JSON": json.dumps(evaluation_mounts),
        "OPENAGENTIC3D_EVAL_DOCKER_ENV_JSON": json.dumps(container_env),
        "OPENAGENTIC3D_EVAL_COMPACT_ARTIFACTS": "1",
    })
    command = [
        "openevolve-run", str(bundle / "initial_program.py"),
        str(bundle / "evaluator.py"), "--config", str(bundle / "config.yaml"),
        "--output", str(output / "openevolve"),
    ]
    plan = {"task": spec.task_id, "profile": args.profile, "iterations": args.iterations,
            "image": args.image, "gpu": args.gpu, "command": command,
            "container_environment": container_env, "mounts": evaluation_mounts}
    (output / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
