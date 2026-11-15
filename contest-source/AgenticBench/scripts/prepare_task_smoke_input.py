#!/usr/bin/env python3
"""Prepare one exact task input in private, credential-free Docker storage.

One bundle can serve compatible tasks. Preparation is never a candidate round.
Historical absolute timing paths are replaced by a separately versioned portable
profile, not by relabeling old measurements. Existing outputs are never retried.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic3d.knowledge import write_json
from local_bench.benchmark_input import load_benchmark_input_profile, prepare_benchmark_input, validate_benchmark_input
from local_bench.workspace import isolated_clone
from scripts.codesign_runtime import Runtime


class PreparationRuntime(Runtime):
    write_root = None

    def docker(self, command, gpu, scratch=None):
        if self.write_root is None:
            raise ValueError("Private preparation directory is required")
        writable = self.contained(self.write_root)
        if writable == self.batch:
            raise ValueError("Shared batch root cannot be writable")
        argv = super().docker(command, gpu, scratch)
        index = argv.index(str(self.batch) + ":" + str(self.batch))
        argv[index] += ":ro"
        index = argv.index(self.image)
        argv[index:index] = ["-e", "OPEN3DBENCH_IMAGE_ID=" + self.image,
                             "-v", str(writable) + ":" + str(writable),
                             "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
        return argv


def prepare_inside(source, profile_id, output):
    """Prepare trusted profile data without historical task-specific patches."""
    source, output = Path(source).resolve(), Path(output).resolve()
    profile = load_benchmark_input_profile(profile_id)
    work = output.parent / "input-source"
    isolated_clone(source, work)
    write_json(output.parent / "preparation_contract.json", {
        "profile": profile.profile_id, "config": str(profile.config),
        "source": str(source), "private_source": str(work), "trusted_patch": None,
        "policy": "Fresh preparation; no candidate evaluation and no automatic retry"})
    return prepare_benchmark_input(profile_id, open3dbench_root=work, output_dir=output,
                                   python_executable=Path("/usr/bin/python3"),
                                   dreamplace_install=Path("/opt/dreamplace"), timeout_seconds=21600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--gpu")
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=48)
    parser.add_argument("--min-gpu-free-mb", type=int, default=2048)
    args = parser.parse_args(argv)
    profile = load_benchmark_input_profile(args.profile)
    if args.inside:
        print(json.dumps(prepare_inside(args.source, profile.profile_id, args.output), indent=2))
        return 0
    if any(value is None for value in (args.batch_root, args.image, args.gpu)):
        parser.error("Host preparation requires --batch-root, --image and --gpu")
    runtime = PreparationRuntime(args.source, args.batch_root, args.image, args.gpu,
                                 args.cpus, args.memory_gb, share_gpu=True,
                                 min_gpu_free_mb=args.min_gpu_free_mb)
    output = runtime.contained(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_root = output.parent
    command = ["/usr/bin/python3", str(Path(__file__).resolve()), "--inside",
               "--source", str(runtime.source), "--profile", profile.profile_id,
               "--output", str(output)]
    code = runtime.execute(command, output, output.parent / "prepare.console.log", {
        "kind": "task-smoke-input", "profile": profile.profile_id,
        "source_commit": runtime.source_commit, "image": runtime.image})
    if code:
        raise RuntimeError("Input preparation failed; inspect " + str(output.parent / "prepare.console.log"))
    report = validate_benchmark_input(output)
    print(json.dumps({"event": "input-qualified", "validation": report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
