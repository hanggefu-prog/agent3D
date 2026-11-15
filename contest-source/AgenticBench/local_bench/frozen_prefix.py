"""Validate a replay's frozen input/floorplan without pretending it reran them."""
import json
from pathlib import Path
from typing import Optional

from .benchmark_input import _sha256, _validate_stage_manifest


PREFIX = {"input": "00_input", "floorplan": "10_floorplan"}


def validate_frozen_prefix(run_dir: Path, frozen_input_run: Optional[Path]) -> None:
    # The source is supplied by the trusted caller, never discovered through a
    # candidate-authored path. Old replay artifacts need no mutation or rerun.
    if frozen_input_run is None:
        raise ValueError("frozen-prefix replay requires its trusted frozen input run")
    root, baseline = run_dir.resolve(), frozen_input_run.resolve()
    if root == baseline:
        raise ValueError("frozen-prefix replay cannot use itself as frozen input")
    run = json.loads((root / "run.json").read_text())
    for stage, directory in PREFIX.items():
        source, copied = baseline / directory, root / directory
        if copied.is_symlink():
            raise ValueError("frozen replay stage must not be a directory symlink: " + stage)
        manifest = source / "stage_manifest.json"
        payload = _validate_stage_manifest(manifest, stage)
        if not isinstance(payload.get("signature"), dict):
            raise ValueError("frozen stage is missing its signature: " + stage)
        for key, run_key in (("design", "design"), ("pdk", "pdk"), ("integration", "type")):
            if not run.get(run_key) or payload.get(key) != run[run_key]:
                raise ValueError("frozen stage case/platform mismatch: " + stage)
        copied_manifest = copied / "stage_manifest.json"
        if not copied_manifest.is_file() or _sha256(manifest) != _sha256(copied_manifest):
            raise ValueError("replay changed frozen stage manifest: " + stage)
        for record in payload["artifacts"].values():
            artifact = (copied / record["path"]).resolve()
            try:
                artifact.relative_to(copied.resolve())
            except ValueError as error:
                raise ValueError("replay artifact escapes frozen stage") from error
            if not artifact.is_file():
                raise ValueError("missing frozen replay artifact: " + str(artifact))
            if record["path"] == "design_params.json":
                original = json.loads((source / "design_params.json").read_text())
                actual = json.loads(artifact.read_text())
                expected = dict(original)
                for key, filename in (("def_input", "design.def"),
                                      ("verilog_input", "design.v"), ("sdc_input", "design.sdc")):
                    target = copied / filename
                    if target.is_file():
                        expected[key] = str(target.resolve())
                if actual != original and actual != expected:
                    raise ValueError("replay changed frozen design parameters: " + stage)
            elif artifact.stat().st_size != record["bytes"] or _sha256(artifact) != record["sha256"]:
                raise ValueError("replay changed frozen artifact: " + str(artifact))
