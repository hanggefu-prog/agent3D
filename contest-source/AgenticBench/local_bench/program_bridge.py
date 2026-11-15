from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Mapping, Optional

from .evolve_protocol import parse_candidate_program, render_candidate_program
from .model import TaskSpec, validate_candidate_payload


def initial_program_text(spec: TaskSpec, initial_candidate: Mapping[str, Any]) -> str:
    """Return the exact program evolved by EvoStage/OpenEvolve for this task."""

    if spec.program_mode == "candidate_literal":
        return render_candidate_program(
            dict(initial_candidate),
            docstring="Return one static OpenAgentic3DBench candidate.",
        )
    assert spec.program_template is not None
    text = spec.program_template.read_text(encoding="utf-8")
    if "EVOLVE-BLOCK-START" not in text or "EVOLVE-BLOCK-END" not in text:
        raise ValueError("task-adapter program template must contain EVOLVE-BLOCK markers")
    return text


def _load_trusted_adapter(spec: TaskSpec) -> ModuleType:
    assert spec.program_adapter is not None
    digest = hashlib.sha256(str(spec.program_adapter).encode("utf-8")).hexdigest()[:16]
    module_spec = importlib.util.spec_from_file_location(
        "openagentic3d_task_adapter_{}".format(digest), spec.program_adapter
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError("cannot load task program adapter: {}".format(spec.program_adapter))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def materialize_program(
    spec: TaskSpec,
    program_text: str,
    open3dbench_root: Optional[Path],
) -> Dict[str, Any]:
    """Parse an untrusted evolved program through a trusted task-owned adapter."""

    if spec.program_mode == "candidate_literal":
        return parse_candidate_program(program_text)
    if open3dbench_root is None:
        raise ValueError("task-adapter programs require an Open3DBench checkout")
    module = _load_trusted_adapter(spec)
    materialize = getattr(module, "materialize", None)
    if not callable(materialize):
        raise TypeError("task program adapter must define materialize(program_text, open3dbench_root)")
    value = materialize(program_text, open3dbench_root.expanduser().resolve())
    return validate_candidate_payload(value, source="task-adapter candidate")
