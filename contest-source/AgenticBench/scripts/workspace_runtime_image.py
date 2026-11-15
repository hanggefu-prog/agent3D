"""Explicit installed-image binding for the latest global workspace tasks.

Canonical scientific identity and actual Docker identity are recorded separately.
No tag lookup, image pull, conversion, or implicit fallback occurs here.
"""
from pathlib import Path

from local_bench import reference_latest as latest
from scripts.verify_release import checked_json, verify


def bind_release(path, sha256, *, source, canonical):
    if canonical != latest.IMAGE_ID:
        raise ValueError("Workspace release requires the canonical reference identity")
    binding = {"path": str(Path(path).absolute()), "sha256": sha256}
    data = checked_json(binding)
    result = verify(binding["path"], sha256, requested_source=source)
    return {"release": binding, "canonical_image": canonical,
            "actual_image": result["runtime_image"],
            "source": str(Path(source).absolute()),
            "framework_sha256": data["framework_sha256"],
            "equivalence": data["runtime"].get("equivalence")}


def execution_image(runtime):
    binding = getattr(runtime, "workspace_runtime_image", None)
    if binding is None:
        return runtime.image
    current = bind_release(binding["release"]["path"], binding["release"]["sha256"],
                           source=runtime.source, canonical=runtime.image)
    if current != binding:
        raise ValueError("Workspace installed runtime changed after admission")
    if getattr(runtime, "reference_baseline", None) != latest.BASELINE_ID:
        raise ValueError("Installed-image equivalence is only supported for the latest family")
    return binding["actual_image"]
