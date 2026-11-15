"""Host-frozen optimization contracts and baseline-relative source verifier.

The prompt front matter selects components; prose never grants permissions.
This is a source/contract verifier, not a proof of algorithmic equivalence.
"""
import ast
import copy
from pathlib import Path
import re

import yaml

from .candidates import HBT_ALGORITHM, verify_scope, verify_snapshot
from .common import digest, read
from .algorithm_scope import FUNCTION_SCOPE, function_scope, placement_python, verify_functions

OBJECTIVE = "paired-ratio-weighted-timing-power-cap-v2"
TIMING_OBJECTIVE = "paired-timing-minmax-overflow-penalty-v3"
CPP_SUFFIXES = (".cc", ".cpp", ".h", ".hh", ".hpp")
COMPONENTS = {
    "partition": (("Place-3D/src/partition/lol/methods/", (".py",)),),
    "placement": (("Place-3D/src/placement/concurrent_dreamplace.py", (".py",)),),
    "hbt": ((HBT_ALGORITHM, (".py",)),),
    "timing_opt": (("OpenROAD-3D/openroad/src/rsz/src/", CPP_SUFFIXES),
                   ("OpenROAD-3D/openroad/src/rsz/include/", CPP_SUFFIXES)),
    "cts": (("OpenROAD-3D/openroad/src/cts/src/", CPP_SUFFIXES),
            ("OpenROAD-3D/openroad/src/cts/include/", CPP_SUFFIXES)),
    "grt": (("OpenROAD-3D/openroad/src/grt/src/", CPP_SUFFIXES),
            ("OpenROAD-3D/openroad/src/grt/include/", CPP_SUFFIXES)),
}


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a SHA-256 identity")
    return value


def belongs(name, component):
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        return False
    if component == 'placement' and placement_python(name):
        return True
    if component == 'partition' and name in FUNCTION_SCOPE and name.startswith('Place-3D/src/partition/'):
        return True
    return any((name.startswith(prefix) if prefix.endswith("/") else name == prefix)
               and Path(name).suffix in suffixes for prefix, suffixes in COMPONENTS[component])


def prompt_scope(prompt):
    lines = prompt.splitlines()
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        raise ValueError("Prompt requires YAML front matter with components")
    end = lines.index("---", 1)
    scope = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(scope, dict) or set(scope) - {"components", "files"}:
        raise ValueError("Prompt supports only components and optional narrower files")
    components = scope.get("components")
    if (not isinstance(components, list) or not components or
            any(not isinstance(c, str) or c not in COMPONENTS for c in components) or
            len(set(components)) != len(components)):
        raise ValueError("Select a nonempty subset of: " + ", ".join(COMPONENTS))
    if not "\n".join(lines[end + 1:]).strip():
        raise ValueError("Prompt must describe the optimization task")
    if "files" in scope and (not isinstance(scope["files"], list) or not scope["files"] or
                            any(not isinstance(f, str) for f in scope["files"])):
        raise ValueError("files must be a nonempty list")
    return scope


def validate_optimization(spec):
    if set(spec) != {"schema_version", "baseline", "prompt", "components", "allowed_files", "objective", "retention"}:
        raise ValueError("Unknown/missing optimization contract fields")
    if spec["schema_version"] != 1 or spec["objective"] not in {OBJECTIVE, TIMING_OBJECTIVE} or spec["retention"] != "all-nondominated":
        raise ValueError("Unsupported optimization contract")
    identity(spec["baseline"])
    scope = prompt_scope(spec["prompt"])
    if spec["components"] != sorted(scope["components"]):
        raise ValueError("Prompt scope differs from frozen components")
    files = spec["allowed_files"]
    if not isinstance(files, list) or not files or any(not isinstance(f, str) for f in files) or files != sorted(set(files)):
        raise ValueError("Expected sorted exact allowed files")
    if "files" in scope and sorted(set(scope["files"])) != files:
        raise ValueError("Prompt file restrictions differ")
    for name in files:
        if not any(belongs(name, c) for c in spec["components"]):
            raise ValueError("File outside maintained component boundary: " + name)
    for component in spec["components"]:
        if not any(belongs(name, component) for name in files):
            raise ValueError("Selected component has no editable files: " + component)


def define_task(base, baseline, prompt, store):
    from .contract import validate, timing_only
    identity(baseline)
    manifest = verify_snapshot(Path(store) / baseline)
    if manifest["parent"] is not None or manifest["scope"] is not None or manifest["policy"] is not None:
        raise ValueError("Task baseline must be a host-imported root snapshot")
    scope = prompt_scope(prompt)
    files = sorted(f for f, row in manifest["files"].items() if "symlink" not in row
                   and any(belongs(f, c) for c in scope["components"]))
    if "files" in scope:
        if not set(scope["files"]) <= set(files):
            raise ValueError("Prompt files must exist inside the selected components")
        files = sorted(set(scope["files"]))
    task = copy.deepcopy(base)
    task["optimization"] = {"schema_version": 1, "baseline": baseline, "prompt": prompt,
        "components": sorted(scope["components"]), "allowed_files": files,
        "objective": TIMING_OBJECTIVE if timing_only(task) else OBJECTIVE, "retention": "all-nondominated"}
    validate(task)
    return task


def edit_policy(task):
    from .contract import validate
    result = {"allowed_paths": task["optimization"]["allowed_files"],
            "native": any(name.startswith("OpenROAD-3D/openroad/")
                          for name in task["optimization"]["allowed_files"]),
            "allow_create": False, "allow_delete": False, "task_digest": validate(task)}
    limits = function_scope(result['allowed_paths'])
    if limits:
        result['function_scope'] = limits
    return result


def verify_candidate(task, store, candidate):
    """Hash actual candidate bytes; audit every content-addressed ancestry edge.

    Ancestor manifests suffice for scope diffs: their SHA identities bind their
    inventories. Rehashing all ancestor trees per matrix cell is unnecessary.
    """
    from .contract import validate
    task_id = validate(task)
    identity(candidate)
    baseline = task["optimization"]["baseline"]
    store = Path(store)
    current = verify_snapshot(store / candidate)
    policy = edit_policy(task)
    lineage = []
    node, name = current, candidate
    while name != baseline:
        if name in {n for n, _ in lineage}:
            raise ValueError("Cyclic candidate ancestry")
        identity(node.get("parent"))
        if node.get("policy") != policy:
            raise ValueError("Candidate policy differs from frozen task")
        lineage.append((name, node))
        name = node["parent"]
        node = read(store / name / "candidate.json")
        if digest(node) != name:
            raise ValueError("Ancestor manifest changed")
    if node.get("parent") is not None or node.get("policy") is not None or node.get("scope") is not None:
        raise ValueError("Invalid baseline root")
    if not set(policy["allowed_paths"]) <= set(node["files"]):
        raise ValueError("Editable files absent from baseline")
    before, native = node["files"], False
    for _, child in reversed(lineage):
        audit = verify_scope(before, child["files"], policy)
        native |= audit["native_build_required"]
        audit["native_build_required"] = native
        if child.get("scope") != audit:
            raise ValueError("Forged scope/native build receipt")
        before = child["files"]
    # Check every ancestry edge, not only the final cumulative edit.
    for child_id, child in lineage:
        verify_functions(store / child['parent'] / 'tree', store / child_id / 'tree', policy)
    cumulative = verify_scope(node["files"], current["files"], policy)
    if candidate != baseline and not cumulative["changes"]:
        raise ValueError("Candidate has no changes relative to baseline")
    for relative in cumulative["changes"]:
        if relative.endswith(".py"):
            ast.parse((store / candidate / "tree" / relative).read_bytes(), filename=relative)
    return {"status": "legal", "task_digest": task_id, "candidate": candidate,
            "baseline": baseline, "parent": current["parent"],
            "changes": cumulative["changes"], "native_build_required": native,
            "policy_digest": digest(policy), "ancestry": [n for n, _ in lineage] + [baseline]}
