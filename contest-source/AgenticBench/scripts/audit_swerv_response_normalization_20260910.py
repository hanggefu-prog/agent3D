"""One-off representation-only repair; emit reviewed artifacts, never overwrite a run."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "output/sta_knowledge_v2_20260910-contract-fix"
sys.path.insert(0, str(BATCH / "framework"))
module = importlib.util.spec_from_file_location("audit_driver", BATCH / "framework/scripts/run_sta_codesign_comparison.py")
d = importlib.util.module_from_spec(module)
sys.modules[module.name] = d
module.loader.exec_module(d)
from agentic3d.sta_protocol import task_spec, decode_updates
from local_bench.evolve_protocol import render_candidate_program

def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()

def recount(section):
    lines = section.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("@@ ")]
    changes, delta = [], 0
    for n, i in enumerate(starts):
        match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)\n", lines[i])
        assert match, lines[i]
        body = lines[i + 1:starts[n + 1] if n + 1 < len(starts) else len(lines)]
        assert all(line.startswith((" ", "+", "-", "\\")) for line in body)
        old = sum(line.startswith((" ", "-")) for line in body)
        new = sum(line.startswith((" ", "+")) for line in body)
        old_start = int(match[1])
        header = f"@@ -{old_start},{old} +{old_start + delta},{new} @@{match[5]}\n"
        if header != lines[i]:
            changes.append({"line": i + 1, "before": lines[i].rstrip(), "after": header.rstrip()})
        lines[i] = header
        delta += new - old
    result = "".join(lines)
    assert [x for x in section.splitlines() if not x.startswith("@@ ")] == [x for x in result.splitlines() if not x.startswith("@@ ")]
    return result, changes

def main():
    run = BATCH / "swerv_wrapper/agentic3d"
    folder = run / "rounds/03/inner/01"
    assert not (folder / "response.json").exists()
    assert not (folder / "evaluation.json").exists()
    rejected_file = folder / "response.validation.json"
    rejected = json.loads(rejected_file.read_text())["last_rejected_response"]
    plan = json.loads((run / "rounds/03/plan.json").read_text())
    incumbent = json.loads((run / "rounds/02/outer-candidate/candidate.json").read_text())
    spec = d.load_task(d.STA_PROFILE)
    selected = task_spec(plan["local_task"], spec)
    candidate = d.materialize_program(spec, rejected["program"], None)
    sections = d._patch_sections(candidate["source_patch"])
    previous = d._patch_sections(incumbent["source_patch"])
    scope = set(plan["local_task"]["source_scope"])
    assert scope == {"Place-3D/src/partition/lol/methods/tp_gnn.py"}
    removed = {}
    for path in set(sections) - scope:
        assert sections[path] == previous[path], "Cannot remove changed out-of-scope content"
        removed[path] = sha(sections[path])
    assert len(removed) == 2
    normalized = copy.deepcopy(candidate)
    parts, changes = [], {}
    for path in sorted(scope):
        section, edits = recount(sections[path])
        parts.append(section)
        changes[path] = edits
    normalized["source_patch"] = "".join(parts)
    merged = d._merge_candidate(incumbent, normalized, plan["hypothesis"], replace_paths=scope)
    original_merged = d._merge_candidate(incumbent, candidate, plan["hypothesis"], replace_paths=scope)
    normalized_original = dict(original_merged)
    normalized_original["source_patch"] = "".join(recount(section)[0] for section in d._patch_sections(original_merged["source_patch"]).values())
    # The untouched incumbent sections need not have canonical header spelling.
    assert {p: recount(s)[0] for p, s in d._patch_sections(merged["source_patch"]).items()} == d._patch_sections(normalized_original["source_patch"])
    d.verify_candidate(selected, normalized)
    d.verify_candidate(spec, merged)
    decode_updates(rejected, spec)
    audit_dir = ROOT / "tmp/sta_noop_scope_recovery_swerv_20260910"
    d._check_source_patch_applies(run / "agent-workspace", normalized["source_patch"], audit_dir / "normalized-local")
    d._check_source_patch_applies(run / "agent-workspace", merged["source_patch"], audit_dir / "normalized-merged")
    # Apply only to isolated temporary copies and parse without imports or execution.
    with tempfile.TemporaryDirectory(prefix="sta-patch-audit-") as tmp:
        tree = Path(tmp)
        for path in d._patch_sections(merged["source_patch"]):
            target = tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / "agent-workspace" / path, target)
        subprocess.run(["git", "apply", "--no-index", str(audit_dir / "normalized-merged/source-apply-check.patch")], cwd=tree, check=True, capture_output=True)
        resulting = {}
        for path in d._patch_sections(merged["source_patch"]):
            content = (tree / path).read_text()
            ast.parse(content, filename=path)
            resulting[path] = sha(content)
    response = copy.deepcopy(rejected)
    response["program"] = render_candidate_program(normalized)
    assert d.materialize_program(selected, response["program"], None) == normalized
    assert {k: v for k, v in response.items() if k != "program"} == {k: v for k, v in rejected.items() if k != "program"}
    audit = {"case": "swerv_wrapper", "round": 3, "inner_attempt": 1,
             "type": "audited_representation_only_response_normalization",
             "source": str(rejected_file.relative_to(ROOT)), "source_sha256": sha(rejected_file.read_text()),
             "removed_exact_incumbent_sections_sha256": removed, "hunk_header_changes": changes,
             "original_patch_sha256": sha(candidate["source_patch"]),
             "normalized_local_patch_sha256": sha(normalized["source_patch"]),
             "merged_patch_sha256": sha(merged["source_patch"]), "resulting_source_sha256": resulting,
             "checks": ["all removed sections byte-identical to incumbent", "every context/addition/deletion content line unchanged", "merged global patch identical after header normalization", "frozen local and global scope/limits checks passed", "strict git apply --check passed on frozen source", "isolated merged patch application and Python AST parse passed", "knowledge and hypothesis unchanged", "static candidate wrapper round-trip passed"],
             "limits": "No physical evaluation or behavioral validation performed by this audit; no extra model call; scoring, budgets, frozen framework and original rejected traces unchanged. This manual interface recovery must be disclosed in experimental reporting."}
    print(json.dumps({"response": response, "audit": audit}))

if __name__ == "__main__":
    main()
