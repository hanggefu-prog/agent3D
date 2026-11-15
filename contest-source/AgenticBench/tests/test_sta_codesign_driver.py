import tempfile
import json
import ast
import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest
from scripts import run_sta_codesign_comparison as driver

from local_bench.evaluation import EVALUATOR_SOURCE_PATCHES
from scripts.run_sta_codesign_comparison import (
    REMOTE_COMPARISON,
    REFERENCE_RUN,
    _check_source_patch_applies,
    _merge_candidate,
    _patch_sections,
)


def test_comparison_uses_exact_cutpoint_reference():
    assert REFERENCE_RUN.endswith("/reference-v8-exact-cutpoint-two-pass/run")
    assert REMOTE_COMPARISON.endswith("/comparison-v8-exact-cutpoint-two-pass")


def test_new_case_cannot_inherit_bp_multi_inputs():
    args = driver.build_parser().parse_args([
        "--mode", "codex", "--output-dir", "/tmp/not-created", "--gpu", "0", "--case", "ariane133"
    ])
    with pytest.raises(ValueError, match="own bundle"):
        driver.configure_remote(args)


def test_explicit_proxy_is_used_for_control_and_candidate_upload(tmp_path, monkeypatch):
    proxy = "docker exec -i nju-easyconnect busybox nc %h %p"
    monkeypatch.setattr(driver, "SSH_PROXY_COMMAND", proxy)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(driver.subprocess, "run", run)
    driver._remote_candidate(tmp_path, "/isolated/candidate")
    assert [command[0] for command in commands] == ["ssh", "scp"]
    for command in commands:
        assert "ProxyCommand=" + proxy in command
        assert "BatchMode=yes" in command
        assert "ConnectTimeout=15" in command


def test_new_case_launcher_preserves_case_reference_and_five_round_budget():
    from scripts.launch_sta_codesign_case import command_for
    command, output = command_for("ariane133", "amax3", "agentic3d", 0)
    assert command[command.index("--rounds") + 1] == "5"
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert command[command.index("--remote-host") + 1] == "amax3-gcr"
    assert "/sta-cases/ariane133/reference/run" in command[command.index("--reference-run") + 1]
    assert output.parts[-2:] == ("ariane133", "agentic3d")
    assert "--ssh-proxy-command" in command
    assert "--resume" not in command
    resumed, resumed_output = command_for("ariane133", "amax3", "agentic3d", 0, resume=True)
    assert resumed_output == output and resumed[-1] == "--resume"


def test_case_reference_gate_checks_profile_and_exact_sta():
    bundle = {"benchmark_id": "ariane133_lol_ng45_tp_gnn_sta"}
    metrics = dict(runs_successfully=1, opt_passes=2, sta_graph_complete=1,
                   sta_unconstrained_endpoints=0, final_global_wns_setup_ns=-2.,
                   final_global_tns_setup_ns=-30., combined_score=1.)
    driver.validate_case_reference("ariane133", bundle, {"metrics": metrics})
    with pytest.raises(ValueError, match="another case"):
        driver.validate_case_reference("bp", bundle, {"metrics": metrics})
    for key, invalid in (("opt_passes", 1), ("sta_graph_complete", 0),
                         ("sta_unconstrained_endpoints", 1), ("combined_score", 1.1),
                         ("final_global_wns_setup_ns", float("nan"))):
        with pytest.raises(ValueError):
            driver.validate_case_reference("ariane133", bundle, {"metrics": {**metrics, key: invalid}})


@pytest.mark.parametrize("case", ["bp", "ariane133", "swerv_wrapper"])
def test_new_case_profiles_keep_the_matched_sta_protocol(case):
    import yaml
    from local_bench.benchmark_input import load_benchmark_input_profile
    profile = load_benchmark_input_profile(case + "_lol_ng45_tp_gnn_sta")
    config = yaml.safe_load(profile.config.read_text())
    assert config["design"] == case
    assert config["type"] == "lol"
    assert config["placement"]["gpu"] is True
    assert config["placement"]["target_density"] == .6
    assert set(profile.compatible_tasks) >= {"tp_gnn_placement_sta", "tp_gnn_sta_inner", "placement_sta_inner"}


def test_case_prompts_regenerate_all_three_contracts_and_freeze_on_resume(tmp_path, monkeypatch):
    seen = []
    def render(spec, candidate, workspace):
        seen.append(spec.benchmark_input_id)
        assert "ariane133_lol_ng45_tp_gnn_sta" in str(spec.base_config)
        return spec.task_id + ":" + spec.benchmark_input_id
    monkeypatch.setattr(driver, "build_problem_description", render)
    args = SimpleNamespace(case="ariane133", agent_workspace=tmp_path)
    driver.write_case_prompts(args, tmp_path)
    assert seen == ["ariane133_lol_ng45_tp_gnn_sta"] * 3
    assert "bp_multi" not in args.instructions.read_text()
    assert args.initial_program.is_file()
    driver.write_case_prompts(args, tmp_path)
    monkeypatch.setattr(driver, "build_problem_description", lambda *args: "changed code")
    with pytest.raises(ValueError, match="during resume"):
        driver.write_case_prompts(args, tmp_path)


def test_sta_codesign_has_a_trusted_two_pass_evaluator_patch():
    patch = EVALUATOR_SOURCE_PATCHES["tp_gnn_placement_sta"]
    text = patch.read_text(encoding="utf-8")
    assert "OPEN3D_OPT_PASSES" in text
    assert '"pass_records": pass_records' in text
    assert "passive_cutpoint_query_exact" in text


def test_trusted_two_pass_patch_applies_to_current_core_without_changing_core(tmp_path):
    root = Path(__file__).resolve().parents[2]
    relative = Path("open3dbench/backend/three_d_opt.py")
    source = root / relative
    original = source.read_bytes()
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    patch = EVALUATOR_SOURCE_PATCHES["tp_gnn_placement_sta"]
    assert patch == EVALUATOR_SOURCE_PATCHES["tp_gnn_placement_cts_thermal"]
    # Check and apply against the current bytes, not a stale textual fixture.
    for options in (("--check",), ()):
        result = subprocess.run(["git", "apply", *options, str(patch)], cwd=tmp_path,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    text = target.read_text()
    compile(text, str(target), "exec")
    tree = ast.parse(text)
    optimize = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "optimize")
    pass_loops = [node for node in ast.walk(optimize) if isinstance(node, ast.For)
                  and isinstance(node.target, ast.Name) and node.target.id == "pass_number"]
    assert len(pass_loops) == 1
    loop = pass_loops[0]
    assert isinstance(loop.iter, ast.Call) and loop.iter.func.id == "range"
    assert isinstance(loop.iter.args[0], ast.Constant) and loop.iter.args[0].value == 1
    assert isinstance(loop.iter.args[1], ast.BinOp) and loop.iter.args[1].left.id == "passes"
    assert loop.iter.args[1].right.value == 1
    assert 'os.environ.get("OPEN3D_OPT_PASSES", "1")' in text
    assert '"pass_records": pass_records' in text
    assert '"die_passes": {"bottom": passes, "top": passes}' in text
    assert "copy_function=copy_with_log_reference" in text
    assert '"authoritative_timing_view": "continuous-passive-hbt"' in text
    assert source.read_bytes() == original


def test_remote_eval_helper_requires_cutpoint_capable_openroad():
    helper = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "open3dbench-sta-evaluation"
        / "scripts"
        / "run_remote_sta.sh"
    ).read_text(encoding="utf-8")

    assert "docker-sta-20260902/docker/runtime-refresh/openroad" in helper
    assert "get_parasitic_cutpoint_timing=get_parasitic_cutpoint_timing" in helper
    assert '-e OPENROAD_EXE="$sta_openroad"' in helper
    assert "-e OPEN3D_OPT_PASSES=2" in helper
    assert '-e OPENROAD_EXE="$runtime/openroad-agentic3d"' not in helper
    assert '--base-config "$profile_config"' in helper.split('if [[ "$input_kind" == bundle ]]')[0]
    assert 'profile_config=${9:-' in helper


def test_legacy_remote_helper_is_only_a_skill_alias():
    helper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_sta_codesign_remote_eval.sh"
    ).read_text(encoding="utf-8")
    assert "skills/open3dbench-sta-evaluation/scripts/run_remote_sta.sh" in helper


def _patch(path: str, old: str, new: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-{old}\n"
        f"+{new}\n"
    )


def test_merge_replaces_complete_specialist_scope_and_preserves_other_scope():
    incumbent = {
        "source_patch": _patch("partition.py", "p0", "p1")
        + _patch("placement_a.py", "a0", "a1")
        + _patch("placement_b.py", "b0", "b1")
    }
    correction = {"source_patch": _patch("placement_b.py", "b0", "b2")}

    merged = _merge_candidate(
        incumbent,
        correction,
        "replace placement candidate",
        replace_paths={"placement_a.py", "placement_b.py"},
    )
    sections = _patch_sections(merged["source_patch"])

    assert set(sections) == {"partition.py", "placement_b.py"}
    assert "+p1" in sections["partition.py"]
    assert "+b2" in sections["placement_b.py"]


def test_patch_sections_round_trips_multi_file_diff():
    patch = _patch("a.py", "a", "aa") + _patch("b.py", "b", "bb")
    sections = _patch_sections(patch)
    assert list(sections) == ["a.py", "b.py"]
    assert "diff --git a/a.py b/a.py" in sections["a.py"]
    assert "diff --git a/b.py b/b.py" in sections["b.py"]


def test_source_apply_check_rejects_corrupt_hunk_before_remote_evaluation():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "a.py").write_text("a\n", encoding="utf-8")
        evidence = root / "evidence"
        _check_source_patch_applies(root, _patch("a.py", "a", "aa"), evidence)
        corrupt = _patch("a.py", "a", "aa").replace("@@ -1 +1 @@", "@@ -1,9 +1 @@")
        with pytest.raises(ValueError, match="does not apply"):
            _check_source_patch_applies(root, corrupt, evidence / "corrupt")


def test_resume_history_rejects_gaps_and_budget_overrun(tmp_path):
    assert driver._history(tmp_path, 5) == []
    (tmp_path / "history.json").write_text(json.dumps([{"round": 2}]))
    with pytest.raises(ValueError, match="contiguous"):
        driver._history(tmp_path, 5)
    (tmp_path / "history.json").write_text(json.dumps([{"round": i} for i in range(1, 7)]))
    with pytest.raises(ValueError, match="budget"):
        driver._history(tmp_path, 5)


def test_resume_recovers_existing_analyzer_plan_without_model_call(tmp_path):
    call = tmp_path / "analyzer/calls/002"
    call.mkdir(parents=True)
    (call / "prompt.md").write_text("Select one bounded specialist task for outer round 2.")
    plan = {"target": "partition", "max_inner_attempts": 2}
    (call / "result.json").write_text(json.dumps({"success": True, "final_message": json.dumps(plan)}))
    def unexpected(*args):
        pytest.fail("must not regenerate a saved plan")
    analyzer = SimpleNamespace(root=tmp_path / "analyzer", ask=unexpected)
    assert driver._round_plan(analyzer, tmp_path, 2, "new") == plan
    assert (tmp_path / "rounds/02/plan.json").is_file()


def test_resume_reuses_matching_remote_result_without_evaluating(tmp_path, monkeypatch):
    candidate = {"schema_version": 1, "source_patch": ""}
    (tmp_path / "candidate.json").write_text(json.dumps(candidate))
    monkeypatch.setattr(driver, "_remote_exists", lambda path: True)
    result = {"metrics": {"combined_score": 1.1}}
    monkeypatch.setattr(driver, "_remote_result", lambda path: candidate if path.endswith("candidate.json") else result)
    def unexpected(*args):
        pytest.fail("must not launch an already completed evaluation")
    monkeypatch.setattr(driver, "_remote_candidate", unexpected)
    actual, run = driver.evaluate_remote(task="tp_gnn_placement_sta", local_candidate=tmp_path,
        remote_root="/round-01", input_kind="bundle", input_path="/input", reference_run="/reference",
        gpu=0, retention="full", log_path=tmp_path / "result.log")
    assert actual == result
    assert run == "/round-01/result/run"
    (tmp_path / "candidate.json").write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError, match="different candidate"):
        driver.evaluate_remote(task="tp_gnn_placement_sta", local_candidate=tmp_path,
            remote_root="/round-01", input_kind="bundle", input_path="/input", reference_run="/reference",
            gpu=0, retention="full", log_path=tmp_path / "result.log")


def test_codex_resume_only_spends_remaining_outer_budget(tmp_path, monkeypatch):
    history = [{"round": i, "score": 1.1 if i == 3 else 1., "accepted_as_incumbent": i == 3}
               for i in range(1, 5)]
    (tmp_path / "history.json").write_text(json.dumps(history))
    best = tmp_path / "rounds/03/candidate"
    best.mkdir(parents=True)
    (best / "program.py").write_text("incumbent")
    instructions = tmp_path / "instructions.md"
    instructions.write_text("initial")
    args = SimpleNamespace(instructions=instructions, model="gpt-5.6-sol", codex_bin="codex",
        agent_workspace=tmp_path, rounds=5, initial_program=instructions, reference_run="ref", gpu=0)
    monkeypatch.setattr(driver, "PersistentCodex", lambda *args: None)
    calls = []
    def candidate(agent, spec, prompt, destination):
        assert '"incumbent_program": "incumbent"' in prompt
        assert destination.parent.name == "05"
        return "new", {}, {"hypothesis": "new"}
    monkeypatch.setattr(driver, "_program_candidate", candidate)
    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"metrics": {"runs_successfully": 1, "combined_score": 1.2}}, "run"
    monkeypatch.setattr(driver, "evaluate_remote", evaluate)
    driver.run_codex_baseline(args, tmp_path)
    assert len(calls) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["final_sta_evaluations"] == 5
    assert json.loads((tmp_path / "history.json").read_text())[:4] == history
