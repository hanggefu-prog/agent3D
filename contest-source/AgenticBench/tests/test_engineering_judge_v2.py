"""Judge v2: independence, redaction, quoted citations and a median panel.

No model call or container is started; the workspace turn and docker are faked.
"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engineering_bench" / "physical"))

from engineering_bench import model
from engineering_bench.judge import (DEFAULT_FAMILIES, aggregate_samples, check_judge_independence,
    legality_checks, load_families, model_family, prepare_review_view, redact_text, redact_value,
    review_prompt, validate_review, validate_families)
from engineering_bench.submission import inventory, seal
from engineering_bench.tasks import load_task


def review(evidence, status="graded", score=3):
    return json.dumps({"status": status, "score": score, "summary": "Review",
        "strengths": [], "issues": [{"explanation": "Finding", "evidence": evidence}] if evidence else [],
        "unverified": []})


# --- independence -----------------------------------------------------------

@pytest.mark.parametrize("name,family", [("gpt-5.6-sol", "openai"), ("openai/o3-pro", "openai"),
    ("codex-mini", "openai"), ("claude-opus-4-1", "anthropic"), ("gemini-2.5-pro", "google"),
    ("Llama-3.1-70B", "meta"), ("o3db", None), ("", None), ("unknown-model", None)])
def test_model_family(name, family):
    assert model_family(name) == family


def test_independence_fails_closed_and_records_override():
    record = check_judge_independence("claude-opus-4", "gpt-5.6-sol")
    assert record["independent"] and record["override_reason"] is None
    with pytest.raises(ValueError, match="same model family"):
        check_judge_independence("gpt-5", "codex-mini")
    with pytest.raises(ValueError, match="Unknown model family"):
        check_judge_independence("house-model", "gpt-5")
    with pytest.raises(ValueError, match="non-empty"):
        check_judge_independence("gpt-5", "gpt-4o", allow_same_family_reason="  ")
    record = check_judge_independence("gpt-5", "gpt-4o", allow_same_family_reason="only licensed judge")
    assert not record["independent"] and record["override_reason"] == "only licensed judge"


def test_configurable_family_map(tmp_path):
    path = tmp_path / "families.json"
    path.write_text(json.dumps({"lab-a": ["alpha"], "lab-b": ["beta"]}))
    families = load_families(path)
    assert check_judge_independence("alpha-2", "beta-1", families)["independent"]
    with pytest.raises(ValueError, match="Unknown"):
        check_judge_independence("gpt-5", "beta-1", families)
    with pytest.raises(ValueError, match="two families"):
        validate_families({"a": ["x"], "b": ["x"]})
    with pytest.raises(ValueError, match="lowercase"):
        validate_families({"a": ["X"]})
    assert load_families() is DEFAULT_FAMILIES


# --- redaction ----------------------------------------------------------------

def test_redaction_is_deterministic_and_line_preserving():
    text = ("# drafted by gpt-5.6-sol and Claude-Opus-4.1 / o3-mini\n"
            "x = o3db_path  # Co-Authored-By: Claude Code <noreply@anthropic.com>\n"
            "thread_id=019a-abc session 0c6f1b2e-1111-2222-3333-444455556666\n"
            "gpt = 3; codex_path = 1; claude_bridge(); mail me@example.org\n"
            "private: house-model-x\n")
    out = redact_text(text, ["house-model-x"])
    assert out == redact_text(text, ["house-model-x"])
    assert out.count("\n") == text.count("\n")
    lines = out.split("\n")
    assert lines[0] == "# drafted by [REDACTED-MODEL] and [REDACTED-MODEL] / [REDACTED-MODEL]"
    assert lines[1] == "x = o3db_path  # Co-Authored-By: [REDACTED-AUTHOR]"
    assert lines[2] == "thread_id=[REDACTED-ID] session [REDACTED-ID]"
    # Ordinary identifiers survive; only versioned model names are touched.
    assert lines[3] == "gpt = 3; codex_path = 1; claude_bridge(); mail [REDACTED-EMAIL]"
    assert lines[4] == "private: [REDACTED-MODEL]"
    assert redact_value({"k": ["gpt-5"], "n": 1}) == {"k": ["[REDACTED-MODEL]"], "n": 1}


def test_review_view_redacts_changed_files_only(tmp_path):
    subject = tmp_path / "subject"
    (subject / "engineering").mkdir(parents=True)
    (subject / ".git").mkdir()
    (subject / ".git" / "config").write_text("author = someone\n")
    (subject / "base.py").write_text("# gpt-4 baseline comment\n")
    changed = subject / "engineering/clock_compensation.py"
    changed.write_text("# written with gpt-5.6-sol\ndef plan():\n    return []\n")
    before = changed.read_text()
    receipt = prepare_review_view(subject, ["engineering/clock_compensation.py"], tmp_path / "view", ["x"])
    view = tmp_path / "view"
    assert receipt["redacted_paths"] == ["engineering/clock_compensation.py"]
    assert not (view / ".git/config").read_text().count("author")
    assert sorted(p.name for p in (view / ".git").iterdir()) == ["HEAD", "config", "objects", "refs"]
    assert (view / "engineering/clock_compensation.py").read_text().splitlines()[0] == "# written with [REDACTED-MODEL]"
    assert (view / "base.py").read_text() == "# gpt-4 baseline comment\n"
    assert changed.read_text() == before


# --- citations ------------------------------------------------------------------

def test_v2_citations_verify_quoted_text(tmp_path):
    (tmp_path / "source.py").write_text("def plan_clock(context):\n    return []\n")
    good = [{"path": "source.py", "line": 1, "quote": "def plan_clock(context):"}]
    assert validate_review(review(good), [tmp_path], version=2)["score"] == 3
    # Whitespace differences are tolerated; whole short line is allowed.
    assert validate_review(review([{"path": "source.py", "line": 2, "quote": "return  []"}]), [tmp_path], version=2)
    for bad, message in [
            ([{"path": "source.py", "line": 2, "quote": "def plan_clock(context):"}], "does not match"),
            ([{"path": "source.py", "line": 1, "quote": "def"}], "does not match"),
            ([{"path": "source.py", "line": 1, "quote": " "}], "non-empty"),
            ([{"path": "source.py", "line": 1}], "quote"),
            ([{"path": "source.py", "line": 9, "quote": "def plan_clock"}], "does not match")]:
        with pytest.raises(ValueError, match=message):
            validate_review(review(bad), [tmp_path], version=2)
    # Version 1 replies (without quotes) remain valid for old receipts.
    assert validate_review(review([{"path": "source.py", "line": 1}]), [tmp_path])["score"] == 3
    with pytest.raises(ValueError):
        validate_review(review(good), [tmp_path])


def test_prompts_by_version():
    task = load_task("clock_compensation")
    assert '"quote"' in review_prompt(task, 2) and '"quote"' not in review_prompt(task)
    with pytest.raises(ValueError):
        review_prompt(task, 3)


# --- panel ------------------------------------------------------------------------

def test_median_and_dispersion():
    rows = [{"status": "graded", "score": s} for s in (2, 4, 5)]
    panel = aggregate_samples(rows)
    assert panel["score"] == 4 and panel["dispersion"]["range"] == 3
    assert panel["dispersion"]["median_absolute_deviation"] == 1
    assert panel["weight_in_headline"] == 0
    rows[1] = {"status": "ungradable", "score": None}
    assert aggregate_samples(rows)["status"] == "insufficient_samples"
    assert aggregate_samples(rows)["score"] is None
    with pytest.raises(ValueError):
        aggregate_samples(rows, 2)


def test_legality_status_map_never_passes_unverified():
    checks = {"a": {"passed": True}, "b": {"passed": False},
              "c": {"status": "unverified", "passed": False},
              "d": {"status": "passed", "passed": True},
              "e": {"status": "passed", "passed": None}}
    state = legality_checks(checks)
    assert state["status_map"] == {"a": "passed", "b": "failed", "c": "unverified", "d": "passed", "e": "unverified"}
    assert state["unverified"] == ["c", "e"] and not state["all_passed"]


def sealed_subject(tmp_path):
    task = load_task("clock_compensation")
    subject = tmp_path / "subject"
    (subject / ".git").mkdir(parents=True)
    (subject / "AGENTS.md").write_text("guide\n")
    before = inventory(subject)
    (subject / "engineering").mkdir()
    (subject / "engineering/clock_compensation.py").write_text(
        "# drafted with gpt-5.6-sol\ndef plan_clock_compensation(context):\n    return []\n")
    sealed = seal(task, subject, before, tmp_path / "sealed.json")
    return task, subject, sealed


def fake_turns(monkeypatch, replies):
    calls = []
    def turn(**kwargs):
        calls.append(kwargs)
        reply = replies[len(calls) - 1]
        kwargs["output"].mkdir(parents=True)
        (kwargs["output"] / "final.txt").write_text(reply or "")
        return {"success": reply is not None, "final_message": reply or ""}
    monkeypatch.setattr(model, "run_workspace_turn", turn)
    return calls


def test_panel_samples_fresh_sessions_and_records_outcomes(tmp_path, monkeypatch):
    task, subject, sealed = sealed_subject(tmp_path)
    quote = [{"path": "engineering/clock_compensation.py", "line": 2,
              "quote": "def plan_clock_compensation(context):"}]
    calls = fake_turns(monkeypatch, [review(quote, score=2), None, review(quote, score=4),
                                     review([{"path": "engineering/clock_compensation.py", "line": 2, "quote": "wrong text here"}])])
    panel = model.judge_panel(task, subject, sealed, model="claude-opus-4", developer_model="gpt-5.6-sol",
        evidence_roots=[subject], output=tmp_path / "panel", samples=4, binary="x", control=tmp_path,
        container="c", terminal_script=tmp_path, python="py")
    assert [c["thread_id"] for c in calls] == [None] * 4
    assert all('"quote"' in c["prompt"] for c in calls)
    assert [o["status"] for o in panel["sample_outcomes"]] == ["graded", "ungradable", "graded", "invalid"]
    assert panel["status"] == "insufficient_samples" and panel["score"] is None
    assert json.loads((tmp_path / "panel/panel.json").read_text())["independence"]["independent"]
    with pytest.raises(ValueError, match="three"):
        model.judge_panel(task, subject, sealed, model="claude-opus-4", developer_model="gpt-5",
            evidence_roots=[subject], output=tmp_path / "p2", samples=2)
    with pytest.raises(ValueError, match="same model family"):
        model.judge_panel(task, subject, sealed, model="gpt-5", developer_model="gpt-5",
            evidence_roots=[subject], output=tmp_path / "p3")
    assert not (tmp_path / "p3").exists()


def test_judge_task_v2_end_to_end_without_docker(tmp_path, monkeypatch):
    import judge_task
    task, subject, sealed = sealed_subject(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setattr(judge_task, "layout_path", lambda root, name: root / name)
    monkeypatch.setattr(judge_task, "verify_release", lambda path: (base, "sha256:" + "0" * 64))
    docker = []
    class Done:
        returncode, stdout, stderr = 0, "judge source read-only\n", ""
    monkeypatch.setattr(judge_task.subprocess, "run", lambda cmd, **kw: docker.append(cmd) or Done())
    final = tmp_path / "final"
    native = {"checks": {"physical_equivalence": {"capability_schema": 2, "status": "passed", "passed": True},
                         "electrical_limits": {"capability_schema": 2, "status": "unverified", "passed": False}}}
    (final / "ledger/bp").mkdir(parents=True)
    (final / "ledger/bp/result.json").write_text(json.dumps({"submission_sha256": sealed["submission_sha256"],
        "status": "succeeded", "legal": True, "error": None}))
    (final / "bp").mkdir()
    (final / "bp/final-native.json").write_text(json.dumps(native))
    view_line = "def plan_clock_compensation(context):"
    reply = review([{"path": "engineering/clock_compensation.py", "line": 2, "quote": view_line}], score=3)
    prompts = fake_turns(monkeypatch, [reply, reply.replace('"score": 3', '"score": 5'), reply])
    args = ["--root", str(tmp_path), "--experiment", str(tmp_path), "--subject", str(subject),
            "--sealed", str(tmp_path / "sealed.json"), "--final-root", str(final), "--task", "clock_compensation",
            "--judge-model", "claude-opus-4", "--review-version", "2", "--developer-model", "gpt-5.6-sol"]
    with pytest.raises(ValueError, match="same model family"):
        judge_task.main(args[:-2] + ["--developer-model", "claude-sonnet-4", "--output", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists() and not docker
    summary = judge_task.main(args + ["--output", str(tmp_path / "out")])
    out = tmp_path / "out"
    assert summary["status"] == "graded" and summary["score"] == 3 and summary["weight_in_headline"] == 0
    assert summary["dispersion"]["max"] == 5 and summary["independence"]["judge_family"] == "anthropic"
    # Three fresh containers, each started, probed and stopped.
    assert sum(1 for c in docker if c[:2] == ["docker", "stop"]) == 3
    assert len(json.loads((out / "container-dispatch.json").read_text())["samples"]) == 3
    view = (out / "subject-view/engineering/clock_compensation.py").read_text()
    assert "gpt-5.6-sol" not in view and view.splitlines()[1] == view_line
    for path in (out / "control/submission.diff", out / "results/legality.json"):
        assert "gpt-5.6-sol" not in path.read_text() and "claude-opus-4" not in path.read_text()
    for prompt in prompts:
        assert "gpt-5.6-sol" not in prompt["prompt"]
    legality = json.loads((out / "results/legality.json").read_text())
    row = [r for r in legality["cases"] if r["case"] == "bp"][0]
    assert row["checks"]["electrical_limits"] == "unverified" and row["unverified_checks"] == ["electrical_limits"]
    receipt = json.loads((out / "redaction-receipt.json").read_text())
    assert receipt["redacted_paths"] == ["engineering/clock_compensation.py"]
    assert "gpt-5.6-sol" not in json.dumps(receipt)


def test_judge_task_v1_path_kept(tmp_path, monkeypatch):
    import judge_task
    task, subject, sealed = sealed_subject(tmp_path)
    monkeypatch.setattr(judge_task, "layout_path", lambda root, name: root / name)
    monkeypatch.setattr(judge_task, "verify_release", lambda path: (tmp_path, "sha256:" + "0" * 64))
    class Done:
        returncode, stdout, stderr = 0, "", ""
    monkeypatch.setattr(judge_task.subprocess, "run", lambda cmd, **kw: Done())
    prompts = fake_turns(monkeypatch, [review([{"path": "engineering/clock_compensation.py", "line": 2}], score=4)])
    summary = judge_task.main(["--root", str(tmp_path), "--experiment", str(tmp_path), "--subject", str(subject),
        "--sealed", str(tmp_path / "sealed.json"), "--final-root", str(tmp_path / "none"),
        "--task", "clock_compensation", "--model", "gpt-5.6-sol",
        "--output", str(tmp_path / "out")])
    assert summary["score"] == 4 and len(prompts) == 1 and '"quote"' not in prompts[0]["prompt"]
    assert json.loads((tmp_path / "out/container-dispatch.json").read_text())["model"] == "gpt-5.6-sol"
