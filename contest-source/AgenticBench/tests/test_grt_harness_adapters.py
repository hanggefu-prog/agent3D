"""Agent adapter, usage/cost accounting and batch model allowlist (no model calls)."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_bench import agent_adapters as adapters
from global_bench import workspace_codex as runtime

THREAD = "019e2d6f-9530-7a21-b44f-c56e4a1dd4e7"
PRICE = {"gpt-5.6-sol": {"input_per_mtok": 2.0, "cached_input_per_mtok": 0.5, "output_per_mtok": 10.0}}


def events_file(tmp_path, events, raw=()):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events) + "".join(r + "\n" for r in raw))
    return path


def usage(i, c, o):
    return {"input_tokens": i, "cached_input_tokens": c, "output_tokens": o}


def test_parse_keeps_last_running_total_and_counts_terminal_calls(tmp_path):
    path = events_file(tmp_path, [
        {"type": "thread.started", "thread_id": THREAD},
        {"type": "item.started", "item": {"type": "mcp_tool_call", "server": "terminal", "tool": "exec"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "terminal", "tool": "exec"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "other", "tool": "exec"}},
        {"type": "turn.completed", "usage": usage(100, 40, 7)},
        {"type": "turn.completed", "usage": dict(usage(150, 40, 10), reasoning_output_tokens=2)},
    ], raw=["not json", "[1, 2]"])
    parsed = adapters.CodexCliAdapter().parse_events(path)
    assert parsed == {"thread_id": THREAD, "completed": True, "tool_calls": 1,
                      "usage": {"turns": 2, "input_tokens": 150, "cached_input_tokens": 40, "output_tokens": 10}}


def test_usage_delta_between_running_totals():
    total = dict(usage(1500, 1200, 90), turns=1)
    assert adapters.usage_delta(total, usage(1000, 800, 60)) == {
        "turns": 1, "input_tokens": 500, "cached_input_tokens": 400, "output_tokens": 30}
    for previous in (None, {}, usage(2000, 0, 0), usage(None, 1, 1)):
        delta = adapters.usage_delta(total, previous)
        assert None in (delta["input_tokens"], delta["cached_input_tokens"], delta["output_tokens"])


@pytest.mark.parametrize("turns", [
    [{"type": "turn.completed"}],
    [{"type": "turn.completed", "usage": usage(10, 1, 1)}, {"type": "turn.completed", "usage": "n/a"}],
    [{"type": "turn.completed", "usage": usage(10, -1, 1)}],
    [{"type": "turn.completed", "usage": usage(10.0, 1, True)}],
])
def test_unknown_usage_is_null_never_fabricated(tmp_path, turns):
    parsed = adapters.CodexCliAdapter().parse_events(events_file(tmp_path, turns))
    values = parsed["usage"]
    assert values["turns"] == len(turns)
    assert None in (values["input_tokens"], values["cached_input_tokens"], values["output_tokens"])
    assert adapters.estimate_cost(values, "gpt-5.6-sol", PRICE) is None


def test_no_events_means_no_usage_and_previous_thread_is_kept(tmp_path):
    parsed = adapters.CodexCliAdapter().parse_events(tmp_path / "missing.jsonl", thread_id=THREAD)
    assert parsed["thread_id"] == THREAD and not parsed["completed"]
    assert parsed["usage"] == {"turns": 0, "input_tokens": None, "cached_input_tokens": None, "output_tokens": None}


def test_cost_requires_explicit_price_for_model():
    values = adapters.sum_usage([usage(1_000_000, 400_000, 100_000)])
    cost = adapters.estimate_cost(values, "gpt-5.6-sol", PRICE)
    assert cost["usd"] == pytest.approx(0.6 * 2.0 + 0.4 * 0.5 + 0.1 * 10.0)
    assert adapters.estimate_cost(values, "other-model", PRICE) is None
    assert adapters.estimate_cost(values, "gpt-5.6-sol", None) is None
    assert adapters.estimate_cost(adapters.sum_usage([usage(1, 2, 0)]), "gpt-5.6-sol", PRICE) is None


@pytest.mark.parametrize("table", [{}, [], {"m": {}}, {"m": dict(PRICE["gpt-5.6-sol"], extra=1)},
    {"m": dict(PRICE["gpt-5.6-sol"], output_per_mtok=-1)}, {"m": dict(PRICE["gpt-5.6-sol"], output_per_mtok=float("inf"))},
    {"m": dict(PRICE["gpt-5.6-sol"], output_per_mtok=True)}, {"bad model": PRICE["gpt-5.6-sol"]}])
def test_price_table_is_strict(table):
    with pytest.raises(ValueError):
        adapters.validate_price_table(table)


def test_adapter_registry_and_protocol():
    assert adapters.adapter_for("codex-cli").name == "codex-cli"
    with pytest.raises(ValueError):
        adapters.adapter_for("openai-http")
    with pytest.raises(NotImplementedError):
        adapters.AgentAdapter().parse_events("x")


def test_codex_command_matches_legacy_builder(tmp_path):
    adapter = adapters.CodexCliAdapter()
    overrides = adapter.overrides(container="c", control=tmp_path, terminal_script=tmp_path / "t.py", python="/p")
    assert overrides == runtime.codex_overrides(container="c", control=tmp_path,
                                                terminal_script=tmp_path / "t.py", python="/p")
    legacy = runtime.build_workspace_command(binary="codex", control=tmp_path, final=tmp_path / "f",
                                             overrides=overrides, thread_id=THREAD, model="gpt-5.6-sol")
    legacy[-1:-1] = ["--output-schema", "s.json"]
    assert adapter.build_command(binary="codex", control=tmp_path, final=tmp_path / "f", overrides=overrides,
                                 thread_id=THREAD, model="gpt-5.6-sol", output_schema="s.json") == legacy


class FakeAdapter(adapters.CodexCliAdapter):
    name = "fake-cli"

    def build_command(self, **kwargs):
        return ["/trusted/fake", "--model", kwargs["model"], "-"]


def run_turn(tmp_path, monkeypatch, events, **kwargs):
    (tmp_path / "control").mkdir()
    seen = {}
    class Process:
        pid, returncode = 4242, 0
        def __init__(self, command, **kw):
            seen["command"] = command; self.kw = kw
        def communicate(self, prompt, timeout):
            for event in events:
                self.kw["stdout"].write(json.dumps(event) + "\n")
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    result = runtime.run_workspace_turn(binary="/trusted/codex", control=tmp_path / "control",
        output=tmp_path / "round", container="o3db-agent-test", prompt="p",
        terminal_script=tmp_path / "terminal.py", python="/trusted/python", **kwargs)
    return result, seen


def test_turn_persists_usage_and_priced_cost(tmp_path, monkeypatch):
    result, seen = run_turn(tmp_path, monkeypatch, [{"type": "thread.started", "thread_id": THREAD},
        {"type": "turn.completed", "usage": usage(2000, 1000, 100)}], price_table=PRICE)
    assert result["success"] and result["adapter"] == "codex-cli" and result["model"] == runtime.MODEL
    assert result["usage"]["input_tokens"] == 2000 and result["session_usage"] == result["usage"]
    assert result["cost"]["usd"] == pytest.approx((1000 * 2.0 + 1000 * 0.5 + 100 * 10.0) / 1e6)
    assert json.loads((tmp_path / "round/result.json").read_text()) == result
    assert seen["command"][seen["command"].index("--model") + 1] == "gpt-5.6-sol"


def test_resumed_turn_prices_only_its_share_of_the_session(tmp_path, monkeypatch):
    events = [{"type": "turn.completed", "usage": usage(3000, 1500, 150)}]
    result, _ = run_turn(tmp_path, monkeypatch, events, thread_id=THREAD, price_table=PRICE,
                         previous_session_usage=dict(usage(2000, 1000, 100), turns=1))
    assert result["success"] and result["session_usage"]["input_tokens"] == 3000
    assert result["usage"] == {"turns": 1, "input_tokens": 1000, "cached_input_tokens": 500, "output_tokens": 50}
    assert result["cost"]["usd"] == pytest.approx((500 * 2.0 + 500 * 0.5 + 50 * 10.0) / 1e6)


def test_resumed_turn_without_previous_total_has_unknown_usage(tmp_path, monkeypatch):
    events = [{"type": "turn.completed", "usage": usage(3000, 1500, 150)}]
    result, _ = run_turn(tmp_path, monkeypatch, events, thread_id=THREAD, price_table=PRICE)
    assert result["success"] and result["session_usage"]["input_tokens"] == 3000
    assert result["usage"]["input_tokens"] is None and result["cost"] is None


def test_turn_without_price_table_has_null_cost_and_custom_adapter(tmp_path, monkeypatch):
    result, seen = run_turn(tmp_path, monkeypatch, [{"type": "thread.started", "thread_id": THREAD},
        {"type": "turn.completed", "usage": usage(1, 0, 1)}], adapter=FakeAdapter(), model="other-model")
    assert result["cost"] is None and result["adapter"] == "fake-cli" and result["model"] == "other-model"
    assert seen["command"] == ["/trusted/fake", "--model", "other-model", "-"]


def test_turn_rejects_invalid_price_table_before_process(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        run_turn(tmp_path, monkeypatch, [], price_table={"m": {}})
    assert not (tmp_path / "round").exists()


# launch_codesign_batch model allowlist
from agentic3d.knowledge import write_json  # noqa: E402
from scripts import launch_codesign_batch as batch  # noqa: E402


def plan(tmp_path, monkeypatch, capsys, config, *argv):
    path = tmp_path / "server.json"
    write_json(path, dict({"cases": ["bp_fe"], "session": "codesign-test"}, **config))
    monkeypatch.setattr(batch.sys, "argv", ["launch_codesign_batch.py", "--config", str(path), *argv])
    monkeypatch.setattr(batch, "Runtime", lambda *a, **kw: pytest.fail("plan must not start runtime"))
    batch.main()
    return json.loads(capsys.readouterr().out)


def test_batch_default_allowlist_keeps_legacy_pin(tmp_path, monkeypatch, capsys):
    assert plan(tmp_path, monkeypatch, capsys, {"model": "gpt-5.6-sol"})["model"] == "gpt-5.6-sol"
    with pytest.raises(ValueError, match="model_allowlist"):
        plan(tmp_path, monkeypatch, capsys, {"model": "gpt-6"})
    with pytest.raises(ValueError, match="model_allowlist"):
        plan(tmp_path, monkeypatch, capsys, {"model": "gpt-5.6-sol"}, "--model", "gpt-6")


def test_batch_model_flag_selects_from_explicit_allowlist(tmp_path, monkeypatch, capsys):
    config = {"model": "gpt-5.6-sol", "model_allowlist": ["gpt-5.6-sol", "gpt-6"]}
    assert plan(tmp_path, monkeypatch, capsys, config, "--model", "gpt-6")["model"] == "gpt-6"
    assert plan(tmp_path, monkeypatch, capsys, {"model_allowlist": ["gpt-6"]}, "--model", "gpt-6")["model"] == "gpt-6"
    for bad in ([], ["gpt-6", "gpt-6"], ["bad model"], "gpt-6"):
        with pytest.raises(ValueError, match="model_allowlist"):
            plan(tmp_path, monkeypatch, capsys, {"model": "gpt-6", "model_allowlist": bad})
