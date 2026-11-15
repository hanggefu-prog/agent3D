"""Agent-client adapters: command construction and event parsing only.

An adapter never talks to a vendor network endpoint itself. It builds the argv
for an already-installed, host-authenticated agent CLI and parses the event log
that CLI wrote. Process control, timeouts and receipts stay in
``workspace_codex.run_workspace_turn`` so every adapter shares the same bounded
and auditable execution path. See docs/AGENT_ADAPTERS.md.

Usage numbers are client-reported. A field is known only when the client
reports it as a non-negative integer; anything else is ``None``. Codex reports
the running total of the whole session, so a resumed round's own usage is the
difference from the previous round's total (``usage_delta``). Costs are
derived only from an explicit price table and never estimated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

USAGE_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")
PRICE_FIELDS = ("input_per_mtok", "cached_input_per_mtok", "output_per_mtok")
SESSION_ID = r"[0-9a-fA-F-]{36}"
MODEL_ID = r"[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}"


class AgentAdapter:
    """Protocol. Subclasses set ``name`` and implement both methods.

    ``build_command`` returns argv whose last element is ``-`` (prompt on stdin).
    ``parse_events`` returns ``{thread_id, completed, tool_calls, usage}``.
    ``usage_scope`` says what that usage covers: ``"invocation"`` or
    ``"session_total"`` (the running total of a resumable session).
    """
    name = None
    usage_scope = "invocation"

    def overrides(self, *, container, control, terminal_script, python):
        return ()

    def build_command(self, *, binary, control, final, overrides, thread_id=None,
                      model, output_schema=None):
        raise NotImplementedError

    def parse_events(self, events_path, thread_id=None):
        raise NotImplementedError


def _count(value):
    return type(value) is int and value >= 0


def sum_usage(records):
    """Per-field sum over turn usage dicts; ``None`` if any turn lacks it."""
    records = list(records)
    usage = {"turns": len(records)}
    for field in USAGE_FIELDS:
        values = [r.get(field) if isinstance(r, dict) else None for r in records]
        usage[field] = sum(values) if records and all(_count(v) for v in values) else None
    return usage


def last_usage(records):
    """The last record of running totals; ``None`` per field it does not report."""
    records = list(records)
    usage = sum_usage(records[-1:])
    usage["turns"] = len(records)
    return usage


def usage_delta(total, previous):
    """Usage between two running totals of one session; ``None`` per unknown field.

    Without a previous total the share of this round cannot be separated, so
    every field is ``None``. A total that decreased is not a valid running total.
    """
    usage = {"turns": total.get("turns")}
    for field in USAGE_FIELDS:
        now = total.get(field)
        before = previous.get(field) if isinstance(previous, dict) else None
        usage[field] = now - before if _count(now) and _count(before) and now >= before else None
    return usage


class CodexCliAdapter(AgentAdapter):
    """Codex CLI ``exec``/``exec resume`` with ``--json`` JSONL events.

    ``turn.completed`` carries the session's running total, including earlier
    resumed invocations (observed with codex-cli 0.162.1 on real campaigns).
    """
    name = "codex-cli"
    usage_scope = "session_total"

    def overrides(self, *, container, control, terminal_script, python):
        from .workspace_codex import codex_overrides
        return codex_overrides(container=container, control=control,
                               terminal_script=terminal_script, python=python)

    def build_command(self, *, binary, control, final, overrides, thread_id=None,
                      model, output_schema=None):
        from .workspace_codex import build_workspace_command
        command = build_workspace_command(binary=binary, control=control, final=final,
                                          overrides=overrides, thread_id=thread_id, model=model)
        if output_schema is not None:
            command[-1:-1] = ["--output-schema", str(output_schema)]
        return command

    def parse_events(self, events_path, thread_id=None):
        completed = False
        observed = thread_id
        tool_calls = 0
        turns = []
        text = Path(events_path).read_text() if Path(events_path).exists() else ""
        for line in text.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "thread.started":
                observed = event.get("thread_id")
            if kind == "turn.completed":
                completed = True
                turns.append(event.get("usage"))
            item = event.get("item")
            if (isinstance(item, dict) and item.get("type") == "mcp_tool_call"
                    and item.get("server") == "terminal" and item.get("tool") == "exec"):
                tool_calls += kind == "item.completed"
        return {"thread_id": observed, "completed": completed, "tool_calls": tool_calls,
                "usage": last_usage(turns)}


ADAPTERS = {CodexCliAdapter.name: CodexCliAdapter}


def adapter_for(name):
    if name not in ADAPTERS:
        raise ValueError("Unknown agent adapter")
    return ADAPTERS[name]()


def validate_price_table(table):
    """``{model: {input_per_mtok, cached_input_per_mtok, output_per_mtok}}`` USD."""
    if not isinstance(table, dict) or not table:
        raise ValueError("price_table must be a non-empty object")
    for model, price in table.items():
        if not isinstance(model, str) or not re.fullmatch(MODEL_ID, model):
            raise ValueError("Invalid price_table model")
        if not isinstance(price, dict) or set(price) != set(PRICE_FIELDS):
            raise ValueError("price_table entries need exactly " + ", ".join(PRICE_FIELDS))
        for value in price.values():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("price_table rates must be finite non-negative numbers")
    return table


def estimate_cost(usage, model, price_table=None):
    """USD cost, or ``None`` without a price for the model or complete usage.

    Codex reports ``cached_input_tokens`` as a subset of ``input_tokens``.
    """
    if price_table is None or model not in price_table or not isinstance(usage, dict):
        return None
    values = [usage.get(f) for f in USAGE_FIELDS]
    if not all(_count(v) for v in values):
        return None
    tokens, cached, output = values
    if cached > tokens:
        return None
    price = price_table[model]
    usd = ((tokens - cached) * price["input_per_mtok"] + cached * price["cached_input_per_mtok"]
           + output * price["output_per_mtok"]) / 1e6
    return {"usd": usd, "model": model, "price": dict(price),
            "basis": "client-reported usage x configured price table"}
