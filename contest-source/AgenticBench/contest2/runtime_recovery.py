"""Shared host recovery for model calls and persistent editing turns.

Locks are held by the controller AND inherited by its Codex process. A new
controller can recover a turn only after all owners have released that lock.
No elapsed-time guess authorizes replay or releases a model reservation.
"""
from contextlib import contextmanager
from functools import wraps
import fcntl
import json
import os
import re
from pathlib import Path

from grt_bench.common import digest, read, write


class LegacyOwnershipError(ValueError):
    pass


@contextmanager
def lease(path, *, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Recovery lock cannot be a symlink")
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield stream.fileno()
        finally:
            # Do not LOCK_UN here: an inherited descriptor must keep the lock
            # held if the host controller is interrupted before its child exits.
            pass


def recover_turns(directory):
    path = Path(directory) / "state.json"
    if not path.exists():
        return
    state = read(path)
    changed = False
    for request_id, turn in state["turns"].items():
        if turn["status"] != "pending":
            continue
        receipt = Path(directory) / request_id / "agent/result.json"
        try:
            result = read(receipt)
            if not isinstance(result, dict) or type(result.get("success")) is not bool:
                raise ValueError("Invalid turn receipt")
            if result["success"] and (not isinstance(result.get("thread_id"), str)
                    or not re.fullmatch(r"[0-9a-fA-F-]{36}", result["thread_id"])):
                raise ValueError("Successful turn receipt has no valid session identity")
        except (OSError, ValueError):
            if turn.get("ownership") != "inherited-process-lock":
                raise LegacyOwnershipError("Legacy pending turn has no process-ownership proof; inspect and reconcile it before restart")
            result = {"success": False, "status": "interrupted", "retryable": True,
                      "error": "Previous editing process ended without a complete receipt; edits and logs retained"}
        turn.update(status="finished", result=result)
        if result.get("success"):
            state["thread_id"] = result["thread_id"]
        changed = True
    if changed:
        write(path, state, replace=True)


def recoverable_turn(method):
    @wraps(method)
    def run(self, prompt, request_id):
        try:
            with lease(self.codex_directory / "turn.lock", nonblocking=True) as descriptor:
                recover_turns(self.codex_directory)
                path = self.codex_directory / "state.json"
                if path.exists():
                    previous = read(path)["turns"].get(request_id)
                    if previous is not None and previous["status"] == "finished":
                        if previous["prompt_digest"] != digest(prompt):
                            raise ValueError("Codex request ID was reused with a different prompt")
                        return previous["result"]
                self._turn_lease_fd = descriptor
                try:
                    return method(self, prompt, request_id)
                finally:
                    self._turn_lease_fd = None
        except LegacyOwnershipError as error:
            return {"success": False, "status": "reconciliation_required", "retryable": False,
                    "error": str(error)}
        except BlockingIOError:
            return {"success": False, "status": "running", "retryable": True,
                    "error": "Previous editing process still owns the turn; no duplicate model dispatch"}
    return run


def provider_directory(ledger):
    return ledger.path.parent / "model-receipts"


def settle_receipt(ledger, request_id, receipt):
    from .model_pricing import MODEL_ID, charge_usage
    if (receipt.get("request_id") != request_id or receipt.get("model") != MODEL_ID
            or type(receipt.get("peak")) is not bool):
        raise ValueError("Provider usage receipt identity differs")
    usage = receipt["usage"]
    cost = charge_usage(usage, peak=receipt["peak"])
    ledger.record_model_call(request_id, MODEL_ID, usage["input_tokens"], usage["output_tokens"], str(cost))


def recover_provider(ledger):
    """Called before serving new requests; never assumes a missing bill is zero."""
    root = provider_directory(ledger)
    for row in ledger.snapshot()["model_calls"]:
        if row["status"] not in {"reserved", "uncertain"}:
            continue
        request_id = row["request_id"]
        # Only gateway-generated hexadecimal IDs are filesystem names.
        if len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id):
            continue
        if not (root / (request_id + ".lock")).exists():
            # A legacy gateway did not hold this lock; absence is not proof
            # that its provider request ended.
            continue
        try:
            with lease(root / (request_id + ".lock"), nonblocking=True):
                receipt = root / (request_id + ".json")
                if receipt.exists():
                    settle_receipt(ledger, request_id, read(receipt))
                elif row["status"] == "reserved":
                    ledger.mark_model_uncertain(request_id, "Gateway ownership ended without a billing receipt")
        except BlockingIOError:
            continue
