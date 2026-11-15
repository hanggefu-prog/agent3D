"""Organizer-owned, transactional phase budgets.

Use one database per team, phase, and GNN baseline. Each baseline has its own
evaluation and model-cost budget. A candidate is charged when its
full-matrix request is accepted, and a later candidate failure never refunds
that charge. Finishing accepted work remains possible after the limit is reached.

The trusted model proxy MUST call ``reserve_model_call`` before provider
dispatch, using an upper bound computed from its pinned pricing/token limits.
``record_model_call`` settles actual provider charges, including unexpected
overruns; it must never be exposed as a participant-reported cost API. Its
unreserved form exists only for trusted bookkeeping/imports and is NOT dispatch
authorization. SQLite protects concurrent controllers on one host; distribute
requests to this owner service rather than sharing its file over network disks.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterator

from .scoring import PUBLIC_SPLITS


TASKS = frozenset({"binary_loss", "loss_decoder"})
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_MONEY_TEXT = 128
MAX_MONEY_DIGITS = 128
MAX_MONEY_SCALE = 18
MAX_MONEY_ADJUSTED = 64


class BudgetExceeded(RuntimeError):
    """A new request cannot be accepted under the remaining phase budget."""


class BudgetConflict(ValueError):
    """An idempotency key or existing database has different immutable data."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{field} must be a nonempty string of at most 512 characters")
    return value


def _money(value: object, field: str, *, positive: bool = False) -> Decimal:
    # Reject floats: converting an inexact binary float would silently change
    # the RMB contract. JSON monetary fields should be decimal strings.
    if not isinstance(value, (str, Decimal)):
        raise ValueError(f"{field} must be a decimal string or Decimal")
    if isinstance(value, str) and len(value) > MAX_MONEY_TEXT:
        raise ValueError(f"{field} exceeds the decimal parser limit")
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be finite decimal RMB") from exc
    if not amount.is_finite() or amount < 0 or (positive and amount == 0):
        raise ValueError(f"{field} must be {'positive' if positive else 'nonnegative'} finite RMB")
    parts = amount.as_tuple()
    # Finite is not sufficient: a tiny exponent string can expand into GBs
    # under format(..., 'f'), or overflow the Decimal context in accounting.
    # These parser limits preserve exact decimal sums, rather than rounding.
    if (len(parts.digits) > MAX_MONEY_DIGITS or parts.exponent < -MAX_MONEY_SCALE
            or parts.exponent > MAX_MONEY_ADJUSTED or amount.adjusted() > MAX_MONEY_ADJUSTED):
        raise ValueError(f"{field} exceeds the bounded decimal range")
    return amount


def _count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _amount(value: Decimal) -> str:
    return format(value, "f")


def _sum_exact(values) -> Decimal:
    values = list(values)
    if not values:
        return Decimal(0)
    exponent = min(value.as_tuple().exponent for value in values)
    precision = max(
        len(value.as_tuple().digits) + value.as_tuple().exponent - exponent
        for value in values
    ) + len(str(len(values))) + 1
    with localcontext() as context:
        context.prec = max(28, precision)
        return sum(values, Decimal(0))


class BudgetLedger:
    def __init__(
        self,
        path: Path | str,
        evaluation_limit: int = 10,
        model_cost_limit_rmb: Decimal | str | None = None,
        *,
        task_id: str,
        allowed_model: str | None = None,
        evaluation_mode: str = "formal",
    ) -> None:
        if type(evaluation_limit) is not int or evaluation_limit <= 0:
            raise ValueError("evaluation_limit must be a positive integer")
        self.path = Path(path)
        if not isinstance(task_id, str) or task_id not in TASKS:
            raise ValueError("Unknown competition task")
        self.task_id = task_id
        if evaluation_mode not in {"formal", "smoke"}:
            raise ValueError("Unknown evaluation mode")
        self.evaluation_mode = evaluation_mode
        self.evaluation_limit = evaluation_limit
        self.model_cost_limit_rmb = _money(
            model_cost_limit_rmb, "model_cost_limit_rmb", positive=True
        )
        self.allowed_model = None if allowed_model is None else _text(allowed_model, "allowed_model")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            statements = (
                "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                """CREATE TABLE IF NOT EXISTS evaluations (
                    request_id TEXT PRIMARY KEY, task TEXT NOT NULL, candidate_id TEXT NOT NULL,
                    status TEXT NOT NULL, accepted_at TEXT NOT NULL, completed_at TEXT,
                    reason TEXT, payload_digest TEXT
                )""",
                """CREATE TABLE IF NOT EXISTS job_outbox (
                    job_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, case_id TEXT NOT NULL,
                    seed INTEGER NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(request_id, case_id, seed)
                )""",
                """CREATE TABLE IF NOT EXISTS model_calls (
                    request_id TEXT PRIMARY KEY, model TEXT NOT NULL, status TEXT NOT NULL,
                    max_cost_rmb TEXT, reserved_at TEXT, input_tokens INTEGER,
                    output_tokens INTEGER, cost_rmb TEXT, recorded_at TEXT
                )""",
                """CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                    kind TEXT NOT NULL, request_id TEXT NOT NULL, payload_json TEXT NOT NULL
                )""",
            )
            # executescript() implicitly commits; run DDL individually so the
            # schema and frozen settings are installed in this transaction.
            for statement in statements:
                db.execute(statement)
            if (db.execute("SELECT 1 FROM settings LIMIT 1").fetchone() and
                    not db.execute("SELECT 1 FROM settings WHERE key = 'evaluation_mode'").fetchone()):
                db.execute("INSERT INTO settings VALUES ('evaluation_mode', 'formal')")
            expected = {
                "schema_version": "2",
                "task_id": self.task_id,
                "evaluation_mode": self.evaluation_mode,
                "evaluation_limit": str(self.evaluation_limit),
                "model_cost_limit_rmb": _amount(self.model_cost_limit_rmb),
                "allowed_model": self.allowed_model or "",
                "frozen_candidate": "",
                "frozen_harness_sha256": "",
            }
            for key, value in expected.items():
                db.execute("INSERT OR IGNORE INTO settings VALUES (?, ?)", (key, value))
                actual = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()[0]
                equal = (Decimal(actual) == Decimal(value) if key == "model_cost_limit_rmb"
                         else True if key in {"frozen_candidate", "frozen_harness_sha256"}
                         else actual == value)
                if not equal:
                    raise BudgetConflict(f"Existing ledger setting differs: {key}")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout = 30000")
            db.execute("PRAGMA synchronous = FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _event(db: sqlite3.Connection, kind: str, request_id: str, payload: dict) -> None:
        db.execute(
            "INSERT INTO audit_events (timestamp, kind, request_id, payload_json) VALUES (?, ?, ?, ?)",
            (_now(), kind, request_id, json.dumps(payload, sort_keys=True, ensure_ascii=False)),
        )

    def _state(self, db: sqlite3.Connection) -> dict:
        accepted = db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        frozen_candidate = db.execute("SELECT value FROM settings WHERE key = 'frozen_candidate'").fetchone()[0]
        rows = db.execute("SELECT status, cost_rmb, max_cost_rmb FROM model_calls").fetchall()
        spent = _sum_exact(Decimal(row["cost_rmb"]) for row in rows if row["cost_rmb"] is not None)
        reserved = _sum_exact(
            Decimal(row["max_cost_rmb"]) for row in rows if row["status"] in {"reserved", "uncertain"}
        )
        committed = _sum_exact((spent, reserved))
        reasons = []
        exhausted = []
        if accepted >= self.evaluation_limit:
            reasons.append("evaluation_limit_reached")
            exhausted.append("evaluation_limit_reached")
        if frozen_candidate:
            reasons.append("candidate_frozen_for_validation")
        if spent >= self.model_cost_limit_rmb:
            reasons.append("model_cost_limit_reached")
            exhausted.append("model_cost_limit_reached")
        elif committed >= self.model_cost_limit_rmb:
            reasons.append("model_cost_fully_reserved")
        available = max(Decimal(0), _sum_exact((self.model_cost_limit_rmb, spent.copy_negate(), reserved.copy_negate())))
        # API admission has a nonzero worst-case reservation; evaluation admission
        # does not spend model tokens. Keep the two permissions explicit.
        from .model_pricing import reserve_full_context
        minimum = Decimal(reserve_full_context(peak=True)) if self.allowed_model == "deepseek-flash" else Decimal(0)
        model_reasons = list(reasons)
        if available < minimum and "model_cost_limit_reached" not in model_reasons:
            model_reasons.append("insufficient_model_reservation")
        evaluation_reasons = [r for r in reasons if r not in {
            "model_cost_fully_reserved", "model_cost_limit_reached"}]
        return {
            "model_dispatch_allowed": not model_reasons,
            "evaluation_dispatch_allowed": not evaluation_reasons,
            "model_stop_reasons": model_reasons,
            "model_reservation_rmb": _amount(minimum),
            "unresolved_model_calls": sum(row["status"] == "uncertain" for row in rows),
            "task_id": self.task_id,
            "evaluation_mode": self.evaluation_mode,
            "evaluation_limit": self.evaluation_limit,
            "evaluations_accepted": accepted,
            "evaluations_remaining": max(0, (1 if self.evaluation_mode == "smoke"
                                              else self.evaluation_limit) - accepted),
            "model_cost_limit_rmb": _amount(self.model_cost_limit_rmb),
            "model_cost_spent_rmb": _amount(spent),
            "model_cost_reserved_rmb": _amount(reserved),
            "model_cost_available_rmb": _amount(max(Decimal(0), _sum_exact((self.model_cost_limit_rmb, spent.copy_negate(), reserved.copy_negate())))),
            "model_cost_overrun_rmb": _amount(max(Decimal(0), _sum_exact((spent, self.model_cost_limit_rmb.copy_negate())))),
            "dispatch_allowed": not model_reasons,
            "stop_reasons": model_reasons,
            "budget_exhausted": bool(exhausted),
            "exhaustion_reasons": exhausted,
            "allowed_model": self.allowed_model,
            "frozen_candidate": frozen_candidate or None,
        }

    def freeze(self, candidate_id: str, harness_sha256: str) -> dict:
        """End all new model/evaluation intake before exposing test feedback."""
        if self.evaluation_mode != "formal":
            raise ValueError("Smoke results cannot enter formal validation")
        candidate_id = _text(candidate_id, "candidate_id")
        if not isinstance(harness_sha256, str) or not _DIGEST.fullmatch(harness_sha256):
            raise ValueError("harness_sha256 must be an official SHA-256 digest")
        with self._transaction() as db:
            current = db.execute("SELECT value FROM settings WHERE key = 'frozen_candidate'").fetchone()[0]
            harness = db.execute("SELECT value FROM settings WHERE key = 'frozen_harness_sha256'").fetchone()[0]
            if current:
                if (current, harness) != (candidate_id, harness_sha256):
                    raise BudgetConflict("A different candidate or harness is already frozen")
                return {"candidate_id": current, "harness_sha256": harness}
            if db.execute("SELECT 1 FROM evaluations WHERE status = 'accepted' LIMIT 1").fetchone():
                raise ValueError("Finish or reconcile accepted evaluations before freezing")
            if db.execute("SELECT 1 FROM model_calls WHERE status = 'reserved' LIMIT 1").fetchone():
                raise ValueError("Reconcile in-flight model calls before freezing")
            if not db.execute("SELECT 1 FROM evaluations WHERE candidate_id = ? AND status = 'succeeded' LIMIT 1",
                              (candidate_id,)).fetchone():
                raise ValueError("Freeze requires a successful training evaluation")
            db.execute("UPDATE settings SET value = ? WHERE key = 'frozen_candidate'", (candidate_id,))
            db.execute("UPDATE settings SET value = ? WHERE key = 'frozen_harness_sha256'", (harness_sha256,))
            self._event(db, "candidate_frozen", candidate_id, {"harness_sha256": harness_sha256})
            return {"candidate_id": candidate_id, "harness_sha256": harness_sha256}

    def _guard(self, db: sqlite3.Connection, *, model=False) -> dict:
        state = self._state(db)
        if not state["model_dispatch_allowed" if model else "evaluation_dispatch_allowed"]:
            raise BudgetExceeded(", ".join(state["stop_reasons"]))
        return state

    def _model(self, model: object) -> str:
        model = _text(model, "model")
        if self.allowed_model is not None and model != self.allowed_model:
            raise ValueError("Model differs from the frozen allowed model")
        return model

    def accept_evaluation(self, request_id: str, task: str, candidate_id: str) -> dict:
        if self.evaluation_mode != "formal":
            raise ValueError("Smoke evaluations require the fixed single-cell matrix")
        request_id = _text(request_id, "request_id")
        candidate_id = _text(candidate_id, "candidate_id")
        if task != self.task_id:
            raise ValueError("Evaluation task differs from the fixed task ledger")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM evaluations WHERE request_id = ?", (request_id,)).fetchone()
            if row is not None:
                if row["task"] != task or row["candidate_id"] != candidate_id:
                    raise BudgetConflict("Evaluation request ID was used for different data")
                return dict(row)
            self._guard(db)
            db.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?, 'accepted', ?, NULL, NULL, NULL)",
                (request_id, task, candidate_id, _now()),
            )
            self._event(db, "evaluation_accepted", request_id, {"task": task, "candidate_id": candidate_id})
            return dict(db.execute("SELECT * FROM evaluations WHERE request_id = ?", (request_id,)).fetchone())

    def accept_matrix(self, request_id: str, candidate_id: str, payload_digest: str,
                      *, baseline=None) -> dict:
        """Atomically charge one evaluation and enqueue all fixed baseline cells.

        The official intake must first verify the source snapshot and derive
        payload_digest from all bound request identities. Participant digests
        are not proof of a valid candidate.
        """
        request_id = _text(request_id, "request_id")
        candidate_id = _text(candidate_id, "candidate_id")
        if not isinstance(payload_digest, str) or not _DIGEST.fullmatch(payload_digest):
            raise ValueError("payload_digest must be an official SHA-256 digest")
        baseline = self.task_id if baseline is None else baseline
        if baseline != self.task_id:
            raise BudgetConflict("Baseline differs from the fixed task ledger")
        from .sampling import SMOKE_CELL, coordinates
        selected = coordinates(baseline, PUBLIC_SPLITS["train"])
        if self.evaluation_mode == "smoke":
            selected = [SMOKE_CELL]
        matrix = [(case, int(seed)) for case, seed in selected]
        with self._transaction() as db:
            row = db.execute("SELECT * FROM evaluations WHERE request_id = ?", (request_id,)).fetchone()
            if row is not None:
                if (row["task"], row["candidate_id"], row["payload_digest"]) != (self.task_id, candidate_id, payload_digest):
                    raise BudgetConflict("Evaluation request ID was used for different bound data")
                jobs = db.execute("SELECT case_id,seed FROM job_outbox WHERE request_id = ?", (request_id,)).fetchall()
                if {(r[0], r[1]) for r in jobs} != set(matrix) or len(jobs) != len(matrix):
                    raise BudgetConflict("Accepted matrix has an incomplete durable outbox")
                return dict(row)
            if self.evaluation_mode == "smoke" and db.execute("SELECT 1 FROM evaluations").fetchone():
                raise BudgetExceeded("Smoke mode accepts exactly one evaluation")
            self._guard(db)
            accepted = _now()
            db.execute("INSERT INTO evaluations VALUES (?, ?, ?, 'accepted', ?, NULL, NULL, ?)",
                       (request_id, self.task_id, candidate_id, accepted, payload_digest))
            for case, seed in matrix:
                job_id = hashlib.sha256(json.dumps([self.task_id, request_id, candidate_id,
                                                     payload_digest, case, seed], separators=(",", ":")).encode()).hexdigest()
                db.execute("INSERT INTO job_outbox VALUES (?, ?, ?, ?, 'pending', ?)",
                           (job_id, request_id, case, seed, accepted))
            self._event(db, "matrix_accepted", request_id, {"task": self.task_id,
                         "candidate_id": candidate_id, "payload_digest": payload_digest,
                         "job_count": len(matrix)})
            return dict(db.execute("SELECT * FROM evaluations WHERE request_id = ?", (request_id,)).fetchone())

    def reject_submission(self, request_id, payload_digest, reason, *, baseline=None):
        """Charge a rejected submission once, without creating native jobs."""
        request_id = _text(request_id, "request_id")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Rejected submission requires a reason")
        reason = reason[:4000]
        if not _DIGEST.fullmatch(payload_digest):
            raise ValueError("Invalid rejected-submission digest")
        baseline = self.task_id if baseline is None else baseline
        if baseline != self.task_id:
            raise BudgetConflict("Baseline differs from the fixed task ledger")
        candidate = "rejected-" + baseline + "-" + payload_digest
        with self._transaction() as db:
            row = db.execute("SELECT * FROM evaluations WHERE request_id=?", (request_id,)).fetchone()
            if row is not None:
                if row["candidate_id"] != candidate or row["status"] != "rejected":
                    raise BudgetConflict("Request ID belongs to another submission")
                return dict(row)
            self._guard(db)
            if self.evaluation_mode == "smoke" and db.execute("SELECT 1 FROM evaluations").fetchone():
                raise BudgetExceeded("Smoke mode accepts exactly one evaluation")
            now = _now()
            db.execute("INSERT INTO evaluations VALUES (?, ?, ?, 'rejected', ?, ?, ?, ?)",
                       (request_id, self.task_id, candidate, now, now, reason, payload_digest))
            self._event(db, "submission_rejected", request_id,
                        {"baseline": baseline, "reason": reason, "charged": True})
            return dict(db.execute("SELECT * FROM evaluations WHERE request_id=?", (request_id,)).fetchone())

    def pending_jobs(self, request_id: str) -> list[dict]:
        request_id = _text(request_id, "request_id")
        with self._transaction() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM job_outbox WHERE request_id = ? ORDER BY case_id, seed", (request_id,))]

    def finish_evaluation(self, request_id: str, status: str, reason: str | None = None) -> dict:
        request_id = _text(request_id, "request_id")
        if not isinstance(status, str) or status not in {"succeeded", "failed"}:
            raise ValueError("Evaluation completion status must be succeeded or failed")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or None")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM evaluations WHERE request_id = ?", (request_id,)).fetchone()
            if row is None:
                raise KeyError("Evaluation was not accepted")
            if row["status"] != "accepted":
                if row["status"] != status or row["reason"] != reason:
                    raise BudgetConflict("Completed evaluation receipt is immutable")
                return dict(row)
            db.execute(
                "UPDATE evaluations SET status = ?, reason = ?, completed_at = ? WHERE request_id = ?",
                (status, reason, _now(), request_id),
            )
            self._event(db, "evaluation_finished", request_id, {"status": status, "reason": reason})
            return dict(db.execute("SELECT * FROM evaluations WHERE request_id = ?", (request_id,)).fetchone())

    def reserve_model_call(self, request_id: str, model: str, max_cost_rmb: Decimal | str) -> dict:
        """Authorize one proxy dispatch with a trusted, positive maximum charge.

        A retry returning an existing reservation authorizes recovery of that
        same provider request, never a second provider invocation. The proxy
        must bind its provider idempotency key to this request ID.
        """
        request_id = _text(request_id, "request_id")
        model = self._model(model)
        maximum = _money(max_cost_rmb, "max_cost_rmb", positive=True)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM model_calls WHERE request_id = ?", (request_id,)).fetchone()
            if row is not None:
                if row["model"] != model or row["max_cost_rmb"] is None or Decimal(row["max_cost_rmb"]) != maximum:
                    raise BudgetConflict("Model request ID was used for different reservation data")
                return dict(row)
            state = self._guard(db, model=True)
            if maximum > Decimal(state["model_cost_available_rmb"]):
                raise BudgetExceeded("Model reservation exceeds available RMB budget")
            db.execute(
                "INSERT INTO model_calls (request_id, model, status, max_cost_rmb, reserved_at) VALUES (?, ?, 'reserved', ?, ?)",
                (request_id, model, _amount(maximum), _now()),
            )
            self._event(db, "model_reserved", request_id, {"model": model, "max_cost_rmb": _amount(maximum)})
            return dict(db.execute("SELECT * FROM model_calls WHERE request_id = ?", (request_id,)).fetchone())

    def record_model_call(
        self,
        request_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_rmb: Decimal | str,
    ) -> dict:
        """Settle actual trusted proxy charges; never discard an in-flight bill.

        New unreserved records are trusted bookkeeping only, gated by the
        current phase limits. They cannot authorize a provider call and may
        reveal an overrun. Real proxy dispatch requires a prior reservation.
        """
        request_id = _text(request_id, "request_id")
        model = self._model(model)
        input_tokens = _count(input_tokens, "input_tokens")
        output_tokens = _count(output_tokens, "output_tokens")
        cost = _money(cost_rmb, "cost_rmb")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM model_calls WHERE request_id = ?", (request_id,)).fetchone()
            if row is not None:
                if row["model"] != model:
                    raise BudgetConflict("Model differs from the reserved or recorded request")
                if row["status"] == "settled":
                    if (row["input_tokens"], row["output_tokens"], Decimal(row["cost_rmb"])) != (input_tokens, output_tokens, cost):
                        raise BudgetConflict("Settled model receipt is immutable")
                    return dict(row)
            else:
                self._guard(db)
                db.execute(
                    "INSERT INTO model_calls (request_id, model, status) VALUES (?, ?, 'reserved')",
                    (request_id, model),
                )
            db.execute(
                "UPDATE model_calls SET status = 'settled', input_tokens = ?, output_tokens = ?, cost_rmb = ?, recorded_at = ? WHERE request_id = ?",
                (input_tokens, output_tokens, _amount(cost), _now(), request_id),
            )
            self._event(db, "model_settled", request_id, {
                "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens,
                "cost_rmb": _amount(cost),
                "exceeded_reservation": row is not None and cost > Decimal(row["max_cost_rmb"]),
            })
            return dict(db.execute("SELECT * FROM model_calls WHERE request_id = ?", (request_id,)).fetchone())

    def mark_model_uncertain(self, request_id: str, reason: str) -> None:
        """Trusted gateway only: request ended without a usable billing receipt.

        The ceiling remains encumbered. This is not a refund or an actual bill.
        A later trusted receipt can settle it using record_model_call().
        """
        with self._transaction() as db:
            row = db.execute("SELECT status FROM model_calls WHERE request_id = ?", (request_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown model reservation")
            if row["status"] == "reserved":
                db.execute("UPDATE model_calls SET status = 'uncertain' WHERE request_id = ?", (request_id,))
                self._event(db, "model_usage_uncertain", request_id, {"reason": reason})

    def snapshot(self) -> dict:
        with self._transaction() as db:
            result = self._state(db)
            result["evaluations"] = [dict(row) for row in db.execute("SELECT * FROM evaluations ORDER BY accepted_at, request_id")]
            result["model_calls"] = [dict(row) for row in db.execute("SELECT * FROM model_calls ORDER BY COALESCE(reserved_at, recorded_at), request_id")]
            result["audit_event_count"] = db.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
            return result

    def audit_events(self) -> list[dict]:
        """Read the append-only event history without changing its receipts."""
        with self._transaction() as db:
            events = []
            for row in db.execute("SELECT * FROM audit_events ORDER BY sequence"):
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json"))
                events.append(item)
            return events
