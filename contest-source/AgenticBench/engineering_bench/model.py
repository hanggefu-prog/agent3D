"""One developer and a fresh, read-only judge; reuse the proven CLI isolation.

Callers own the Docker lifecycle and resource admission. This module never
copies login files, grants host-shell access, resumes another role's session,
or automatically retries a failed paid request.
"""
import contextlib
from pathlib import Path

from global_bench.workspace_codex import agent_container_command, run_workspace_turn
from .judge import (aggregate_samples, check_judge_independence, review_prompt, validate_review,
                    MIN_JUDGE_SAMPLES)
from .protocol import qualify, write_new
from .submission import verify_seal
from .scoring import (CURRENT_SCORING_VERSION, qualified_contract, selected_contract,
                      scoring_prompt, development_instruction)


def cpu_container_command(**kwargs):
    """CTS and data-fanout development do not reserve a placement-training GPU."""
    if "gpu" in kwargs:
        raise ValueError("This engineering worker is CPU-only")
    command = agent_container_command(gpu="0", **kwargs)
    index = command.index("--gpus")
    del command[index:index + 2]
    return command


def developer_prompt(task, version=CURRENT_SCORING_VERSION):
    contract = selected_contract(task, version)
    return development_instruction(task, contract) + '\n\n' + scoring_prompt(task, contract) + """

## Working environment

Your private source checkout is /workspace. Task documentation and the qualified
interface description are in /task. Frozen per-case context and reference
artifacts are read-only under /inputs/reference. Put build outputs, pytest caches
and temporary work under /scratch, not the submitted source tree. The trusted
framework is mounted read-only at /workspace/AgenticBench. Use the container
terminal. No credentials, host filesystem or Docker socket are exposed there.

Edit only the task's editable_paths in /task/task.json. You may add helpers and
tests within that scope. Do not commit changes or edit Git metadata. There is no
language-expression whitelist, patch-length limit or required heuristic. Do not
add dependencies that need network access. The host will seal all modified and
new source files, not just the files you list. No final-case search or model calls
may be launched from the workspace. The host runs the sole final submission.
"""


def develop(task, qualification_root, *, model, output, scoring_version=CURRENT_SCORING_VERSION, **runtime):
    # Missing native support is reported before consuming a developer call.
    proofs = [qualify(task, c, Path(qualification_root) / task.task_id / c / "qualification.json")
              for c in task.data["cases"]]
    contract = qualified_contract(task, proofs, scoring_version)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "dispatch.json", {"task": task.task_id, "task_digest": task.digest(),
        "model": model, "role": "developer", "qualifications": [p["sha256"] for p in proofs]})
    write_new(output / "scoring-contract.json", contract)
    return run_workspace_turn(**runtime, output=output / "turn", model=model,
        thread_id=None, prompt=developer_prompt(task, scoring_version), timeout_seconds=task.data["development_timeout_seconds"])


def develop_source(task, *, model, output, **runtime):
    """Explicit preliminary source work, never a qualified/final submission.

    This is a separate phase, not a fallback from ``develop``. The caller must
    retain the draft and session for integration after native adapters qualify.
    No final ledger or judge is reachable from this entry point.
    """
    note = (task.root / "development.md").read_text()
    prompt = task.instruction + "\n\n" + note + """

## Private development environment

Your independent source checkout is /workspace. The source is read-only except
for the task's permitted contribution directories/files. Git metadata, the
framework at /workspace/AgenticBench, /task and /inputs are read-only. Use the
container terminal; build outputs, pytest caches and temporary files belong in
/scratch. No host shell, credentials, network or Docker socket is available.
Read /workspace/AGENTS.md and relevant component guidance before editing. Inspect
/task/environment.json for the exact source/runtime and any uninitialized
optional submodules. Missing dependencies are not permission to bypass a build
check; report them and keep native build success unclaimed until verified.

This turn produces a source draft and small tests, not the sealed final
submission. The physical adapters and numerical guard limits are still being
qualified. Do not invent missing native APIs, reference measurements, acceptable
timing thresholds or qualification receipts. Isolate any unresolved dependency
behind a clearly documented function and report it. Do not claim physical
improvement without evidence. You may inspect existing frozen artifacts but
must not execute a full-case EDA evaluation or start model calls.

Follow /task/task.json for editable paths. Keep the public entrypoint in the task
instruction. Add useful implementation and tests where possible; do not simply
write a proposal. No code-length limit or required heuristic applies. Do not
commit, modify Git metadata or install network dependencies. Finish with a short
English summary naming changed files, tests and precise remaining integration
requirements. The host will check the complete source diff. A later integration
step in this same task must precede sealing and the sole final per case.
"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from .protocol import digest
    write_new(output / "dispatch.json", {"task": task.task_id, "task_digest": task.digest(),
        "model": model, "role": "developer", "phase": "preliminary_source_development",
        "development_note_sha256": digest(note), "candidate_finals": 0,
        "physical_qualification": False, "submission_sealed": False})
    return run_workspace_turn(**runtime, output=output / "turn", model=model,
        thread_id=None, prompt=prompt, timeout_seconds=task.data["development_timeout_seconds"])


def _judge_sample(task, subject, sealed, *, model, evidence_roots, output, version, **runtime):
    verify_seal(subject, sealed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "dispatch.json", {"task": task.task_id, "task_digest": task.digest(),
        "model": model, "role": "independent_judge", "submission_sha256": sealed["submission_sha256"]})
    result = run_workspace_turn(**runtime, output=output / "turn", model=model,
        thread_id=None, prompt=review_prompt(task, version), timeout_seconds=task.data["judge_timeout_seconds"])
    verify_seal(subject, sealed)
    if not result["success"]:
        # API/tool failures are not a score of 1 and are not silently resampled.
        return {"status": "ungradable", "score": None, "reason": "Judge turn did not complete"}
    try:
        review = validate_review(result["final_message"], evidence_roots, version=version)
    except ValueError as exc:
        if version == 1:
            raise
        # A panel sample with an unverifiable reply is an outcome, not a retry.
        review = {"status": "invalid", "score": None, "reason": str(exc)}
        write_new(output / "invalid-review.json", review)
        return review
    write_new(output / "review.json", review)
    return review


def judge(task, subject, sealed, *, model, evidence_roots, output, **runtime):
    """Version-1 single review (kept to reproduce frozen judge receipts)."""
    return _judge_sample(task, subject, sealed, model=model, evidence_roots=evidence_roots,
                         output=output, version=1, **runtime)


def judge_panel(task, subject, sealed, *, model, developer_model, evidence_roots, output,
                samples=MIN_JUDGE_SAMPLES, families=None, allow_same_family_reason=None,
                sample_runtime=None, **runtime):
    """Version-2 panel: independent family, >=3 fresh samples, median score.

    Each sample is a fresh session with no access to the other samples. A failed
    turn or an invalid reply is recorded as that sample's outcome (never resampled);
    fewer than ``samples`` graded replies yields status insufficient_samples and
    no score. ``sample_runtime(i)`` may return a context manager that yields
runtime overrides (a fresh container and scratch per sample). The panel
    never contributes to the objective headline.
    """
    if type(samples) is not int or samples < MIN_JUDGE_SAMPLES:
        raise ValueError("The judge panel needs at least three samples")
    independence = check_judge_independence(model, developer_model, families, allow_same_family_reason)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "independence.json", independence)
    reviews = []
    for index in range(samples):
        scope = sample_runtime(index) if sample_runtime is not None else contextlib.nullcontext({})
        with scope as overrides:
            options = dict(runtime)
            options.update(overrides or {})
            reviews.append(_judge_sample(task, subject, sealed, model=model, evidence_roots=evidence_roots,
                                         output=output / ("sample-%d" % index), version=2, **options))
    panel = aggregate_samples(reviews, samples)
    panel.update(review_version=2, judge_model=model, independence=independence,
                 sample_outcomes=[{"status": r["status"], "score": r["score"]} for r in reviews],
                 submission_sha256=sealed["submission_sha256"], task_digest=task.digest())
    write_new(output / "panel.json", panel)
    return panel
