"""Opt-in STA knowledge-v2 runner; legacy experiments never enter this module."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .knowledge import KnowledgeStore, canonical, digest, write_json
from .enhanced_task import (CONTEXT_MODE, knowledge_catalog, compile_enhanced_task,
                            compact_history, local_context, read_enhanced_task)
from .sta_protocol import (COMMON, ANALYZER, LOCAL, REFLECTOR, KNOWLEDGE_REVIEW,
                           ANALYZER_SCHEMA, LOCAL_SCHEMA, REFLECTION_SCHEMA, REVIEW_SCHEMA,
                           decode_updates, validate_task,
                           task_spec, proxy_score, PARTITION_PATH)
from .research_feedback import (KNOWLEDGE_AUTHORITY, validate_feedback,
                                index_suggestions, validate_review)
from .evidence import collect_evidence
from .research_io import (ResearchBlocked, RESPONSE_INSTRUCTION, response_schema,
                          unwrap_response, check_research_access)


class StatelessResearcher:
    """Fresh ephemeral model calls with all permitted memory supplied explicitly.

    Avoid hidden conversation carry-over when a frozen test excludes newly
    proposed knowledge. Each prompt/response is still persisted for audit.
    """
    def __init__(self, name, root, schema, model, binary, workspace, driver):
        self.name, self.root, self.schema = name, root, schema
        self.model, self.binary, self.workspace, self.driver = model, binary, workspace, driver
        root.mkdir(parents=True, exist_ok=True)
        self.schema_path = root / "output.schema.json"
        write_json(self.schema_path, response_schema(schema))
        self.access_check = None

    def ask(self, prompt):
        if self.access_check is not None:
            self.access_check()
        directories = list((self.root / "calls").glob("*"))
        number = max([0] + [int(p.name) for p in directories if p.name.isdigit()]) + 1
        result = self.driver.run_codex(prompt + RESPONSE_INSTRUCTION, self.driver.CodexExecConfig(
            workspace=self.workspace, run_dir=self.root / "calls" / f"{number:03d}",
            codex_binary=self.binary, model=self.model, sandbox="read-only",
            config_overrides=("model_reasoning_effort=xhigh",), output_schema=self.schema_path,
            ephemeral=True, skip_git_repo_check=True, timeout_seconds=3600))
        if not result.success:
            raise RuntimeError("research model call failed; inspect " + result.metadata_path)
        return unwrap_response(json.loads(result.final_message))


def configured_seed(args, repo):
    source = getattr(args, "knowledge_seed", None)
    files = json.loads(source.read_text()) if source else seed_snapshot(
        repo, getattr(args, "objective", None), getattr(args, "rounds", 5))
    if not isinstance(files, dict) or not files or any(not isinstance(v, str) for v in files.values()):
        raise ValueError("knowledge seed must be a nonempty text-file snapshot")
    return files


def final_score(result):
    metrics = result.get("metrics", {})
    if any(metrics.get(key) != value for key, value in {
        "runs_successfully": 1, "opt_passes": 2, "sta_graph_complete": 1,
        "sta_unconstrained_endpoints": 0}.items()):
        return None
    for key in ("combined_score", "final_global_wns_setup_ns", "final_global_tns_setup_ns"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
    return metrics["combined_score"] if metrics["combined_score"] > 0 else None


def seed_snapshot(repo: Path, objective=None, rounds=5) -> dict[str, str]:
    files = {}
    for name in ("optimize-mol-partition", "optimize-mol-layout"):
        root = repo / "skills" / name
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("seed skill contains symlink")
            if path.is_file() and path.suffix in {".md", ".py", ".sh", ".json", ".yaml"}:
                files["skills/" + name + "/" + path.relative_to(root).as_posix()] = path.read_text()
    for name in (
        "tp_gnn_partition_proxy",
        "placement_proxy",
        "tp_gnn_placement_cts_thermal" if objective == "thermal" else "tp_gnn_placement_two_pass",
    ):
        for filename in ("profile.yaml", "prompt.md"):
            files["evaluation_profiles/" + name + "/" + filename] = (
                repo / "evaluation_profiles" / name / filename
            ).read_text()
    files["tools.json"] = canonical({
        "source_inspection": "rg/sed and bounded read-only shell checks in the private workspace",
        "partition": "driver-owned tp_gnn_sta_inner; frozen case input; cut/balance metrics",
        "placement": "driver-owned placement_sta_inner; incumbent partition; native HPWL/overflow",
        "final": "driver-owned tp_gnn_placement_sta; original case reference; exactly two OPT passes",
        "skill_helpers": "seed scripts are available under knowledge/seed; check regime and dependencies",
        "generated_helpers": "syntax-checked only, never automatically executed by evaluator",
    })
    files["protocol/common.md"] = COMMON
    files["protocol/analyzer.md"] = ANALYZER
    files["protocol/local.md"] = LOCAL
    files["protocol/reflector.md"] = REFLECTOR
    files["protocol/knowledge-review.md"] = KNOWLEDGE_REVIEW
    if objective is not None:
        from .codesign_endpoint import common, role_prompt, TARGETS
        files["protocol/common.md"] = common(objective, rounds)
        for role, name in (("ANALYZER", "analyzer"), ("LOCAL", "local"),
                           ("REFLECTOR", "reflector"), ("KNOWLEDGE_REVIEW", "knowledge-review")):
            files[f"protocol/{name}.md"] = role_prompt(role, objective)
        catalog = json.loads(files["tools.json"])
        catalog["final"] = "driver-owned " + TARGETS[objective] + "; original case reference; " + objective + " contract"
        files["tools.json"] = canonical(catalog)
    return files


def initialize(args, root, driver):
    spec = driver.load_task(driver.STA_PROFILE)
    seeds = configured_seed(args, driver.ROOT)
    split = json.loads(args.knowledge_split.read_text()) if args.knowledge_split else None
    store = KnowledgeStore.initialize(root / "knowledge", seed_files=seeds,
        case=args.case, phase=args.knowledge_phase, split=split,
        imported=args.knowledge_import, adapt_test=args.test_knowledge_adaptation)
    # Resumes check the exact source and prompt bytes; new runs get fresh agent sessions.
    sources = {}
    for name in spec.allowed_source_paths:
        source = args.agent_workspace / name
        if source.is_symlink():
            raise ValueError("source snapshot contains symlink")
        sources[name] = source.read_text()
    contract = {"source_sha256": digest(sources),
                "instructions_sha256": digest(args.instructions.read_text()),
                "implementation_sha256": implementation_digest(driver.ROOT),
                "protocol": "knowledge-v2", "context_mode": CONTEXT_MODE,
                "knowledge_authority": KNOWLEDGE_AUTHORITY}
    path = root / "knowledge/source_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("source/prompt changed on knowledge resume")
    write_json(path, contract)
    workspace = root / "agent-workspace"
    for name, content in sources.items():
        destination = workspace / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_text() != content:
            raise ValueError("private source snapshot changed")
        destination.write_text(content)
    for name, content in seeds.items():
        destination = workspace / "knowledge/seed" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_text() != content:
            raise ValueError("private seed snapshot changed")
        destination.write_text(content)
    return spec, store, workspace


def implementation_digest(repo):
    paths = ["agentic3d/" + name for name in
             ("knowledge.py", "sta_protocol.py", "sta_knowledge_runner.py", "enhanced_task.py",
              "research_feedback.py", "research_io.py", "evidence.py")]
    paths += ["scripts/run_sta_codesign_comparison.py"]
    return digest({name: (repo / name).read_text() for name in paths})


def model_usage(root):
    """Actual trace accounting; missing usage is unavailable, not zero tokens."""
    counts = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    calls = seconds = observed = 0
    for role in root.glob("*-v2"):
        for path in role.glob("calls/*/result.json"):
            value = json.loads(path.read_text())
            calls += 1
            seconds += value.get("duration_seconds", 0.)
        for path in role.glob("calls/*/events.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                    observed += 1
                    for key in counts:
                        counts[key] += event["usage"].get(key, 0)
    return {"completed_model_calls": calls, "model_wall_seconds": seconds,
            "turns_with_usage": observed, "tokens": counts if observed else None}


def publish_knowledge(store, workspace, spec):
    records = store.records()
    for record in records:
        for task in record["tasks"]:
            validate_task(task, spec)
    # Append-only event names preserve revisions rather than silently overriding seeds.
    write_json(workspace / "knowledge/active.json", records)
    return knowledge_catalog(store, workspace)


def require_evidence(report, workspace):
    """Reject unavailable/tampered evidence before any model reasons about it."""
    if not isinstance(report, dict) or not isinstance(report.get("files"), list):
        raise ResearchBlocked("Evidence unavailable: malformed file index")
    if report.get("error") or not report.get("files"):
        raise ResearchBlocked("Evidence unavailable: " + str(report.get("error") or "empty file index"))
    for item in report["files"]:
        try:
            path = Path(item["local_file"])
            if not path.is_absolute():
                raise ValueError("evidence path must be absolute")
            path.resolve().relative_to(workspace.resolve())
            if path.is_symlink() or digest(path.read_text()) != item["sha256"]:
                raise ValueError("hash mismatch or symlink")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ResearchBlocked("Evidence unreadable, malformed or changed: " + str(item)) from error
    return report


def ensure_evidence(driver, remote_run, destination, workspace):
    saved = destination / "index.json"
    if saved.exists():
        try:
            report = json.loads(saved.read_text())
        except (OSError, ValueError) as error:
            raise ResearchBlocked("Evidence index unreadable: " + str(saved)) from error
        if not isinstance(report, dict):
            raise ResearchBlocked("Evidence index malformed: " + str(saved))
        # Explicit resume may retry a failed read; never silently replace good,
        # hash-mismatched evidence or run another physical evaluation.
        if not report.get("error") and report.get("files"):
            return require_evidence(report, workspace)
    return require_evidence(collect_evidence(driver, remote_run, destination), workspace)


def validate_evidence_citation(name, workspace):
    """Accept a local file or an unambiguous original path in our trusted index."""
    workspace = workspace.resolve()
    path = Path(name)
    local = path if path.is_absolute() else workspace / path
    try:
        local.resolve().relative_to(workspace)
    except ValueError:
        raise ValueError("evidence citation escapes this experiment")
    if local.is_file():
        return local
    matches = set()
    for index in (workspace / "evidence").rglob("index.json"):
        if index.is_symlink():
            continue
        index.resolve().relative_to(workspace)
        for item in json.loads(index.read_text()).get("files", []):
            if item.get("path") != name:
                continue
            target = Path(item["local_file"])
            target = target if target.is_absolute() else workspace / target
            try:
                target.resolve().relative_to(workspace)
            except ValueError:
                raise ValueError("indexed evidence escapes this experiment")
            if target.is_file() and digest(target.read_text()) == item.get("sha256"):
                matches.add(target.resolve())
    if len(matches) == 1:
        return matches.pop()
    raise ValueError("evidence citation missing or ambiguous: " + name +
                     "; use the exact local_file from the relevant evidence index")


class ValidationExhausted(ValueError):
    """A model response was rejected, not an infrastructure failure."""


class SearchBudgetExhausted(RuntimeError):
    """Bounded search ended without a new admissible candidate."""


MAX_TASK_PLANS_PER_FINAL = 3
MAX_RESPONSE_SAMPLES = 3


def cached_call(agent, path, prompt, validate, *, resample=False):
    if path.exists():
        response = json.loads(path.read_text())
        validate(response)
        return response
    ledger = path.with_suffix(".sampling.json")
    state = {"prompt_sha256": digest(prompt), "resample": resample, "samples": [], "pending": False}
    if ledger.exists():
        state = json.loads(ledger.read_text())
        if state["prompt_sha256"] != digest(prompt) or state["resample"] != resample:
            raise RuntimeError("sampling contract changed: " + str(ledger))
        if state["pending"]:
            raise RuntimeError("model call interrupted/failed; inspect before retry: " + str(ledger))
    errors = []
    for index in range(MAX_RESPONSE_SAMPLES):
        repair = ""
        if errors:
            instruction = (
                "The previous candidate is invalid (selection score 0, no physical metrics). "
                "You may abandon its hypothesis and sample a different solution within the SAME task scope. "
                "Inspect the frozen source; return a complete patch, not an incremental diff. "
                if resample else
                "Repair the rejected response below, preserving its hypothesis and evidence wherever "
                "compatible with the complete output contract. Do not restart research or invent a "
                "different task merely to fix serialization. "
            )
            repair = ("\n# Bounded response repair/resampling\n" + instruction +
                      "Fix ALL accumulated errors. Return the full response.\n" +
                      canonical({"validation_errors": errors,
                                 "rejected_response": state["samples"][index - 1]["response"]}))
        if index == len(state["samples"]):
            # Persist before dispatch: uncertain API calls must not silently get
            # a fresh budget on resume. OSError/RuntimeError deliberately escape.
            state["pending"] = True
            write_json(ledger, state)
            try:
                response = agent.ask(prompt + repair)
                sample = {"response": response, "parse_error": None}
            except json.JSONDecodeError as error:
                sample = {"response": None, "parse_error": str(error)}
            except ResearchBlocked as error:
                state["pending"] = False
                state.setdefault("blocked", []).append(str(error))
                write_json(ledger, state)
                # A blocked role produced no candidate and spent no repair slot.
                # Explicit resume may ask again; earlier invalid samples remain.
                raise
            except ValueError as error:
                sample = {"response": None, "parse_error": str(error)}
            except (RuntimeError, OSError) as error:
                if not resample:
                    # A known failed planning/reflection API call pauses this
                    # invocation; a later explicit resume may retry it. It is
                    # not a rejected candidate and cannot consume final budget.
                    state["pending"] = False
                    state.setdefault("infrastructure_errors", []).append(str(error))
                    write_json(ledger, state)
                raise
            state["samples"].append(sample)
            state["pending"] = False
            write_json(ledger, state)
        sample = state["samples"][index]
        response = sample["response"]
        try:
            if sample["parse_error"]:
                raise ValueError("invalid model JSON: " + sample["parse_error"])
            validate(response)
        except (ValueError, KeyError, TypeError, SyntaxError) as error:
            errors.append(str(error))
            write_json(path.with_suffix(".validation.json"), {"errors": errors,
                       "last_rejected_response": response})
            continue
        write_json(path, response)
        return response
    raise ValidationExhausted("structured response failed three validations: " + "; ".join(errors))


def candidate_identity(candidate):
    """Notes/hypotheses are not changes to the evaluated algorithm."""
    return digest({key: value for key, value in candidate.items() if key != "notes"})


def invalid_trial(attempt, error):
    return {"attempt": attempt, "status": "invalid_candidate", "valid": False,
            "selection_score": 0., "proxy_score": None, "physical_evaluated": False,
            "result": {"metrics": {}, "error_type": "CandidateValidationError",
                       "error_message": str(error)}}


def run(args, root, driver):
    try:
        return _run(args, root, driver)
    except ResearchBlocked as error:
        path = root / "history.json"
        history = json.loads(path.read_text()) if path.exists() else []
        write_json(root / "search_status.json", {
            "state": "needs-attention", "reason": "research-blocked", "message": str(error),
            "completed_final": len(history), "remaining_final": args.rounds - len(history)})
        raise


def _run(args, root, driver):
    spec, store, workspace = initialize(args, root, driver)
    shared = getattr(driver, "COMMON_CONTRACT", COMMON) + "\n# Exact case contract/source\n" + args.instructions.read_text()
    role_prompt = getattr(driver, "role_prompt", lambda role: globals()[role])
    reference = driver._remote_result(str(Path(args.reference_run).parent / "evaluation_result.json"))
    driver.validate_case_reference(args.case, {"benchmark_id": args.case + "_lol_ng45_tp_gnn_sta"}, reference)
    history = driver._history(root, args.rounds)
    reference_metrics = reference["metrics"]
    incumbent = json.loads((driver.STA_PROFILE / "initial_candidate/candidate.json").read_text())
    incumbent_score, incumbent_run = 1., args.reference_run
    incumbent_metrics = reference_metrics
    for record in history:
        if record["accepted_as_incumbent"]:
            incumbent = json.loads((root / record.get("round_path", f'rounds/{record["round"]:02d}') / "outer-candidate/candidate.json").read_text())
            incumbent_score = record["outer_score"]
            incumbent_run = record["remote_run"]
            incumbent_metrics = record["outer_result"]["metrics"]
    access_checked = False
    def check_access():
        nonlocal access_checked
        if not access_checked:
            report = require_evidence(json.loads((workspace / "evidence/reference/index.json").read_text()), workspace)
            check_research_access(args, root / "research-preflight-v2", workspace, driver,
                                  workspace / spec.allowed_source_paths[0], Path(report["files"][0]["local_file"]))
            access_checked = True
    def agent(name, schema):
        role = StatelessResearcher(name, root / name, schema, args.model, args.codex_bin, workspace, driver)
        role.access_check = check_access
        return role
    analyzer = agent("global-analyzer-v2", ANALYZER_SCHEMA)
    specialists = {stage: agent(stage + "-researcher-v2", LOCAL_SCHEMA)
                   for stage in ("partition", "placement", "joint")}
    # Reflection has a separate structured session and sees the actual outer feedback.
    reflector = agent("knowledge-reflector-v2", REFLECTION_SCHEMA)
    reviewer = agent("global-analyzer-review-v2", REVIEW_SCHEMA)
    evidence_index = root / "agent-workspace/evidence/index.json"
    ensure_evidence(driver, args.reference_run, workspace / "evidence/reference", workspace)
    if not evidence_index.exists():
        write_json(evidence_index, {"reference": "reference/index.json", "rounds": []})
    # Both phases are recoverable after a history write, including the last round.
    # Only Analyzer planning/review calls publish into KnowledgeStore.
    research_history = []
    def reflect(record):
        number = record["round"]
        round_dir = root / record.get("round_path", f"rounds/{number:02d}")
        research_id = record.get("research_id", f"round-{number:02d}")
        final_evaluated = record.get("final_evaluated", True)
        if final_evaluated:
            record["evidence"] = ensure_evidence(driver, record["remote_run"],
                workspace / "evidence" / f"round-{number:02d}", workspace)
            write_json(root / "history.json", history)
            write_json(evidence_index, {"reference": "reference/index.json",
                                       "rounds": [r["evidence"] for r in history]})
        citation = lambda name: validate_evidence_citation(name, workspace)
        feedback = []
        for path in sorted((round_dir / "inner").glob("*/response.json")):
            response = json.loads(path.read_text())
            advice = validate_feedback(response, "local", citation)
            candidate = json.loads((path.parent / "merged_candidate.json").read_text())
            feedback.append({"attempt": int(path.parent.name), "feedback": advice,
                             "candidate_sha256": digest(candidate),
                             "candidate": candidate})
        selected = json.loads((round_dir / "outer-candidate/candidate.json").read_text())
        # Exact source patches stay in experiment evidence, not reusable knowledge.
        artifact = workspace / "evidence" / f"{research_id}-research.json"
        write_json(artifact, {"local_attempts": feedback, "selected_candidate": selected,
                             "selected_candidate_sha256": digest(selected),
                             "rejected_trials": record.get("inner_trials", []),
                             "final_evaluated": final_evaluated,
                             "round": compact_history([record])[0]})
        current_history = [r for r in history if r["round"] < number or
                           (final_evaluated and r["round"] == number)]
        accepted_history = [r for r in current_history if r["accepted_as_incumbent"]]
        best_record = accepted_history[-1] if accepted_history else None
        context = {"round": compact_history([record])[0],
                   "enhanced_task": read_enhanced_task(round_dir / "enhanced_task.json"),
                   "case": args.case, "phase": args.knowledge_phase,
                   "final_evaluated": final_evaluated,
                   "budget": {"total_final": args.rounds, "completed_final": len(current_history),
                              "remaining_final": args.rounds - len(current_history),
                              "completed_inner": sum(r["inner_attempts"] for r in current_history)},
                   "reference_metrics": reference_metrics,
                   "incumbent_metrics": best_record["outer_result"]["metrics"] if best_record else reference_metrics,
                   "incumbent_score": best_record["outer_score"] if best_record else 1.,
                   "history": compact_history(current_history),
                   "candidate_evidence": {"path": artifact.relative_to(workspace).as_posix(),
                                          "sha256": digest(json.loads(artifact.read_text())),
                                          "selected_candidate_sha256": digest(selected)},
                   "local_feedback": [{k: v for k, v in item.items() if k != "candidate"} for item in feedback],
                   "knowledge_catalog": publish_knowledge(store, workspace, spec),
                   "evidence_index": str(evidence_index)}
        reflection = cached_call(reflector, round_dir / "reflection.json",
                                 shared + "\n" + role_prompt("REFLECTOR") + "\n" + canonical(context),
                                 lambda r: validate_feedback(r, "reflector", citation))
        suggestions = index_suggestions(context["local_feedback"], reflection)
        review = cached_call(reviewer, round_dir / "knowledge_review.json",
                             shared + "\n" + role_prompt("KNOWLEDGE_REVIEW") + "\n" + canonical({
                                 **context, "reflection": reflection, "suggestions": suggestions}),
                             lambda r: validate_review(r, suggestions, spec))
        skills, tasks = validate_review(review, suggestions, spec)
        store.save(f"{research_id}-review", skills, tasks,
                   {"round": number, "author": "global-analyzer", "author_phase": "knowledge-review",
                    "knowledge_authority": KNOWLEDGE_AUTHORITY,
                    "outer_score": record["outer_score"], "accepted": record["accepted_as_incumbent"],
                    "review_sha256": digest(review), "suggestions_sha256": digest(suggestions),
                    "selected_candidate_sha256": digest(selected),
                    "validation": "Analyzer reviewed physical outcome; generalization not independently proven"})
        # Within-run advice/outcomes are feedback, not reusable definitions. In
        # frozen tests no test-proposal task/skill bodies enter the next plan.
        research_history.append({"round": number, "research_id": research_id,
                                 "final_evaluated": final_evaluated,
                                 "rejected_trials": record.get("inner_trials", []) if not final_evaluated else [],
                                 "suggestions": suggestions,
                                 "reflection_assessment": reflection["assessment"],
                                 "decisions": review["decisions"]})
    for record in history:
        for rejected_path in sorted((root / "rounds" / f'{record["round"]:02d}').glob("**/rejected.json")):
            reflect(json.loads(rejected_path.read_text()))
        reflect(record)
    while len(history) < args.rounds:
        number = len(history) + 1
        base_dir = root / "rounds" / f"{number:02d}"
        rejected = [json.loads(p.read_text()) for p in sorted(base_dir.glob("**/rejected.json"))]
        for record in rejected:
            if not any(r.get("research_id") == record["research_id"] for r in research_history):
                reflect(record)
        if len(rejected) >= MAX_TASK_PLANS_PER_FINAL:
            write_json(root / "search_status.json", {
                "state": "needs-attention", "reason": "candidate-search-budget-exhausted",
                "completed_final": len(history), "remaining_final": args.rounds - len(history),
                "rejected_task_plans": len(rejected), "incumbent_score": incumbent_score})
            raise SearchBudgetExhausted("candidate search exhausted; no extra final charged")
        task_plan = len(rejected) + 1
        remaining_inner = 3 - sum(t.get("physical_evaluated", False)
                                  for r in rejected for t in r["inner_trials"])
        research_id = f"round-{number:02d}" + (f"-replan-{task_plan:02d}" if task_plan > 1 else "")
        learned = publish_knowledge(store, workspace, spec)
        round_dir = base_dir if task_plan == 1 else base_dir / "replans" / f"{task_plan:02d}"
        state = {"case": args.case, "phase": args.knowledge_phase,
                 "budget": {"total_final": args.rounds, "completed_final": len(history),
                            "remaining_final_including_this": args.rounds-len(history),
                            "max_inner_this_round": 3,
                            "remaining_inner_physical": remaining_inner,
                            "task_plan": task_plan, "max_task_plans_per_final": MAX_TASK_PLANS_PER_FINAL,
                            "max_response_samples_per_candidate": MAX_RESPONSE_SAMPLES,
                            "completed_inner": sum(r["inner_attempts"] for r in history)},
                 "reference_metrics": reference_metrics, "incumbent_metrics": incumbent_metrics,
                 "incumbent_score": incumbent_score, "incumbent_candidate": incumbent,
                 "history": compact_history(history), "knowledge_catalog": learned,
                 "recent_research_feedback": research_history[-5:],
                 "tool_catalog": "knowledge/seed/tools.json",
                 "evidence_index": str(evidence_index)}
        def validate_plan(response):
            validate_task(response["local_task"], spec)
            decode_updates(response, spec)
            for name in response["evidence_paths"]:
                validate_evidence_citation(name, workspace)
            attempts = response["max_inner_attempts"]
            if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
                raise ValueError("invalid inner budget")
            if response["local_task"]["evaluation"] == "final" and attempts != 1:
                raise ValueError("final-only task has one candidate, not extra STA trials")
            if response["local_task"]["evaluation"] != "final" and attempts > remaining_inner:
                raise ValueError("remaining inner physical budget is " + str(remaining_inner) +
                                 "; reduce attempts or choose a final-only task")
            compile_enhanced_task(response, learned, spec, workspace)
        plan = cached_call(analyzer, round_dir / "plan.json",
                           shared + "\n" + role_prompt("ANALYZER") + "\nSTATE:\n" + canonical(state), validate_plan)
        enhanced = compile_enhanced_task(plan, learned, spec, workspace)
        handoff_path = round_dir / "enhanced_task.json"
        if handoff_path.exists() and json.loads(handoff_path.read_text()) != enhanced:
            raise ValueError("enhanced task changed on resume")
        write_json(handoff_path, enhanced)
        task = plan["local_task"]
        skills, tasks = decode_updates(plan, spec)
        store.save(f"{research_id}-plan", skills, [task] + tasks,
                   {"round": number, "author": "global-analyzer", "validation": "proposed",
                    "knowledge_authority": KNOWLEDGE_AUTHORITY})
        selected_spec = task_spec(task, spec)
        local_incumbent = dict(incumbent)
        local_incumbent["source_patch"] = "".join(text for path, text in
            driver._patch_sections(incumbent["source_patch"]).items() if path in task["source_scope"])
        attempts = plan["max_inner_attempts"] if task["evaluation"] != "final" else 1
        primitive = task["evaluation"]
        primitive_spec = None if primitive == "final" else driver.load_task(
            driver.TP_GNN_PROXY_PROFILE if primitive == "partition" else driver.PLACEMENT_PROXY_PROFILE)
        def project_inner(merged):
            stage_paths = {p for p in spec.allowed_source_paths
                           if (p == PARTITION_PATH) == (primitive == "partition")}
            candidate = dict(merged)
            candidate["source_patch"] = "".join(text for path, text in
                driver._patch_sections(merged["source_patch"]).items() if path in stage_paths)
            if primitive == "placement":
                candidate["config_overrides"] = {}
            return candidate
        trials, best = [], None
        for attempt in range(1, attempts + 1):
            folder = round_dir / "inner" / f"{attempt:02d}"
            prompt = shared + "\n" + role_prompt("LOCAL") + "\n" + canonical(local_context(
                state, enhanced, local_incumbent, attempt, trials,
                {key: getattr(selected_spec, key) for key in
                 ("max_added_lines", "max_deleted_lines", "max_changed_files")}))
            def validate_candidate(response):
                validate_feedback(response, "local", lambda name: validate_evidence_citation(name, workspace))
                candidate = driver.materialize_program(selected_spec, response["program"], None)
                driver.verify_candidate(selected_spec, candidate)
                driver._check_source_patch_applies(workspace, candidate.get("source_patch", ""), folder / "apply-check")
                merged_check = driver._merge_candidate(incumbent, candidate, plan["hypothesis"],
                                                       replace_paths=set(task["source_scope"]))
                driver.verify_candidate(spec, merged_check)
                driver._check_source_patch_applies(workspace, merged_check.get("source_patch", ""), folder / "merged-apply-check")
                if candidate_identity(merged_check) == candidate_identity(incumbent):
                    raise ValueError("candidate is unchanged from incumbent; no redundant physical evaluation")
                if primitive_spec is not None:
                    driver.verify_candidate(primitive_spec, project_inner(merged_check))
            try:
                response = cached_call(specialists[task["stage"]], folder / "response.json", prompt,
                                       validate_candidate, resample=True)
            except ValidationExhausted as error:
                trial = invalid_trial(attempt, error)
                write_json(folder / "rejection.json", trial)
                trials.append(trial)
                write_json(round_dir / "inner_trials.json", trials)
                continue
            candidate = driver.materialize_program(selected_spec, response["program"], None)
            # Both cheap helpers and the final evaluator see a globally valid candidate.
            merged = driver._merge_candidate(incumbent, candidate, plan["hypothesis"],
                                             replace_paths=set(task["source_scope"]))
            driver.verify_candidate(spec, merged)
            write_json(folder / "merged_candidate.json", merged)
            # Cheap helpers accept their stage's full source scope, not the other stage.
            if primitive == "final":
                best = (None, merged)
                break
            inner_candidate = project_inner(merged)
            driver.verify_candidate(primitive_spec, inner_candidate)
            write_json(folder / "candidate/candidate.json", inner_candidate)
            cached = folder / "evaluation.json"
            if cached.exists():
                result = json.loads(cached.read_text())
                run_path = json.loads((folder / "evaluation_run.json").read_text())["run"]
            else:
                result, run_path = driver.evaluate_remote(
                    task="tp_gnn_sta_inner" if primitive == "partition" else "placement_sta_inner",
                    local_candidate=folder / "candidate",
                    remote_root=driver.REMOTE_COMPARISON + f"/agentic3d-knowledge-v2/{research_id}/inner/{attempt:02d}",
                    input_kind="bundle" if primitive == "partition" else "run",
                    input_path=driver.REMOTE_BUNDLE if primitive == "partition" else incumbent_run,
                    reference_run=incumbent_run, gpu=args.gpu, retention="compact",
                    log_path=folder / "remote_eval.log")
                write_json(folder / "evaluation_run.json", {"run": run_path})
                write_json(cached, result)
            ensure_evidence(driver, run_path, workspace / "evidence" / f"{research_id}-inner-{attempt:02d}", workspace)
            score = proxy_score(task, result, incumbent_metrics)
            trials.append({"attempt": attempt, "proxy_score": score, "physical_evaluated": True,
                           "valid": score is not None, "selection_score": score if score is not None else 0.,
                           "result": driver._compact_result(result)})
            if score is not None and (best is None or score > best[0]):
                best = (score, merged)
            if score is not None and score > 1. + args.inner_tolerance:
                break
        proposed = best[1] if best is not None else dict(incumbent)
        write_json(round_dir / "outer-candidate/candidate.json", proposed)
        write_json(round_dir / "inner_trials.json", trials)
        if best is None:
            record = {"round": number, "research_id": research_id,
                      "round_path": round_dir.relative_to(root).as_posix(), "final_evaluated": False,
                      "global_plan": plan, "target": task["stage"], "inner_attempts": len(trials),
                      "inner_trials": trials, "outer_score": None, "accepted_as_incumbent": False,
                      "outer_result": {"metrics": {}, "error_type": "NoValidCandidate",
                                       "error_message": "Replan using rejected candidate/inner feedback; no final performed"}}
            write_json(round_dir / "rejected.json", record)
            print(canonical({"event": "task_rejected", **record}), flush=True)
            continue
        result, run_path = driver.evaluate_remote(task=getattr(driver, "FINAL_TARGET", "tp_gnn_placement_sta"),
            local_candidate=round_dir / "outer-candidate",
            remote_root=driver.REMOTE_COMPARISON + f"/agentic3d-knowledge-v2/{research_id}/outer",
            input_kind="bundle", input_path=driver.REMOTE_BUNDLE,
            reference_run=args.reference_run, gpu=args.gpu, retention="full",
            log_path=round_dir / "outer_eval.log")
        write_json(round_dir / "outer_result.json", result)
        score = getattr(driver, "final_score", final_score)(result)
        accepted = score is not None and score > incumbent_score
        record = {"round": number, "protocol": "knowledge-v2", "global_plan": plan,
                  "round_path": round_dir.relative_to(root).as_posix(), "research_id": research_id,
                  "final_evaluated": True,
                  "target": task["stage"], "inner_attempts": len(trials), "inner_trials": trials,
                  "outer_score": score, "accepted_as_incumbent": accepted,
                  "outer_result": driver._compact_result(result), "remote_run": run_path,
                  "evidence": {"remote_run": run_path, "files": [], "error": "collection pending"}}
        # Physical accounting is durable BEFORE evidence or model calls. An
        # explicit resume repairs those reads/reviews, never repeats this final.
        history.append(record)
        driver._write_json(root / "history.json", history)
        write_json(evidence_index, {"reference": "reference/index.json",
                                   "rounds": [r["evidence"] for r in history]})
        if accepted:
            incumbent, incumbent_score, incumbent_run = proposed, score, run_path
            incumbent_metrics = result["metrics"]
        print(canonical({"event": "outer_round", **record}), flush=True)
        reflect(record)
    publish_knowledge(store, workspace, spec)
    summary = {"protocol": "knowledge-v2", "context_mode": CONTEXT_MODE,
               "knowledge_authority": KNOWLEDGE_AUTHORITY,
               "mode": "agentic3d", "model": args.model,
               "final_sta_evaluations": len(history), "incumbent_score": incumbent_score,
               "valid_final_sta_evaluations": sum(r["outer_score"] is not None for r in history),
               "failed_final_sta_evaluations": sum(r["outer_score"] is None for r in history),
               "inner_proxy_evaluations": sum(t.get("physical_evaluated", True)
                   for p in (root / "rounds").glob("**/inner_trials.json") for t in json.loads(p.read_text())),
               "invalid_candidates": len(list((root / "rounds").glob("**/rejection.json"))),
               "rejected_task_plans": len(list((root / "rounds").glob("**/rejected.json"))),
               "history": history, "knowledge_manifest": store.manifest,
               "model_usage": model_usage(root)}
    if getattr(args, "objective", None) is not None:
        summary["objective"] = args.objective
        for prefix in ("", "valid_", "failed_"):
            summary[prefix + "final_evaluations"] = summary.pop(prefix + "final_sta_evaluations")
    write_json(root / "summary.json", summary)
    write_json(root / "search_status.json", {"state": "complete", "completed_final": len(history),
                                           "remaining_final": 0})
    if args.knowledge_export:
        store.export(args.knowledge_export)
