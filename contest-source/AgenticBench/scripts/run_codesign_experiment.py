#!/usr/bin/env python3
"""New, endpoint-aware co-design run. Does not resume historical STA experiments."""
from __future__ import annotations

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic3d import codesign_endpoint as endpoint
from agentic3d.knowledge import digest, write_json
from agentic3d import sta_knowledge_runner as knowledge
from scripts import run_sta_codesign_comparison as legacy
from scripts.codesign_runtime import CASES, Runtime


def immutable_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("immutable run contract differs: " + str(path))
    write_json(path, value)


def qualify_reference(case, objective, bundle, reference_run):
    from local_bench.metrics import EXTRACTORS
    from local_bench.benchmark_input import validate_benchmark_input
    from local_bench.tp_gnn_contract import validate_tp_gnn_method
    validated_input = validate_benchmark_input(bundle, task_id=endpoint.TARGETS[objective])
    metadata = json.loads((bundle / "benchmark_input.json").read_text())
    result = json.loads((reference_run.parent / "evaluation_result.json").read_text())
    endpoint.validate_reference(case, metadata, result, objective)
    if result.get("artifacts", {}).get("exit_code") != 0:
        raise ValueError("reference lacks successful evaluator process evidence")
    partition_manifest = json.loads((reference_run / "20_partition/stage_manifest.json").read_text())
    placement_manifest = json.loads((reference_run / "30_placement/stage_manifest.json").read_text())
    if placement_manifest.get("method") != "concurrent-dreamplace":
        raise ValueError("reference used unexpected method: 30_placement")
    report = json.loads((reference_run / "20_partition/tp_gnn_report.json").read_text())
    # Accept each genuine algorithm identity; a configuration alias is not
    # permission to relabel a historical reference or omit native flat proof.
    validate_tp_gnn_method(report, partition_manifest.get("method"))
    if report.get("config", {}).get("decoder") != "global-2way":
        raise ValueError("reference requires proven TP-GNN global-2way decoder")
    # Re-read trusted raw artifacts; a successful-looking summary alone is insufficient.
    kwargs = ({"frozen_input_run": Path(validated_input["run_dir"])}
              if objective == "thermal" else {})
    measured = EXTRACTORS[endpoint.TARGETS[objective]](reference_run, None, **kwargs)
    for key, value in measured.items():
        if result["metrics"].get(key) != value:
            raise ValueError("reference result differs from raw artifacts: " + key)
    # Extractors validate physical artifacts, but do not own process success.
    # Preserve that independently checked envelope field, not a fabricated score.
    endpoint.validate_reference(case, metadata, {"metrics": {
        **measured, "runs_successfully": result["metrics"]["runs_successfully"]}}, objective)
    relative = Path(metadata.get("run_dir", "run"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid bundle run path")
    for stage in ("00_input", "10_floorplan"):
        source = bundle / relative / stage / "stage_manifest.json"
        dest = reference_run / stage / "stage_manifest.json"
        if json.loads(source.read_text()) != json.loads(dest.read_text()):
            raise ValueError("reference/frozen input mismatch: " + stage)
    return result


def make_driver(args, runtime):
    driver = SimpleNamespace(**vars(legacy))
    driver.STA_PROFILE = ROOT / "evaluation_profiles" / endpoint.PROFILES[args.objective]
    driver.FINAL_TARGET = endpoint.TARGETS[args.objective]
    driver.COMMON_CONTRACT = endpoint.common(args.objective, args.rounds)
    driver.role_prompt = lambda role: endpoint.role_prompt(role, args.objective)
    driver.final_score = lambda result: endpoint.final_score(result, args.objective)
    driver.validate_case_reference = lambda case, bundle, result: endpoint.validate_reference(case, bundle, result, args.objective)
    driver.REMOTE_BUNDLE = str(args.bundle)
    driver.REMOTE_COMPARISON = str(args.output / "physical")
    def with_proxy_context(result, run):
        if args.objective == "thermal" and result.get("metrics", {}).get("runs_successfully") == 1:
            from local_bench.metrics import _placement_native
            metrics = dict(result["metrics"])
            try:
                metrics.update(_placement_native(run))
            except (OSError, ValueError):
                pass  # unavailable diagnostics are NOT invented or treated as zero
            partition = run / "20_partition/partition_metrics.json"
            if partition.exists():
                raw = json.loads(partition.read_text())
                if "timing_weighted_cut" in raw:
                    metrics["timing_weighted_cut"] = raw["timing_weighted_cut"]
            return {**result, "metrics": metrics}
        return result
    driver._remote_result = lambda path: with_proxy_context(json.loads(Path(path).read_text()), Path(path).parent / "run")
    # collect_evidence supplies shell-quoted, trusted argv, not model-generated commands.
    driver._remote = lambda *argv: subprocess.run(shlex.split(" ".join(argv)),
                                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    def evaluate(**kw):
        root = Path(kw["remote_root"])
        result = runtime.evaluate(kw["task"], kw["local_candidate"], root,
                                  kw["input_kind"], Path(kw["input_path"]),
                                  Path(kw["reference_run"]) if kw["reference_run"] else None,
                                  args.case, kw["log_path"])
        return with_proxy_context(result, root / "result/run"), str(root / "result/run")
    driver.evaluate_remote = evaluate
    return driver


def run_codex(args, driver, reference):
    """Persistent single agent, identical source/judge, one final attempt per round."""
    root = args.output
    spec = driver.load_task(driver.STA_PROFILE)
    agent = driver.PersistentCodex("single-codex", root / "agent", driver.PROGRAM_SCHEMA,
                                   args.model, args.codex_bin, args.agent_workspace)
    history = driver._history(root, args.rounds)
    incumbent = json.loads((driver.STA_PROFILE / "initial_candidate/candidate.json").read_text())
    best = 1.
    for record in history:
        if record["accepted_as_incumbent"]:
            best = record["score"]
            incumbent = json.loads((root / record.get("round_path", f'rounds/{record["round"]:02d}') / "candidate/candidate.json").read_text())
    while len(history) < args.rounds:
        number = len(history) + 1
        base_dir = root / "rounds" / f"{number:02d}"
        rejected = [json.loads(p.read_text()) for p in sorted(base_dir.glob("**/rejected.json"))]
        if len(rejected) >= knowledge.MAX_TASK_PLANS_PER_FINAL:
            write_json(root / "search_status.json", {"state": "needs-attention",
                "reason": "candidate-search-budget-exhausted", "completed_final": len(history),
                "remaining_final": args.rounds - len(history), "rejected_candidates": rejected})
            raise knowledge.SearchBudgetExhausted("candidate search exhausted; no extra final charged")
        sample_group = len(rejected) + 1
        folder = base_dir if sample_group == 1 else base_dir / "resamples" / f"{sample_group:02d}"
        prompt = (driver.COMMON_CONTRACT + "\n# Single-agent baseline\n"
                  "You independently implement partition/placement changes. The multi-agent role/knowledge "
                  "rules above do not require you to emit tasks or skills: return hypothesis and program only. "
                  "You have no physical proxy calls. One final evaluation per round, including failure. "
                  "Read only this private source workspace and evidence; no parent/other-run inspection.\n"
                  + args.instructions.read_text() + "\n" + json.dumps({
                      "budget": {"total_final": args.rounds, "completed_final": len(history)},
                      "reference_metrics": reference["metrics"], "incumbent": incumbent,
                      "incumbent_score": best, "history": history,
                      "invalid_candidates": rejected,
                      "sampling_budget": {"group": sample_group,
                          "max_groups": knowledge.MAX_TASK_PLANS_PER_FINAL,
                          "max_responses_per_group": knowledge.MAX_RESPONSE_SAMPLES}}, ensure_ascii=False))
        def validate(response):
            candidate = driver.materialize_program(spec, response["program"], None)
            driver.verify_candidate(spec, candidate)
            driver._check_source_patch_applies(args.agent_workspace, candidate.get("source_patch", ""), folder / "apply-check")
            if knowledge.candidate_identity(candidate) == knowledge.candidate_identity(incumbent):
                raise ValueError("candidate is unchanged from incumbent; no redundant physical evaluation")
        try:
            response = knowledge.cached_call(agent, folder / "response.json", prompt, validate, resample=True)
        except knowledge.ValidationExhausted as error:
            write_json(folder / "rejected.json", knowledge.invalid_trial(sample_group, error))
            continue
        candidate = driver.materialize_program(spec, response["program"], None)
        write_json(folder / "candidate/candidate.json", candidate)
        (folder / "candidate/program.py").write_text(response["program"])
        result, run_path = driver.evaluate_remote(task=driver.FINAL_TARGET,
            local_candidate=folder / "candidate", remote_root=str(root / "physical" / f"round-{number:02d}"),
            input_kind="bundle", input_path=str(args.bundle), reference_run=args.reference_run,
            gpu=args.gpu, retention="full", log_path=folder / "evaluation.log")
        write_json(folder / "evaluation.json", result)
        evidence = knowledge.collect_evidence(driver, run_path, args.agent_workspace / "evidence" / f"round-{number:02d}")
        score = driver.final_score(result)
        accepted = score is not None and score > best
        if accepted:
            best, incumbent = score, candidate
        record = {"round": number, "round_path": folder.relative_to(root).as_posix(),
                  "score": score, "accepted_as_incumbent": accepted,
                  "hypothesis": response["hypothesis"], "remote_run": run_path,
                  "evidence": evidence, **driver._compact_result(result)}
        history.append(record)
        write_json(root / "history.json", history)
        print(json.dumps({"event": "outer_round", **record}), flush=True)
    write_json(root / "summary.json", {"mode": "codex", "objective": args.objective,
        "model": args.model, "final_evaluations": len(history),
        "valid_final_evaluations": sum(r["score"] is not None for r in history),
        "failed_final_evaluations": sum(r["score"] is None for r in history),
        "incumbent_score": best, "history": history,
        "rejected_candidate_groups": len(list((root / "rounds").glob("**/rejected.json")))})
    write_json(root / "search_status.json", {"state": "complete", "completed_final": len(history),
                                           "remaining_final": 0})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("codex", "agentic3d"), required=True)
    p.add_argument("--objective", choices=tuple(endpoint.PROFILES), required=True)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--rounds", type=int, default=endpoint.DEFAULT_ROUNDS,
                   help="Positive final-attempt budget, including failures (default: 10)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--reference-run", required=True)
    p.add_argument("--gpus", required=True, help="Comma-separated approved physical GPU IDs; shared locks")
    p.add_argument("--share-gpu", action="store_true",
                   help="Allow existing GPU processes when memory/disk admission checks pass")
    p.add_argument("--min-gpu-free-mb", type=int, default=16000,
                   help="Positive minimum free GPU memory in MiB; admission only, not a usage cap")
    p.add_argument("--image", required=True, help="Pinned Docker image ID")
    p.add_argument("--codex-bin", required=True)
    p.add_argument("--model", default="gpt-5.6-sol")
    p.add_argument("--cpus", type=int, default=32)
    p.add_argument("--memory-gb", type=int, default=96)
    p.add_argument("--resume", action="store_true")
    return p


def main():
    args = parser().parse_args()
    endpoint.validate_rounds(args.rounds)
    args.output, args.source, args.bundle, args.batch_root = [p.resolve() for p in
        (args.output, args.source, args.bundle, args.batch_root)]
    args.gpu = int(args.gpus.split(",")[0])  # runtime allocates the actual free approved slot
    runtime = Runtime(args.source, args.batch_root, args.image, args.gpus, args.cpus, args.memory_gb,
                      share_gpu=args.share_gpu, min_gpu_free_mb=args.min_gpu_free_mb)
    reference = qualify_reference(args.case, args.objective, args.bundle, Path(args.reference_run))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = (args.output / "run_config.json").exists()
        if existing and not args.resume:
            raise ValueError("existing experiment requires explicit --resume")
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
        config.update(protocol="codesign-endpoints-v1", source_commit=runtime.source_commit,
                      reference_sha256=digest(reference), bundle_sha256=digest(json.loads(
                          (args.bundle / "benchmark_input.json").read_text())),
                      implementation_sha256=digest({str(p.relative_to(ROOT)): p.read_text() for p in
                          [*sorted((ROOT / "agentic3d").glob("*.py")), Path(__file__),
                           ROOT / "scripts/codesign_runtime.py", ROOT / "scripts/run_sta_codesign_comparison.py"]}))
        immutable_json(args.output / "run_config.json", config)
        driver = make_driver(args, runtime)
        spec = replace(driver.load_task(driver.STA_PROFILE),
                       base_config=runtime.profile(args.case) / "config.yaml",
                       benchmark_input_id=args.case + "_lol_ng45_tp_gnn_sta")
        # Equal private source availability. Never grant either method access to other runs.
        args.agent_workspace = args.output / "source-workspace"
        for name in spec.allowed_source_paths:
            dest = args.agent_workspace / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = (args.source / name).read_text()
            if dest.exists() and dest.read_text() != content:
                raise ValueError("private source changed")
            dest.write_text(content)
        args.instructions = args.output / "instructions.md"
        initial = json.loads((spec.initial_candidate / "candidate.json").read_text())
        prompt = driver.build_problem_description(spec, initial, args.source)
        if args.instructions.exists() and args.instructions.read_text() != prompt:
            raise ValueError("source prompt changed")
        args.instructions.write_text(prompt)
        args.knowledge_seed = args.knowledge_split = args.knowledge_import = args.knowledge_export = None
        args.knowledge_phase = "independent"
        args.test_knowledge_adaptation = False
        args.inner_tolerance = 1e-6
        try:
            if args.mode == "agentic3d":
                knowledge.run(args, args.output, driver)
            else:
                knowledge.collect_evidence(driver, args.reference_run, args.agent_workspace / "evidence/reference")
                run_codex(args, driver, reference)
        except BaseException as error:
            write_json(args.output / f"failure-{time.time_ns()}.json",
                       {"type": type(error).__name__, "message": str(error)})
            raise


if __name__ == "__main__":
    main()
