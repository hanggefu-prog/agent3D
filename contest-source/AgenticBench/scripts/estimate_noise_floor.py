"""Estimate the v3 dominance noise floor from repeated reference runs.

Each native RUN file is a complete matrix envelope:
{"outcomes": [...], "provenance": {"root": "/trusted/run1", "task": <complete source task object>}}.
Rows retain verified candidate/task/endpoint/split identity, complete frozen seed
vectors and all five raw metrics. Pass each campaign root separately through
--trusted-root: data files cannot authorize their own directories. Queue leases,
immutable verifier receipts and installed image identities bind physical repeats;
copying a directory or replaying analysis does not create a new execution.

The trust boundary is host-owned metadata in operator-authorized directories.
This tool detects inconsistent evidence; it cannot authenticate a forged or
compromised host store. Bare fixtures need --synthetic and are explicitly marked
as non-physical results. Source-task objects may differ in analysis-only fields
for explicit rescoring; physical settings and source/seed/image identity may not.
The per-cell mean is the pseudo-reference, and the sample standard deviation of
run vectors is the sigma of one evaluation in the target scoring space.

A meaningful noise floor needs at least 5 repeated runs of every cell; 2-4 runs are
accepted with a warning. --across-seeds instead treats the seeds of one or more runs as
repeats of each case (a proxy that also contains true seed sensitivity).

  estimate_noise_floor.py --task task.json [--scoring v3-scoring.json] run1.json run2.json ... \
      --trusted-root /trusted/run1 --trusted-root /trusted/run2 \
      [--split train] [--k 2] [--output noise_floor.json]
  estimate_noise_floor.py --task task.json --synthetic synthetic1.json synthetic2.json
"""
import argparse
import json
import math
import copy
import re
import sqlite3
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.common import confined, digest, finite, read, sha, verify_records
from grt_bench.contract import METRICS, OBJECTIVES, validate
from grt_bench.scoring_v3 import cell, compute

RECOMMENDED_REPEATS = 5


def _physical_contract(task):
    """Measurements may be explicitly rescored, but physical settings stay fixed."""
    value = copy.deepcopy(task)
    for key in ("id", "evaluator_sha256", "scoring", "normalization_floors", "epsilon"):
        value.pop(key, None)
    if "optimization" in value:
        # Algorithm source/ancestry remains bound by candidate and host receipts.
        value["optimization"].pop("prompt", None)
    return value


def _native_provenance(task, run, rows, split, trusted_roots):
    """Check host-owned queue/receipts; never infer executions from paths/inodes.

    The caller must supply an unmodified trusted campaign directory. This is
    offline verification of host evidence, not a signature/authentication system.
    Lease identities are hashed in output so credentials are never exported.
    """
    proof = run.get("provenance") if isinstance(run, dict) else None
    if not isinstance(proof, dict) or set(proof) != {"root", "task"}:
        raise ValueError("Native repeats require provenance with host root and source task; use --synthetic for fixtures")
    source_task = proof["task"]
    source_id = validate(source_task)
    if _physical_contract(source_task) != _physical_contract(task):
        raise ValueError("Repeated runs changed the physical measurement contract")
    root = Path(proof["root"]).resolve(strict=True)
    if root not in trusted_roots:
        raise ValueError("Noise provenance root is not operator-authorized; supply --trusted-root separately")
    database = confined(root, "control/queue.sqlite").resolve(strict=True)
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        jobs = {r["id"]: dict(r) for r in connection.execute("SELECT * FROM jobs")}
    finally:
        connection.close()
    candidate = rows[0]["candidate"] if rows else None
    selected = []
    for row in jobs.values():
        payload = json.loads(row["payload"])
        if (payload.get("candidate"), payload.get("task_digest"), payload.get("split")) == (candidate, source_id, split):
            selected.append((row, payload))
    if len(selected) != len(rows):
        raise ValueError("Native repeated matrix is incomplete or has duplicate host attempts")
    supplied = {(r["case"], r["seed"]["id"]): r for r in rows}
    identities, images, physical_evaluators = {}, {}, {}
    from grt_bench.incremental import physical_source
    for row, payload in selected:
        key = (payload["case"], payload["seed"]["id"])
        if key in identities or key not in supplied or row["state"] != "succeeded":
            raise ValueError("Each native repeat requires unique successful host cells")
        if digest({"payload": payload, "resources": json.loads(row["resources"])}) != row["id"]:
            raise ValueError("Host job identity changed")
        attempt = confined(root, "runs/" + row["id"])
        result = json.loads(row["result"])
        verified = read(confined(attempt, "verified/result.json"))
        if result != read(confined(attempt, "result.json")) or any(result.get(k) != v for k, v in payload.items()):
            raise ValueError("Host result differs from submitted cell")
        if {k: v for k, v in result.items() if k not in {"accounting", "candidate_verification_sha256"}} != verified:
            raise ValueError("Host result differs from independent verification")
        if supplied[key] != verified and supplied[key] != result:
            raise ValueError("Supplied noise metrics differ from verified host evidence")
        plan_path = confined(attempt, "control/plan.json")
        plan = read(plan_path)
        evidence = verified["evidence"]
        analysis = confined(attempt, "output/analysis")
        if (sha(plan_path) != evidence["plan_sha256"] or
                sha(confined(analysis, "manifest.json")) != evidence["analysis_manifest_sha256"] or
                read(confined(attempt, "control/task.json")) != source_task or
                any(plan.get(k) != v for k, v in payload.items())):
            raise ValueError("Verified native task, plan or analysis changed")
        verify_records(analysis, read(confined(analysis, "manifest.json"))["artifacts"])
        physical_job, output = physical_source(root, row["id"])
        original = jobs.get(physical_job)
        if (not original or not isinstance(original.get("token"), str) or
                not re.fullmatch(r"[0-9a-f]{32}", original["token"]) or not original.get("host") or
                original.get("started") is None or original.get("finished") is None):
            raise ValueError("Native repeat lacks a completed physical host lease")
        started = finite(original["started"], "physical lease start", 0)
        if finite(original["finished"], "physical lease finish", 0) < started:
            raise ValueError("Physical lease finished before it started")
        physical = confined(root, "runs/" + physical_job)
        native = json.loads(original["result"] or "null")
        execution = native.get("accounting", {}).get("execution") if isinstance(native, dict) else None
        if (not isinstance(execution, dict) or type(execution.get("exit_code")) is not int or
                execution["exit_code"] != 0 or not (output / "execution.json").is_file()):
            raise ValueError("Native repeat lacks successful physical execution evidence")
        execution_record = read(confined(output, "execution.json"))
        if execution_record.get("physical_execution_reused"):
            raise ValueError("Analysis replay is not a new physical execution")
        runtime = read(confined(physical, "control/python-runtime.json"))
        physical_plan = read(confined(physical, "control/plan.json"))
        physical_task = read(confined(physical, "control/task.json"))
        if validate(physical_task) != physical_plan.get("task_digest"):
            raise ValueError("Physical source task identity differs")
        if (digest(runtime) != physical_plan.get("python_runtime_digest") or
                not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime.get("image", "")) or
                runtime.get("candidate") != candidate):
            raise ValueError("Native repeat runtime identity differs")
        identities[key] = digest({"host": original["host"], "lease": original["token"]})
        images[key] = runtime["image"]
        physical_evaluators[key] = physical_task["evaluator_sha256"]
    return {"task_digest": source_id, "physical_attempts": identities, "images": images,
            "measurement_evaluator": source_task["evaluator_sha256"],
            "physical_evaluators": physical_evaluators}


def _index(task, run, split, synthetic=False, trusted_roots=()):
    rows = run.get("outcomes") if isinstance(run, dict) else run
    if not isinstance(rows, list):
        raise ValueError("Run requires an outcome list")
    proof = run.get("provenance") if isinstance(run, dict) else None
    if proof is not None and not isinstance(proof, dict):
        raise ValueError("Noise provenance must be an object")
    source_task = (proof or {}).get("task", task)
    source_id = validate(source_task)
    if _physical_contract(source_task) != _physical_contract(task):
        raise ValueError("Repeated runs changed the physical measurement contract")
    expected = {(c["id"], s["id"]) for c in task["cases"] if c["split"] == split for s in task["seeds"]}
    result, candidates = {}, set()
    seeds = {s["id"]: s for s in task["seeds"]}
    for row in rows:
        seed = row.get("seed")
        if not isinstance(seed, dict) or seed != seeds.get(seed.get("id")):
            raise ValueError("Full seed vector differs from the frozen task")
        key = (row.get("case"), seed["id"])
        if (row.get("task_digest") != source_id or row.get("endpoint") != task["endpoint"] or
                row.get("split") != split or row.get("status") != "verified"):
            raise ValueError("Noise outcome task/endpoint/split/verification differs")
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate):
            raise ValueError("Noise outcome candidate is not bound")
        candidates.add(candidate)
        if key in result or set(row.get("metrics", ())) != set(METRICS):
            raise ValueError("Duplicate cell or raw metric set differs")
        result[key] = {"metrics": {m: finite(row["metrics"][m], m, 0) for m in METRICS}}
    if set(result) != expected or len(candidates) != 1:
        raise ValueError("Every run requires exactly the case/seed cells of one candidate")
    provenance = ({"task_digest": source_id, "synthetic": True} if synthetic else
                  _native_provenance(task, run, rows, split, trusted_roots))
    return result, next(iter(candidates)), provenance


def _mean_row(rows):
    return {"metrics": {m: statistics.mean(r["metrics"][m] for r in rows) for m in METRICS}}


def estimate(task, runs, split="train", across_seeds=False, k=2.0, *, synthetic=False, trusted_roots=()):
    task_id = validate(task)
    if "version" not in task["scoring"]:
        raise ValueError("Noise floors are defined in scoring v3 space (pass a v3 scoring contract)")
    finite(k, "k", 0)
    if split not in {"train", "test", "auxiliary"}:
        raise ValueError("Unknown noise-estimation split")
    # The input envelopes cannot grant themselves authority. Trust is supplied
    # separately by the operator/caller, never inferred from run JSON or inodes.
    roots = {Path(root).resolve(strict=True) for root in trusted_roots}
    if synthetic and roots:
        raise ValueError("Synthetic fixtures cannot declare trusted native roots")
    checked = [_index(task, run, split, synthetic, roots) for run in runs]
    if len({candidate for _, candidate, _ in checked}) != 1:
        raise ValueError("Repeated runs must evaluate the same candidate")
    indexed = [value for value, _, _ in checked]
    proofs = [proof for _, _, proof in checked]
    if not synthetic:
        if len({proof["measurement_evaluator"] for proof in proofs}) != 1:
            raise ValueError("Repeated runs changed measurement evaluator implementation")
        for key in indexed[0]:
            identities = [proof["physical_attempts"][key] for proof in proofs]
            if len(set(identities)) != len(identities):
                raise ValueError("Repeated analysis/copy of the same physical attempt is not independent noise")
            if len({proof["images"][key] for proof in proofs}) != 1:
                raise ValueError("Repeated runs changed installed runtime image")
            if len({proof["physical_evaluators"][key] for proof in proofs}) != 1:
                raise ValueError("Repeated runs changed physical execution implementation")
    cells = sorted(indexed[0]) if indexed else []
    seeds = [s["id"] for s in task["seeds"]]
    warnings = []
    if across_seeds:
        if len(seeds) < 2:
            raise ValueError("Across-seed estimation needs at least two seeds")
        cases = sorted({c for c, _ in cells})
        # Repeat j: seed j of every case (and run), each against its case mean over seeds and runs.
        refs = {(c, s): _mean_row([r[(c, t)] for r in indexed for t in seeds]) for c in cases for s in seeds}
        samples = [compute(task, [(c, s) for c in cases], r, refs, split) for r in indexed for s in seeds]
        scale = 1 / math.sqrt(len(seeds))  # one evaluation averages every seed of a case
        warnings.append("Across-seed proxy: includes seed sensitivity, not only run-to-run nondeterminism")
    else:
        if len(indexed) < 2:
            raise ValueError("Requires at least two repeated runs")
        refs = {key: _mean_row([r[key] for r in indexed]) for key in cells}
        samples = [compute(task, cells, r, refs, split) for r in indexed]
        scale = 1.0
    if len(samples) < RECOMMENDED_REPEATS:
        warnings.append("Fewer than %d repeats: the noise floor is a rough estimate" % RECOMMENDED_REPEATS)
    sigma = lambda values: scale * statistics.stdev(values)
    floor = {o: sigma([s["vector"][o] for s in samples]) for o in OBJECTIVES}
    metric = {m: sigma([s["metric_ratios"][m] for s in samples]) for m in METRICS}
    if across_seeds:
        groups = {c: (c, [r[(c, t)] for r in indexed for t in seeds], refs[(c, seeds[0])]) for c in cases}
    else:
        groups = {c + "/" + s: (c, [r[(c, s)] for r in indexed], refs[(c, s)]) for c, s in cells}
    per_cell = {name: {m: statistics.stdev([cell(task, case, m, row["metrics"][m], ref["metrics"][m])[0]
                                            for row in rows]) for m in METRICS}
                for name, (case, rows, ref) in sorted(groups.items())}
    scoring = {key: task["scoring"][key] for key in ("temperature", "aggregation", "overflow")}
    return {"noise_floor": floor, "metric_sigma": metric, "per_cell_sigma": per_cell,
            "repeats": len(samples), "runs": len(indexed), "across_seeds": across_seeds, "split": split,
            "task_digest": task_id, "candidate": checked[0][1], "synthetic": synthetic,
            "provenance": [{"task_digest": p["task_digest"],
                            **({"physical_attempts": {c + "/" + seed: value for (c, seed), value in p["physical_attempts"].items()}}
                               if not synthetic else {})} for p in proofs],
            "scoring_basis": scoring, "scoring_basis_digest": digest(scoring),
            "pseudo_reference": "per-cell mean of the repeated raw metrics", "warnings": warnings,
            "dominance": {"mode": "noise", "k": k, "noise_floor": floor}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=Path, required=True, help="Target peak-endpoint scoring v3 task JSON")
    parser.add_argument("--trusted-root", type=Path, action="append", default=[],
                        help="Operator-authorized host campaign root; repeat for every native root (exact paths only)")
    parser.add_argument("--scoring", type=Path, help="v3 scoring object replacing the task's scoring")
    parser.add_argument("--split", default="train")
    parser.add_argument("--across-seeds", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Explicit fixture-only estimate; no physical qualification")
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    task = json.loads(args.task.read_text())
    if args.scoring:
        task["scoring"] = json.loads(args.scoring.read_text())
    result = estimate(task, [json.loads(p.read_text()) for p in args.runs], args.split, args.across_seeds,
                      finite(args.k, "k", 0), synthetic=args.synthetic, trusted_roots=args.trusted_root)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return result


if __name__ == "__main__": main()
