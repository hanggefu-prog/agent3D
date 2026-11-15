"""Rank agents from independent optimization campaigns (hypervolume, HV-vs-budget, local log scores).

Campaign file: a JSON list of admissions in admission order, or {"admissions": [...]}.
Admission: {"candidate": optional id, "vector": {"timing", "overflow", "temperature_peak_k"}
or [t, o, T], "feasible": bool, "cost": optional {"evaluations", "wall_seconds", "tokens"}}.
Missing evaluation cost counts one per admission. Vectors are normalized to the frozen
baseline (1, 1, 1), e.g. archive aggregate rows. Local-score file: a JSON list of positive
scores (lower is better), or {"scores": [...]}.

  rank_campaigns.py --agent A a1.json a2.json a3.json --agent B b1.json b2.json b3.json \
      [--budget evaluations|wall_seconds|tokens] [--budget-limit X] [--scores A a.json] [--output out.json]
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ranking import BUDGETS, REFERENCE, campaign_summary, mean_log_score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", nargs="+", action="append", required=True, metavar=("NAME", "CAMPAIGN"),
                        help="agent name followed by one JSON file per independent campaign")
    parser.add_argument("--scores", nargs=2, action="append", default=[], metavar=("NAME", "FILE"))
    parser.add_argument("--budget", choices=BUDGETS, default="evaluations")
    parser.add_argument("--budget-limit", type=float)
    parser.add_argument("--reference", type=float, nargs=3, default=list(REFERENCE))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    agents = {}
    for name, *files in args.agent:
        if not files or name in agents:
            parser.error("--agent needs a unique name and at least one campaign file")
        agents[name] = [json.loads(Path(f).read_text()) for f in files]
    result = campaign_summary(agents, args.budget, args.budget_limit, tuple(args.reference),
                              args.resamples, args.confidence, args.seed)
    if args.scores:
        result["local_scores"] = {}
        for name, path in args.scores:
            scores = json.loads(Path(path).read_text())
            scores = scores.get("scores") if isinstance(scores, dict) else scores
            result["local_scores"][name] = mean_log_score(scores, args.resamples, args.confidence, args.seed)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return result


if __name__ == "__main__": main()
