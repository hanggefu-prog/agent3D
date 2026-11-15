"""Compact, read-only view of the active GRT matrix and native stage progress."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

from grt_bench.common import read, digest
from grt_bench.queue import Queue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    active = read(read(args.root / "control/active.json")["provision"])
    selected = []
    for row in Queue(args.root / "control/queue.sqlite").rows():
        payload = json.loads(row["payload"])
        if payload.get("task_digest") != active["task"] or payload.get("candidate") != active["candidate"]:
            continue
        item = {"case": payload["case"], "seed": payload["seed"]["id"], "state": row["state"]}
        if row["state"] in {"running", "uncertain", "cancelling"}:
            item["elapsed_minutes"] = round((time.time() - row["started"]) / 60, 1)
            run = args.root / "runs" / row["id"]
            phase = "source/prefix preparation"
            prefix_plan = run / "control/prefix-plan.json"
            if prefix_plan.exists():
                image = read(args.root / "control/runtimes" / (payload["candidate"] + ".json"))["image"]
                key = digest({"candidate":payload["candidate"], "task":payload["task_digest"],
                              "case":payload["case"], "io_seed":payload["seed"]["io"], "image":image})
                log = args.root / "cache/frontend-prefixes" / key / "output/prefix.log"
                if log.exists():
                    matches = re.findall(r"\[open3dbench\] (stage [^\n]+|synthesis:[^\n]+)", log.read_text())
                    if matches: phase = "shared prefix: " + matches[-1]
            for name in ("flow-0.log", "flow-1.log"):
                log = run / "output" / name
                if log.exists():
                    matches = re.findall(r"\[open3dbench\] (stage [^\n]+|global_routing:[^\n]+)", log.read_text())
                    if matches: phase = matches[-1]
            if (run / "output/analysis").is_dir(): phase = "joint timing, power and thermal analysis"
            if (run / "output/analysis/metrics.json").exists(): phase = "independent verification"
            item["phase"] = phase
        if row["state"] == "failed":
            item["error"] = json.loads(row["result"]).get("error")
            item["evidence"] = str(args.root / "runs" / row["id"])
        selected.append(item)
    print(json.dumps({"task": active["task"], "counts": dict(Counter(row["state"] for row in selected)),
                      "active_or_complete": [row for row in selected if row["state"] != "queued"]}, indent=2))


if __name__ == "__main__":
    main()
