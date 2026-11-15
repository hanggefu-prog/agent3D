"""Host/admin CLI. No model-controlled evaluator changes or implicit launches."""
import argparse
from pathlib import Path
from .common import digest, read, write
from .contract import matrix, validate
from .queue import Queue
from .pareto import aggregate, frontier


def submission_order(task, candidate, split="train"):
    # Start different cases before replicas contend for the same prefix lock.
    # Stable sorting preserves the host's case order within each seed round.
    seeds = {seed["id"]: index for index, seed in enumerate(task["seeds"])}
    return sorted(matrix(task, candidate, split), key=lambda cell: seeds[cell["seed"]["id"]])


def main():
    p=argparse.ArgumentParser(prog="python -m grt_bench")
    commands=p.add_subparsers(dest="command",required=True)
    q=commands.add_parser("validate");q.add_argument("task",type=Path)
    q=commands.add_parser("prepare");q.add_argument("spec",type=Path);q.add_argument("bindings",type=Path);q.add_argument("root",type=Path)
    q=commands.add_parser("freeze");q.add_argument("source",type=Path);q.add_argument("store",type=Path);q.add_argument("--parent");q.add_argument("--policy",type=Path);q.add_argument("--paths",type=Path)
    q=commands.add_parser("register-host");q.add_argument("database");q.add_argument("host");q.add_argument("capacity",type=Path)
    q=commands.add_parser("submit");q.add_argument("database");q.add_argument("task",type=Path);q.add_argument("candidate");q.add_argument("resources",type=Path);q.add_argument("--split",choices=["train","test","auxiliary"],default="train");q.add_argument("--root",type=Path)
    q=commands.add_parser("status");q.add_argument("database")
    q=commands.add_parser("cancel");q.add_argument("database");q.add_argument("job")
    q=commands.add_parser("reconcile");q.add_argument("database");q.add_argument("job");q.add_argument("receipt",type=Path)
    q=commands.add_parser("build");q.add_argument("snapshot",type=Path);q.add_argument("base_image");q.add_argument("store",type=Path);q.add_argument("cache",type=Path);q.add_argument("receipt",type=Path);q.add_argument("--threads",type=int,default=8)
    q=commands.add_parser("aggregate");q.add_argument("task",type=Path);q.add_argument("candidate");q.add_argument("outcomes",type=Path);q.add_argument("reference",type=Path);q.add_argument("output",type=Path);q.add_argument("--split",default="train",choices=["train","test","auxiliary"])
    q=commands.add_parser("frontier");q.add_argument("rows",type=Path);q.add_argument("output",type=Path);q.add_argument("--epsilon",type=float,default=0)
    q=commands.add_parser("task-define",help="Freeze a Pareto task and prompt-selected source scope")
    q.add_argument("root",type=Path);q.add_argument("base",type=Path);q.add_argument("baseline");q.add_argument("prompt",type=Path)
    for name in ("candidate-verify", "candidate-checkout", "candidate-freeze", "archive-ingest", "archive-show", "archive-select"):
        q=commands.add_parser(name);q.add_argument("root",type=Path);q.add_argument("task",type=Path)
        if name in {"candidate-verify", "archive-ingest", "archive-select"}: q.add_argument("candidate")
        if name == "archive-select": q.add_argument("--reason",required=True)
        if name in {"candidate-checkout", "candidate-freeze"}: q.add_argument("source",type=Path);q.add_argument("--selection",required=True)
    a=p.parse_args()
    if a.command == "validate":
        print(validate(read(a.task)))
    elif a.command == "prepare":
        from .common import sha, verify_records, evaluator_identity
        task=read(a.spec);bindings=read(a.bindings)
        task["evaluator_sha256"]=evaluator_identity()
        for case in task["cases"]:
            binding=bindings[case["id"]]
            config=Path(binding["config"]).resolve(strict=True)
            manifest=Path(binding["manifest"]).resolve(strict=True)
            inputs=read(manifest)
            verify_records(manifest.parent,inputs["artifacts"])
            case["config_sha256"]=sha(config)
            case["input_identity"]=digest(inputs)
        identity=validate(task)
        write(a.root / "control/tasks" / (identity + ".json"),task)
        print(identity)
    elif a.command == "freeze":
        from .candidates import freeze
        print(freeze(a.source,a.store,paths=read(a.paths) if a.paths else None,
                     parent=a.parent,policy=read(a.policy) if a.policy else None))
    elif a.command == "register-host":
        Queue(a.database).register_host(a.host,read(a.capacity))
    elif a.command == "submit":
        task=read(a.task)
        if "optimization" in task:
            from .archive import Archive
            if a.root is None or Path(a.database).resolve() != (a.root / "control/queue.sqlite").resolve():
                raise ValueError("Optimization submission requires --root and its authoritative queue")
            if read(a.root / "control/tasks" / (validate(task) + ".json")) != task:
                raise ValueError("Task is not registered in the worker control root")
            Archive(a.root,task).admit(a.candidate)
        q=Queue(a.database)
        for payload in submission_order(task,a.candidate,a.split):
            print(q.submit(payload,read(a.resources)))
    elif a.command == "status":
        import json
        print(json.dumps([{k:r[k] for k in ("id","state","host","submitted","started","finished")} for r in Queue(a.database).rows()],indent=2))
    elif a.command == "cancel":
        Queue(a.database).cancel(a.job)
    elif a.command == "reconcile":
        Queue(a.database).reconcile(a.job,read(a.receipt))
    elif a.command == "build":
        from .builder import build
        write(a.receipt,build(a.snapshot,a.base_image,a.store,a.cache,a.threads))
    elif a.command == "aggregate":
        write(a.output,aggregate(read(a.task),a.candidate,read(a.outcomes),read(a.reference),a.split))
    elif a.command == "frontier":
        write(a.output,frontier(read(a.rows),a.epsilon))
    elif a.command == "task-define":
        from .common import evaluator_identity
        from .optimization import define_task
        base=read(a.base);base["evaluator_sha256"]=evaluator_identity()
        task=define_task(base,a.baseline,a.prompt.read_text(),a.root / "candidates")
        identity=validate(task)
        destination=a.root / "control/tasks" / (identity + ".json")
        if not destination.exists(): write(destination,task)
        elif read(destination) != task: raise ValueError("Task identity collision")
        print(identity)
    elif a.command in {"candidate-verify", "candidate-checkout", "candidate-freeze", "archive-ingest", "archive-show", "archive-select"}:
        import json
        from .archive import Archive
        archive=Archive(a.root,read(a.task))
        if a.command == "candidate-verify": result=archive.admit(a.candidate)
        elif a.command == "candidate-checkout": result=archive.checkout(a.source,a.selection)
        elif a.command == "candidate-freeze": result={"candidate":archive.freeze(a.source,a.selection)}
        elif a.command == "archive-ingest": result=archive.ingest(a.candidate)
        elif a.command == "archive-select": result=archive.select(a.candidate,a.reason)
        else: result=archive.show()
        print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__": main()
