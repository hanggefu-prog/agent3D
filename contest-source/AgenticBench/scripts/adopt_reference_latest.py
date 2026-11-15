"""Zero-EDA import of an explicitly qualified latest GRT reference."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_bench.reference_latest import adopt_latest_input


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-proof", type=Path, required=True)
    parser.add_argument("--execution-proof-sha256", required=True)
    parser.add_argument("--runtime-proof", type=Path)
    parser.add_argument("--runtime-proof-sha256")
    args = parser.parse_args(argv)
    print(json.dumps(adopt_latest_input(args.reference_run, args.output, args.case, args.source,
        execution_proof=args.execution_proof, execution_proof_sha256=args.execution_proof_sha256,
        runtime_proof=args.runtime_proof, runtime_proof_sha256=args.runtime_proof_sha256), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
