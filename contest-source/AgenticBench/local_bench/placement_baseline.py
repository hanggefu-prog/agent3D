"""Read-only actual-provenance guard for new TP-GNN-frozen placement studies."""
import argparse
import hashlib
import json
from pathlib import Path

from .tp_gnn_contract import validate_tp_gnn_method


def validate_tp_gnn_partition(run, *, expected_method="tp-gnn", require_bound_report=False,
                              expected_decoder="global-2way"):
    run = Path(run)
    def read(name):
        return json.loads((run / "20_partition" / name).read_text())
    stage, artifact, report = (read(name) for name in
                               ("stage_manifest.json", "partition_manifest.json", "tp_gnn_report.json"))
    if require_bound_report:
        # The native producer embeds the report in its hash-published method
        # manifest. An untracked, independently edited report is not evidence.
        path = run / "20_partition/partition_manifest.json"
        record = stage.get("artifacts", {}).get("method_manifest", {})
        if (not isinstance(record, dict) or not isinstance(record.get("path"), str)
                or (path.parent / record["path"]).resolve() != path.resolve()
                or record.get("bytes") != path.stat().st_size
                or record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or artifact.get("method_report") != report):
            raise ValueError("placement partition requires a hash-bound native method report")
    method = validate_tp_gnn_method(report, stage.get("method"), artifact.get("method"), expected_method=expected_method)
    if (stage.get("status") != "success"
            or report.get("config", {}).get("decoder") != expected_decoder):
        raise ValueError("new placement baseline requires actual TP-GNN {} partition artifacts".format(expected_decoder))
    if expected_decoder == "spatial-bin":
        bins = report.get("clustering", {}).get("spatial_bins", {})
        if (bins.get("postclosure", {}).get("accepted") is not True
                or bins.get("handoff_constraints", {}).get("accepted") is not True):
            raise ValueError("Spatial-bin input failed feasibility checks")
    return method


def main():
    from .benchmark_input import validate_benchmark_input
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = validate_benchmark_input(args.bundle)
    if not bundle.get("partition_method"):
        raise ValueError("select a TP-GNN placement input profile with qualified producer identity")
    print(json.dumps({"status": "ok", "benchmark_id": bundle["benchmark_id"],
                      "partition": bundle["partition_method"] + "/global-2way"}))


if __name__ == "__main__":
    main()
