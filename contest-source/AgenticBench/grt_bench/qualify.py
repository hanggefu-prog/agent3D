"""Native endpoint smoke on an immutable CTS reference, not seed qualification."""
import argparse
from pathlib import Path
from .common import write
from . import ENDPOINT, TIMING_ENDPOINT
from .endpoint import run
from .verify import verify_analysis


def main():
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "platform", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--pdk", required=True)
    p.add_argument("--hotspot", type=Path, default=Path("/opt/open3dbench-assets/hotspot"))
    p.add_argument("--endpoint",choices=(ENDPOINT,TIMING_ENDPOINT),default=ENDPOINT)
    a = p.parse_args()
    from open3dbench.backend.three_d_route import route_two_dies
    from open3dbench.pdks import get_pdk
    from open3dbench.stage_manifest import validate_stage_manifest
    validate_stage_manifest(a.checkpoint.parent / "stage_manifest.json", expected_stage="cts", verify_artifacts=True)
    a.output.mkdir(parents=True, exist_ok=False)
    pdk = get_pdk(a.pdk)
    checkpoint = route_two_dies(a.checkpoint, a.platform, a.output / "grt",
        bottom_layers=pdk.bottom_routing_layers, top_layers=pdk.top_routing_layers,
        bottom_antenna_cell=pdk.bottom_antenna_cell, top_antenna_cell=pdk.top_antenna_cell,
        bottom_ndr_policy=pdk.routing_ndr_policy[0], top_ndr_policy=pdk.routing_ndr_policy[1],
        threads=4, congestion_iterations=4, grt_only=True)
    run(checkpoint, a.platform, a.hotspot, a.output / "analysis", endpoint=a.endpoint)
    metrics = verify_analysis(a.output / "analysis", expected_endpoint=a.endpoint)
    write(a.output / "qualification.json", {"status":"verified", "kind":"cts-replay-endpoint-smoke",
        "seed_matrix_qualified":False, "metrics":metrics})


if __name__ == "__main__": main()
