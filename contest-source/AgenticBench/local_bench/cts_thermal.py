"""Trusted post-CTS power/HotSpot boundary, separate from routed thermal v1."""
from pathlib import Path


def snapshot_command(workspace, run_dir, config, python, environment, *, power_threads=32):
    from open3dbench.pdks import get_pdk

    if type(power_threads) is not int or power_threads < 1:
        raise ValueError("Power extraction threads must be an explicit positive integer")

    flow = Path(workspace) / "OpenROAD-3D/flow"
    platform = flow / "platforms" / (get_pdk(config["pdk"]).openroad_platform + "_3D")
    return [python, "-m", "open3dbench.backend.thermal_snapshot",
            "--checkpoint", str(Path(run_dir) / "70_cts/checkpoint.json"),
            "--platform-dir", str(platform),
            "--hotspot-root", environment.get("OPEN3DBENCH_HOTSPOT_ROOT", str(flow / "HotSpot")),
            "--output-dir", str(Path(run_dir) / "95_thermal_snapshot"),
            "--openroad", environment.get("OPENROAD_EXE", "/usr/local/bin/openroad"),
            "--grid-size", "10", "--ambient-k", "318.15", "--threads", str(power_threads)]
