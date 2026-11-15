#!/usr/bin/env bash
set -euo pipefail
node_root=${1:?node root required}
case "$node_root" in /data/gcr/agentic3d-compute-20260908|/data/agentic3d/agentic3d-compute-20260908) ;; *) exit 2 ;; esac
pool=/nas1/gcr/agentic3d-dev/compute-pool-20260908
[[ -f "$pool/sta-framework-base.ready" && -f "$pool/sta-overlay.ready" ]]
exec 9>"$node_root/framework-sta.lock"
flock -n 9 || exit 3
[[ ! -f "$node_root/framework-sta.ready" ]] || exit 0
mkdir -p "$node_root/framework-sta" "$node_root/tools-sta"
rsync -a "$pool/sta-framework-base/" "$node_root/framework-sta/"
rsync -a "$pool/sta-overlay/local_bench/benchmark_inputs/" "$node_root/framework-sta/local_bench/benchmark_inputs/"
cp "$pool/sta-overlay/scripts/run_sta_codesign_remote_eval.sh" "$node_root/framework-sta/scripts/"
cp "$pool/sta-overlay/deploy/compute_pool/sta_node_env.sh" "$node_root/tools-sta/"
cp "$pool/sta-overlay/deploy/compute_pool/sta_eval_node.sh" "$node_root/tools-sta/"
cp "$pool/sta-overlay/deploy/compute_pool/prepare_sta_case.sh" "$node_root/tools-sta/"
chmod u+x "$node_root/tools-sta/sta_eval_node.sh" "$node_root/tools-sta/prepare_sta_case.sh"
printf '%s  %s\n' \
  97898022eef119490c836368638d5afb1688a4ab2506f94a5f74fba68b9b6884 "$node_root/framework-sta/local_bench/evaluation.py" \
  a3709659829a7a439a3624af36dacf705880ec85331c26bd44ee87910a244d8b "$node_root/framework-sta/local_bench/metrics.py" | sha256sum -c -
docker run --rm --network none -v "$node_root:$node_root" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH="$node_root/framework-sta:$node_root/framework-sta/.deps-py38" \
  agentic3d/compute-pool-runtime:20260908 /opt/conda/bin/python3.8 -c '
from local_bench.benchmark_input import load_benchmark_input_profile
for case in ("bp", "ariane133", "swerv_wrapper"):
    profile = load_benchmark_input_profile(case + "_lol_ng45_tp_gnn_sta")
    assert "tp_gnn_placement_sta" in profile.compatible_tasks
    print(profile.profile_id)
'
date -Is > "$node_root/framework-sta.ready"
echo "sta_framework_ready $(date -Is)"
