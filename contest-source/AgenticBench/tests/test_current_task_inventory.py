"""Keep initial algorithms stable without pinning incidental Git diff context."""
import hashlib
import json
from pathlib import Path

import pytest

from local_bench.cli import main
from local_bench.model import load_task
from local_bench.program_bridge import materialize_program
from local_bench.task_inventory import CORE_TASK_DIRECTORIES, RETIRED_TASK_DIRECTORIES
from local_bench.verifier import verify_candidate

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "AgenticBench/local_tasks"

# Captured before extracting shared adapters and deleting retired task files.
INITIAL_DIGESTS = {
    "tp_gnn_post_placement_timing": "d3d1f6a44a95c2a278c3556df12aac981aec8f94446804521e96f049d873f993",
    "placement_initialization": "fccdcf21b5f17ac46e46cac0a947c8db519512a53eae14079a9e5da86ca1ceea",
    "placement_preconditioner": "e34d93c897d9bbd20057b432a00ee29b318f170673a7afc82b830efb38d49873",
    "placement_parameters_reference_20260923": "21aef4a6c1e174ac21d1faf7e78bef9593da905639b7d76f082d5457ff0c0dfe",
    "hbt_assignment_policy": "3dabaf50abacf6dfe5b13501fc454a05eadbae69edd02fb9d10276772aaf21af",
    "global_routing_ripup_policy": "ffbdab8a92ceda6ef2b1cfd2b5293e08be1b503c9eb282805b60f949bbf65be9",
    "global_routing_maze_cost": "bed7d285dffd498cbdcdf60c62bc6f42b720692e8007b61bb7308f56bc06d583",
    "global_routing_iteration_controller": "fab03e88da992d90d885532d514d2dc3a593928069cf5e310a5b94928773176a",
}

# Native source patches acquire different context when OpenROAD is advanced.
# Pin each program and its exclusive edit site, not the surrounding source lines.
ROUTING_PROGRAMS = {
    "global_routing_ripup_policy": ("aaa9e309ed707cc5bed26aba44ab624907cbe426b9d109b33080f36fed2c6912", "RipUp.cpp"),
    "global_routing_maze_cost": ("185ec748df9f1d6b8c4e251aa1bf97fa37893491d0761e560fc4c045b979756b", "maze.cpp"),
    "global_routing_iteration_controller": ("51dac7e0a5a76befa0bcbc1274f3c6bb0a858b2930f341a7484d7f3cc8314fda", "FastRoute.cpp"),
}


def test_only_current_task_definitions_are_shipped(capsys):
    assert {p.parent.name for p in TASKS.glob("*/task.yaml")} == set(CORE_TASK_DIRECTORIES)
    assert set(INITIAL_DIGESTS) == set(CORE_TASK_DIRECTORIES) - {
        "tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}
    main(["list"])
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 10
    assert all(r["availability"] == "available" for r in rows)


@pytest.mark.parametrize("name", RETIRED_TASK_DIRECTORIES)
def test_retired_tasks_cannot_be_loaded(name):
    assert not (TASKS / name).exists()
    with pytest.raises(FileNotFoundError):
        load_task(TASKS / name)


@pytest.mark.parametrize("name", INITIAL_DIGESTS)
def test_initial_materialization_is_unchanged(name):
    spec = load_task(TASKS / name)
    candidate = materialize_program(spec, spec.program_template.read_text(), ROOT)
    verified = verify_candidate(spec, candidate)
    if name == "tp_gnn_post_placement_timing":
        from test_tp_gnn_native_adapter_compat import assert_historical_initial_candidate
        assert_historical_initial_candidate(spec, candidate, INITIAL_DIGESTS[name])
        return
    if name in ROUTING_PROGRAMS:
        program_sha, source = ROUTING_PROGRAMS[name]
        assert hashlib.sha256(spec.program_template.read_bytes()).hexdigest() == program_sha
        assert verified["patch"]["paths"] == ["OpenROAD-3D/openroad/src/grt/src/fastroute/src/" + source]
        assert not candidate["config_overrides"]
        return
    digest = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
    assert digest == INITIAL_DIGESTS[name]
