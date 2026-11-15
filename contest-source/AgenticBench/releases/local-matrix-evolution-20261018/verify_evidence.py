"""Check the archived reading view without changing native receipts or files."""

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def verify():
    evidence = json.loads((ROOT / "evidence.json").read_text())
    tasks = {name: task for name, task in evidence["tasks"].items() if ":" not in name}
    assert len(tasks) == 8
    reference_cells = []
    candidate_cells = []
    complete_matrices = 0

    for name, task in tasks.items():
        assert task["summary"]["status"] == "completed", name
        assert len(task["reference_cells"]) == 6, name
        assert all(cell["status"] == "succeeded" for cell in task["reference_cells"]), name
        assert len(task["rounds"]) == 2, name
        reference_cells.extend(task["reference_cells"])

        for round_result in task["rounds"]:
            round_id = "round-%02d" % round_result["number"]
            cells = round_result["cells"]
            assert len(cells) == 6, (name, round_id)
            assert {(cell["case"], cell["seed"]) for cell in cells} == {
                (case, seed)
                for case in ("bp_fe-ng45", "ariane133-ng45")
                for seed in ("s1", "s2", "s3")
            }, (name, round_id)
            candidate_cells.extend(cells)

            program = ROOT / "candidates" / name / (round_id + ".py")
            assert hashlib.sha256(program.read_bytes()).hexdigest() == round_result["program_sha256"]
            diff = program.with_name(round_id + "-vs-reference.diff")
            assert diff.stat().st_size > 0
            record = next(
                item for item in task["proposal_history"] if item["proposal_id"] == round_id
            )
            assert record["program_sha256"] == round_result["program_sha256"]

            if round_result["complete"]:
                complete_matrices += 1
                assert all(cell["status"] == "succeeded" for cell in cells)
                case_means = [
                    statistics.mean(cell["score"] for cell in cells if cell["case"] == case)
                    for case in ("bp_fe-ng45", "ariane133-ng45")
                ]
                aggregate = statistics.mean(case_means)
                assert abs(round_result["score"] - aggregate) < 1e-12
                assert abs(round_result["aggregate_recomputed"] - aggregate) < 1e-12
            else:
                assert round_result["score"] is None

    assert len(reference_cells) == 48
    assert len(candidate_cells) == 96
    assert sum(cell["status"] == "succeeded" for cell in candidate_cells) == 89
    assert complete_matrices == 14
    assert len(evidence["model_call_receipts"]) == 17

    resources = evidence["resources"]
    assert resources["peak_gpu_slots"] == 6
    assert resources["peak_gpu_slots_per_device"] == {"0": 3, "1": 3}
    assert all(
        job["state"] not in ("running", "queued", "uncertain", "cancelling")
        for job in resources["jobs"].values()
    )
    assert evidence["extra_diagnostics"]["final-resource-audit.json"]["all_owned_reservations_returned"]

    tp_task = tasks["tp_gnn_post_placement_timing"]
    tp_cells = tp_task["reference_cells"] + [
        cell for result in tp_task["rounds"] for cell in result["cells"]
    ]
    assert len(tp_cells) == 18
    assert all(
        cell["gpu_release"]["evidence"]["terminated"]
        and not cell["container_phases"]["backend"]["has_gpu_flag"]
        for cell in tp_cells
    )
    backend_seconds = [cell["container_phases"]["backend"]["wall_seconds"] for cell in tp_cells]

    # Test counts retain their separately recorded log provenance.
    summary = json.loads((ROOT / "summary.json").read_text())
    assert summary["scores"] == {
        name: [result["score"] for result in task["rounds"]] for name, task in tasks.items()
    }
    assert abs(summary["tp_gpu_slot_minutes_avoided"] - sum(backend_seconds) / 60) < 1e-12
    assert abs(summary["tp_cpu_backend_median_seconds"] - statistics.median(backend_seconds)) < 1e-12
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    verify()
