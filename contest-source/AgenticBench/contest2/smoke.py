"""Prepare and verify the outer harness's complete automatic feedback test."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

from grt_bench.common import read, sha
from local_tasks.tp_gnn_local_task_definitions import (
    materialize_binary_loss, materialize_loss_decoder, validate_program,
)
from .sampling import SMOKE_CELL


def stage_agent(controller):
    folder = controller.root / "participant-agent" / controller.baseline
    folder.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).resolve().parents[1] / "agent" / controller.baseline / "agent.py"
    shutil.copy2(template, folder / "agent.py")
    shutil.copy2(controller.task_dir / "initial_program.py", folder / "program.py")
    for name in ("agent.py", "program.py"):
        (folder / name).chmod(0o600)
    return folder


def verify_loop(controller, folder, candidate):
    def require(condition, message):
        if not condition:
            raise ValueError("Automatic smoke check failed: " + message)

    raw = []
    for name in ("initial_program.py", "first_program.py", "second_program.py"):
        path = folder / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 18000,
                "missing or invalid code version: " + name)
        raw.append(path.read_text())
    programs = [validate_program(value, controller.task_id) for value in raw]
    trees = [ast.dump(ast.parse(value), include_attributes=False) for value in programs]
    require(programs[0] == controller.initial_source, "initial program differs from the baseline")
    require(trees[0] != trees[1] and trees[1] != trees[2], "both model turns must change code")
    require((folder / "program.py").read_text() == raw[2], "final program differs from the next draft")
    template = Path(__file__).resolve().parents[1] / "agent" / controller.baseline / "agent.py"
    require(sha(folder / "agent.py") == sha(template), "default Agent changed during self-check")
    materialize = materialize_binary_loss if controller.baseline == "binary_loss" else materialize_loss_decoder
    require(read(controller.candidates / candidate / "candidate.json") == materialize(programs[1]),
            "evaluated candidate differs from the first edit")
    feedback = controller.evaluation_status("smoke-evaluation")
    require(feedback["status"] == "succeeded" and feedback["candidate_id"] == candidate,
            "native evaluation did not succeed")
    require(read(folder / "feedback.json") == feedback, "Agent feedback differs from the evaluator")
    result = read(controller.evaluations / "smoke-evaluation/evaluation_result.json")
    matrix = result["metrics"]["matrix"]
    require(result["metrics"]["runs_successfully"] == 1 and matrix["expected_cells"] == 1
            and matrix["succeeded_cells"] == 1, "single-cell matrix is incomplete")
    require([(row["case"], row["seed"]) for row in matrix["cells"]] == [SMOKE_CELL],
            "diagnostic cell differs")
    state = read(controller.codex_directory / "state.json")
    keys = successful_turns(state)
    turns = [state["turns"][key]["result"] for key in keys]
    require(all(row["success"] and row["terminal_tool_calls"] >= 1 for row in turns),
            "Codex did not successfully edit in both turns")
    require(turns[0]["thread_id"] == turns[1]["thread_id"], "Codex session was not resumed")
    prompt = (controller.codex_directory / keys[1] / "agent/prompt.md").read_text()
    require(prompt.endswith(json.dumps(feedback, sort_keys=True, ensure_ascii=False)),
            "second Codex turn did not receive the actual feedback")
    budget = controller.ledger.snapshot()
    require(budget["evaluation_mode"] == "smoke" and budget["evaluations_accepted"] == 1,
            "diagnostic budget differs")
    require(len(controller.ledger.pending_jobs("smoke-evaluation")) == 1,
            "diagnostic dispatch count differs")
    return {"passed": True, "case": SMOKE_CELL[0], "seed": SMOKE_CELL[1],
        "first_edit_completed": True, "native_evaluation_succeeded": True,
        "feedback_received": True, "second_edit_completed": True,
        "second_candidate_evaluated": False, "same_codex_session": True,
        "terminal_tool_calls": [row["terminal_tool_calls"] for row in turns],
        "source_file_sha256": [hashlib.sha256(value.encode()).hexdigest() for value in raw],
        "validated_program_sha256": [hashlib.sha256(value.encode()).hexdigest() for value in programs],
        "feedback": feedback["feedback"], "formal_evaluations_added": 0}


def successful_turns(state):
    """Failed model attempts are retained; each smoke phase must succeed once."""
    import re
    names = []
    for phase in ("first-edit", "feedback-edit"):
        selected = [key for key, value in state["turns"].items()
                    if re.fullmatch(phase + r"(?:-retry-[1-9][0-9]*)?", key)
                    and value.get("result", {}).get("success")]
        if len(selected) != 1:
            raise ValueError("Automatic smoke check needs one successful " + phase)
        names.extend(selected)
    if any(not re.fullmatch(r"(?:first-edit|feedback-edit)(?:-retry-[1-9][0-9]*)?", key)
           or value.get("status") != "finished" for key, value in state["turns"].items()):
        raise ValueError("Unexpected or unfinished smoke model turn")
    return names
