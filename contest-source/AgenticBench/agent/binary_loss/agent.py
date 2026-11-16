"""Default participant loop: edit, evaluate, and feed results into the next edit."""
#只需提交这个
import json
from pathlib import Path
import uuid
import time

from contest2.agent_api import AgentAPIError


def evaluate(context, source, request_id):
    try:
        context.submit_evaluation(source, request_id=request_id)
    except AgentAPIError as error:
        if not any(row["request_id"] == request_id for row in context.status()["history"]):
            return {"status": "rejected", "reason": str(error)}
    result = context.wait_evaluation(request_id)
    return result


class Agent:
    def run(self, context):
        folder = Path(__file__).parent
        program = folder / "program.py"
        initial = program.read_text()
        smoke = context.evaluation_mode == "smoke"
        if smoke:
            (folder / "initial_program.py").write_text(initial)
        feedback = None
        best = None
        best_source = initial
        for row in context.status()["history"]:
            if row.get("baseline", context.task) != context.task:
                continue
            result = context.wait_evaluation(row["request_id"])
            if result["status"] == "succeeded":
                if best is None or result["feedback"]["score"] > best["feedback"]["score"]:
                    best, best_source = result, result["source"]
            feedback = result
        attempts = {}
        model_failure = None
        while True:
            state = context.status()
            if not state.get("codex_available", True):
                raise RuntimeError("Default Agent requires Codex CLI; install it on the host or use a custom Agent with context.model")
            draft_only = smoke and best is not None
            if not state["dispatch_allowed"] or (state["evaluations_remaining"] == 0 and not draft_only):
                if state.get("unresolved_model_calls") and "insufficient_model_reservation" in state.get("model_stop_reasons", []):
                    raise RuntimeError(
                        "Model usage is unresolved and blocks further calls. "
                        "Inspect control/model-receipts/*.error.json before resuming this work directory; "
                        "verified candidates and evaluation history are retained.")
                break
            program.write_text(best_source)
            task = ("tp_gnn_binary_loss_timing" if context.task == "binary_loss"
                    else "tp_gnn_loss_decoder_timing")
            if smoke and not draft_only:
                instruction = ("Multiply only the binarization regularizer contribution by 1.01."
                    if context.task == "binary_loss" else
                    "Multiply the existing loss returned by compute_tp_gnn_loss by 1.01. "
                    "Keep decode_tp_gnn_embeddings unchanged.")
            elif draft_only:
                instruction = ("Use the measured feedback to choose and implement a further small "
                    "numerical loss-function adjustment. Keep the decoder unchanged if present. "
                    "Cite at least one measured value in your explanation.")
            else:
                instruction = ("Propose and implement one reasoned improvement to the loss function."
                    if context.task == "binary_loss" else
                    "Propose and implement one reasoned improvement to the loss function and/or "
                    "clustering decoder, using previous feedback.")
            interface = (
                "The program must contain only compute_binary_partition_loss. "
                "Its return must be a finite differentiable scalar tensor. "
                if context.task == "binary_loss" else
                "The program must contain compute_tp_gnn_loss followed by decode_tp_gnn_embeddings. "
                "The loss returns a finite differentiable scalar tensor. The decoder returns one "
                "integer label per group (1 top, 0 bottom), with both sides nonempty. It may call "
                "reference_global_two_way once with the exact five unchanged inputs, or use at most "
                "one for-loop over range(max_iterations) without calling that reference helper. ")
            prompt = (
                "Read /workspace/agent/program.py and /opt/contest/AgenticBench/local_tasks/" + task +
                "/prompt.md for function inputs and numerical APIs. " + instruction + " " + interface +
                "Submit Python function source in program.py, not JSON; the outer harness packages "
                "the JSON capsule. Keep the complete program within 18000 UTF-8 bytes and 3000 AST "
                "nodes. No imports, extra functions, input mutation, rebinding inputs or trusted "
                "helpers, external I/O or external state. Loss functions have no loops. The exact validators "
                "are /opt/contest/AgenticBench/local_tasks/tp_gnn_local_task_definitions.py and "
                "tp_gnn_evolution_loss_language.py in that same directory. "
                "Preserve signatures, function order, and pure-function "
                "rules. Use the container terminal to edit only program.py. Do not call Agent API "
                "operations or submit/wait for an official evaluation. Return after editing and explain "
                "the change. The Python Agent submits and waits after this Codex call ends.\n"
                "Contest scheduling uses the fixed evaluation cells below, not historical sampling "
                "or timeout descriptions in the task prompt: formal training has 9 cells and public "
                "validation has 6 for either baseline; smoke uses one cell. Only individual model API "
                "requests have a timeout. Agent sessions, terminal commands, resource queues and "
                "physical evaluations have no execution deadline.\n"
                "Legal cell scores are max(raw_score, -100); confirmed terminal physical failures "
                "score -100 and stay in the fixed matrix mean. A scored matrix may contain invalid "
                "cells and remains eligible as a parent or final candidate. Inspect valid, raw_score "
                "and failure reasons in the feedback. Unknown execution or missing evidence has no score.\n"
                "Fixed evaluation cells:\n" + json.dumps(context.evaluation_cells, ensure_ascii=False) + "\n"
                "Remaining evaluations: " + str(state["evaluations_remaining"]) + "\n")
            prompt += ("For a terminal result marked running, poll exec with its session_id "
                       "until the command finishes before returning.\n")
            if model_failure is not None:
                prompt += "PREVIOUS MODEL FAILURE:\n" + json.dumps(model_failure, ensure_ascii=False) + "\n"
            prompt += ("Read /workspace/README.md for the contest workflow and /opt/contest/source "
                       "for the complete physical source. Evaluation feedback includes artifacts_dir "
                       "and files_index: inspect the original logs there with terminal tools. "
                       "Large text files are complete gzip files, readable with zcat or Python gzip.\n")
            if feedback is not None:
                prompt += "EVALUATOR FEEDBACK:\n" + json.dumps(feedback, sort_keys=True, ensure_ascii=False)
            phase = "feedback-edit" if draft_only else "first-edit"
            attempt = attempts.get(phase, 0)
            attempts[phase] = attempt + 1
            turn_id = (phase + ("-retry-" + str(attempt) if attempt else "")) if smoke else uuid.uuid4().hex
            try:
                turn = context.codex_turn(prompt, request_id=turn_id)
            except AgentAPIError as error:
                turn = {"success": False, "error": str(error)}
            if not turn.get("success"):
                if turn.get("retryable") is False:
                    raise RuntimeError(turn.get("error", "Model adapter requires host reconciliation"))
                model_failure = {"status": "model_failed", "turn": turn}
                (folder / "model_failure.json").write_text(json.dumps(model_failure, ensure_ascii=False))
                time.sleep(5)
                continue
            model_failure = None
            source = program.read_text()
            if smoke:
                (folder / ("second_program.py" if draft_only else "first_program.py")).write_text(source)
            if draft_only:
                break
            request_id = "smoke-evaluation" if smoke else uuid.uuid4().hex
            feedback = evaluate(context, source, request_id)
            (folder / "feedback.json").write_text(json.dumps(feedback, sort_keys=True, ensure_ascii=False))
            if feedback["status"] == "succeeded":
                if best is None or feedback["feedback"]["score"] > best["feedback"]["score"]:
                    best, best_source = feedback, source
                    (folder / "best_program.py").write_text(source)
            elif smoke:
                raise RuntimeError("Single-cell native evaluation failed")
            # Ordinary rejections and failed matrices are feedback, not termination.
        if best is None:
            raise RuntimeError("No candidate completed a verified training evaluation")
        if not smoke:
            program.write_text(best_source)
        return best["candidate_id"]
