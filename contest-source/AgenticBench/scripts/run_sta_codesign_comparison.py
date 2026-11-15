#!/usr/bin/env python3
"""Run matched-model Codex and Agentic3D STA co-design baselines.

The expensive budget counts only ``tp_gnn_placement_sta`` evaluations.  The
Agentic3D mode may run bounded partition/placement proxy evaluations and stops
each inner loop at its first strict improvement.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic3d.codex_exec import CodexExecConfig, run_codex
from local_bench.evolve_protocol import render_candidate_program
from local_bench.model import TaskSpec, load_task
from local_bench.policy import patch_paths
from local_bench.program_bridge import materialize_program
from local_bench.prompting import build_problem_description
from local_bench.verifier import verify_candidate


TASK_ROOT = ROOT / "local_tasks"
EVALUATION_PROFILE_ROOT = ROOT / "evaluation_profiles"
STA_PROFILE = EVALUATION_PROFILE_ROOT / "tp_gnn_placement_two_pass"
TP_GNN_PROXY_PROFILE = EVALUATION_PROFILE_ROOT / "tp_gnn_partition_proxy"
PLACEMENT_PROXY_PROFILE = EVALUATION_PROFILE_ROOT / "placement_proxy"
REMOTE_HOST = "ta1-gcr"
REMOTE_EXPERIMENT = "/data/gcr/agentic3d-dev/runs/sta_codesign_bp_multi_20260902-v1"
REMOTE_FRAMEWORK = "/data/gcr/agentic3d-dev/framework-sta-codesign-20260902-v1"
REMOTE_HELPER = os.environ.get("OPENAGENTIC3D_STA_REMOTE_HELPER", REMOTE_FRAMEWORK + "/scripts/run_sta_codesign_remote_eval.sh")
REMOTE_BUNDLE = REMOTE_EXPERIMENT + "/benchmark-input"
REMOTE_PROFILE_CONFIG = None
SSH_PROXY_COMMAND = None
REMOTE_ENV: tuple[str, ...] = ()
REMOTE_COMPARISON = os.environ.get("OPENAGENTIC3D_STA_COMPARISON", REMOTE_EXPERIMENT + "/comparison-v8-exact-cutpoint-two-pass")
REFERENCE_RUN = REMOTE_EXPERIMENT + "/reference-v8-exact-cutpoint-two-pass/run"


PROGRAM_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["hypothesis", "program"],
    "properties": {
        "hypothesis": {"type": "string"},
        "program": {"type": "string"},
    },
}

PLAN_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "target",
        "hypothesis",
        "evidence",
        "specialist_instruction",
        "max_inner_attempts",
    ],
    "properties": {
        "target": {"type": "string", "enum": ["partition", "placement"]},
        "hypothesis": {"type": "string"},
        "evidence": {"type": "string"},
        "specialist_instruction": {"type": "string"},
        "max_inner_attempts": {"type": "integer", "minimum": 1, "maximum": 3},
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _compact_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = result.get("metrics", {})
    artifacts = result.get("artifacts", {})
    keep = (
        "combined_score",
        "normalized_sta_cost",
        "normalized_wns_cost",
        "normalized_tns_cost",
        "final_global_wns_setup_ns",
        "final_global_tns_setup_ns",
        "pass1_global_wns_setup_ns",
        "pass1_global_tns_setup_ns",
        "cut_net_count",
        "timing_weighted_cut",
        "area_balance_error",
        "placement_total_hpwl_um",
        "placement_native_hpwl",
        "placement_overflow_bottom",
        "placement_overflow_top",
        "placement_overflow_max",
        "hbt_count",
        "runs_successfully",
        "opt_passes", "sta_graph_complete", "sta_unconstrained_endpoints",
        "normalized_average_temperature_cost", "global_average_temperature_k",
        "global_average_temperature_c", "bottom_average_temperature_k",
        "top_average_temperature_k", "global_maximum_temperature_c", "total_power_w",
        "thermal_grid_size", "thermal_ambient_k", "thermal_model",
    )
    return {
        "metrics": {key: metrics[key] for key in keep if key in metrics},
        "error_type": artifacts.get("error_type"),
        "error_message": artifacts.get("error_message"),
    }


def _score(result: Mapping[str, Any]) -> Optional[float]:
    metrics = result.get("metrics", {})
    if not metrics.get("runs_successfully"):
        return None
    value = metrics.get("combined_score")
    return float(value) if isinstance(value, (int, float)) else None


@dataclass
class PersistentCodex:
    name: str
    root: Path
    schema: Mapping[str, Any]
    model: str
    codex_binary: str
    workspace: Path
    timeout_seconds: int = 3600

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema_path = self.root / "output.schema.json"
        _write_json(self.schema_path, self.schema)
        self.state_path = self.root / "session.json"
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            state = {"thread_id": None, "calls": 0, "runs": []}
        self.thread_id = state.get("thread_id")
        self.calls = int(state.get("calls", 0))
        self.runs = list(state.get("runs", []))

    def ask(self, prompt: str) -> Dict[str, Any]:
        # A failed call may have written evidence before session.json was saved.
        existing = self.root / "calls"
        if existing.is_dir():
            self.calls = max([self.calls] + [int(p.name) for p in existing.iterdir()
                                           if p.is_dir() and p.name.isdigit()])
        self.calls += 1
        run_dir = self.root / "calls" / "{:03d}".format(self.calls)
        result = run_codex(
            prompt,
            CodexExecConfig(
                workspace=self.workspace,
                run_dir=run_dir,
                codex_binary=self.codex_binary,
                model=self.model,
                sandbox="read-only",
                config_overrides=("model_reasoning_effort=xhigh",),
                output_schema=self.schema_path,
                timeout_seconds=self.timeout_seconds,
                skip_git_repo_check=True,
                resume_thread_id=self.thread_id,
            ),
        )
        if not result.success:
            raise RuntimeError(
                "{} Codex call failed with exit {}; see {}".format(
                    self.name, result.exit_code, result.metadata_path
                )
            )
        payload = json.loads(result.final_message)
        if not isinstance(payload, dict):
            raise ValueError("{} returned a non-object".format(self.name))
        self.thread_id = result.thread_id or self.thread_id
        self.runs.append(result.to_dict())
        _write_json(
            self.state_path,
            {"thread_id": self.thread_id, "calls": self.calls, "runs": self.runs},
        )
        return payload


def _check_source_patch_applies(
    source_root: Path, source_patch: str, destination: Path
) -> None:
    if not source_patch.strip():
        return
    destination.mkdir(parents=True, exist_ok=True)
    preflight_patch = destination / "source-apply-check.patch"
    preflight_patch.write_text(source_patch, encoding="utf-8")
    apply_check = subprocess.run(
        [
            "git",
            "apply",
            "--check",
            "--no-index",
            "--unsafe-paths",
            str(preflight_patch.resolve()),
        ],
        cwd=str(source_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (destination / "source-apply-check.log").write_text(
        apply_check.stdout or "clean\n", encoding="utf-8"
    )
    if apply_check.returncode:
        raise ValueError(
            "candidate source patch does not apply to the frozen source: "
            + (apply_check.stdout or "git apply --check failed")
        )


def _program_candidate(
    agent: PersistentCodex,
    spec: TaskSpec,
    prompt: str,
    destination: Path,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    if (destination / "candidate.json").is_file() and (destination / "program.py").is_file():
        program = (destination / "program.py").read_text(encoding="utf-8")
        candidate = json.loads((destination / "candidate.json").read_text())
        if materialize_program(spec, program, None) != candidate:
            raise ValueError("saved program and candidate disagree")
        verify_candidate(spec, candidate)
        _check_source_patch_applies(agent.workspace, str(candidate.get("source_patch", "")),
                                   destination / "resume-check")
        response_path = destination / "agent_response.json"
        response = json.loads(response_path.read_text()) if response_path.is_file() else {
            "hypothesis": candidate.get("notes", ""), "program": program,
        }
        return program, candidate, response
    current_prompt = prompt
    for repair in range(3):
        response = agent.ask(current_prompt)
        program = str(response["program"])
        try:
            candidate = materialize_program(spec, program, None)
            verification = verify_candidate(spec, candidate)
            destination.mkdir(parents=True, exist_ok=True)
            source_patch = str(candidate.get("source_patch", ""))
            _check_source_patch_applies(agent.workspace, source_patch, destination)
            (destination / "program.py").write_text(program, encoding="utf-8")
            _write_json(destination / "candidate.json", candidate)
            _write_json(destination / "verification.json", verification)
            _write_json(destination / "agent_response.json", response)
            return program, candidate, response
        except Exception as error:
            current_prompt = (
                "Your previous candidate failed the deterministic benchmark verifier: "
                "{}: {}. Return a corrected complete program now. Check the outer "
                "Python wrapper with ast.parse; source_patch is one string value, "
                "not Python code to execute. Do not leave extra closing braces. "
                "Do not explain it."
            ).format(type(error).__name__, error)
    raise RuntimeError("{} failed candidate verification three times".format(agent.name))


def _remote(*args: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
    proxy = ["-o", "ProxyCommand=" + SSH_PROXY_COMMAND] if SSH_PROXY_COMMAND else []
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=20",
            "-o",
            "ConnectTimeout=15",
            *proxy,
            REMOTE_HOST,
            *args,
        ],
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _remote_candidate(local_dir: Path, remote_dir: str) -> None:
    created = _remote("mkdir", "-p", remote_dir)
    if created.returncode:
        raise RuntimeError(created.stdout)
    copied = subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            *(["-o", "ProxyCommand=" + SSH_PROXY_COMMAND] if SSH_PROXY_COMMAND else []),
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=20",
            str(local_dir / "candidate.json"),
            REMOTE_HOST + ":" + remote_dir + "/candidate.json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if copied.returncode:
        raise RuntimeError(copied.stdout)


def _remote_result(path: str) -> Dict[str, Any]:
    read = _remote("cat", path)
    if read.returncode:
        raise RuntimeError("remote evaluation produced no result: {}".format(read.stdout))
    value = json.loads(read.stdout)
    if not isinstance(value, dict):
        raise ValueError("remote evaluation result is not an object")
    return value


def _remote_exists(path: str) -> bool:
    result = _remote("test", "-e", path)
    if result.returncode not in (0, 1):
        raise RuntimeError("remote existence check failed: {}".format(result.stdout))
    return result.returncode == 0


def evaluate_remote(
    *,
    task: str,
    local_candidate: Path,
    remote_root: str,
    input_kind: str,
    input_path: str,
    reference_run: str,
    gpu: int,
    retention: str,
    log_path: Path,
) -> tuple[Dict[str, Any], str]:
    print(
        json.dumps(
            {"event": "remote_evaluation_started", "task": task, "output": remote_root},
            sort_keys=True,
        ),
        flush=True,
    )
    remote_candidate = remote_root + "/candidate"
    remote_output = remote_root + "/result"
    result_path = remote_output + "/evaluation_result.json"
    if _remote_exists(result_path):
        saved_candidate = _remote_result(remote_candidate + "/candidate.json")
        current_candidate = json.loads((local_candidate / "candidate.json").read_text())
        if saved_candidate != current_candidate:
            raise ValueError("refusing to reuse evaluation for a different candidate")
        result = _remote_result(result_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():
            _write_json(log_path, result)
        print(json.dumps({"event": "remote_evaluation_reused", "output": remote_root}), flush=True)
        return result, remote_output + "/run"
    if _remote_exists(remote_output):
        raise RuntimeError("incomplete remote evaluation requires inspection: " + remote_output)
    _remote_candidate(local_candidate, remote_candidate)
    command = [
        *(["env", *REMOTE_ENV] if REMOTE_ENV else []),
        REMOTE_HELPER,
        task,
        remote_candidate,
        remote_output,
        input_kind,
        input_path,
        reference_run,
        str(gpu),
        retention,
    ]
    if REMOTE_PROFILE_CONFIG:
        command.append(REMOTE_PROFILE_CONFIG)
    started = time.time()
    run = _remote(*command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(run.stdout or "", encoding="utf-8")
    result = _remote_result(remote_output + "/evaluation_result.json")
    result.setdefault("driver", {})
    result["driver"].update(
        {
            "remote_exit_code": run.returncode,
            "remote_output": remote_output,
            "wall_seconds": time.time() - started,
        }
    )
    print(
        json.dumps(
            {
                "event": "remote_evaluation_finished",
                "task": task,
                "output": remote_root,
                "exit_code": run.returncode,
                "score": _score(result),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result, remote_output + "/run"


def _patch_sections(text: str) -> Dict[str, str]:
    if not text.strip():
        return {}
    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not starts:
        starts = [index for index, line in enumerate(lines) if line.startswith("--- ")]
    sections: Dict[str, str] = {}
    for order, start in enumerate(starts):
        end = starts[order + 1] if order + 1 < len(starts) else len(lines)
        section = "".join(lines[start:end]).strip() + "\n"
        paths = patch_paths(section)
        if len(paths) != 1:
            raise ValueError("every merged patch section must describe exactly one file")
        sections[paths[0]] = section
    if set(sections) != set(patch_paths(text)):
        raise ValueError("could not split every file in the candidate patch")
    return sections


def _merge_candidate(
    incumbent: Mapping[str, Any],
    correction: Mapping[str, Any],
    hypothesis: str,
    replace_paths: Optional[set[str]] = None,
) -> Dict[str, Any]:
    sections = _patch_sections(str(incumbent.get("source_patch", "")))
    if replace_paths is not None:
        for path in replace_paths:
            sections.pop(path, None)
    sections.update(_patch_sections(str(correction.get("source_patch", ""))))
    return {
        "schema_version": 1,
        "config_overrides": {"partition": {"algorithm": "tp-gnn"}},
        "candidate_files": {},
        "source_patch": "".join(sections[path] for path in sorted(sections)),
        "notes": hypothesis[:4000],
    }


def _history(root: Path, total: int) -> list[Dict[str, Any]]:
    path = root / "history.json"
    rows = json.loads(path.read_text()) if path.is_file() else []
    if not isinstance(rows, list) or [r.get("round") for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("outer history must be a contiguous round sequence")
    if len(rows) > total:
        raise ValueError("history exceeds the fixed outer budget")
    return rows


def _round_plan(analyzer: PersistentCodex, root: Path, number: int, prompt: str) -> Dict[str, Any]:
    path = root / "rounds" / "{:02d}".format(number) / "plan.json"
    if path.is_file():
        return json.loads(path.read_text())
    # Older drivers only persisted analyzer calls, not per-round plans.
    marker = "bounded specialist task for outer round {}.".format(number)
    for prior in sorted((analyzer.root / "calls").glob("*/prompt.md"), reverse=True):
        if marker in prior.read_text():
            result_path = prior.parent / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                if result.get("success"):
                    plan = json.loads(result["final_message"])
                    _write_json(path, plan)
                    return plan
    plan = analyzer.ask(prompt)
    _write_json(path, plan)
    return plan


def run_codex_baseline(args: argparse.Namespace, root: Path) -> None:
    spec = load_task(STA_PROFILE)
    instructions = args.instructions.read_text(encoding="utf-8")
    agent = PersistentCodex(
        "single-codex",
        root / "agent",
        PROGRAM_SCHEMA,
        args.model,
        args.codex_bin,
        args.agent_workspace,
    )
    history = _history(root, args.rounds)
    incumbent_score = 1.0
    incumbent_program = args.initial_program.read_text(encoding="utf-8")
    for record in history:
        if record.get("accepted_as_incumbent"):
            incumbent_score = float(record["score"])
            incumbent_program = (root / "rounds" / "{:02d}".format(record["round"])
                                 / "candidate/program.py").read_text()
    for round_number in range(len(history) + 1, args.rounds + 1):
        if round_number == 1:
            prompt = (
                "You are the single-agent Codex baseline. Independently optimize the "
                "following benchmark. You receive exactly five final STA evaluations "
                "and no local proxy evaluations. Return JSON matching the supplied "
                "schema; program must be the complete evolve_candidate function. "
                "Your current directory is the complete clean read-only source snapshot; "
                "do not inspect parent directories or experiment outputs.\n\n"
                + instructions
            )
        else:
            prompt = (
                "Continue the same five-evaluation optimization. Here is the latest "
                "black-box final-STA feedback and compact history:\n"
                + json.dumps(
                    {
                        "last": history[-1],
                        "history": history,
                        "incumbent_score": incumbent_score,
                        "incumbent_program": incumbent_program,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\nReturn one materially revised complete candidate."
            )
        candidate_dir = root / "rounds" / "{:02d}".format(round_number) / "candidate"
        program, _candidate, response = _program_candidate(
            agent, spec, prompt, candidate_dir
        )
        remote_root = REMOTE_COMPARISON + "/codex/round-{:02d}".format(
            round_number
        )
        result, _run = evaluate_remote(
            task="tp_gnn_placement_sta",
            local_candidate=candidate_dir,
            remote_root=remote_root,
            input_kind="bundle",
            input_path=REMOTE_BUNDLE,
            reference_run=args.reference_run,
            gpu=args.gpu,
            retention="full",
            log_path=candidate_dir.parent / "remote_eval.log",
        )
        compact = _compact_result(result)
        score = _score(result)
        accepted = score is not None and score > incumbent_score
        if accepted:
            incumbent_score = float(score)
            incumbent_program = program
        record = {
            "round": round_number,
            "hypothesis": response.get("hypothesis"),
            "score": score,
            "remote_run": _run,
            "accepted_as_incumbent": accepted,
            **compact,
        }
        history.append(record)
        _write_json(root / "history.json", history)
        print(json.dumps({"event": "outer_round", **record}, sort_keys=True), flush=True)
    _write_json(
        root / "summary.json",
        {
            "mode": "codex",
            "model": args.model,
            "final_sta_evaluations": len(history),
            "incumbent_score": incumbent_score,
            "history": history,
        },
    )


def run_agentic3d(args: argparse.Namespace, root: Path) -> None:
    global_spec = load_task(STA_PROFILE)
    local_specs = {
        "partition": load_task(TP_GNN_PROXY_PROFILE),
        "placement": load_task(PLACEMENT_PROXY_PROFILE),
    }
    local_instructions = {
        "partition": args.partition_instructions.read_text(encoding="utf-8"),
        "placement": args.placement_instructions.read_text(encoding="utf-8"),
    }
    analyzer = PersistentCodex(
        "global-analyzer", root / "global-analyzer", PLAN_SCHEMA,
        args.model, args.codex_bin, args.agent_workspace,
    )
    specialists = {
        target: PersistentCodex(
            target + "-specialist",
            root / (target + "-specialist"),
            PROGRAM_SCHEMA,
            args.model,
            args.codex_bin,
            args.agent_workspace,
        )
        for target in ("partition", "placement")
    }
    incumbent_candidate: Dict[str, Any] = {
        "schema_version": 1,
        "config_overrides": {"partition": {"algorithm": "tp-gnn"}},
        "candidate_files": {},
        "source_patch": "",
        "notes": "Maintained full-flow reference.",
    }
    incumbent_local_program = {
        target: render_candidate_program(
            json.loads(
                (TASK_ROOT / local_specs[target].task_id / "initial_candidate/candidate.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        for target in ("partition", "placement")
    }
    incumbent_score = 1.0
    incumbent_run = args.reference_run
    history = _history(root, args.rounds)
    inner_evaluations = sum(int(r.get("inner_attempts", 0)) for r in history)
    for record in history:
        if record.get("accepted_as_incumbent"):
            number = int(record["round"])
            incumbent_score = float(record["outer_score"])
            incumbent_candidate = json.loads((root / "rounds" / "{:02d}".format(number)
                                              / "outer-candidate/candidate.json").read_text())
            incumbent_run = record.get("remote_run") or (REMOTE_EXPERIMENT
                + "/comparison-v8-exact-cutpoint-two-pass/agentic3d/round-{:02d}/outer/result/run".format(number))
            sections = _patch_sections(incumbent_candidate["source_patch"])
            for target, spec in local_specs.items():
                local_candidate = dict(incumbent_candidate)
                local_candidate["source_patch"] = "".join(
                    text for path, text in sections.items() if path in spec.allowed_source_paths
                )
                incumbent_local_program[target] = render_candidate_program(local_candidate)

    for round_number in range(len(history) + 1, args.rounds + 1):
        plan_prompt = (
            "You are the read-only Global Analyzer in Agentic3D. Select exactly one "
            "bounded specialist task for outer round {}. Final setup WNS/TNS after "
            "two OPT passes is authoritative. Cut and placement metrics are only "
            "diagnostics. Choose max_inner_attempts 1..3; the local loop stops at "
            "the first strict proxy improvement. Do not write code. Use only the supplied "
            "state and the clean current source snapshot; do not inspect parent directories "
            "or other experiment outputs.\n\nSTATE:\n{}"
        ).format(
            round_number,
            json.dumps(
                {
                    "incumbent_score": incumbent_score,
                    "incumbent_run": incumbent_run,
                    "prior_outer_rounds": history,
                    "partition_patch_present": bool(
                        _patch_sections(incumbent_candidate["source_patch"]).get(
                            "Place-3D/src/partition/lol/methods/tp_gnn.py"
                        )
                    ),
                    "placement_patch_files": [
                        path
                        for path in _patch_sections(incumbent_candidate["source_patch"])
                        if "placement" in path or "DREAMPlace" in path
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        plan = _round_plan(analyzer, root, round_number, plan_prompt)
        target = str(plan["target"])
        specialist = specialists[target]
        local_spec = local_specs[target]
        attempts = int(plan["max_inner_attempts"])
        best_local: Optional[tuple[float, str, Dict[str, Any], Dict[str, Any], Path]] = None
        last_inner_result: Optional[Dict[str, Any]] = None
        first_improvement = False

        for attempt in range(1, attempts + 1):
            specialist_prompt = (
                (local_instructions[target] + "\n\n") if specialist.calls == 0 else ""
            ) + (
                "Use only the clean current source snapshot and this session's feedback; "
                "do not inspect parent directories or other experiment outputs.\n\n"
            ) + json.dumps(
                {
                    "outer_round": round_number,
                    "inner_attempt": attempt,
                    "global_plan": plan,
                    "current_complete_local_program": incumbent_local_program[target],
                    "first_improvement_threshold": 1.0,
                    "previous_attempt": _compact_result(last_inner_result)
                    if last_inner_result is not None
                    else None,
                },
                ensure_ascii=False,
                indent=2,
            )
            candidate_dir = (
                root / "rounds" / "{:02d}".format(round_number) / "inner"
                / target / "{:02d}".format(attempt) / "candidate"
            )
            program, candidate, response = _program_candidate(
                specialist, local_spec, specialist_prompt, candidate_dir
            )
            inner_evaluations += 1
            remote_root = (
                REMOTE_COMPARISON
                + "/agentic3d/round-{:02d}/inner/{}/attempt-{:02d}".format(
                    round_number, target, attempt
                )
            )
            if target == "partition":
                input_kind, input_path = "bundle", REMOTE_BUNDLE
            else:
                input_kind, input_path = "run", incumbent_run
            result, _ = evaluate_remote(
                task=local_spec.task_id,
                local_candidate=candidate_dir,
                remote_root=remote_root,
                input_kind=input_kind,
                input_path=input_path,
                reference_run=incumbent_run,
                gpu=args.gpu,
                retention="compact",
                log_path=candidate_dir.parent / "remote_eval.log",
            )
            last_inner_result = result
            score = _score(result)
            if score is not None and (best_local is None or score > best_local[0]):
                best_local = (score, program, candidate, result, candidate_dir)
            _write_json(candidate_dir.parent / "agent_response.json", response)
            if score is not None and score > 1.0 + args.inner_tolerance:
                first_improvement = True
                break

        round_dir = root / "rounds" / "{:02d}".format(round_number)
        if best_local is not None:
            local_score, local_program, local_candidate, local_result, _ = best_local
            proposed_candidate = _merge_candidate(
                incumbent_candidate,
                local_candidate,
                "{}; local {} score {:.9g}; first_improvement={}".format(
                    plan["hypothesis"], target, local_score, first_improvement
                ),
                replace_paths=set(local_spec.allowed_source_paths),
            )
            proposed_local_program = local_program
        else:
            local_score, local_result = None, {"metrics": {"runs_successfully": 0.0}}
            proposed_candidate = dict(incumbent_candidate)
            proposed_local_program = incumbent_local_program[target]
            proposed_candidate["notes"] = (
                str(plan["hypothesis"]) + "; no valid local correction"
            )[:4000]
        verify_candidate(global_spec, proposed_candidate)
        outer_candidate_dir = round_dir / "outer-candidate"
        outer_candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_json(outer_candidate_dir / "candidate.json", proposed_candidate)
        (outer_candidate_dir / "program.py").write_text(
            render_candidate_program(proposed_candidate), encoding="utf-8"
        )
        remote_root = REMOTE_COMPARISON + "/agentic3d/round-{:02d}/outer".format(
            round_number
        )
        outer_result, outer_run = evaluate_remote(
            task="tp_gnn_placement_sta",
            local_candidate=outer_candidate_dir,
            remote_root=remote_root,
            input_kind="bundle",
            input_path=REMOTE_BUNDLE,
            reference_run=args.reference_run,
            gpu=args.gpu,
            retention="full",
            log_path=round_dir / "outer_eval.log",
        )
        outer_score = _score(outer_result)
        accepted = outer_score is not None and outer_score > incumbent_score
        if accepted:
            incumbent_score = float(outer_score)
            incumbent_candidate = proposed_candidate
            incumbent_run = outer_run
            incumbent_local_program[target] = proposed_local_program
        record = {
            "round": round_number,
            "global_plan": plan,
            "target": target,
            "inner_attempts": attempt,
            "inner_score": local_score,
            "first_improvement": first_improvement,
            "inner_result": _compact_result(local_result),
            "outer_score": outer_score,
            "remote_run": outer_run,
            "accepted_as_incumbent": accepted,
            "outer_result": _compact_result(outer_result),
        }
        history.append(record)
        _write_json(root / "history.json", history)
        print(json.dumps({"event": "outer_round", **record}, sort_keys=True), flush=True)

    _write_json(
        root / "summary.json",
        {
            "mode": "agentic3d",
            "model": args.model,
            "final_sta_evaluations": len(history),
            "inner_proxy_evaluations": inner_evaluations,
            "inner_policy": "first_strict_improvement",
            "incumbent_score": incumbent_score,
            "history": history,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("legacy", "knowledge-v2"), default="legacy",
                        help="Opt-in evolvable skills/tasks; never change an existing run's protocol.")
    parser.add_argument("--knowledge-phase", choices=("independent", "train", "test"), default="independent")
    parser.add_argument("--knowledge-split", type=Path, help="JSON with disjoint train/test case lists.")
    parser.add_argument("--knowledge-seed", type=Path, help="Explicit immutable seed/snapshot.json; default snapshots maintained seeds.")
    parser.add_argument("--knowledge-import", type=Path, help="Explicit checksummed training package only.")
    parser.add_argument("--knowledge-export", type=Path, help="Export reusable training knowledge after completion.")
    parser.add_argument("--test-knowledge-adaptation", action="store_true",
                        help="Opt in to test-local knowledge updates; never modify the imported package.")
    parser.add_argument("--mode", required=True, choices=("codex", "agentic3d"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue saved outer history without repeating evaluations.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--codex-bin", default=str(Path.home() / ".local/bin/codex"))
    parser.add_argument(
        "--instructions",
        type=Path,
        default=ROOT / "tmp/sta_codesign_bp_multi_20260902-v1/instructions.md",
    )
    parser.add_argument(
        "--initial-program",
        type=Path,
        default=ROOT / "tmp/sta_codesign_bp_multi_20260902-v1/initial_program.py",
    )
    parser.add_argument(
        "--partition-instructions",
        type=Path,
        default=ROOT / "tmp/sta_codesign_bp_multi_20260902-v1/tp_gnn_sta_inner.md",
    )
    parser.add_argument(
        "--placement-instructions",
        type=Path,
        default=ROOT / "tmp/sta_codesign_bp_multi_20260902-v1/placement_sta_inner.md",
    )
    parser.add_argument("--reference-run", default=REFERENCE_RUN)
    parser.add_argument("--case", default="bp_multi", choices=("bp_multi", "bp", "ariane133", "swerv_wrapper"))
    parser.add_argument("--remote-host", default=REMOTE_HOST)
    parser.add_argument("--remote-bundle", default=REMOTE_BUNDLE)
    parser.add_argument("--remote-comparison", default=REMOTE_COMPARISON)
    parser.add_argument("--remote-helper", default=REMOTE_HELPER)
    parser.add_argument("--remote-profile-config")
    parser.add_argument("--ssh-proxy-command", help="Optional existing proxy, used consistently for SSH and SCP.")
    parser.add_argument("--inner-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--agent-workspace",
        type=Path,
        default=ROOT / "tmp/sta_codesign_clean_workspace",
    )
    return parser


def configure_remote(args: argparse.Namespace) -> None:
    """Keep old bp_multi invocations compatible, but never inherit its inputs for a new case."""
    global REMOTE_HOST, REMOTE_BUNDLE, REMOTE_COMPARISON, REMOTE_HELPER, REMOTE_PROFILE_CONFIG, SSH_PROXY_COMMAND
    if args.case != "bp_multi":
        if (args.remote_bundle == REMOTE_EXPERIMENT + "/benchmark-input"
                or args.reference_run == REFERENCE_RUN
                or args.remote_comparison.startswith(REMOTE_EXPERIMENT + "/")
                or not args.remote_profile_config):
            raise ValueError("new cases require their own bundle, reference, comparison and profile config")
        expected = args.case + "_lol_ng45_tp_gnn_sta/config.yaml"
        if not args.remote_profile_config.endswith("/" + expected):
            raise ValueError("remote profile does not match case")
    REMOTE_HOST = args.remote_host
    REMOTE_BUNDLE = args.remote_bundle
    REMOTE_COMPARISON = args.remote_comparison
    REMOTE_HELPER = args.remote_helper
    REMOTE_PROFILE_CONFIG = args.remote_profile_config
    SSH_PROXY_COMMAND = args.ssh_proxy_command


def validate_case_reference(case: str, bundle: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    if bundle.get("benchmark_id") != case + "_lol_ng45_tp_gnn_sta":
        raise ValueError("frozen input belongs to another case/profile")
    metrics = result.get("metrics", {})
    if (metrics.get("runs_successfully") != 1 or metrics.get("opt_passes") != 2
            or metrics.get("sta_graph_complete") != 1
            or metrics.get("sta_unconstrained_endpoints") != 0):
        raise ValueError("case reference failed exact two-pass STA validity gates")
    for key in ("final_global_wns_setup_ns", "final_global_tns_setup_ns", "combined_score"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("case reference has invalid " + key)
    if not math.isclose(metrics["combined_score"], 1.0, rel_tol=1e-9):
        raise ValueError("bootstrap reference must have self-normalized score 1")


def write_case_prompts(args: argparse.Namespace, root: Path) -> None:
    """Generate case-correct contracts and source excerpts for both matched baselines."""
    profile_id = args.case + "_lol_ng45_tp_gnn_sta"
    profile_config = ROOT / "local_bench/benchmark_inputs" / profile_id / "config.yaml"
    destination = root / "case_prompts"
    destination.mkdir(exist_ok=True)
    for task, attribute in (("tp_gnn_placement_sta", "instructions"),
                            ("tp_gnn_sta_inner", "partition_instructions"),
                            ("placement_sta_inner", "placement_instructions")):
        definitions = {
            "tp_gnn_placement_sta": STA_PROFILE,
            "tp_gnn_sta_inner": TP_GNN_PROXY_PROFILE,
            "placement_sta_inner": PLACEMENT_PROXY_PROFILE,
        }
        definition = definitions[task]
        spec = replace(load_task(definition), base_config=profile_config,
                       benchmark_input_id=profile_id)
        candidate = json.loads((spec.initial_candidate / "candidate.json").read_text())
        prompt = "Current design: {}. Use only this case's own reference and feedback.\n\n".format(args.case)
        prompt += build_problem_description(spec, candidate, args.agent_workspace)
        path = destination / (task + ".md")
        if path.exists() and path.read_text() != prompt:
            raise ValueError("refusing to change source context or task contract during resume")
        path.write_text(prompt, encoding="utf-8")
        setattr(args, attribute, path)
    initial = json.loads((STA_PROFILE / "initial_candidate/candidate.json").read_text())
    args.initial_program = destination / "initial_program.py"
    program = render_candidate_program(initial)
    if args.initial_program.exists() and args.initial_program.read_text() != program:
        raise ValueError("refusing to change initial candidate during resume")
    args.initial_program.write_text(program, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.protocol == "knowledge-v2" and args.mode != "agentic3d":
        raise ValueError("knowledge-v2 currently implements the Agentic3D baseline only")
    if args.protocol == "legacy" and (args.knowledge_phase != "independent" or args.knowledge_split
            or args.knowledge_import or args.knowledge_export or args.knowledge_seed or args.test_knowledge_adaptation):
        raise ValueError("knowledge options require --protocol knowledge-v2")
    if args.knowledge_export and args.knowledge_phase != "train":
        raise ValueError("only training runs may export knowledge")
    if args.test_knowledge_adaptation and args.knowledge_phase != "test":
        raise ValueError("test adaptation requires test phase")
    configure_remote(args)
    if args.rounds != 5:
        raise ValueError("the comparison contract requires exactly five outer rounds")
    if args.gpu not in range(8):
        raise ValueError("GPU must be between 0 and 7")
    args.agent_workspace = args.agent_workspace.expanduser().resolve()
    if not args.agent_workspace.is_dir():
        raise FileNotFoundError(
            "clean agent workspace is missing: {}".format(args.agent_workspace)
        )
    root = args.output_dir.expanduser().resolve()
    config = {
            "mode": args.mode,
            "model": args.model,
            "reasoning_effort": "xhigh",
            "rounds": args.rounds,
            "gpu": args.gpu,
            "reference_run": args.reference_run,
            "remote_bundle": REMOTE_BUNDLE,
            "agent_workspace": str(args.agent_workspace),
            "outer_budget_semantics": "only final two-pass STA evaluations count",
        }
    if args.protocol == "knowledge-v2":
        from agentic3d.knowledge import digest, read_package
        from agentic3d.sta_knowledge_runner import configured_seed, implementation_digest
        config.update(protocol=args.protocol, knowledge_phase=args.knowledge_phase,
                      seed_sha256=digest(configured_seed(args, ROOT)),
                      implementation_sha256=implementation_digest(ROOT),
                      knowledge_split=json.loads(args.knowledge_split.read_text()) if args.knowledge_split else None,
                      knowledge_import_sha256=read_package(args.knowledge_import)["sha256"] if args.knowledge_import else None,
                      test_knowledge_adaptation=args.test_knowledge_adaptation)
    if args.case != "bp_multi" or args.remote_profile_config:
        config.update(case=args.case, remote_host=REMOTE_HOST, remote_profile_config=REMOTE_PROFILE_CONFIG,
                      remote_helper=REMOTE_HELPER, remote_comparison=REMOTE_COMPARISON,
                      ssh_proxy_command=SSH_PROXY_COMMAND)
        # No API calls until this case has its own successful physical reference.
        bundle = _remote_result(REMOTE_BUNDLE + "/benchmark_input.json")
        validate_case_reference(args.case, bundle,
                                _remote_result(str(Path(args.reference_run).parent / "evaluation_result.json")))
        run_dir = Path(bundle.get("run_dir", "run"))
        if run_dir.is_absolute() or ".." in run_dir.parts:
            raise ValueError("frozen run escapes its bundle")
        for stage in ("00_input", "10_floorplan"):
            frozen = _remote_result(str(Path(REMOTE_BUNDLE) / run_dir / stage / "stage_manifest.json"))
            reference = _remote_result(str(Path(args.reference_run) / stage / "stage_manifest.json"))
            if frozen != reference:
                raise ValueError("reference does not share frozen " + stage)
    if args.resume:
        previous = json.loads((root / "run_config.json").read_text())
        if previous.get("protocol", "legacy") != args.protocol:
            raise ValueError("cannot change experiment protocol on resume")
        for key in config:
            if previous.get(key) != config[key]:
                raise ValueError("resume configuration differs for " + key)
    else:
        root.mkdir(parents=True, exist_ok=False)
        _write_json(root / "run_config.json", config)
    with (root / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.case != "bp_multi" or args.remote_profile_config:
                write_case_prompts(args, root)
            _write_json(root / ("driver_start_{}.json".format(time.time_ns())), {
                "remote_comparison": REMOTE_COMPARISON, "remote_helper": REMOTE_HELPER,
                "resume": args.resume, "timestamp": time.time(),
            })
            if args.mode == "codex":
                run_codex_baseline(args, root)
            elif args.protocol == "knowledge-v2":
                from agentic3d.sta_knowledge_runner import run as run_knowledge_v2
                run_knowledge_v2(args, root, sys.modules[__name__])
            else:
                run_agentic3d(args, root)
        except BaseException as error:
            _write_json(root / ("driver_failure_{}.json".format(time.time_ns())), {
                "type": type(error).__name__, "message": str(error),
                "completed_outer_rounds": len(_history(root, args.rounds)),
                "timestamp": time.time(),
            })
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
