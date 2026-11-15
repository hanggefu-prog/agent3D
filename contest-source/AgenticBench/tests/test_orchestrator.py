from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic3d.orchestrator import (  # noqa: E402
    Agentic3DOrchestrator,
    WorkflowConfig,
    _repository_state_root,
)
from agentic3d.analyzers import GlobalAnalyzer, LocalAnalyzer  # noqa: E402


FRAMEWORK = Path(__file__).resolve().parents[1]


class OrchestratorTests(unittest.TestCase):
    def test_default_state_root_is_external_to_target_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = root / "checkout" / "Open3DBench-V2.0"
            repository.mkdir(parents=True)
            state_home = root / "state"
            with patch.dict(
                os.environ,
                {"XDG_STATE_HOME": str(state_home)},
                clear=False,
            ):
                os.environ.pop("AGENTIC3D_RUNS_ROOT", None)
                state_root = _repository_state_root(repository)

            self.assertTrue(
                state_root.is_relative_to((state_home / "agentic3d").resolve())
            )
            self.assertFalse(state_root.is_relative_to(repository))
            self.assertIn("Open3DBench-V2.0-", state_root.name)

    def test_analyzers_cannot_accept_an_incomplete_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "candidate.json"
            candidate.write_text(
                json.dumps({"task_status": "failed"}), encoding="utf-8"
            )
            local_feedback = root / "local.json"
            local_feedback.write_text(
                json.dumps(
                    {"analysis_phase": "post_task", "decision": "accept"}
                ),
                encoding="utf-8",
            )
            local = object.__new__(LocalAnalyzer)
            local.stage = "layout"
            local_output = {
                "stage": "layout",
                "analysis_phase": "post_task",
                "decision": "accept",
                "single_hypothesis": "audit the dispatched layout hypothesis",
                "scope_proposal": {"source_paths": [], "config_paths": []},
            }
            with self.assertRaisesRegex(RuntimeError, "incomplete Task Agent"):
                local._validate(
                    local_output,
                    1,
                    {
                        "kind": "post_task_local_analysis",
                        "candidate_update": str(candidate),
                    },
                )

            global_analyzer = object.__new__(GlobalAnalyzer)
            global_output = {
                "round": 1,
                "global_plan": {"target_stage": "layout"},
                "decision": {
                    "status": "dispatch",
                    "candidate_status": "accept",
                },
            }
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                global_analyzer._validate(
                    global_output,
                    1,
                    {
                        "candidate_update": str(candidate),
                        "local_feedback": str(local_feedback),
                    },
                )

    def _repo(self, root: Path) -> Path:
        (root / "Place-3D" / "src").mkdir(parents=True)
        (root / "Place-3D" / "src" / "api.py").write_text("# api\n")
        (root / "Place-3D" / "src" / "partition").mkdir()
        (root / "Place-3D" / "src" / "placement").mkdir()
        (root / "Place-3D" / "src" / "placement" / "placer.py").write_text(
            "DENSITY = 0.70\n"
        )
        (root / "Place-3D" / "config").mkdir()
        (root / "Place-3D" / "config" / "partition").mkdir()
        (root / "Place-3D" / "config" / "partition" / "base.json").write_text(
            "{}\n"
        )
        (root / "Place-3D" / "results").mkdir()
        (root / "OpenROAD-3D" / "flow").mkdir(parents=True)
        (root / "OpenROAD-3D" / "flow" / "Makefile").write_text("all:\n")
        subprocess.run(
            ["git", "-C", str(root), "init"], check=True, capture_output=True
        )
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            [
                "git", "-C", str(root), "-c", "user.name=test", "-c",
                "user.email=test@example.com", "commit", "-m", "baseline",
            ],
            check=True,
            capture_output=True,
        )
        return root

    def _fake_codex(self, root: Path) -> Path:
        executable = root / "fake-codex"
        executable.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import pathlib
                import re
                import sys

                args = sys.argv[1:]
                prompt = sys.stdin.read()
                output = pathlib.Path(args[args.index('--output-last-message') + 1])
                if 'Local Analyzer' in prompt and 'Monitor → Guide' in prompt:
                    phase = 'post_task' if 'analysis_phase=post_task' in prompt else 'pre_task'
                    result = {{
                        'stage': 'layout',
                        'analysis_phase': phase,
                        'conclusion': 'inspect layout evidence',
                        'evidence': ['metrics'],
                        'single_hypothesis': 'reduce overflow without HPWL regression',
                        'scope_proposal': {{
                            'source_paths': ['Place-3D/src/placement/placer.py'],
                            'config_paths': [],
                            'allow_create': False,
                            'allow_delete': False,
                            'allow_rename': False,
                            'allow_mode_change': False,
                            'allow_symlink': False,
                            'allowed_extensions': ['.py'],
                            'max_changed_files': 2,
                            'max_added_lines': 20,
                            'max_deleted_lines': 20,
                        }},
                        'allowed_changes': ['placement config'],
                        'forbidden_changes': ['partition', 'routing'],
                        'acceptance_gates': ['HPWL <= baseline'],
                        'correctness_gates': ['valid DEF'],
                        'replay_rule': 'replay after pass',
                        'rollback_rule': 'rollback on any failed gate',
                        'decision': 'accept' if phase == 'post_task' else 'recommend',
                    }}
                    thread = 'local-thread'
                elif '你是 Agentic3D 的 Global Analyzer' in prompt:
                    round_number = int(re.search(r'全局分析第 (\\d+) 轮', prompt).group(1))
                    result = {{
                        'round': round_number,
                        'diagnosis': {{
                            'summary': 'layout is selected',
                            'ppa_objective': 'improve HPWL with correctness',
                            'bottlenecks': [{{'stage': 'layout', 'severity': 'high', 'reason': 'baseline'}}],
                            'evidence': ['repository context'],
                        }},
                        'global_plan': {{
                            'target_stage': 'layout',
                            'rationale': 'layout bottleneck',
                            'action_hint': 'one placement hypothesis',
                            'success_criteria': ['HPWL improves'],
                            'required_artifacts': ['placement metrics'],
                            'frozen_constraints': ['partition'],
                            'local_tasks': [] if round_number else [{{
                                'task_id': 'layout-source-round-0',
                                'template': 'placement_preconditioner',
                                'stage': 'layout',
                                'hypothesis': 'reduce overflow without HPWL regression',
                                'objective': {{
                                    'goal': 'improve placement objective',
                                    'metric_weights': {{'hpwl': 0.9, 'overflow': 0.1}},
                                }},
                                'acceptance_gates': ['HPWL improves'],
                                'correctness_gates': ['valid DEF'],
                                'evaluation_budget': 1,
                                'source_scope': [
                                    'Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py'
                                ],
                                'config_scope': [],
                                'context_artifacts': ['placement metrics'],
                            }}],
                        }},
                        'decision': {{
                            'status': 'dispatch' if round_number == 0 else 'complete',
                            'candidate_status': 'not_applicable' if round_number == 0 else 'accept',
                            'reason': 'dispatch layout' if round_number == 0 else 'goal proven',
                        }},
                    }}
                    thread = 'global-thread'
                else:
                    match = re.search(r"创建 `([^`]+/plan[.]md)`", prompt)
                    plan = pathlib.Path.cwd() / match.group(1)
                    plan.write_text('# Plan before edits\\n', encoding='utf-8')
                    target = pathlib.Path.cwd() / 'Place-3D/src/placement/placer.py'
                    target.write_text('DENSITY = 0.75\\n', encoding='utf-8')
                    result = {{
                        'round': 1,
                        'plan': {{
                            'analysis': 'read adapter context',
                            'hypothesis': 'one layout change',
                            'components': ['Place-3D'],
                            'files_to_modify': ['Place-3D/src/placement/placer.py'],
                            'changes': ['density source constant'],
                            'validation': ['test'],
                            'rollback': 'restore config',
                        }},
                        'execution': {{
                            'summary': 'validated candidate',
                            'files_changed': ['Place-3D/src/placement/placer.py'],
                            'commands_run': ['make verify'],
                            'validation_results': [{{'name': 'correctness', 'passed': True, 'evidence': 'ok'}}],
                        }},
                        'decision': {{
                            'status': 'goal_achieved',
                            'reason': 'candidate passes',
                            'progress': 'improved',
                            'goal_evidence': ['metrics'],
                            'next_focus': '',
                        }},
                    }}
                    thread = 'task-thread'
                print(json.dumps({{'type': 'thread.started', 'thread_id': thread}}), flush=True)
                output.write_text(json.dumps(result), encoding='utf-8')
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return executable

    def test_round_zero_user_ingress_and_fixed_collaboration_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = self._repo(Path(temp) / "repo")
            fake = self._fake_codex(repo)
            workflow = repo / ".agentic3d" / "workflows" / "test"
            state = Agentic3DOrchestrator(
                WorkflowConfig(
                    repository=repo,
                    framework_root=FRAMEWORK,
                    user_task="用户命令：优化布局 HPWL",
                    workflow_dir=workflow,
                    codex_binary=str(fake),
                    max_cycles=2,
                    task_max_rounds=1,
                )
            ).run()
            self.assertEqual(state.status, "completed")
            self.assertEqual([item["stage"] for item in state.cycles], ["layout"])
            self.assertIn("layout", state.accepted_candidates)
            round0_prompt = (
                workflow
                / "global-analyzer"
                / "runs"
                / "000"
                / "prompt.md"
            ).read_text(encoding="utf-8")
            self.assertIn("用户命令：优化布局 HPWL", round0_prompt)
            self.assertIn("解析 orchestrator 传入的 `user_task`", round0_prompt)
            self.assertIn(
                "成功标准（`success_criteria`）、正确性门槛、基线与候选的可比性",
                round0_prompt,
            )
            pre_prompt = (
                workflow
                / "local-analyzers"
                / "layout"
                / "runs"
                / "000"
                / "prompt.md"
            ).read_text(encoding="utf-8")
            post_prompt = (
                workflow
                / "local-analyzers"
                / "layout"
                / "runs"
                / "001"
                / "prompt.md"
            ).read_text(encoding="utf-8")
            self.assertIn("analysis_phase=pre_task", pre_prompt)
            self.assertIn("analysis_phase=post_task", post_prompt)
            self.assertIn("不得另行提出下一轮假设", post_prompt)
            task_dir = Path(state.cycles[0]["task_state"]).parent
            task_prompt = (
                task_dir
                / "rounds"
                / "001"
                / "codex"
                / "prompt.md"
            ).read_text(encoding="utf-8")
            self.assertIn("你无权声明端到端的全局任务已经完成", task_prompt)
            self.assertIn("输入与决策优先级", task_prompt)
            self.assertIn("不得静默替换该假设", task_prompt)
            self.assertTrue(
                (
                    task_dir
                    / "rounds"
                    / "001"
                    / "plan.md"
                ).is_file()
            )
            self.assertTrue((workflow / "global-ppa-report.json").is_file())
            self.assertTrue((workflow / "local-task-catalog.json").is_file())
            self.assertTrue((workflow / "cycles" / "000" / "scope.json").is_file())
            candidate = json.loads(
                (workflow / "cycles" / "000" / "candidate-update.json").read_text()
            )
            self.assertTrue(candidate["integrated_commit"])
            self.assertTrue(
                json.loads(Path(candidate["scope_report"]).read_text())["passed"]
            )

    def test_direct_scope_mode_preserves_unrestricted_workspace_edits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = self._repo(Path(temp) / "repo")
            fake = self._fake_codex(repo)
            workflow = repo / ".agentic3d" / "workflows" / "direct"
            state = Agentic3DOrchestrator(
                WorkflowConfig(
                    repository=repo,
                    framework_root=FRAMEWORK,
                    user_task="直接修改布局源码",
                    workflow_dir=workflow,
                    codex_binary=str(fake),
                    max_cycles=1,
                    task_max_rounds=1,
                    scope_mode="direct",
                )
            ).run()
            self.assertEqual(state.status, "completed")
            candidate = json.loads(
                (workflow / "cycles" / "000" / "candidate-update.json").read_text()
            )
            self.assertEqual(candidate["scope_mode"], "direct")
            self.assertIsNone(candidate["worktree"])
            self.assertIsNone(candidate["integrated_commit"])
            report = json.loads(Path(candidate["scope_report"]).read_text())
            self.assertTrue(report["bypassed"])
            self.assertEqual(
                (repo / "Place-3D" / "src" / "placement" / "placer.py").read_text(),
                "DENSITY = 0.75\n",
            )
            self.assertIn(
                "Place-3D/src/placement/placer.py",
                subprocess.run(
                    ["git", "-C", str(repo), "status", "--short"],
                    text=True,
                    stdout=subprocess.PIPE,
                    check=True,
                ).stdout,
            )


if __name__ == "__main__":
    unittest.main()
