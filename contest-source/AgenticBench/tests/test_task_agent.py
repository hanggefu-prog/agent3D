from __future__ import annotations

import json
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic3d.task_agent import TaskAgent, TaskAgentConfig  # noqa: E402


class TaskAgentTests(unittest.TestCase):
    def _fake_codex(self, directory: Path) -> Path:
        executable = directory / "fake-codex"
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
                output_path = pathlib.Path(args[args.index('--output-last-message') + 1])
                match = re.search(r"创建 `([^`]+/plan[.]md)`", prompt)
                if not match:
                    raise SystemExit('plan path missing from prompt')
                plan_path = pathlib.Path.cwd() / match.group(1)
                plan_path.write_text('# Round plan\\n\\nPlan before code.\\n', encoding='utf-8')
                resumed = 'resume' in args
                round_number = 2 if resumed else 1
                status = 'goal_achieved' if resumed else 'continue'
                result = {{
                    'round': round_number,
                    'plan': {{
                        'analysis': 'baseline evidence',
                        'hypothesis': 'small coherent change',
                        'components': ['component'],
                        'files_to_modify': [],
                        'changes': ['change'],
                        'validation': ['test'],
                        'rollback': 'revert candidate',
                    }},
                    'execution': {{
                        'summary': 'executed after planning',
                        'files_changed': ['candidate.cfg'],
                        'commands_run': ['test'],
                        'validation_results': [
                            {{'name': 'correctness', 'passed': True, 'evidence': 'ok'}}
                        ],
                    }},
                    'decision': {{
                        'status': status,
                        'reason': 'continue once' if not resumed else 'goal proven',
                        'progress': 'improved',
                        'goal_evidence': ['metric improved'],
                        'next_focus': 'second hypothesis' if not resumed else '',
                    }},
                }}
                print(json.dumps({{'type': 'thread.started', 'thread_id': 'same-thread'}}), flush=True)
                output_path.write_text(json.dumps(result), encoding='utf-8')
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return executable

    def test_two_rounds_share_context_and_require_plans(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            skill = workspace / "skills" / "layout" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("# Layout skill\n", encoding="utf-8")
            fake_codex = self._fake_codex(workspace)
            task_dir = workspace / ".agentic3d" / "tasks" / "layout-test"

            state = TaskAgent(
                TaskAgentConfig(
                    name="Layout Task Agent",
                    goal="Improve HPWL while passing density gates",
                    workspace=workspace,
                    task_dir=task_dir,
                    skill_path=skill,
                    codex_binary=str(fake_codex),
                    max_rounds=3,
                )
            ).run()

            self.assertEqual(state.status, "completed")
            self.assertEqual(state.thread_id, "same-thread")
            self.assertEqual(len(state.rounds), 2)
            first_prompt = (
                task_dir / "rounds" / "001" / "codex" / "prompt.md"
            ).read_text(encoding="utf-8")
            self.assertIn("输入与决策优先级", first_prompt)
            self.assertIn("本阶段候选达到 handoff 中的", first_prompt)
            self.assertIn("不得静默替换该假设", first_prompt)
            self.assertTrue((task_dir / "rounds" / "001" / "plan.md").is_file())
            self.assertTrue((task_dir / "rounds" / "002" / "plan.md").is_file())
            second_metadata = json.loads(
                (task_dir / "rounds" / "002" / "codex" / "result.json").read_text()
            )
            self.assertEqual(second_metadata["resume_thread_id"], "same-thread")
            self.assertEqual(second_metadata["command"][:3], [str(fake_codex), "exec", "resume"])


if __name__ == "__main__":
    unittest.main()
