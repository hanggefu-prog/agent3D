from __future__ import annotations

import json
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic3d.codex_exec import (  # noqa: E402
    CodexExecConfig,
    build_codex_command,
    run_codex,
)
from local_bench.llm_client import CodexExecLLM  # noqa: E402


class CodexExecTests(unittest.TestCase):
    def _fake_codex(self, directory: Path, *, sleep_seconds: float = 0.0) -> Path:
        executable = directory / "fake-codex"
        executable.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import pathlib
                import sys
                import time

                args = sys.argv[1:]
                output = pathlib.Path(args[args.index('--output-last-message') + 1])
                prompt = sys.stdin.read()
                time.sleep({sleep_seconds})
                print(json.dumps({{'type': 'thread.started', 'thread_id': 'test-thread'}}), flush=True)
                print(json.dumps({{'type': 'item.completed', 'text': 'working'}}), flush=True)
                print('fake diagnostic', file=sys.stderr, flush=True)
                output.write_text('finished: ' + prompt, encoding='utf-8')
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return executable

    def test_build_command_uses_stdin_and_safe_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = CodexExecConfig(
                workspace=root,
                run_dir=root / "run",
                model="test-model",
                sandbox="read-only",
                config_overrides=("model_reasoning_effort=high",),
                ephemeral=True,
            )
            command = build_codex_command(config, root / "final.md")

        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertIn("--json", command)
        self.assertIn("read-only", command)
        self.assertIn("test-model", command)
        self.assertEqual(command[-1], "-")

    def test_run_codex_persists_trace_and_final_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_codex = self._fake_codex(root)
            events = []
            diagnostics = []
            result = run_codex(
                "hello agent",
                CodexExecConfig(
                    workspace=root,
                    run_dir=root / "run-001",
                    codex_binary=str(fake_codex),
                ),
                on_event=events.append,
                on_stderr=diagnostics.append,
            )

            self.assertTrue(result.success)
            self.assertEqual(result.final_message, "finished: hello agent")
            self.assertEqual(result.thread_id, "test-thread")
            self.assertEqual(events[0]["type"], "thread.started")
            self.assertIn("fake diagnostic", "".join(diagnostics))
            metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["run_id"], "run-001")
            self.assertEqual(
                (root / "run-001" / "prompt.md").read_text(encoding="utf-8"),
                "hello agent",
            )
            self.assertEqual(
                len((root / "run-001" / "events.jsonl").read_text().splitlines()),
                2,
            )

    def test_timeout_returns_a_failed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_codex = self._fake_codex(root, sleep_seconds=1.0)
            result = run_codex(
                "slow",
                CodexExecConfig(
                    workspace=root,
                    run_dir=root / "run-timeout",
                    codex_binary=str(fake_codex),
                    timeout_seconds=0.05,
                ),
            )

            self.assertFalse(result.success)
            self.assertTrue(result.timed_out)
            self.assertNotEqual(result.exit_code, 0)

    def test_resume_command_uses_the_existing_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = CodexExecConfig(
                workspace=root,
                run_dir=root / "run",
                sandbox="workspace-write",
                profile="initial-only-profile",
                resume_thread_id="thread-123",
            )
            command = build_codex_command(config, root / "final.md")

        self.assertEqual(command[:3], ["codex", "exec", "resume"])
        self.assertIn("thread-123", command)
        self.assertNotIn("--profile", command)
        self.assertNotIn("--sandbox", command)
        self.assertEqual(command[-1], "-")

    def test_rejects_danger_full_access(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                CodexExecConfig(
                    workspace=root,
                    run_dir=root / "run",
                    sandbox="danger-full-access",
                )

    def test_codex_llm_provider_uses_read_only_observable_exec(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_codex = self._fake_codex(root)
            provider = CodexExecLLM(
                root,
                root / "samples",
                codex_binary=str(fake_codex),
                timeout_seconds=10,
            )
            response = provider.draw_sample([
                {"role": "system", "content": "system contract"},
                {"role": "user", "content": "return candidate"},
            ])
            sample_dirs = list((root / "samples").iterdir())

        self.assertIn("# SYSTEM", response)
        self.assertIn("# USER", response)
        self.assertEqual(len(sample_dirs), 1)
        self.assertEqual(provider.public_dict()["sandbox"], "read-only")


if __name__ == "__main__":
    unittest.main()
