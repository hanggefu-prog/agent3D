"""Safe, observable subprocess adapter for ``codex exec``.

This module is intentionally independent of the future scheduler. Every agent
invocation goes through :func:`run_codex` and produces the same run-artifact
layout, so an orchestrator can consume results without knowing subprocess
details.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple


JsonEventCallback = Callable[[Dict[str, Any]], None]
TextCallback = Callable[[str], None]


class CodexExecError(RuntimeError):
    """Base error raised before a Codex process can produce a run result."""


class CodexNotFoundError(CodexExecError):
    """Raised when the configured Codex executable cannot be found."""


@dataclass(frozen=True)
class CodexExecConfig:
    """Configuration for one isolated Codex invocation."""

    workspace: Path
    run_dir: Path
    codex_binary: str = "codex"
    model: Optional[str] = None
    sandbox: str = "workspace-write"
    profile: Optional[str] = None
    config_overrides: Sequence[str] = field(default_factory=tuple)
    images: Sequence[Path] = field(default_factory=tuple)
    output_schema: Optional[Path] = None
    timeout_seconds: Optional[float] = None
    ephemeral: bool = False
    skip_git_repo_check: bool = False
    strict_config: bool = False
    resume_thread_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.sandbox not in {"read-only", "workspace-write"}:
            raise ValueError(
                "sandbox must be 'read-only' or 'workspace-write'; "
                "danger-full-access is deliberately unsupported"
            )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.resume_thread_id is not None and not self.resume_thread_id.strip():
            raise ValueError("resume_thread_id must not be empty")


@dataclass(frozen=True)
class CodexRunResult:
    """Machine-readable outcome of one Codex invocation."""

    run_id: str
    success: bool
    exit_code: int
    timed_out: bool
    started_at: str
    finished_at: str
    duration_seconds: float
    final_message: str
    run_dir: str
    events_path: str
    stderr_path: str
    final_message_path: str
    metadata_path: str
    thread_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def build_codex_command(config: CodexExecConfig, final_message_path: Path) -> List[str]:
    """Build an argv list without invoking a shell or embedding the prompt."""

    if config.resume_thread_id:
        # `codex exec resume` intentionally exposes fewer flags than the first
        # invocation. The session retains its original workspace/profile/sandbox;
        # cwd is also set on Popen as a second, explicit workspace boundary.
        command = [
            config.codex_binary,
            "exec",
            "resume",
            "--json",
            "--output-last-message",
            str(final_message_path.resolve()),
        ]
    else:
        command = [
            config.codex_binary,
            "exec",
            "--json",
            "--color",
            "never",
            "--sandbox",
            config.sandbox,
            "--cd",
            str(config.workspace.resolve()),
            "--output-last-message",
            str(final_message_path.resolve()),
        ]

    if config.model:
        command.extend(["--model", config.model])
    if config.profile and not config.resume_thread_id:
        command.extend(["--profile", config.profile])
    if config.strict_config:
        command.append("--strict-config")
    if config.ephemeral:
        command.append("--ephemeral")
    if config.skip_git_repo_check:
        command.append("--skip-git-repo-check")
    if config.output_schema:
        command.extend(["--output-schema", str(config.output_schema.resolve())])
    for override in config.config_overrides:
        command.extend(["--config", override])
    for image in config.images:
        command.extend(["--image", str(image.resolve())])

    if config.resume_thread_id:
        command.append(config.resume_thread_id)

    # Reading the prompt from stdin avoids shell quoting and argv length issues.
    command.append("-")
    return command


def _write_metadata(path: Path, data: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _pump_stream(
    stream: TextIO,
    stream_name: str,
    messages: "queue.Queue[Tuple[str, Optional[str]]]",
) -> None:
    try:
        for line in iter(stream.readline, ""):
            messages.put((stream_name, line))
    finally:
        messages.put((stream_name, None))
        stream.close()


def run_codex(
    prompt: str,
    config: CodexExecConfig,
    *,
    on_event: Optional[JsonEventCallback] = None,
    on_stderr: Optional[TextCallback] = None,
) -> CodexRunResult:
    """Run Codex once, stream its output, and persist a reproducible trace.

    A non-zero Codex exit is represented in :class:`CodexRunResult`; launch
    failures raise :class:`CodexExecError` because no agent run occurred.
    """

    workspace = config.workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise CodexExecError(f"workspace is not a directory: {workspace}")

    run_dir = config.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    run_id = run_dir.name
    events_path = run_dir / "events.jsonl"
    stderr_path = run_dir / "stderr.log"
    final_message_path = run_dir / "final_message.md"
    metadata_path = run_dir / "result.json"
    prompt_path = run_dir / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    normalized_config = CodexExecConfig(
        workspace=workspace,
        run_dir=run_dir,
        codex_binary=config.codex_binary,
        model=config.model,
        sandbox=config.sandbox,
        profile=config.profile,
        config_overrides=config.config_overrides,
        images=config.images,
        output_schema=config.output_schema,
        timeout_seconds=config.timeout_seconds,
        ephemeral=config.ephemeral,
        skip_git_repo_check=config.skip_git_repo_check,
        strict_config=config.strict_config,
        resume_thread_id=config.resume_thread_id,
    )
    command = build_codex_command(normalized_config, final_message_path)
    started_at = _utc_now()
    started_monotonic = time.monotonic()

    try:
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
            env=os.environ.copy(),
        )
    except FileNotFoundError as exc:
        raise CodexNotFoundError(
            f"Codex executable not found: {config.codex_binary!r}"
        ) from exc
    except OSError as exc:
        raise CodexExecError(f"failed to launch Codex: {exc}") from exc

    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        process.stdin.write(prompt)
        process.stdin.close()
    except BrokenPipeError:
        # The process may reject invalid CLI arguments before reading stdin.
        pass

    messages: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
    threads = [
        threading.Thread(
            target=_pump_stream,
            args=(process.stdout, "stdout", messages),
            daemon=True,
        ),
        threading.Thread(
            target=_pump_stream,
            args=(process.stderr, "stderr", messages),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    thread_id: Optional[str] = None
    completed_streams = set()
    with events_path.open("w", encoding="utf-8") as events_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        while len(completed_streams) < 2:
            if (
                not timed_out
                and config.timeout_seconds is not None
                and time.monotonic() - started_monotonic >= config.timeout_seconds
            ):
                timed_out = True
                process.terminate()

            try:
                stream_name, line = messages.get(timeout=0.1)
            except queue.Empty:
                if timed_out and process.poll() is None:
                    # Give Codex a brief chance to terminate cleanly, then force it.
                    if time.monotonic() - started_monotonic >= config.timeout_seconds + 2.0:  # type: ignore[operator]
                        process.kill()
                continue

            if line is None:
                completed_streams.add(stream_name)
                continue

            if stream_name == "stderr":
                stderr_file.write(line)
                stderr_file.flush()
                if on_stderr:
                    on_stderr(line)
                continue

            events_file.write(line)
            events_file.flush()
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                parsed = {"type": "unparsed_stdout", "text": line.rstrip("\n")}
            if parsed.get("type") == "thread.started" and parsed.get("thread_id"):
                thread_id = str(parsed["thread_id"])
            if on_event:
                on_event(parsed)

    for thread in threads:
        thread.join(timeout=1.0)
    exit_code = process.wait()
    finished_at = _utc_now()
    duration = time.monotonic() - started_monotonic
    final_message = (
        final_message_path.read_text(encoding="utf-8")
        if final_message_path.exists()
        else ""
    )

    result = CodexRunResult(
        run_id=run_id,
        success=exit_code == 0 and not timed_out,
        exit_code=exit_code,
        timed_out=timed_out,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(duration, 3),
        final_message=final_message,
        run_dir=str(run_dir),
        events_path=str(events_path),
        stderr_path=str(stderr_path),
        final_message_path=str(final_message_path),
        metadata_path=str(metadata_path),
        thread_id=thread_id or config.resume_thread_id,
    )
    _write_metadata(
        metadata_path,
        {
            **result.to_dict(),
            "command": command,
            "prompt_path": str(prompt_path),
            "workspace": str(workspace),
            "sandbox": config.sandbox,
            "model": config.model,
            "resume_thread_id": config.resume_thread_id,
        },
    )
    return result


def _load_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        try:
            return args.prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            parser.error(f"cannot read prompt file: {exc}")
    if sys.stdin.isatty():
        parser.error("provide --prompt, --prompt-file, or pipe a prompt on stdin")
    return sys.stdin.read()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic3d-codex-exec",
        description="Run one observable, reproducible Codex agent invocation.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="Prompt text (prefer a file or stdin)")
    prompt_group.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file")
    parser.add_argument(
        "-C", "--workspace", type=Path, default=Path.cwd(), help="Agent workspace"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="New artifact directory (default: WORKSPACE/.agentic3d/runs/RUN_ID)",
    )
    parser.add_argument("--codex-bin", default="codex", help="Codex executable")
    parser.add_argument("--model", help="Codex model override")
    parser.add_argument(
        "--sandbox",
        choices=("read-only", "workspace-write"),
        default="workspace-write",
    )
    parser.add_argument("--profile", help="Codex configuration profile")
    parser.add_argument(
        "--config", action="append", default=[], metavar="KEY=VALUE", help="Codex config override"
    )
    parser.add_argument("--image", action="append", default=[], type=Path)
    parser.add_argument("--output-schema", type=Path)
    parser.add_argument("--timeout", type=float, help="Wall-clock timeout in seconds")
    parser.add_argument("--ephemeral", action="store_true")
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument("--strict-config", action="store_true")
    parser.add_argument(
        "--resume-session",
        metavar="THREAD_ID",
        help="Continue an existing Codex exec session while writing a new trace",
    )
    parser.add_argument(
        "--stream-events",
        action="store_true",
        help="Emit wrapped Codex JSON events to stdout before the result",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not relay Codex stderr")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    prompt = _load_prompt(args, parser)
    workspace = args.workspace.expanduser().resolve()
    run_dir = args.run_dir
    if run_dir is None:
        artifact_root = os.environ.get("AGENTIC3D_RUNS_ROOT")
        run_root = (
            Path(artifact_root).expanduser().resolve()
            if artifact_root
            else workspace / ".agentic3d"
        )
        run_dir = run_root / "codex" / new_run_id()

    def emit_event(event: Dict[str, Any]) -> None:
        if args.stream_events:
            print(
                json.dumps({"kind": "codex_event", "event": event}, ensure_ascii=False),
                flush=True,
            )

    def emit_stderr(line: str) -> None:
        if not args.quiet:
            print(line, end="", file=sys.stderr, flush=True)

    config = CodexExecConfig(
        workspace=workspace,
        run_dir=run_dir,
        codex_binary=args.codex_bin,
        model=args.model,
        sandbox=args.sandbox,
        profile=args.profile,
        config_overrides=tuple(args.config),
        images=tuple(args.image),
        output_schema=args.output_schema,
        timeout_seconds=args.timeout,
        ephemeral=args.ephemeral,
        skip_git_repo_check=args.skip_git_repo_check,
        strict_config=args.strict_config,
        resume_thread_id=args.resume_session,
    )
    try:
        result = run_codex(
            prompt, config, on_event=emit_event, on_stderr=emit_stderr
        )
    except (CodexExecError, ValueError, OSError) as exc:
        print(json.dumps({"kind": "launch_error", "error": str(exc)}), file=sys.stderr)
        return 127

    print(
        json.dumps(
            {"kind": "run_result", **result.to_dict()}, ensure_ascii=False
        ),
        flush=True,
    )
    if result.timed_out:
        return 124
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
