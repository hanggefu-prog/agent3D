from __future__ import annotations

import copy
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml

from agentic3d.codex_exec import CodexExecConfig, new_run_id, run_codex

from .evolve_protocol import render_candidate_program
from .model import validate_candidate_payload


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key_env: str
    api_mode: str
    timeout_seconds: int = 120
    max_tokens: int = 8192
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    reasoning_effort: Optional[str] = None
    max_attempts: int = 3
    stream: bool = False

    def public_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_mode": self.api_mode,
            "api_key_configured": bool(os.environ.get(self.api_key_env)),
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort,
            "max_attempts": self.max_attempts,
            "stream": self.stream,
        }


def load_llm_config(path: Path) -> LLMConfig:
    data = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(data, dict) or int(data.get("schema_version", 0)) != 1:
        raise ValueError("LLM config must be a schema_version 1 object: {}".format(path))
    provider = str(data.get("provider", "qwen"))
    if provider not in {"qwen", "openai", "deepseek"}:
        raise ValueError("unsupported configured provider: {}".format(provider))
    base_url = str(data.get("base_url", "")).rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("LLM base_url must be an https URL")
    api_key_env = str(data.get("api_key_env", "DASHSCOPE_API_KEY"))
    if not api_key_env or not api_key_env.replace("_", "").isalnum():
        raise ValueError("api_key_env must be an environment-variable name")
    api_mode = str(
        data.get(
            "api_mode",
            "responses" if provider == "openai" else "chat_completions",
        )
    )
    if api_mode not in {"responses", "chat_completions"}:
        raise ValueError("api_mode must be responses or chat_completions")
    reasoning_effort = data.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort not in {"none", "low", "medium", "high", "max"}:
        raise ValueError("reasoning_effort must be none, low, medium, high, or max")
    temperature = data.get("temperature")
    top_p = data.get("top_p")
    return LLMConfig(
        provider=provider,
        base_url=base_url,
        model=str(data.get("model", "qwen-plus")),
        api_key_env=api_key_env,
        api_mode=api_mode,
        timeout_seconds=int(data.get("timeout_seconds", 120)),
        max_tokens=int(data.get("max_tokens", 8192)),
        temperature=float(temperature) if temperature is not None else None,
        top_p=float(top_p) if top_p is not None else None,
        reasoning_effort=str(reasoning_effort) if reasoning_effort is not None else None,
        max_attempts=int(data.get("max_attempts", 3)),
        stream=bool(data.get("stream", False)),
    )


class OpenAICompatibleLLM:
    """Credential-safe client for OpenAI Responses and compatible chat APIs."""

    do_auto_trim = True

    def __init__(self, config: LLMConfig):
        if not os.environ.get(config.api_key_env):
            raise RuntimeError(
                "missing {} API key; set environment variable {}".format(
                    config.provider, config.api_key_env
                )
            )
        self._config = config
        self.debug_mode = False

    def _messages(self, prompt: Any) -> List[Dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt.strip()}]
        if not isinstance(prompt, list):
            raise TypeError("LLM prompt must be text or a message list")
        rows = []
        for row in prompt:
            if not isinstance(row, Mapping):
                raise TypeError("LLM message must be an object")
            rows.append({"role": str(row.get("role", "user")), "content": str(row.get("content", row.get("message", "")))})
        return rows

    def draw_sample(self, prompt: Any, *args, **kwargs) -> str:
        api_key = os.environ.get(self._config.api_key_env)
        if not api_key:
            raise RuntimeError(
                "{} API key environment variable is no longer set".format(
                    self._config.provider
                )
            )
        messages = self._messages(prompt)
        if self._config.api_mode == "responses":
            request_body: Dict[str, Any] = {
                "model": self._config.model,
                "input": messages,
                "max_output_tokens": self._config.max_tokens,
                "store": False,
            }
            if self._config.reasoning_effort and self._config.reasoning_effort != "none":
                request_body["reasoning"] = {"effort": self._config.reasoning_effort}
            url = self._config.base_url + "/responses"
        else:
            request_body = {
                "model": self._config.model,
                "messages": messages,
                "max_tokens": self._config.max_tokens,
                "stream": self._config.stream,
            }
            if self._config.temperature is not None:
                request_body["temperature"] = self._config.temperature
            if self._config.top_p is not None:
                request_body["top_p"] = self._config.top_p
            if self._config.provider == "deepseek" and self._config.reasoning_effort:
                if self._config.reasoning_effort == "none":
                    request_body["thinking"] = {"type": "disabled"}
                else:
                    effort = self._config.reasoning_effort
                    if effort == "medium":
                        effort = "high"
                    request_body["thinking"] = {
                        "type": "enabled",
                    }
                    request_body["reasoning_effort"] = effort
            url = self._config.base_url + "/chat/completions"
        empty_thinking_completion = False
        for attempt in range(self._config.max_attempts):
            attempt_body = dict(request_body)
            if (
                empty_thinking_completion
                and self._config.provider == "deepseek"
                and "thinking" in attempt_body
            ):
                # A reasoning stream can exhaust its output budget without ever
                # emitting final content.  Preserve one full thinking attempt,
                # then request a direct answer instead of repeating the same
                # empty, costly completion up to max_attempts times.
                attempt_body["thinking"] = {"type": "disabled"}
                attempt_body.pop("reasoning_effort", None)
            payload = json.dumps(attempt_body).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                    "User-Agent": "OpenAgentic3DBench/0.1",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self._config.timeout_seconds, context=ssl.create_default_context()) as response:
                    if self._config.api_mode == "chat_completions" and self._config.stream:
                        content = self._stream_chat_text(response)
                    else:
                        result = json.load(response)
                        content = self._response_text(result)
                if not isinstance(content, str) or not content.strip():
                    empty_thinking_completion = True
                    if attempt + 1 >= self._config.max_attempts:
                        raise RuntimeError(
                            "{} returned an empty completion after {} attempts".format(
                                self._config.provider, self._config.max_attempts
                            )
                        )
                    time.sleep(min(8, 2 ** attempt))
                    continue
                return content
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt + 1 >= self._config.max_attempts:
                    raise RuntimeError(
                        "{} request failed with HTTP {}".format(
                            self._config.provider, error.code
                        )
                    ) from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt + 1 >= self._config.max_attempts:
                    raise RuntimeError(
                        "{} request failed after {} attempts".format(
                            self._config.provider, self._config.max_attempts
                        )
                    ) from error
            time.sleep(min(8, 2 ** attempt))
        raise RuntimeError("{} request failed".format(self._config.provider))

    @staticmethod
    def _stream_chat_text(response: Any) -> str:
        """Collect OpenAI-compatible SSE chat deltas without logging them."""

        chunks: List[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
            line = line.strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices", []) if isinstance(event, Mapping) else []
            if not choices or not isinstance(choices[0], Mapping):
                continue
            delta = choices[0].get("delta", {})
            if not isinstance(delta, Mapping):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                chunks.append(content)
        return "".join(chunks)

    def _response_text(self, result: Mapping[str, Any]) -> str:
        if self._config.api_mode == "chat_completions":
            return str(result["choices"][0]["message"]["content"])
        texts = []
        for item in result.get("output", []):
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if isinstance(content, Mapping) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        texts.append(text)
        return "\n".join(texts)

    def draw_samples(self, prompts: List[Any], *args, **kwargs) -> List[str]:
        return [self.draw_sample(prompt, *args, **kwargs) for prompt in prompts]

    def close(self) -> None:
        return None


class CodexExecLLM:
    """LLM4AD-compatible provider backed by an existing Codex login.

    Each sample is an independent, read-only ``codex exec`` invocation. The
    evolution harness already carries current candidates, metrics, artifacts,
    and history in the prompt, so sharing a hidden Codex thread would make the
    benchmark less reproducible rather than more capable.
    """

    do_auto_trim = True

    def __init__(
        self,
        workspace: Path,
        run_root: Path,
        *,
        codex_binary: str = "codex",
        model: Optional[str] = None,
        timeout_seconds: int = 900,
    ):
        self._workspace = workspace.expanduser().resolve()
        self._run_root = run_root.expanduser().resolve()
        if not self._workspace.is_dir():
            raise FileNotFoundError(self._workspace)
        if timeout_seconds <= 0:
            raise ValueError("Codex timeout must be positive")
        self._run_root.mkdir(parents=True, exist_ok=True)
        self._codex_binary = codex_binary
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._counter = 0
        self.debug_mode = False

    @staticmethod
    def _prompt_text(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if not isinstance(prompt, list):
            raise TypeError("Codex prompt must be text or a message list")
        rows = []
        for message in prompt:
            if not isinstance(message, Mapping):
                raise TypeError("Codex message must be an object")
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", message.get("message", "")))
            rows.append("# {}\n{}".format(role, content))
        return "\n\n".join(rows)

    def draw_sample(self, prompt: Any, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        with self._lock:
            self._counter += 1
            order = self._counter
        run_dir = self._run_root / "sample-{:04d}-{}".format(order, new_run_id())
        result = run_codex(
            self._prompt_text(prompt),
            CodexExecConfig(
                workspace=self._workspace,
                run_dir=run_dir,
                codex_binary=self._codex_binary,
                model=self._model,
                sandbox="read-only",
                timeout_seconds=self._timeout_seconds,
                ephemeral=True,
                skip_git_repo_check=True,
            ),
        )
        if not result.success:
            raise RuntimeError(
                "Codex sample failed with exit {}{}; see {}".format(
                    result.exit_code,
                    " after timeout" if result.timed_out else "",
                    result.metadata_path,
                )
            )
        if not result.final_message.strip():
            raise RuntimeError("Codex sample returned an empty final message")
        return result.final_message

    def draw_samples(self, prompts: List[Any], *args: Any, **kwargs: Any) -> List[str]:
        return [self.draw_sample(prompt, *args, **kwargs) for prompt in prompts]

    def public_dict(self) -> Dict[str, Any]:
        return {
            "provider": "codex",
            "authentication": "existing_codex_login",
            "model": self._model or "codex_default",
            "workspace": str(self._workspace),
            "run_root": str(self._run_root),
            "sandbox": "read-only",
            "timeout_seconds": self._timeout_seconds,
        }

    def close(self) -> None:
        return None


class ReplayLLM:
    """Deterministic no-network provider for verifier and adapter smoke tests."""

    do_auto_trim = True

    def __init__(self, initial_candidate: Union[Dict[str, Any], str]):
        self._initial_program = initial_candidate if isinstance(initial_candidate, str) else None
        self._initial = (
            None
            if isinstance(initial_candidate, str)
            else validate_candidate_payload(initial_candidate)
        )
        self._counter = 0
        self._lock = threading.Lock()
        self.debug_mode = False

    def draw_sample(self, prompt: Any, *args, **kwargs) -> str:
        with self._lock:
            self._counter += 1
            index = self._counter
        if self._initial_program is not None:
            return self._initial_program
        candidate = copy.deepcopy(self._initial)
        assert candidate is not None
        candidate["notes"] = "deterministic replay candidate {}".format(index)
        return render_candidate_program(candidate)

    def draw_samples(self, prompts: List[Any], *args, **kwargs) -> List[str]:
        return [self.draw_sample(prompt, *args, **kwargs) for prompt in prompts]

    def close(self) -> None:
        return None
