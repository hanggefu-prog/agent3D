#!/usr/bin/env python3
"""Inject the maintained DeepSeek transport into an exported task bundle."""

from pathlib import Path
import sys

import yaml


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: configure_openevolve_deepseek.py CONFIG_YAML")
    path = Path(sys.argv[1]).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("OpenEvolve configuration must be an object")

    # Preserve task-owned database, prompt, evaluator, code-size, and search
    # settings.  The task owns its experiment budget; this helper resolves only
    # the transport plus function-task compatibility settings.
    if int(data.get("max_iterations", 0)) < 1:
        raise ValueError("task max_iterations must be positive")
    data["checkpoint_interval"] = 1
    data["diff_based_evolution"] = False
    llm = data.setdefault("llm", {})
    llm.update(
        {
            "models": [{"name": "deepseek-v4-flash", "weight": 1.0}],
            "evaluator_models": [
                {"name": "deepseek-v4-flash", "weight": 1.0}
            ],
            "api_base": "https://api.deepseek.com",
            "api_key": "${DEEPSEEK_API_KEY}",
            # Function-mode tasks return one short Python block.  A 32K output
            # allowance made DeepSeek continue generating until the transport
            # timeout even when the useful function was already complete.
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 4096,
            "timeout": 300,
            "retries": 2,
            "retry_delay": 5,
        }
    )
    evaluator = data.setdefault("evaluator", {})
    evaluator["timeout"] = 21600
    evaluator["parallel_evaluations"] = 1
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
