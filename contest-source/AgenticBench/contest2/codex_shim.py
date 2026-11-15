#!/usr/bin/env python3
"""Inject the organizer's metered provider without editing frozen v14 control modules."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request


def command(argv, environment):
    binary = environment.get("CONTEST_CODEX_BINARY", "codex")
    url = environment.get("CONTEST_GATEWAY_URL", "")
    token = environment.get("CONTEST_GATEWAY_TOKEN", "")
    if not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{4,5}/v1", url) or len(token) < 32:
        raise ValueError("Organizer metered gateway is not configured")
    if argv == ["login", "status"]:
        request = urllib.request.Request(url[:-3] + "/health", headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 200:
                raise ValueError("Metered gateway is unavailable")
        return None
    if not argv or argv[0] != "exec" or argv[-1] != "-" or "--ignore-user-config" not in argv:
        raise ValueError("Only frozen Codex campaign execution is allowed")
    models = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--model"]
    if models != ["deepseek-flash"]:
        raise ValueError("Campaign model differs from the frozen DeepSeek model")
    settings = [
        'model_provider="contest_gateway"',
        'model_providers.contest_gateway.name="DeepSeek Flash Contest"',
        'model_providers.contest_gateway.base_url=' + json.dumps(url),
        'model_providers.contest_gateway.env_key="CONTEST_GATEWAY_TOKEN"',
        'model_providers.contest_gateway.wire_api="responses"',
        'web_search="disabled"',
    ]
    return [binary, *argv[:-1], *(part for setting in settings for part in ("--config", setting)), "-"]


def main():
    try:
        result = command(sys.argv[1:], os.environ)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if result is None:
        return 0
    environment = dict(os.environ)
    environment.pop("DEEPSEEK_API_KEY", None)
    environment.pop("CONTEST_CODEX_BINARY", None)
    os.execvpe(result[0], result, environment)


if __name__ == "__main__":
    raise SystemExit(main())
