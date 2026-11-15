"""Check provider connectivity and credentials before any model reservation.

The documented GET /models endpoint does not generate model output:
https://api-docs.deepseek.com/api/list-models/
"""
from http.client import HTTPException
import json
from pathlib import Path
import urllib.error
import urllib.request

from grt_bench.common import write
from .model_pricing import MODEL_ID


def check_provider(key, root):
    request = urllib.request.Request("https://api.deepseek.com/models",
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"})
    evidence = {"endpoint": "https://api.deepseek.com/models", "model_requests": 0}
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("Provider model list exceeds the startup limit")
        value = json.loads(data)
        if not isinstance(value, dict) or not isinstance(value.get("data"), list):
            raise ValueError("Provider returned an invalid model list")
        available = [row.get("id") for row in value.get("data", []) if isinstance(row, dict)]
        if MODEL_ID not in available:
            raise ValueError("The configured DeepSeek model is unavailable")
        evidence.update(passed=True, model=MODEL_ID)
    except (OSError, ValueError, HTTPException) as error:
        evidence.update(passed=False, error_type=type(error).__name__)
        if isinstance(error, urllib.error.HTTPError):
            evidence["http_status"] = error.code
        write(Path(root) / "control/provider-preflight.json", evidence, replace=True)
        message = ("DeepSeek credential check failed" if evidence.get("http_status") in (401, 403)
                   else "DeepSeek startup check failed; check network access or HTTPS_PROXY")
        raise RuntimeError(message + "; no model request was dispatched") from None
    write(Path(root) / "control/provider-preflight.json", evidence, replace=True)
    return evidence
