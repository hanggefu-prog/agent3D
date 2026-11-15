"""Small I/O boundary shared by the existing research roles, not a scheduler."""
from __future__ import annotations

import hashlib
import json
import secrets

from .knowledge import write_json


class ResearchBlocked(RuntimeError):
    """A role cannot research; this is neither a task nor a bad candidate."""


RESPONSE_INSTRUCTION = """
# Research response transport
Return {"blocked": null, "response": <your normal role response>} when able to work.
If source/evidence/tool access is unavailable, return {"blocked": "specific reason",
"response": null}. Do not invent a pause task, dummy candidate, or knowledge update.
The controller stops this invocation on blocked; it does not ask Local to fix the
environment. Missing optional evidence alone is not a blocker if the available
evidence suffices. Never report a candidate syntax/physical failure as infrastructure.
"""


def response_schema(role_schema):
    return {"type": "object", "additionalProperties": False,
            "required": ["blocked", "response"], "properties": {
                "blocked": {"type": ["string", "null"]},
                "response": {"anyOf": [role_schema, {"type": "null"}]}}}


def unwrap_response(value):
    if not isinstance(value, dict) or set(value) != {"blocked", "response"}:
        raise ValueError("Return the research envelope with blocked and response fields")
    reason, response = value["blocked"], value["response"]
    if reason is not None:
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
            raise ValueError("blocked must be null or a nonempty reason of at most 4000 characters")
        if response is not None:
            raise ValueError("A blocked response must have response=null; no task/candidate/knowledge")
        raise ResearchBlocked(reason)
    if not isinstance(response, dict):
        raise ValueError("An unblocked response must contain the normal role object")
    return response


def check_research_access(args, root, workspace, driver, source_file, evidence_file):
    """One real read-only probe per invocation before research; no EDA or repair.

    The nonce and expected hashes are never in the prompt. This tests actual file
    access in the same model/workspace/sandbox as every research role, not login.
    Each explicit resume gets a fresh probe and retains all previous traces.
    """
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name) for p in (root / "calls").glob("*") if p.name.isdigit()]
    run_dir = root / "calls" / f"{max(numbers, default=0) + 1:03d}"
    proof = workspace / "research-access.json"
    nonce = secrets.token_hex(16)
    write_json(proof, {"nonce": nonce})
    files = {"source_sha256": source_file, "evidence_sha256": evidence_file,
             "knowledge_sha256": workspace / "knowledge/seed/tools.json"}
    expected = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}
    expected["nonce"] = nonce
    schema = {"type": "object", "additionalProperties": False,
              "properties": {key: {"type": "string"} for key in expected}, "required": list(expected)}
    schema_path = root / "output.schema.json"
    write_json(schema_path, response_schema(schema))
    prompt = ("Read-only research-tool preflight, not an optimization task. Use your file/shell tools "
              "to read the nonce from research-access.json and compute SHA256 over the exact bytes "
              "of these workspace-local files. Do not modify files, run EDA, inspect other runs, "
              "or infer a hash from memory. If tools fail report blocked, never guess.\n" +
              json.dumps({key: path.relative_to(workspace).as_posix() for key, path in files.items()}) +
              RESPONSE_INSTRUCTION)
    try:
        result = driver.run_codex(prompt, driver.CodexExecConfig(
            workspace=workspace, run_dir=run_dir, codex_binary=args.codex_bin, model=args.model,
            sandbox="read-only", config_overrides=("model_reasoning_effort=xhigh",),
            output_schema=schema_path, ephemeral=True, skip_git_repo_check=True, timeout_seconds=180))
        if not result.success:
            raise ResearchBlocked("Research access probe failed; inspect " + str(run_dir))
        measured = unwrap_response(json.loads(result.final_message))
        if measured != expected:
            raise ResearchBlocked("Research access probe did not read the exact source/evidence/knowledge files")
    except (ValueError, RuntimeError, OSError) as error:
        write_json(run_dir / "access_check.json", {"passed": False, "reason": str(error)})
        raise ResearchBlocked(str(error)) from error
    write_json(run_dir / "access_check.json", {"passed": True, "files": list(files)})
