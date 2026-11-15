"""Load only the participant Agent in the untrusted container process."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

from .agent_api import AgentContext


def run(agent_file=Path("/workspace/agent/agent.py"),
        result_file=Path("/workspace/scratch/agent-result.json")):
    agent_file = Path(agent_file)
    if agent_file.is_symlink() or not agent_file.is_file():
        raise ValueError("Participant agent.py must be a regular file")
    spec = importlib.util.spec_from_file_location("participant_agent", agent_file)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load participant Agent")
    sys.path.insert(0, str(agent_file.parent))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    agent = module.Agent()
    result = agent.run(AgentContext())
    if isinstance(result, str):
        candidate = result
    elif isinstance(result, dict):
        candidate = result.get("candidate_id")
    else:
        candidate = getattr(result, "candidate_id", None)
    import re
    if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate):
        raise ValueError("Agent.run must return a verified candidate_id")
    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_file.with_name("." + result_file.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with temporary.open("x") as stream:
            json.dump({"candidate_id": candidate}, stream)
            stream.write("\n")
        os.replace(temporary, result_file)
    finally:
        temporary.unlink(missing_ok=True)
    return candidate


if __name__ == "__main__":
    run()
