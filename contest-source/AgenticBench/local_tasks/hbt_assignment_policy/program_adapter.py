"""Offline policy packaging only; native hook integration is not yet qualified."""
from pathlib import Path

from local_bench.hbt_policy import materialize_policy


def materialize(program_text: str, open3dbench_root: Path) -> dict:
    return materialize_policy(program_text)
