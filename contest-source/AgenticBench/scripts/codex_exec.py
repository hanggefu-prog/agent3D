#!/usr/bin/env python3
"""Source-tree entry point for Agentic3D's Codex adapter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic3d.codex_exec import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
