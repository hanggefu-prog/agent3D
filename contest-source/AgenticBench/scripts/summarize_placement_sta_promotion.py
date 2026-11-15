#!/usr/bin/env python3
"""Compatibility alias for the Open3DBench STA evaluation Skill."""

from pathlib import Path
import runpy


target = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "open3dbench-sta-evaluation"
    / "scripts"
    / "summarize_placement.py"
)
runpy.run_path(str(target), run_name="__main__")
