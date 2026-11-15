#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic3d.open3dbench import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
