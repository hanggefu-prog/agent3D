"""The native adapter is a flat script bundle, also mounted at /adapter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'engineering_bench/physical'))
