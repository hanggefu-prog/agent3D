#!/usr/bin/env bash
# Compatibility alias. New automation should call the Skill entry point.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "$root/skills/open3dbench-sta-evaluation/scripts/validate_ta3d_objective.sh" "$@"
