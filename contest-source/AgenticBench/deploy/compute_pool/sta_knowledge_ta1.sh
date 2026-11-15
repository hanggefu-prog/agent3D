#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/sta_knowledge_admission.sh" ta1 "$@"
