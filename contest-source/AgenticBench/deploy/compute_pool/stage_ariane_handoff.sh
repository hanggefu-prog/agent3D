#!/usr/bin/env bash
# Stage immutable data only. This does not stop, start, or redirect evaluations.
set -euo pipefail
task_stage=/nas1/gcr/agentic3d-dev/compute-pool-20260908/ariane-handoff-20260909
case "${1:-}" in
  export)
    task_source=/data/agentic3d/agentic3d-compute-20260908/sta-cases/ariane133
    [[ ! -e "$task_stage" ]] || { echo 'staging destination already exists'; exit 3; }
    mkdir "$task_stage"
    rsync -aH "$task_source/benchmark-input" "$task_source/reference" "$task_stage/"
    cd "$task_stage"
    find benchmark-input reference -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > files.sha256
    sha256sum -c files.sha256 > export-verify.log
    date -Is > export.ready
    ;;
  import)
    task_destination=/data/gcr/agentic3d-compute-20260908/sta-cases/ariane133
    [[ -f "$task_stage/export.ready" && -f "$task_stage/files.sha256" ]]
    [[ ! -e "$task_destination" ]] || { echo 'case destination already exists'; exit 3; }
    task_free_kib=$(df -Pk /data/gcr/agentic3d-compute-20260908 | awk 'NR==2 {print $4}')
    (( task_free_kib >= 52428800 )) || { echo 'less than50GiB free'; exit 4; }
    mkdir "$task_destination"
    rsync -aH "$task_stage/benchmark-input" "$task_stage/reference" "$task_destination/"
    cp "$task_stage/files.sha256" "$task_destination/handoff-files.sha256"
    cd "$task_destination"
    sha256sum -c handoff-files.sha256 > handoff-verify.log
    date -Is > handoff-inputs.ready
    ;;
  *) echo 'expected export or import'; exit 2 ;;
esac
echo "HANDOFF_DATA_STAGED $1 $(date -Is)"
