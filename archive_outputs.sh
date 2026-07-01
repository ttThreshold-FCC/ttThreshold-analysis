#!/bin/bash
# Move outputs/ to zz_old/outputs_<TIMESTAMP>/ to start the pipeline clean
# while keeping the previous run as a snapshot.
set -e
cd "$(dirname "$0")"

if [ ! -d outputs ]; then
    echo "outputs/ does not exist — nothing to archive"
    exit 0
fi

mkdir -p zz_old
ts=$(date +%Y-%m-%d_%H%M%S)
target="zz_old/outputs_${ts}"
if [ -e "$target" ]; then
    echo "ERROR: $target already exists; refusing to clobber" >&2
    exit 1
fi

mv outputs "$target"
echo "archived outputs/ → $target/"
echo "the pipeline will recreate outputs/ on next run"
