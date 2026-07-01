#!/bin/bash
cd "$(dirname "$0")"
source setup.sh
# Ensure the kinfit BW-normalization log Z table is present + up to date.
# No-op when the table exists and is newer than tools/build_logz_table.cxx.
tools/ensure_logz_table.sh
mkdir -p logs
pids=()
for ecm in 157 160 163; do
    WW_ECM=$ecm fccanalysis run treemaker_lnuqq_step2.py --ncores 4 \
        > logs/step2${WW_TAG:+_$WW_TAG}_ecm${ecm}.log 2>&1 &
    pids+=($!)
    echo "  launched ecm${ecm} (PID=${pids[-1]})"
done
wait "${pids[@]}"
echo "All step2 ECMs done"
