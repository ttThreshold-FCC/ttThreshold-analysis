#!/bin/bash
# Move the contents of the EOS publish target
# (/eos/user/<initial>/<user>/www/mW/) into a timestamped zz_old/ snapshot,
# preserving the previous publish state. The next plot run will
# recreate the published subtrees from scratch. Skips zz_old itself
# so multiple archives can stack.
set -e

# Default to the invoking user's EOS web area; override with PUBLISH_DIR=... .
PUBLISH_DIR=${PUBLISH_DIR:-/eos/user/${USER:0:1}/${USER}/www/mW}

if [ ! -d "$PUBLISH_DIR" ]; then
    echo "$PUBLISH_DIR does not exist — nothing to archive"
    exit 0
fi

ts=$(date +%Y-%m-%d_%H%M%S)
target="$PUBLISH_DIR/zz_old/$ts"
if [ -e "$target" ]; then
    echo "ERROR: $target already exists; refusing to clobber" >&2
    exit 1
fi
mkdir -p "$target"

# Skip zz_old (target), report.pdf (kept live so the report URL
# stays valid across pipeline reruns), and index.php (EOS web index).
shopt -s nullglob dotglob
moved=0
for entry in "$PUBLISH_DIR"/*; do
    name=$(basename "$entry")
    case "$name" in
        zz_old|report.pdf|index.php) continue ;;
    esac
    mv "$entry" "$target/"
    moved=$((moved + 1))
done

echo "archived $moved entries in $PUBLISH_DIR → $target/"
echo "next publish run will recreate the published artefacts"
echo "(report.pdf and index.php preserved live)"
