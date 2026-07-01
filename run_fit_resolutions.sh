#!/bin/bash
cd "$(dirname "$0")"
source setup.sh
exec python3 fit_resolutions.py "$@"
