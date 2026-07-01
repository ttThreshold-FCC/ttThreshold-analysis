#!/bin/bash
cd "$(dirname "$0")"
source setup.sh
exec fccanalysis run treemaker_lnuqq_step1.py
