#!/usr/bin/env bash
# Build the m_W forward-folding slides. TeXLive is taken from CVMFS (no local install needed).
set -e
export PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2024/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error mw_forward_fold.tex
pdflatex -interaction=nonstopmode -halt-on-error mw_forward_fold.tex   # 2nd pass: section nav / refs
echo "==> mw_forward_fold.pdf ($(grep -oE '\([0-9]+ pages' mw_forward_fold.log | tail -1 | tr -d '('))"
