#!/bin/bash
# Build both versions (TeX Live 2026 module):
#   root.pdf       extended / arXiv version, with appendices
#   conference.pdf 8-page ICRA cut, no appendices
set -e
export PATH=/opt/modules/texlive-2026/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"
for doc in root conference; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$doc.tex" >/dev/null 2>&1 \
    || { echo "FAILED: $doc"; latexmk -pdf -interaction=nonstopmode "$doc.tex" 2>&1 | tail -30; exit 1; }
  echo "OK -> $(pwd)/$doc.pdf  ($(grep -o '([0-9]* pages' "$doc.log" | tail -1 | tr -d '('))"
done
