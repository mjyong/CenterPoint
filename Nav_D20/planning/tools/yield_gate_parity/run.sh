#!/bin/bash
# Build the C++ yield gate with the parity checker, make the golden data with the Python reference, compare.
#   bash Nav_D20/planning/tools/yield_gate_parity/run.sh [extra make_golden.py arguments, e.g. --scenes ...]
# Needs g++ (C++17) and Eigen 3; PYTHON (default python3) needs numpy (and, with --scenes, the repository's deps).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
planning="$(cd "${here}/../.." && pwd)"
repo="$(cd "${planning}/../.." && pwd)"
out="${YIELD_PARITY_DIR:-$(mktemp -d)}"
mkdir -p "${out}"
eigen="${EIGEN_DIR:-/usr/include/eigen3}"
g++ -std=c++17 -O2 -Wall -Wextra -I"${planning}" -I"${eigen}" \
  "${here}/parity.cc" "${planning}/src/algorithm/yield_gate/yield_gate.cc" -o "${out}/yield_gate_parity"
(cd "${repo}" && PYTHONPATH="${repo}" "${PYTHON:-python3}" "${here}/make_golden.py" --out "${out}/golden.txt" "$@")
"${out}/yield_gate_parity" "${out}/golden.txt"
