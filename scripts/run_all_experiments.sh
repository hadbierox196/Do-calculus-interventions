#!/usr/bin/env bash
# Runs every experiment in order and then builds the summary report.
# Usage: bash scripts/run_all_experiments.sh [--mimic-path /path/to/mimic-iv-demo]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIMIC_ARGS=()
if [[ "${1:-}" == "--mimic-path" && -n "${2:-}" ]]; then
  MIMIC_ARGS=(--mimic-path "$2")
else
  MIMIC_ARGS=(--synthetic)
fi

mkdir -p results

echo "=== [1/7] Toy SCM validation (Claim 1) ==="
python experiments/01_toy_scm_validation.py

echo "=== [2/7] MLP baseline on IHDP (Claim 2) ==="
python experiments/02_ihdp_mlp_baseline.py

echo "=== [3/7] PCNetworkV2 vs MLP on IHDP (Claim 3) ==="
python experiments/03_ihdp_pc_vs_mlp.py

echo "=== [4/7] Ablations (Claim 4) ==="
python experiments/04_ablations.py

echo "=== [5/7] Topology robustness (Claim 5) ==="
python experiments/05_topology_robustness.py

echo "=== [6/7] MIMIC-IV benchmark (Claim 6) ==="
python experiments/06_mimic_iv.py "${MIMIC_ARGS[@]}"

echo "=== [7/7] Building summary report ==="
python experiments/generate_report.py

echo ""
echo "Done. See results/REPORT.md for the full summary and results/*.json for raw output."
