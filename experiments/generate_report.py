"""Aggregates results/*.json from all six experiments into results/REPORT.md,
formatted to mirror the paper's Tables 2-7. Run this last, after all other
experiments/*.py scripts have produced their JSON output.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(x, nd=4):
    if x is None:
        return "--"
    return f"{x:.{nd}f}"


def build_report() -> str:
    r1 = _load("01_toy_scm_validation.json")
    r2 = _load("02_ihdp_mlp_baseline.json")
    r3 = _load("03_ihdp_pc_vs_mlp.json")
    r4 = _load("04_ablations.json")
    r5 = _load("05_topology_robustness.json")
    r6 = _load("06_mimic_iv.json")

    lines = ["# Results Report\n",
             "Auto-generated from `results/*.json`. Run `scripts/run_all_experiments.sh` "
             "to regenerate every section below.\n"]

    # --- Claim 1: toy SCM ---
    lines.append("## Claim 1: Proposition 1 validation (toy SCM)\n")
    if r1:
        lines.append(f"- Weight recovery: `W_DS_hat={_fmt(r1['w_ds_hat'], 3)}` (true 0.8), "
                      f"`W_ST_hat={_fmt(r1['w_st_hat'], 3)}` (true 0.6)")
        lines.append(f"- `E[D | do(S=1)]` = {_fmt(r1['e_D_do_S1'], 3)}  (expected 0.0)")
        lines.append(f"- `E[D | S=1]` (observational) = {_fmt(r1['e_D_cond_S1'], 3)}  (paper: ~0.640)")
        lines.append(f"- KS test vs N(0,1): statistic={_fmt(r1['ks_stat'], 3)}, p={_fmt(r1['ks_p'], 3)}")
        lines.append(f"- Baseline parity: MLP-like MAE={_fmt(r1['mlp_mae'])}, PC MAE={_fmt(r1['pc_mae'])}\n")
    else:
        lines.append("_Not yet run — `python experiments/01_toy_scm_validation.py`_\n")

    # --- Claims 2/3: IHDP MLP vs PC (Table 4) ---
    lines.append("## Claims 2-3: Interventional gap on IHDP (Table 4)\n")
    lines.append("| Model | Obs MAE | Int MAE | Gap | 95% CI |")
    lines.append("|---|---|---|---|---|")
    if r2:
        lines.append(f"| MLP | {_fmt(r2['obs_mae_mean'])} | {_fmt(r2['int_mae_mean'])} | "
                      f"{_fmt(r2['gap_mean'])} | [{_fmt(r2['gap_ci'][0],3)}, {_fmt(r2['gap_ci'][1],3)}] |")
    else:
        lines.append("| MLP | -- | -- | -- | -- |")
    if r3:
        lines.append(f"| PCNetworkV2 | {_fmt(r3['obs_mae_mean'])} | {_fmt(r3['int_mae_mean'])} | "
                      f"{_fmt(r3['gap_mean'])} | [{_fmt(r3['gap_ci'][0],3)}, {_fmt(r3['gap_ci'][1],3)}] |")
    else:
        lines.append("| PCNetworkV2 | -- | -- | -- | -- |")
    lines.append("")
    if r3 and "gap_reduction" in r3:
        lines.append(f"Gap reduction (MLP - PC): **{_fmt(r3['gap_reduction'])}**, "
                      f"p={_fmt(r3['p_value_vs_mlp'], 4)}, "
                      f"seeds favoring PC: {r3['seeds_favoring_pc']}\n")
    else:
        lines.append("_Run experiments 02 and 03 (in that order) to populate the gap-reduction row._\n")

    # --- Claim 4: ablations (Table 5) ---
    lines.append("## Claim 4: Ablations (Table 5)\n")
    if r4:
        for section_name, key, label in [
            ("Confounding strength (gamma)", "confounding_strength", "gamma"),
            ("Interventional training fraction", "interventional_fraction", "interventional_fraction"),
            ("Latent dimension (d_z)", "latent_dim", "d_z"),
        ]:
            lines.append(f"**{section_name}**\n")
            lines.append(f"| {label} | Gap | 95% CI |")
            lines.append("|---|---|---|")
            for row in r4[key]:
                lines.append(f"| {row[label]} | {_fmt(row['gap_mean'])} | "
                              f"[{_fmt(row['gap_ci'][0],3)}, {_fmt(row['gap_ci'][1],3)}] |")
            lines.append("")
    else:
        lines.append("_Not yet run — `python experiments/04_ablations.py`_\n")

    # --- Claim 5: topology robustness (Table 6) ---
    lines.append("## Claim 5: Topology robustness (Table 6)\n")
    if r5:
        lines.append("| Topology | True ATE | Gap | ATE error |")
        lines.append("|---|---|---|---|")
        for topo, s in r5["summary"].items():
            lines.append(f"| {topo} | {_fmt(s['true_ate'], 2)} | {_fmt(s['gap_mean'])} "
                          f"[{_fmt(s['gap_ci'][0],3)}, {_fmt(s['gap_ci'][1],3)}] | "
                          f"{_fmt(s['ate_error_mean'])} |")
        lines.append("")
    else:
        lines.append("_Not yet run — `python experiments/05_topology_robustness.py`_\n")

    # --- Claim 6: MIMIC-IV (Table 7) ---
    lines.append("## Claim 6: MIMIC-IV benchmark (Table 7)\n")
    if r6:
        lines.append("| Model | Gap | ATE error | E[do(T=1)-do(T=0)] |")
        lines.append("|---|---|---|---|")
        for model_name, s in r6["summary"].items():
            flag = ""
            if model_name == "PCNetworkV2" and abs(s["do1_minus_do0_std"]) < 1e-3:
                flag = " *(degenerate)*"
            lines.append(f"| {model_name}{flag} | {_fmt(s['gap_mean'])} "
                          f"[{_fmt(s['gap_ci'][0],3)}, {_fmt(s['gap_ci'][1],3)}] | "
                          f"{_fmt(s['ate_error_mean'])} | "
                          f"{_fmt(s['mean_do1_minus_do0'])} +/- {_fmt(s['do1_minus_do0_std'])} |")
        lines.append("")
    else:
        lines.append("_Not yet run — `python experiments/06_mimic_iv.py --synthetic`_\n")

    return "\n".join(lines)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    report = build_report()
    out_path = RESULTS_DIR / "REPORT.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\n\nSaved to {out_path}")
