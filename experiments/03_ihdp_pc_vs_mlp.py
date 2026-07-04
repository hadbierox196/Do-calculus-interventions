"""Claim 3 / Figure 3 / Table 4 (PC rows): PCNetworkV2 closes the
interventional gap relative to the MLP baseline on IHDP.

Uses the same splits/seeds convention as experiment 02, but with
PCNetworkV2 (dz=4, 200 epochs -- Table 1 "best stable hyperparameters").
Runs 10 primary seeds + 3 additional verification seeds (13 total, matching
the paper's stability check in Sec 3.3 / Table 4 footnote).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data_ihdp import load_ihdp
from src.pc_networks import PCNetworkV2, train_pc_v2
from src.metrics import mae, interventional_gap, bootstrap_ci, permutation_test_gap_difference

N_SEEDS_PRIMARY = 10
N_SEEDS_VERIFY = 3
D_Z = 4
EPOCHS = 200


def run_one_seed(seed: int) -> dict:
    data = load_ihdp(seed=seed)
    X, T, Y, mu1 = data["X"], data["T"], data["Y"], data["mu1"]
    X_full = np.concatenate([T.reshape(-1, 1), X], axis=1)

    idx_train, idx_test = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=42)
    x_train = torch.tensor(X_full[idx_train], dtype=torch.float32)
    y_train = torch.tensor(Y[idx_train], dtype=torch.float32)
    x_test = torch.tensor(X_full[idx_test], dtype=torch.float32)
    y_test_obs = Y[idx_test]
    mu1_test = mu1[idx_test]

    torch.manual_seed(seed)
    model = PCNetworkV2(d_x=X_full.shape[1], d_z=D_Z, treatment_col=0)
    train_pc_v2(model, x_train, y_train, epochs=EPOCHS, batch_size=32,
                t_inf=100, lr_infer=0.05, lr_weight=0.001)

    y_hat_obs = model.predict_observational(x_test, t_inf=100, lr=0.05).numpy()
    y_hat_do1 = model.predict_interventional(x_test, t_value=1.0, t_inf=100, lr=0.05).numpy()

    obs_mae = mae(y_test_obs, y_hat_obs)
    int_mae = mae(mu1_test, y_hat_do1)
    gap = interventional_gap(int_mae, obs_mae)
    return {"seed": seed, "obs_mae": obs_mae, "int_mae": int_mae, "gap": gap}


def run(mlp_results_path: Path | None = None):
    all_seeds = list(range(N_SEEDS_PRIMARY)) + list(range(100, 100 + N_SEEDS_VERIFY))
    per_seed = [run_one_seed(s) for s in all_seeds]
    primary = per_seed[:N_SEEDS_PRIMARY]

    gaps_primary = np.array([r["gap"] for r in primary])
    obs_maes = np.array([r["obs_mae"] for r in primary])
    int_maes = np.array([r["int_mae"] for r in primary])
    gaps_all = np.array([r["gap"] for r in per_seed])

    rng = np.random.default_rng(0)
    gap_mean, gap_lo, gap_hi = bootstrap_ci(gaps_primary, n_resamples=1000, rng=rng)

    print(f"PCNetworkV2 (IHDP, dz={D_Z}, {N_SEEDS_PRIMARY} primary seeds "
          f"+ {N_SEEDS_VERIFY} verification seeds)")
    print(f"  Obs MAE: {obs_maes.mean():.4f} +/- {obs_maes.std():.4f}")
    print(f"  Int MAE: {int_maes.mean():.4f} +/- {int_maes.std():.4f}")
    print(f"  Gap: {gap_mean:.4f}  95% CI [{gap_lo:.3f}, {gap_hi:.3f}]")
    print(f"  All {len(gaps_all)} seed gaps: {np.round(gaps_all, 3).tolist()}")

    result = {
        "per_seed": per_seed,
        "obs_mae_mean": float(obs_maes.mean()), "obs_mae_sd": float(obs_maes.std()),
        "int_mae_mean": float(int_maes.mean()), "int_mae_sd": float(int_maes.std()),
        "gap_mean": gap_mean, "gap_ci": [gap_lo, gap_hi],
    }

    # If MLP results (experiment 02) are available, run the between-model
    # permutation test comparing PC gaps vs MLP gaps (Sec 2.6, Sec 3.3).
    if mlp_results_path is not None and mlp_results_path.exists():
        with open(mlp_results_path) as f:
            mlp_results = json.load(f)
        mlp_gaps = np.array([r["gap"] for r in mlp_results["per_seed"]])
        obs_stat, p_value = permutation_test_gap_difference(mlp_gaps, gaps_primary,
                                                              n_permutations=10000, rng=rng)
        reduction = mlp_gaps.mean() - gaps_primary.mean()
        n_favor = int(np.sum(gaps_all < mlp_gaps.mean()))
        print(f"\n  PC - MLP gap reduction: {reduction:.4f}  p={p_value:.4g} "
              f"(permutation test on gap difference)")
        print(f"  {n_favor}/{len(gaps_all)} seeds show PC gap < MLP mean gap")
        result.update({
            "mlp_gap_mean": float(mlp_gaps.mean()),
            "gap_reduction": float(reduction),
            "p_value_vs_mlp": float(p_value),
            "seeds_favoring_pc": f"{n_favor}/{len(gaps_all)}",
        })
    else:
        print("\n  (Run experiment 02 first, then pass its results path to compare against MLP.)")

    return result


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    mlp_path = out_dir / "02_ihdp_mlp_baseline.json"
    results = run(mlp_results_path=mlp_path)
    with open(out_dir / "03_ihdp_pc_vs_mlp.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / '03_ihdp_pc_vs_mlp.json'}")
