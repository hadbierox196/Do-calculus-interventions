"""Claim 2 / Figure 2 / Table 4 (MLP rows): MLP baseline interventional gap
on IHDP.

Trains an MLP on IHDP across 20 seeds, computes observational vs.
interventional MAE using the true potential outcomes (mu0, mu1), and the
resulting interventional gap with a bootstrap CI and permutation test.
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
from src.mlp_baseline import MLPBaseline, train_mlp
from src.metrics import mae, interventional_gap, bootstrap_ci, permutation_test_within_model

N_SEEDS = 20


def run_one_seed(seed: int) -> dict:
    data = load_ihdp(seed=seed)
    X, T, Y, mu0, mu1 = data["X"], data["T"], data["Y"], data["mu0"], data["mu1"]

    # Feature matrix: covariates + treatment column (treatment_col=0 convention
    # -> prepend T as the first feature so MLPBaseline can locate/override it)
    X_full = np.concatenate([T.reshape(-1, 1), X], axis=1)

    idx_train, idx_test = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=42)

    x_train = torch.tensor(X_full[idx_train], dtype=torch.float32)
    y_train = torch.tensor(Y[idx_train], dtype=torch.float32)
    x_test = torch.tensor(X_full[idx_test], dtype=torch.float32)
    y_test_obs = Y[idx_test]
    mu1_test = mu1[idx_test]

    torch.manual_seed(seed)
    model = MLPBaseline(d_x=X_full.shape[1], treatment_col=0)
    train_mlp(model, x_train, y_train, epochs=300, batch_size=64, lr=0.001)

    y_hat_obs = model.predict_observational(x_test).numpy()
    y_hat_do1 = model.predict_interventional(x_test, t_value=1.0).numpy()

    obs_mae = mae(y_test_obs, y_hat_obs)
    int_mae = mae(mu1_test, y_hat_do1)  # Eq. 14: compare to true counterfactual mu1
    gap = interventional_gap(int_mae, obs_mae)
    return {"seed": seed, "obs_mae": obs_mae, "int_mae": int_mae, "gap": gap}


def run():
    per_seed = [run_one_seed(s) for s in range(N_SEEDS)]
    obs_maes = np.array([r["obs_mae"] for r in per_seed])
    int_maes = np.array([r["int_mae"] for r in per_seed])
    gaps = np.array([r["gap"] for r in per_seed])

    rng = np.random.default_rng(0)
    gap_mean, gap_lo, gap_hi = bootstrap_ci(gaps, n_resamples=1000, rng=rng)
    _, p_value = permutation_test_within_model(obs_maes, int_maes, n_permutations=10000, rng=rng)

    print(f"MLP Baseline (IHDP, {N_SEEDS} seeds)")
    print(f"  Obs MAE: {obs_maes.mean():.4f} +/- {obs_maes.std():.4f}")
    print(f"  Int MAE: {int_maes.mean():.4f} +/- {int_maes.std():.4f}")
    print(f"  Gap: {gap_mean:.4f}  95% CI [{gap_lo:.3f}, {gap_hi:.3f}]  p={p_value:.4g}")

    return {
        "per_seed": per_seed,
        "obs_mae_mean": float(obs_maes.mean()), "obs_mae_sd": float(obs_maes.std()),
        "int_mae_mean": float(int_maes.mean()), "int_mae_sd": float(int_maes.std()),
        "gap_mean": gap_mean, "gap_ci": [gap_lo, gap_hi], "p_value": p_value,
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    results = run()
    with open(out_dir / "02_ihdp_mlp_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / '02_ihdp_mlp_baseline.json'}")
