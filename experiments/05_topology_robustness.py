"""Claim 5 / Figure 5 / Table 6: Topology robustness.

Runs PCNetworkV2 across the four SCM topologies (chain, mediator, collider,
fork) at default confounding, and reports the interventional gap plus the
ATE estimation error against the *true* ATE for each topology (0.0 for
fork, ~2.0 for the others -- Sec 3.5, Limitation L7 null-effect check).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import torch

from src.scm import SyntheticSCM, SyntheticSCMConfig, TOPOLOGIES
from src.pc_networks import PCNetworkV2, train_pc_v2
from src.metrics import mae, interventional_gap, ate_estimation_error, bootstrap_ci

N_SEEDS = 5


def run_one(topology: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cfg = SyntheticSCMConfig(n=1000, d_c=5, gamma=1.0, delta=2.0, noise_sd=0.3, topology=topology)
    scm = SyntheticSCM(cfg, rng)
    data = scm.sample()
    C, T, Y, mu0, mu1, true_ate = data["C"], data["T"], data["Y"], data["mu0"], data["mu1"], data["true_ate"]

    n = len(Y)
    idx = rng.permutation(n)
    idx_train, idx_test = idx[:int(0.8 * n)], idx[int(0.8 * n):]

    X_full = np.concatenate([T.reshape(-1, 1), C], axis=1)
    x_train = torch.tensor(X_full[idx_train], dtype=torch.float32)
    y_train = torch.tensor(Y[idx_train], dtype=torch.float32)
    x_test = torch.tensor(X_full[idx_test], dtype=torch.float32)

    torch.manual_seed(seed)
    model = PCNetworkV2(d_x=X_full.shape[1], d_z=4, treatment_col=0)
    train_pc_v2(model, x_train, y_train, epochs=150, batch_size=32, t_inf=80,
                lr_infer=0.05, lr_weight=0.001)

    y_hat_obs = model.predict_observational(x_test, t_inf=80, lr=0.05).numpy()
    y_hat_do0 = model.predict_interventional(x_test, t_value=0.0, t_inf=80, lr=0.05).numpy()
    y_hat_do1 = model.predict_interventional(x_test, t_value=1.0, t_inf=80, lr=0.05).numpy()

    obs_mae = mae(Y[idx_test], y_hat_obs)
    int_mae = mae(mu1[idx_test], y_hat_do1)
    gap = interventional_gap(int_mae, obs_mae)
    ate_err = ate_estimation_error(y_hat_do1, y_hat_do0, true_ate)

    return {"topology": topology, "seed": seed, "obs_mae": obs_mae, "int_mae": int_mae,
            "gap": gap, "true_ate": true_ate, "ate_error": ate_err}


def run():
    all_rows = []
    summary = {}
    for topo in TOPOLOGIES:
        rows = [run_one(topo, seed=s) for s in range(N_SEEDS)]
        all_rows.extend(rows)
        gaps = np.array([r["gap"] for r in rows])
        ate_errs = np.array([r["ate_error"] for r in rows])
        gap_mean, gap_lo, gap_hi = bootstrap_ci(gaps, n_resamples=500, rng=np.random.default_rng(0))
        ate_mean, ate_lo, ate_hi = bootstrap_ci(ate_errs, n_resamples=500, rng=np.random.default_rng(0))
        summary[topo] = {
            "true_ate": rows[0]["true_ate"],
            "gap_mean": gap_mean, "gap_ci": [gap_lo, gap_hi],
            "ate_error_mean": ate_mean, "ate_error_ci": [ate_lo, ate_hi],
        }
        print(f"{topo:>10s}: true_ATE={rows[0]['true_ate']:.2f}  "
              f"gap={gap_mean:.4f} CI=[{gap_lo:.3f},{gap_hi:.3f}]  "
              f"ATE_err={ate_mean:.4f} CI=[{ate_lo:.3f},{ate_hi:.3f}]")

    return {"per_run": all_rows, "summary": summary}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    results = run()
    with open(out_dir / "05_topology_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / '05_topology_robustness.json'}")
