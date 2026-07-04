"""Claim 4 / Figure 4 / Table 5: Ablation studies.

Three sweeps, each holding the other two factors at their default value:
  1. Confounding strength gamma in {0.5, 1.0, 2.0, 4.0} -- does the
     interventional gap grow with confounding strength?
  2. Interventional training fraction in {0.0, 0.05, 0.1, 0.2} -- does a
     small amount of interventional data during training close the gap?
  3. Latent dimension d_z in {2, 4, 8, 16} -- is there a capacity sweet spot?

Uses the SyntheticSCM (Eq. 11, chain topology) and PCNetworkV2, 5 seeds per
configuration (Table 5's per-cell seed count).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import torch

from src.scm import SyntheticSCM, SyntheticSCMConfig
from src.pc_networks import PCNetworkV2, train_pc_v2
from src.metrics import mae, interventional_gap, bootstrap_ci

N_SEEDS = 5
DEFAULTS = dict(gamma=1.0, interventional_fraction=0.0, d_z=4)


def run_one(gamma: float, interventional_fraction: float, d_z: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    cfg = SyntheticSCMConfig(n=1000, d_c=5, gamma=gamma, delta=2.0, noise_sd=0.3,
                              topology="chain", interventional_fraction=interventional_fraction)
    scm = SyntheticSCM(cfg, rng)
    data = scm.sample()
    C, T, Y, mu0, mu1 = data["C"], data["T"], data["Y"], data["mu0"], data["mu1"]

    n = len(Y)
    idx = rng.permutation(n)
    idx_train, idx_test = idx[:int(0.8 * n)], idx[int(0.8 * n):]

    X_full = np.concatenate([T.reshape(-1, 1), C], axis=1)
    x_train = torch.tensor(X_full[idx_train], dtype=torch.float32)
    y_train = torch.tensor(Y[idx_train], dtype=torch.float32)
    x_test = torch.tensor(X_full[idx_test], dtype=torch.float32)

    torch.manual_seed(seed)
    model = PCNetworkV2(d_x=X_full.shape[1], d_z=d_z, treatment_col=0)
    train_pc_v2(model, x_train, y_train, epochs=150, batch_size=32, t_inf=80,
                lr_infer=0.05, lr_weight=0.001)

    y_hat_obs = model.predict_observational(x_test, t_inf=80, lr=0.05).numpy()
    y_hat_do1 = model.predict_interventional(x_test, t_value=1.0, t_inf=80, lr=0.05).numpy()

    obs_mae = mae(Y[idx_test], y_hat_obs)
    int_mae = mae(mu1[idx_test], y_hat_do1)
    return interventional_gap(int_mae, obs_mae)


def sweep(param_name: str, values: list):
    rows = []
    for v in values:
        kwargs = dict(DEFAULTS)
        kwargs[param_name] = v
        gaps = np.array([run_one(seed=s, **kwargs) for s in range(N_SEEDS)])
        mean, lo, hi = bootstrap_ci(gaps, n_resamples=500, rng=np.random.default_rng(0))
        rows.append({param_name: v, "gap_mean": mean, "gap_ci": [lo, hi], "gaps": gaps.tolist()})
        print(f"  {param_name}={v}: gap={mean:.4f} CI=[{lo:.3f},{hi:.3f}]")
    return rows


def run():
    print("Ablation 1: confounding strength (gamma)")
    ablation_gamma = sweep("gamma", [0.5, 1.0, 2.0, 4.0])

    print("\nAblation 2: interventional training fraction")
    ablation_int_frac = sweep("interventional_fraction", [0.0, 0.05, 0.1, 0.2])

    print("\nAblation 3: latent dimension (d_z)")
    ablation_dz = sweep("d_z", [2, 4, 8, 16])

    return {
        "confounding_strength": ablation_gamma,
        "interventional_fraction": ablation_int_frac,
        "latent_dim": ablation_dz,
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    results = run()
    with open(out_dir / "04_ablations.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / '04_ablations.json'}")
