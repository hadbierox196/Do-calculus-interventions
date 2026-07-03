"""MIMIC-IV semi-synthetic benchmark (Sec 2.4.4, Eq. 12).

Real ICU covariate structure from the MIMIC-IV Clinical Database Demo v2.2
(n=140 ICU stays, 11 covariates). Treatment = high-intensity ICU unit
assignment (MICU, SICU, CCU, CVICU, Neuro SICU) vs. standard care.

    mu0 = X @ beta + 1.5
    mu1 = mu0 + 1.0 + 0.3 * X_age
    True ATE = 1.0 log-LOS unit

Real data requires a free PhysioNet credentialed download:
    https://physionet.org/content/mimic-iv-demo/
Set MIMIC_DEMO_PATH to the extracted directory (containing icustays.csv,
patients.csv, admissions.csv) and call `load_mimic_real(path)`. Without
access, `load_mimic_synthetic()` produces a structurally matched synthetic
fallback (11 covariates, n=140, same Eq. 12 DGP) so the pipeline runs
end-to-end.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

N_MIMIC = 140
D_COVARIATES = 11
TRUE_ATE = 1.0
NOISE_SD = 0.3

HIGH_INTENSITY_UNITS = {"MICU", "SICU", "CCU", "CVICU", "Neuro SICU"}

# Table 1: MIMIC-IV beta (11 covariates)
BETA = np.array([0.3, -0.1, 0.2, 0.4, -0.1, 0.1, 0.1, -0.05, 0.2, 0.3, 0.5])
AGE_COL = 0  # convention: column 0 of X is (standardised) age


def _counterfactual_outcomes(X: np.ndarray) -> dict[str, np.ndarray]:
    """Eq. 12."""
    mu0 = X @ BETA + 1.5
    mu1 = mu0 + TRUE_ATE + 0.3 * X[:, AGE_COL]
    return {"mu0": mu0, "mu1": mu1}


def load_mimic_synthetic(seed: int = 0) -> dict[str, np.ndarray]:
    """Structurally matched synthetic fallback: same n, same covariate count,
    same Eq. 12 DGP, used when PhysioNet access is unavailable."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(N_MIMIC, D_COVARIATES))
    cf = _counterfactual_outcomes(X)
    # Weak confounding by design (paper reports confounding gap = 0.066)
    logits = 0.3 * X[:, AGE_COL]
    p = 1.0 / (1.0 + np.exp(-logits))
    T = rng.binomial(1, p).astype(np.float64)
    noise = rng.normal(0, NOISE_SD, size=N_MIMIC)
    Y = T * cf["mu1"] + (1 - T) * cf["mu0"] + noise
    return {
        "X": X, "T": T, "Y": Y,
        "mu0": cf["mu0"], "mu1": cf["mu1"],
        "true_ate": TRUE_ATE,
    }


def load_mimic_real(demo_path: str | Path, seed: int = 0) -> dict[str, np.ndarray]:
    """Load real MIMIC-IV Demo covariates and apply the same Eq. 12 DGP for
    counterfactual outcomes (real covariate *structure*, simulated outcomes
    -- this is a semi-synthetic benchmark, matching the paper's design).

    Expects `icustays.csv`, `patients.csv`, `admissions.csv` under demo_path
    (as distributed by PhysioNet's mimic-iv-demo package).
    """
    demo_path = Path(demo_path)
    icustays = pd.read_csv(demo_path / "icustays.csv")
    patients = pd.read_csv(demo_path / "patients.csv")
    admissions = pd.read_csv(demo_path / "admissions.csv")

    df = icustays.merge(patients, on="subject_id", how="left")
    df = df.merge(admissions, on=["subject_id", "hadm_id"], how="left")

    df["treatment"] = df["first_careunit"].isin(HIGH_INTENSITY_UNITS).astype(float)

    # Build an 11-column covariate matrix from available demo fields;
    # missing/derived columns are filled with dataset means or simple flags
    # so the pipeline runs on whatever subset of fields the Demo provides.
    cols = []
    cols.append(pd.to_numeric(df.get("anchor_age", pd.Series(np.nan, index=df.index)), errors="coerce"))
    cols.append((df.get("gender", "M") == "F").astype(float))
    los = pd.to_numeric(df.get("los", pd.Series(np.nan, index=df.index)), errors="coerce")
    cols.append(los)
    for cat_col in ["admission_type", "admission_location", "insurance", "marital_status",
                     "race", "language", "first_careunit", "last_careunit"]:
        if cat_col in df.columns:
            codes = df[cat_col].astype("category").cat.codes.astype(float)
            cols.append(codes)
        if len(cols) >= D_COVARIATES:
            break
    X = pd.concat(cols[:D_COVARIATES], axis=1).to_numpy(dtype=np.float64)
    # standardise + impute
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    if X.shape[1] < D_COVARIATES:
        pad = np.zeros((X.shape[0], D_COVARIATES - X.shape[1]))
        X = np.concatenate([X, pad], axis=1)

    rng = np.random.default_rng(seed)
    cf = _counterfactual_outcomes(X)
    T = df["treatment"].to_numpy(dtype=np.float64)
    noise = rng.normal(0, NOISE_SD, size=len(df))
    Y = T * cf["mu1"] + (1 - T) * cf["mu0"] + noise
    return {
        "X": X, "T": T, "Y": Y,
        "mu0": cf["mu0"], "mu1": cf["mu1"],
        "true_ate": TRUE_ATE,
    }


def load_mimic(demo_path: str | Path | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    """Convenience wrapper: uses real MIMIC-IV Demo data if demo_path is
    provided and valid, otherwise falls back to the synthetic generator."""
    if demo_path is not None and Path(demo_path).exists():
        return load_mimic_real(demo_path, seed=seed)
    return load_mimic_synthetic(seed=seed)
