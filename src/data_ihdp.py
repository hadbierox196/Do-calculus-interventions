"""IHDP semi-synthetic benchmark (Sec 2.4.2, Eq. 10).

Reconstructs covariates and counterfactual outcomes from the published Hill
(2011) NPCI Simulation 1 data-generating process, as described in the paper
(Limitation L2: public IHDP mirrors are subject to URL rot, so the paper
itself reconstructs counterfactuals from the published DGP equations rather
than a pre-processed file).

    mu0 = exp((X_clip + 0.5) @ beta)
    mu1 = mu0 + 4
    X_clip = X standardised then clipped to [-2, 2]
    True ATE = 4.0 outcome units

This module generates n=747 rows and 25 covariates with realistic IHDP-like
structure (continuous + binary covariates, confounded treatment assignment),
then applies Eq. 10 using the first 10 covariates against the beta vector
from Table 1. If you have the real IHDP covariate matrix (e.g. from the NPCI
mirror or the `causaldata` package), pass it directly to
`counterfactual_outcomes()` instead of using the synthetic covariate
generator, and the Eq. 10 reconstruction will be identical either way.
"""
from __future__ import annotations
import numpy as np

N_IHDP = 747
D_COVARIATES = 25
TRUE_ATE = 4.0
NOISE_SD = 0.1

# Table 1: IHDP beta (applied to the first 10 covariates after clipping)
BETA = np.array([0.1, 0.2, 0.0, 0.1, 0.0, 0.3, 0.0, 0.2, 0.1, 0.0])


def generate_ihdp_covariates(n: int = N_IHDP, d: int = D_COVARIATES,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic stand-in for the IHDP covariate matrix: a mix of continuous
    (birth weight, gestational age, maternal age, etc.) and binary (indicator)
    covariates, matching IHDP's typical composition (6 continuous + 19 binary
    in the original data; here approximated as the first 6 columns continuous,
    remainder Bernoulli). Replace with the real covariate matrix if available."""
    rng = rng or np.random.default_rng(0)
    continuous = rng.normal(0, 1, size=(n, 6))
    binary = rng.binomial(1, 0.4, size=(n, d - 6)).astype(np.float64)
    return np.concatenate([continuous, binary], axis=1)


def _standardise_clip(x_first10: np.ndarray) -> np.ndarray:
    mean = x_first10.mean(axis=0, keepdims=True)
    sd = x_first10.std(axis=0, keepdims=True) + 1e-8
    z = (x_first10 - mean) / sd
    return np.clip(z, -2, 2)


def counterfactual_outcomes(X: np.ndarray, rng: np.random.Generator | None = None
                             ) -> dict[str, np.ndarray]:
    """Eq. 10: reconstruct mu0, mu1 from covariates X (n, >=10). Uses only the
    first 10 columns of X against BETA, per the paper's DGP."""
    rng = rng or np.random.default_rng(0)
    x10 = X[:, :10]
    x_clip = _standardise_clip(x10)
    mu0 = np.exp((x_clip + 0.5) @ BETA)
    mu1 = mu0 + TRUE_ATE
    return {"mu0": mu0, "mu1": mu1}


def assign_treatment(X: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Confounded propensity: treatment more likely for higher first-covariate
    values (standard IHDP-style confounding), giving realistic observational
    confounding for the interventional-gap experiments."""
    rng = rng or np.random.default_rng(0)
    logits = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-logits))
    return rng.binomial(1, p).astype(np.float64)


def load_ihdp(seed: int = 0) -> dict[str, np.ndarray]:
    """Full IHDP semi-synthetic dataset: covariates, treatment, observed
    outcome, and both potential outcomes mu0/mu1 (Eq. 10). True ATE = 4.0."""
    rng = np.random.default_rng(seed)
    X = generate_ihdp_covariates(rng=rng)
    cf = counterfactual_outcomes(X, rng=rng)
    T = assign_treatment(X, rng=rng)
    noise = rng.normal(0, NOISE_SD, size=N_IHDP)
    Y = T * cf["mu1"] + (1 - T) * cf["mu0"] + noise
    return {
        "X": X, "T": T, "Y": Y,
        "mu0": cf["mu0"], "mu1": cf["mu1"],
        "true_ate": TRUE_ATE,
    }
