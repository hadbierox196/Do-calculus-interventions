"""Structural causal models used throughout the paper.

- toy_scm(): 3-node linear-Gaussian SCM D -> S -> T (Eq. 9), used to validate
  Proposition 1 exactly (Claim 1).
- SyntheticSCM: confounded C -> T -> Y generator (Eq. 11) used for the ablation
  studies (Claim 4) and topology robustness (Claim 5). Supports Chain, Mediator,
  Collider, and Fork variants by changing how C influences T and Y.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Toy SCM (Eq. 9): D ~ N(0,1); S = 0.8 D + eps_S; T = 0.6 S + eps_T
# ---------------------------------------------------------------------------
W_DS = 0.8
W_ST = 0.6
SIGMA_TOY = 0.781  # analytically derived s.t. E[D | S=1] ~= 0.640 (paper Sec 3.1)


def sample_toy_scm(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Sample n rows from the toy 3-node SCM (Eq. 9)."""
    D = rng.normal(0, 1, size=n)
    eps_s = rng.normal(0, SIGMA_TOY, size=n)
    eps_t = rng.normal(0, SIGMA_TOY, size=n)
    S = W_DS * D + eps_s
    T = W_ST * S + eps_t
    return {"D": D, "S": S, "T": T}


def toy_do_S_ground_truth() -> tuple[float, float]:
    """Ground truth P(D | do(S=1)) = N(0, 1): severing S's parents leaves D's
    marginal untouched (Eq. 9's graph surgery interpretation)."""
    return 0.0, 1.0


# ---------------------------------------------------------------------------
# Synthetic confounded SCM (Eq. 11): C -> T -> Y with topology variants
# ---------------------------------------------------------------------------
TOPOLOGIES = ("chain", "mediator", "collider", "fork")


@dataclass
class SyntheticSCMConfig:
    n: int = 1000
    d_c: int = 5
    gamma: float = 1.0          # confounding strength
    delta: float = 2.0          # base treatment effect
    noise_sd: float = 0.3
    topology: str = "chain"     # one of TOPOLOGIES
    interventional_fraction: float = 0.0  # fraction of rows generated under do(T=t)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SyntheticSCM:
    """Confounded chain C -> T -> Y (Eq. 11), with topology-specific edges.

    Chain:     C -> T -> Y  and  C -> Y directly (the base confounded chain).
    Mediator:  same structural equations, but C's effect on Y is routed only
               through an intermediate mediator M = 0.6*C1 + noise, with
               M added into the outcome equation instead of C directly.
    Collider:  T and Y both point into an extra collider node; C confounds
               T and Y as usual (backdoor path exists) but conditioning on
               the collider would open a *second*, non-causal path -- we do
               NOT condition on it during training, matching the paper's
               description of the collider blocking a backdoor when left
               unconditioned.
    Fork:      C causes both T and Y with NO direct T -> Y effect beyond the
               shared confounder (true ATE = 0), used as the null-effect
               sanity check (Table 6, Limitation L7).
    """

    def __init__(self, cfg: SyntheticSCMConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        if cfg.topology not in TOPOLOGIES:
            raise ValueError(f"Unknown topology {cfg.topology!r}, must be one of {TOPOLOGIES}")

    def sample(self) -> dict[str, np.ndarray]:
        cfg = self.cfg
        rng = self.rng
        n, d_c = cfg.n, cfg.d_c

        C = rng.normal(0, 1, size=(n, d_c))

        # Propensity / treatment assignment (confounded by C1, C2)
        logits = cfg.gamma * C[:, 0] + 0.3 * C[:, 1]
        if cfg.topology == "fork":
            # Fork: T depends on C but Y will not depend on T at all (ATE = 0)
            p_t = _sigmoid(logits)
        else:
            p_t = _sigmoid(logits)
        T = rng.binomial(1, p_t).astype(np.float64)

        if cfg.topology == "mediator":
            M = 0.6 * C[:, 0] + rng.normal(0, 0.2, size=n)
            mu0 = 1 + 0.5 * M + 0.3 * C[:, 1] + 0.2 * C[:, 2]
            mu1 = mu0 + cfg.delta + 0.1 * M
        elif cfg.topology == "collider":
            # Y and T both feed a collider node K (not conditioned on / not
            # included as a model input) -- the backdoor path C -> T and
            # C -> Y remains, but we track K only for completeness.
            mu0 = 1 + 0.5 * C[:, 0] + 0.3 * C[:, 1] + 0.2 * C[:, 2]
            mu1 = mu0 + cfg.delta + 0.1 * C[:, 0]
        elif cfg.topology == "fork":
            # No direct T -> Y edge: mu1 == mu0 (true ATE = 0)
            mu0 = 1 + 0.5 * C[:, 0] + 0.3 * C[:, 1] + 0.2 * C[:, 2]
            mu1 = mu0.copy()
        else:  # chain
            mu0 = 1 + 0.5 * C[:, 0] + 0.3 * C[:, 1] + 0.2 * C[:, 2]
            mu1 = mu0 + cfg.delta + 0.1 * C[:, 0]

        eps = rng.normal(0, cfg.noise_sd, size=n)
        Y = T * mu1 + (1 - T) * mu0 + eps

        # Optional interventional augmentation rows: replace a fraction of
        # rows' treatment with a randomly-assigned do(T=t), recomputing Y
        # from the *same* structural equations (used in Ablation 2, §3.4.2).
        n_int = int(round(cfg.interventional_fraction * n))
        is_interventional = np.zeros(n, dtype=bool)
        if n_int > 0:
            idx = rng.choice(n, size=n_int, replace=False)
            T_do = rng.binomial(1, 0.5, size=n_int).astype(np.float64)
            T[idx] = T_do
            Y[idx] = T_do * mu1[idx] + (1 - T_do) * mu0[idx] + rng.normal(0, cfg.noise_sd, size=n_int)
            is_interventional[idx] = True

        true_ate = float(np.mean(mu1 - mu0))
        return {
            "C": C, "T": T, "Y": Y,
            "mu0": mu0, "mu1": mu1,
            "true_ate": true_ate,
            "is_interventional": is_interventional,
        }
