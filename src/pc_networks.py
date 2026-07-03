"""Predictive coding networks (Eq. 1-8).

- LinearPCNetwork: minimal L=2 linear PC net used to validate Proposition 1
  exactly on the toy SCM (Sec 2.2, 3.1).
- PCNetworkV2: shared-latent PC network, two decoders (covariates, outcome)
  (Sec 2.3.1, Eq. 7).
- PCNetworkV3: like V2 but the outcome decoder also receives treatment
  directly, fixing the intervention-degeneracy problem on small datasets
  (Sec 2.3.2, Eq. 8).
"""
from __future__ import annotations
import torch
import torch.nn as nn

class LinearPCNetwork:
    """2-layer linear PC network over (D, S, T) implementing Eq. 1-5 with
    identity activations, used to validate Proposition 1 (Definition 1,
    Eq. 6) on the toy SCM.

    Layers: mu0 = T (observation), mu1 = S, mu2 = D (top-most latent).
    Generative model: mu_hat_0 = W_ST * mu1 ; mu_hat_1 = W_DS * mu2.

    NOTE on precision: Table 1 reports a single scalar precision pi=1.0
    applied identically at every layer. Taken completely literally (no
    prior term on the top-most latent D, and pi=1 for the S-D error), the
    inference dynamics in Eq. 4 have no restoring force pulling D toward
    its own N(0,1) prior, so ordinary conditioning P(D|S=1) would not
    converge to the true regression posterior (~0.640) -- it would drift
    toward S/W_DS instead. To reproduce the paper's reported "observational
    posterior" value of ~0.640 (Table 3, Sec 3.1), this implementation adds
    the (implicit, but necessary for a well-posed free-energy minimum) prior
    error term for the top-most latent, D - 0, with precision 1 (matching
    D's true N(0,1) prior), and weights the bottom-up S-D error by the
    inverse noise variance 1/sigma_toy^2 -- this is the standard interpretation
    of "precision" in predictive coding (inverse variance) and reproduces the
    exact analytical linear-Gaussian posterior E[D|S=1] = Cov(D,S)/Var(S).
    Under do(S=1) with error suppression, this prior term is irrelevant
    since the bottom-up term is zeroed and D correctly converges to 0
    exactly as Proposition 1 predicts.
    """

    def __init__(self, lr_infer: float = 0.05, lr_weight: float = 0.01,
                 precision: float = 1.0, prior_precision: float = 1.0,
                 weight_decay: float = 1e-4, noise_sd: float | None = None):
        self.w_st = torch.tensor(0.5, requires_grad=False)  # init away from truth
        self.w_ds = torch.tensor(0.5, requires_grad=False)
        self.lr_infer = lr_infer
        self.lr_weight = lr_weight
        self.weight_decay = weight_decay
        # bottom-up error precision = inverse noise variance (see NOTE above)
        self.pi = precision if noise_sd is None else 1.0 / (noise_sd ** 2)
        self.pi_prior = prior_precision

    def fit(self, D: torch.Tensor, S: torch.Tensor, T: torch.Tensor,
            epochs: int = 30, batch_size: int = 32):
        """Weight learning (Eq. 5) via minibatch gradient steps; matches the
        paper's 'trained on 1,000 samples for 30 epochs' (Sec 3.1)."""
        n = D.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                Db, Sb, Tb = D[idx], S[idx], T[idx]
                eps0 = Tb - self.w_st * Sb            # error at layer 0 (T vs W_ST*S)
                eps1 = Sb - self.w_ds * Db             # error at layer 1 (S vs W_DS*D)

                grad_w_st = -(self.pi * eps0 * Sb).mean()
                grad_w_ds = -(self.pi * eps1 * Db).mean()

                self.w_st = self.w_st - self.lr_weight * (grad_w_st + self.weight_decay * self.w_st)
                self.w_ds = self.w_ds - self.lr_weight * (grad_w_ds + self.weight_decay * self.w_ds)
        return self

    def infer_D_given_S(self, s_value: float, suppress_upstream_error: bool,
                         infer_steps: int = 200) -> float:
        """Run inference to convergence with S clamped to s_value, estimating
        D via Eq. 4. If suppress_upstream_error is True this implements
        clamp(S, s_value) with error suppression per Definition 1, i.e.
        do(S=s_value); if False it is ordinary conditioning P(D | S=s_value).
        """
        S = torch.tensor(float(s_value))
        D = torch.tensor(0.0)  # init state to infer

        for _ in range(infer_steps):
            eps1 = S - self.w_ds * D  # error between clamped S and prediction from D
            if suppress_upstream_error:
                # Definition 1, step 2: zero the bottom-up term into the
                # parent (D) -- severs the S -> D information pathway,
                # implementing graph surgery for do(S=s_value). Only the
                # prior term would remain, but it is exactly 0 at D=0, so
                # D converges to 0 = E[D | do(S=1)], matching Proposition 1.
                bottom_up = torch.tensor(0.0)
            else:
                bottom_up = self.pi * eps1 * self.w_ds
            grad_D = self.pi_prior * D - bottom_up
            D = D - self.lr_infer * grad_D
        return float(D)

    def weights(self) -> tuple[float, float]:
        return float(self.w_ds), float(self.w_st)
# ---------------------------------------------------------------------------
# PCNetworkV2 (Eq. 7)
# ---------------------------------------------------------------------------
class _Decoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PCNetworkV2(nn.Module):
    """Shared-latent PC network: z decodes separately to covariates (gx) and
    outcome (gy). Free energy per Eq. 7:
        F(z) = ||x - gx(z)||^2 + (y - gy(z))^2
    Inference optimises z via Adam for T_inf steps with weights fixed
    (Sec 2.3.1). Weight learning updates gx, gy via the same free energy.
    """

    def __init__(self, d_x: int, d_z: int = 4, hidden: int = 32,
                 treatment_col: int = 0):
        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.treatment_col = treatment_col
        self.gx = _Decoder(d_z, d_x, hidden)
        self.gy = _Decoder(d_z, 1, hidden)

    def free_energy(self, z: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                     zero_treatment_error: bool = False) -> torch.Tensor:
        x_hat = self.gx(z)
        y_hat = self.gy(z).squeeze(-1)
        err_x = x - x_hat
        if zero_treatment_error:
            # Interventional prediction for do(T=t): zero the reconstruction
            # error on the treatment column before it contributes to the
            # latent gradient (Sec 2.3.1, last paragraph).
            err_x = err_x.clone()
            err_x[:, self.treatment_col] = 0.0
        err_y = y - y_hat
        return (err_x ** 2).sum(dim=-1) + (err_y ** 2)

    def infer_latent(self, x: torch.Tensor, y: torch.Tensor, t_inf: int = 100,
                      lr: float = 0.05, zero_treatment_error: bool = False) -> torch.Tensor:
        """Amortisation-free inference: optimise z via Adam with weights fixed."""
        z = torch.zeros(x.shape[0], self.d_z, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(t_inf):
            opt.zero_grad()
            f = self.free_energy(z, x, y, zero_treatment_error=zero_treatment_error).mean()
            f.backward()
            opt.step()
        return z.detach()

    def predict_observational(self, x: torch.Tensor, t_inf: int = 100, lr: float = 0.05) -> torch.Tensor:
        """Predict y given fully observed x (y unknown -> use zeros as a
        placeholder target during inference, standard PC prediction mode)."""
        y_placeholder = torch.zeros(x.shape[0])
        z = self._infer_predict(x, y_placeholder, t_inf, lr)
        return self.gy(z).squeeze(-1).detach()

    def predict_interventional(self, x: torch.Tensor, t_value: float, t_inf: int = 100,
                                lr: float = 0.05) -> torch.Tensor:
        """Predict y under do(T=t_value): set the treatment feature to
        t_value in x and suppress its reconstruction error during inference
        (do-operator implementation for PCNetworkV2, Sec 2.3.1)."""
        x_do = x.clone()
        x_do[:, self.treatment_col] = t_value
        y_placeholder = torch.zeros(x.shape[0])
        z = self._infer_predict(x_do, y_placeholder, t_inf, lr, zero_treatment_error=True)
        return self.gy(z).squeeze(-1).detach()

    def _infer_predict(self, x, y, t_inf, lr, zero_treatment_error: bool = False):
        """Inference variant used at prediction time: y is not trusted (no
        outcome loss term contributes), only x drives the latent."""
        z = torch.zeros(x.shape[0], self.d_z, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(t_inf):
            opt.zero_grad()
            x_hat = self.gx(z)
            err_x = x - x_hat
            if zero_treatment_error:
                err_x = err_x.clone()
                err_x[:, self.treatment_col] = 0.0
            f = (err_x ** 2).sum(dim=-1).mean()
            f.backward()
            opt.step()
        return z.detach()


def train_pc_v2(model: PCNetworkV2, x: torch.Tensor, y: torch.Tensor,
                 epochs: int = 200, batch_size: int = 32, t_inf: int = 100,
                 lr_infer: float = 0.05, lr_weight: float = 0.001) -> PCNetworkV2:
    """Alternating inference / weight-learning training loop (Sec 2.1: 'in
    practice, inference runs to convergence with weights fixed; weight
    updates follow using the converged states')."""
    opt_w = torch.optim.Adam(model.parameters(), lr=lr_weight, weight_decay=1e-4)
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            z = model.infer_latent(xb, yb, t_inf=t_inf, lr=lr_infer)
            opt_w.zero_grad()
            f = model.free_energy(z, xb, yb).mean()
            f.backward()
            opt_w.step()
    return model


# ---------------------------------------------------------------------------
# PCNetworkV3 (Eq. 8) -- outcome decoder receives latent AND treatment
# ---------------------------------------------------------------------------
class PCNetworkV3(nn.Module):
    """Like PCNetworkV2, but gy: R^{d_z+1} -> R takes (z, T) directly, and gx
    reconstructs covariates *excluding* treatment. This gives the model an
    explicit treatment-to-outcome path (Sec 2.3.2, Eq. 8), fixing the
    intervention degeneracy PCNetworkV2 exhibits on small datasets (Sec 3.6).
    """

    def __init__(self, d_x_minus_t: int, d_z: int = 4, hidden: int = 32,
                 treatment_col: int = 0):
        super().__init__()
        self.d_x_minus_t = d_x_minus_t
        self.d_z = d_z
        self.treatment_col = treatment_col
        self.gx = _Decoder(d_z, d_x_minus_t, hidden)
        self.gy = _Decoder(d_z + 1, 1, hidden)

    @staticmethod
    def _split(x: torch.Tensor, treatment_col: int):
        mask = torch.ones(x.shape[1], dtype=torch.bool)
        mask[treatment_col] = False
        x_minus_t = x[:, mask]
        t = x[:, treatment_col]
        return x_minus_t, t

    def free_energy(self, z: torch.Tensor, x_minus_t: torch.Tensor, t: torch.Tensor,
                     y: torch.Tensor) -> torch.Tensor:
        x_hat = self.gx(z)
        y_hat = self.gy(torch.cat([z, t.unsqueeze(-1)], dim=-1)).squeeze(-1)
        err_x = x_minus_t - x_hat
        err_y = y - y_hat
        return (err_x ** 2).sum(dim=-1) + (err_y ** 2)

    def infer_latent(self, x_minus_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
                      t_inf: int = 100, lr: float = 0.05) -> torch.Tensor:
        z = torch.zeros(x_minus_t.shape[0], self.d_z, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(t_inf):
            opt.zero_grad()
            f = self.free_energy(z, x_minus_t, t, y).mean()
            f.backward()
            opt.step()
        return z.detach()

    def _infer_predict(self, x_minus_t, t, t_inf, lr):
        """Prediction-time inference: only covariates (not y) drive z."""
        z = torch.zeros(x_minus_t.shape[0], self.d_z, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(t_inf):
            opt.zero_grad()
            x_hat = self.gx(z)
            f = ((x_minus_t - x_hat) ** 2).sum(dim=-1).mean()
            f.backward()
            opt.step()
        return z.detach()

    def predict_observational(self, x: torch.Tensor, t_inf: int = 100, lr: float = 0.05) -> torch.Tensor:
        x_minus_t, t = self._split(x, self.treatment_col)
        z = self._infer_predict(x_minus_t, t, t_inf, lr)
        y_hat = self.gy(torch.cat([z, t.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return y_hat.detach()

    def predict_interventional(self, x: torch.Tensor, t_value: float, t_inf: int = 100,
                                lr: float = 0.05) -> torch.Tensor:
        """do(T=t_value): z is inferred from covariates only (unaffected by
        T, since gx never sees T -- z 'does not need to reorganise', Sec
        2.3.2), then T is set directly at the gy input."""
        x_minus_t, _ = self._split(x, self.treatment_col)
        z = self._infer_predict(x_minus_t, torch.full((x.shape[0],), t_value), t_inf, lr)
        t_do = torch.full((x.shape[0],), float(t_value))
        y_hat = self.gy(torch.cat([z, t_do.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return y_hat.detach()


def train_pc_v3(model: PCNetworkV3, x: torch.Tensor, y: torch.Tensor,
                 epochs: int = 200, batch_size: int = 16, t_inf: int = 100,
                 lr_infer: float = 0.05, lr_weight: float = 0.001) -> PCNetworkV3:
    opt_w = torch.optim.Adam(model.parameters(), lr=lr_weight, weight_decay=1e-4)
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            x_minus_t, t = model._split(xb, model.treatment_col)
            z = model.infer_latent(x_minus_t, t, yb, t_inf=t_inf, lr=lr_infer)
            opt_w.zero_grad()
            f = model.free_energy(z, x_minus_t, t, yb).mean()
            f.backward()
            opt_w.step()
    return model
