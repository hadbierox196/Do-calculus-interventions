"""MLP baseline (Sec 2.3.3): fully-connected associational model.

Input(dx) -> Linear(16) -> ReLU -> Linear(8) -> ReLU -> Linear(1)
Trained with Adam + MSE loss. Because it has no internal mechanism for graph
surgery, its "interventional prediction" for do(T=t) is simply a forward
pass with the treatment feature set to t -- i.e. associational extrapolation,
not a true intervention. This is exactly the gap the paper's Claim 2/3 probe.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, d_x: int, treatment_col: int = 0):
        super().__init__()
        self.treatment_col = treatment_col
        self.net = nn.Sequential(
            nn.Linear(d_x, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_observational(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)

    def predict_interventional(self, x: torch.Tensor, t_value: float) -> torch.Tensor:
        """Associational stand-in for do(T=t): overwrite the treatment
        column and forward-pass. No causal guarantee -- this is precisely
        what motivates the paper's interventional-gap metric."""
        x_do = x.clone()
        x_do[:, self.treatment_col] = t_value
        with torch.no_grad():
            return self.forward(x_do)


def train_mlp(model: MLPBaseline, x: torch.Tensor, y: torch.Tensor,
              epochs: int = 300, batch_size: int = 64, lr: float = 0.001) -> MLPBaseline:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model
