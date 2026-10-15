"""Feed-forward network built from PyTorch primitives (deliverable 1).

No nn.Linear, nn.Dropout or nn.Sequential: each layer is an explicit affine map on nn.Parameter
tensors with an explicit initialisation, so that every design choice can be shown and diagnosed:

    x  -> standardise (train mean/std, stored as buffers: shipped with the model)
       -> [ h = act(x W^T + b) -> inverted dropout ] x L
       -> logits = h W_out^T + b_out          (softmax is inside the cross-entropy loss)

Autograd computes the gradients; the training loop is in training.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

ACTIVATIONS = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


class Linear(nn.Module):
    """y = x W^T + b with an explicit initialisation scheme.

    init='he'      : W ~ N(0, 2/fan_in)       keeps the activation variance constant through ReLUs
    init='xavier'  : W ~ N(0, 2/(fan_in+fan_out))   suited to tanh / sigmoid
    init='zeros'   : W = 0     (diagnosis only: symmetric neurons, nothing learns)
    init='large'   : W ~ N(0, 1)              (diagnosis only: exploding activations)
    init='small'   : W ~ N(0, 1e-4)           (diagnosis only: vanishing signal)
    """

    def __init__(self, in_features: int, out_features: int, init: str = "he"):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters(init)

    @torch.no_grad()
    def reset_parameters(self, init: str) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        std = {"he": math.sqrt(2.0 / fan_in), "xavier": math.sqrt(2.0 / (fan_in + fan_out)),
               "large": 1.0, "small": 1e-4, "zeros": 0.0}[init]
        self.weight.normal_(0.0, std) if std > 0 else self.weight.zero_()
        self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


def inverted_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Zero each unit with probability p and rescale by 1/(1-p) so that E[output] is unchanged.
    At evaluation time the layer is the identity."""
    if not training or p == 0.0:
        return x
    mask = (torch.rand_like(x) >= p).to(x.dtype)
    return x * mask / (1.0 - p)


class MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int, hidden=(512, 256), dropout: float = 0.3,
                 activation: str = "relu", init: str = "he", input_dropout: float = 0.0):
        super().__init__()
        dims = [in_features, *hidden]
        self.hidden = nn.ModuleList(Linear(a, b, init) for a, b in zip(dims[:-1], dims[1:]))
        out_init = "zeros" if init == "zeros" else ("xavier" if init in ("he", "xavier") else init)
        self.out = Linear(dims[-1], n_classes, out_init)
        if init in ("he", "xavier"):
            # Small output weights -> near-uniform softmax at step 0, so the initial loss is ~ln(K).
            # (Checked in the diagnosis: with plain Xavier the initial loss is ~2.5 instead of 1.61.)
            with torch.no_grad():
                self.out.weight.mul_(0.1)
        self.act = ACTIVATIONS[activation]
        self.activation, self.dropout, self.input_dropout = activation, dropout, input_dropout
        # Input standardisation, fitted on the training set (set_normalization); identity by default.
        self.register_buffer("mean", torch.zeros(in_features))
        self.register_buffer("std", torch.ones(in_features))
        self._record = False
        self.last_hidden: list[torch.Tensor] = []

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> "MLP":
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        x = inverted_dropout(x, self.input_dropout, self.training)
        self.last_hidden = []
        for layer in self.hidden:
            x = self.act(layer(x))
            if self._record:
                self.last_hidden.append(x.detach())
            x = inverted_dropout(x, self.dropout, self.training)
        return self.out(x)

    @torch.no_grad()
    def activation_stats(self, x: torch.Tensor) -> dict:
        """Per hidden layer: share of 'dead' units (never active on this batch, ReLU) or saturated
        units (|tanh|>0.99, sigmoid<0.01 or >0.99), and the activation standard deviation."""
        was_training = self.training
        self.eval()
        self._record = True
        self(x)
        self._record = False
        self.train(was_training)
        stats = {}
        for i, h in enumerate(self.last_hidden):
            if self.activation == "relu":
                inactive = (h <= 0).all(dim=0).float().mean()
            elif self.activation == "tanh":
                inactive = (h.abs() > 0.99).float().mean()
            else:
                inactive = ((h < 0.01) | (h > 0.99)).float().mean()
            stats[f"hidden{i}"] = {"dead_or_saturated": float(inactive), "act_std": float(h.std())}
        return stats

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def standardization_stats(X_train: torch.Tensor):
    """Per-feature mean and std computed on the TRAINING set only (no information from val/test)."""
    return X_train.mean(0), X_train.std(0)
