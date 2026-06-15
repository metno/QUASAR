from typing import Callable

import msgspec
import numpy as np
import scipy
import torch
from jaxtyping import Float
from torch import Tensor

type Scalar = Float[Tensor, ""]

type LossFunction = Callable[
    [Float[Tensor, "batch output"], Float[Tensor, " batch"]], Scalar
]


def quantile_loss(
    quantiles: Float[Tensor, "batch mesh"],
    y_obs: Float[Tensor, " batch"],
    prob: Float[Tensor, " mesh"],
) -> Scalar:
    error = y_obs - quantiles
    pos = error * prob
    neg = error * (prob - 1.0)

    # NOTE: the original Keras code returned the mean error per sample,
    # but in the end that engine averages the computed loss per batch,
    # so we can simply take the total mean.
    return torch.mean(torch.maximum(pos, neg))


# Calculates bernstein polynomials up to degree `degree` at probabilities `prob`.
def bernstein(
    degree: int, prob: Float[np.ndarray, " points"]
) -> Float[Tensor, "degree+1 points"]:
    tensors = [
        torch.from_numpy(scipy.stats.binom.pmf(i, degree, prob)).float()
        for i in range(degree + 1)
    ]

    return torch.vstack(tensors)


class QuantileLossConfig(msgspec.Struct):
    # Max degree of the Bernstein polynomial
    degree: int = 8

    # Number of discrete points for the probability axis
    prob: int = 100

    # Number of discrete points for the probability axis during inference
    inference_mesh_mul: int = 4

    # Probability range
    min_prob: float = 0.05
    max_prob: float = 0.95

    # Whether to enforce increasing order quantile coefficients
    strict: bool = False

    def output_layer_size(self) -> int:
        return self.degree + 1


class QuantileLoss:
    def __init__(
        self, config: QuantileLossConfig, device: torch.device, mesh_mul: int = 1
    ):
        prob_mesh = np.linspace(
            config.min_prob,
            config.max_prob,
            (mesh_mul * config.prob) + 1,
        )

        bern = bernstein(config.degree, prob_mesh).to(device)
        prob_mesh = torch.from_numpy(prob_mesh).to(device)

        self.strict = config.strict
        self.degree = config.degree
        self.bern = bern
        self.mesh_mul = mesh_mul
        self.prob_mesh = prob_mesh

        # TODO: maybe like this???
        self.training_mesh = ...
        self.inference_mesh = ...

    def submesh(self):
        return self.prob_mesh[:: self.mesh_mul]

    def __call__(self, y_pred, y_obs):
        quantiles = self.calculate_quantiles(y_pred)
        return quantile_loss(quantiles, y_obs, self.prob_mesh)

    def calculate_quantiles(self, y_pred: Float[Tensor, "batch degree+1"]):
        if self.strict:
            y_pred = torch.nn.functional.softplus(y_pred)
            y_pred = torch.cumsum(y_pred, dim=1)

        quantiles: Float[Tensor, "batch mesh"] = torch.matmul(y_pred, self.bern)
        return quantiles

    def inference(
        self,
        y_pred: Float[Tensor, "batch degree+1"],
        y_obs: Float[Tensor, " batch"],
    ) -> tuple[
        float,
        Float[Tensor, " batch"],
        Float[Tensor, " batch"],
        Float[Tensor, "batch mesh"],
    ]:
        quantiles = self.calculate_quantiles(y_pred)

        # Get mean of the quantiles on fine grid
        forecasts: Float[Tensor, " batch"] = torch.mean(quantiles, dim=1)
        forecasts = torch.unsqueeze(forecasts, dim=1)

        # Calculate PITs (quantile of the observation) on fine grid
        idxs = torch.searchsorted(quantiles, y_obs)

        # TODO: set to 1 if idx >= len(self.prob_mesh) or to 0 if idx == -1?
        idxs = torch.where(idxs >= len(self.prob_mesh), len(self.prob_mesh) - 1, idxs)
        pits = self.prob_mesh[idxs]

        # Calculate loss on coarser grid so it's "comparable" to training loss
        quantiles = quantiles[:, :: self.mesh_mul]
        loss = quantile_loss(quantiles, y_obs, self.prob_mesh[:: self.mesh_mul]).item()
        return loss, forecasts, pits, quantiles
