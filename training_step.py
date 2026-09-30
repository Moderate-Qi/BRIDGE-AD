"""Base-objective update extracted from the BRIDGE-AD training implementation.

The imported bridge_ad models, losses, and conditioning operations belong to
the internal package, which will be released upon acceptance of the paper.
The stage engines use the corresponding packaged training-update function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from bridge_ad.models import PMCDControlBranch
from bridge_ad.pmcd import (
    ControlResidualBranch,
    LossWeights,
    clean_latent_from_noise_prediction,
    pmcd_loss,
    pmcd_noise_prediction,
)
from bridge_ad.types import BipolarRole


@dataclass
class TrainingBatch:
    target_latent: torch.Tensor
    source_latent: torch.Tensor
    masked_source_latent: torch.Tensor
    support_latent: torch.Tensor
    core_latent: torch.Tensor
    visible_target_latent: torch.Tensor
    context_ring: torch.Tensor
    relation_condition: torch.Tensor
    texture_condition: torch.Tensor
    text_embedding: torch.Tensor
    role: BipolarRole


def train_pmcd_microstep(
    pipe: Any,
    active_branch: PMCDControlBranch,
    frozen_texture_branch: ControlResidualBranch,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    generator: torch.Generator,
    weights: LossWeights,
    texture_timestep_cutoff: int,
) -> dict[str, float]:
    timestep = torch.randint(
        50,
        950,
        (batch.target_latent.shape[0],),
        device=batch.target_latent.device,
        generator=generator,
    )
    noise = torch.randn(
        batch.target_latent.shape,
        device=batch.target_latent.device,
        dtype=batch.target_latent.dtype,
        generator=generator,
    )
    noisy = pipe.scheduler.add_noise(batch.target_latent, noise, timestep)
    model_input = torch.cat([noisy, batch.support_latent, batch.masked_source_latent], dim=1)
    prediction = pmcd_noise_prediction(
        pipe.unet,
        active_branch,
        frozen_texture_branch,
        model_input,
        timestep,
        batch.text_embedding,
        batch.relation_condition,
        batch.texture_condition,
        batch.support_latent,
        batch.context_ring,
        texture_active=bool(torch.all(timestep < texture_timestep_cutoff)),
    )
    alpha = pipe.scheduler.alphas_cumprod.to(timestep.device)[timestep]
    clean = clean_latent_from_noise_prediction(noisy, prediction, alpha)
    loss = pmcd_loss(
        prediction,
        noise,
        clean,
        batch.target_latent,
        batch.source_latent,
        batch.support_latent,
        batch.core_latent,
        batch.visible_target_latent,
        batch.role,
        weights,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(active_branch.parameters(), 1.0)
    optimizer.step()
    return {
        "total": float(loss.total.detach().cpu()),
        "noise": float(loss.noise.detach().cpu()),
        "clean_latent": float(loss.clean_latent.detach().cpu()),
        "edge": float(loss.edge.detach().cpu()),
        "operation": float(loss.operation.detach().cpu()),
    }
