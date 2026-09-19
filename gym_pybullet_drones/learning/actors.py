"""Shared local actors; optional model dependencies are imported lazily."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ActorConfig:
    actor_type: str = "mlp"
    model_path: str | None = None
    backbone_dtype: str = "float32"
    attention_backend: str = "eager"
    trainable_last_n_layers: int = 2
    max_prompt_tokens: int = 1024
    max_prompt_chars: int = 1024
    actor_backbone_lr: float = 1e-5
    actor_head_lr: float = 1e-4

    def __post_init__(self):
        if self.actor_type not in ("mlp", "qwen"):
            raise ValueError("actor_type must be mlp or qwen")
        if self.backbone_dtype not in ("float32", "bfloat16"):
            raise ValueError("Supported backbone dtypes: float32, bfloat16")
        if self.attention_backend not in ("eager", "sdpa"):
            raise ValueError("Supported attention backends: eager, sdpa")
        if self.trainable_last_n_layers != 2:
            raise ValueError("This implementation trains exactly the last two text layers")
        if any(not 1 <= limit <= 1024 for limit in (self.max_prompt_tokens, self.max_prompt_chars)):
            raise ValueError("Prompt limits must be in [1,1024]")
        if min(self.actor_backbone_lr, self.actor_head_lr) <= 0:
            raise ValueError("Actor learning rates must be positive")


def mlp(input_size, output_size, hidden_size, output_gain):
    layers = []
    for index, size in enumerate((hidden_size, hidden_size, output_size)):
        linear = nn.Linear(input_size, size)
        nn.init.orthogonal_(linear.weight, gain=output_gain if index == 2 else np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if index < 2:
            layers.append(nn.Tanh())
        input_size = size
    return layers


class MLPActor(nn.Sequential):
    """Sequential keys deliberately retain legacy MAPPO checkpoint compatibility."""

    def encode(self, observations):
        return torch.as_tensor(observations, dtype=torch.float32,
                               device=next(self.parameters()).device)


def build_actor(config, obs_dim, action_dim, hidden_size, device, spec=None):
    if config.actor_type == "mlp":
        return MLPActor(*mlp(obs_dim, action_dim, hidden_size, 0.01)).to(device).eval()
    if spec is None or spec.obs_dim != obs_dim or spec.action_dim != action_dim:
        raise ValueError("Qwen requires a matching ObservationSpec")
    from gym_pybullet_drones.learning.qwen_actor import load_qwen_actor

    return load_qwen_actor(config, spec, device)
