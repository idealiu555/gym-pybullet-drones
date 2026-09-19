"""Text-only Qwen3.5 actor, without generation, vocabulary logits or Swift."""

from pathlib import Path

import torch
from torch import nn

from gym_pybullet_drones.learning.observation_prompt import ObservationPromptEncoder


class QwenActor(nn.Module):
    """One text backbone shared by every row of an independent local batch."""

    def __init__(self, backbone, tokenizer, config, spec, base_identity):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.spec = spec
        self.base_identity = base_identity
        self.encoder = ObservationPromptEncoder(spec, tokenizer, config.max_prompt_tokens,
                                                config.max_prompt_chars)
        if len(backbone.layers) < config.trainable_last_n_layers:
            raise ValueError("Not enough text decoder layers")
        backbone.requires_grad_(False)
        for layer in backbone.layers[-config.trainable_last_n_layers:]:
            # Adam updates and its moments must retain sub-BF16-ULP increments.
            layer.float().requires_grad_(True)
        hidden = backbone.config.hidden_size
        layers = []
        for _ in range(3):
            layers.extend((nn.Linear(hidden, hidden), nn.SiLU()))
        layers.append(nn.Linear(hidden, spec.action_dim))
        self.score = nn.Sequential(*layers).to(device=next(backbone.parameters()).device,
                                                dtype=torch.float32)
        nn.init.orthogonal_(self.score[-1].weight, gain=0.01)
        nn.init.zeros_(self.score[-1].bias)
        self.eval()

    def train(self, mode=True):
        # eval disables dropout without disabling gradients in the trainable suffix.
        return super().train(False)

    def encode(self, observations):
        if isinstance(observations, torch.Tensor):
            observations = observations.detach().cpu().numpy()
        return self.encoder(observations)

    def forward(self, sequences):
        device = next(self.backbone.parameters()).device
        lengths = torch.tensor([len(s) for s in sequences], device=device)
        ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True,
                                              padding_value=self.encoder.tokenizer.pad_token_id).to(device=device, dtype=torch.long)
        mask = torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]
        positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 0)
        kwargs = dict(input_ids=ids, attention_mask=mask.long(),
                      position_ids=positions, use_cache=False, return_dict=True)
        if self.config.backbone_dtype == "bfloat16":
            # Differentiable, temporary compute copies; registered/master weights
            # stay FP32 for both optimizers and checkpoints. Casting explicitly
            # also covers custom kernels that do not participate in autocast.
            weights = {name: p.to(torch.bfloat16) for name, p in self.backbone.named_parameters()
                       if p.requires_grad}
            hidden = torch.func.functional_call(self.backbone, weights, (), kwargs).last_hidden_state
        else:
            hidden = self.backbone(**kwargs).last_hidden_state
        pooled = hidden[torch.arange(len(sequences), device=device), lengths - 1]
        return self.score(pooled.float())

    def parameter_groups(self):
        return [dict(params=[p for p in self.backbone.parameters() if p.requires_grad],
                     lr=self.config.actor_backbone_lr),
                dict(params=list(self.score.parameters()), lr=self.config.actor_head_lr)]


def load_qwen_actor(config, spec, device, tokenizer_path=None, expected_identity=None):
    """Load only local weights with the audited Transformers 5.3.0 structure."""
    from gym_pybullet_drones.learning.actor_checkpoint import base_fingerprint

    if not config.model_path or not Path(config.model_path).is_dir():
        raise ValueError("model_path must be a local Qwen3.5-0.8B directory")
    identity = base_fingerprint(config.model_path)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Base model files differ from the checkpoint fingerprint")
    try:
        import transformers
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise ImportError('Install the optional Qwen dependencies: pip install -e ".[qwen]"') from exc
    if transformers.__version__ != "5.3.0":
        raise RuntimeError("Qwen adapter requires transformers==5.3.0")
    if config.backbone_dtype == "bfloat16" and torch.device(device).type != "cuda":
        raise ValueError("Use float32 for CPU development; BF16 is reserved for CUDA")
    wrapper, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        config.model_path, local_files_only=True, use_safetensors=True, output_loading_info=True,
        dtype=getattr(torch, config.backbone_dtype), attn_implementation=config.attention_backend)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Base model did not load strictly: {info}")
    backbone = wrapper.model.language_model
    if backbone.config.hidden_size != 1024 or len(backbone.layers) != 24:
        raise ValueError("Expected Qwen3.5-0.8B: hidden_size=1024, text layers=24")
    del wrapper
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or config.model_path,
                                              local_files_only=True, padding_side="right")
    return QwenActor(backbone.to(device), tokenizer, config, spec, identity)
