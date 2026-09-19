"""Versioned local actor checkpoints shared by SFT, MAPPO and inference."""

from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
import torch

from gym_pybullet_drones.learning.actors import ActorConfig
from gym_pybullet_drones.learning.observation_prompt import ObservationSpec


def base_fingerprint(directory):
    """Bind to local configuration and all safetensors shards, independent of path."""
    root = Path(directory)
    files = sorted(root.glob("*.safetensors"))
    if not files or not (root / "config.json").is_file():
        raise ValueError("Base requires config.json and safetensors weights")
    files += [root / "config.json"]
    index = root / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
    result = {}
    for path in files:
        with path.open("rb") as stream:
            result[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def read_manifest(path):
    with (Path(path) / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format_version") != 1 or manifest.get("stage") not in ("sft", "mappo"):
        raise ValueError("Unsupported checkpoint format or stage")
    return manifest


def save_actor(path, actor, log_std, stage, extra=None, critic=None, training=None):
    """Save the complete trainable suffix, never duplicate the frozen base."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if stage not in ("sft", "mappo"):
        raise ValueError("Invalid checkpoint stage")
    manifest = dict(format_version=1, stage=stage, actor_config=asdict(actor.config),
                    observation_spec=asdict(actor.spec), base_identity=actor.base_identity,
                    torch_version=str(torch.__version__), **(extra or {}))
    manifest["dependencies"] = {}
    for package in ("transformers", "tokenizers", "safetensors"):
        try:
            manifest["dependencies"][package] = version(package)
        except PackageNotFoundError:
            manifest["dependencies"][package] = None  # CPU substitute tests need no optional packages.
    actor.encoder.tokenizer.save_pretrained(path / "tokenizer")
    manifest["tokenizer_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((path / "tokenizer").iterdir()) if p.is_file()
    }
    state = {name: param.detach().cpu() for name, param in actor.named_parameters()
             if param.requires_grad}
    payload = dict(actor=state, log_std=log_std.detach().cpu())
    if critic is not None:
        payload["critic"] = critic.state_dict()
    torch.save(payload, path / "trainable.pt")
    if training is not None:
        torch.save(training, path / "optimizer.pt")
    elif (path / "optimizer.pt").exists():
        (path / "optimizer.pt").unlink()
    with (path / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


def load_actor(path, device="cpu", model_path=None, expected_spec=None):
    """Return actor, log_std, manifest and remaining (critic-only) CPU state."""
    from gym_pybullet_drones.learning.qwen_actor import load_qwen_actor

    path = Path(path)
    manifest = read_manifest(path)
    spec = ObservationSpec(**manifest["observation_spec"])
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("Checkpoint observation/action semantics do not match the environment")
    config = ActorConfig(**manifest["actor_config"])
    if config.actor_type != "qwen":
        raise ValueError("Directory checkpoint must contain a Qwen actor")
    if model_path is not None:
        config.model_path = str(model_path)
    tokenizer_files = {p.name for p in (path / "tokenizer").iterdir() if p.is_file()}
    if tokenizer_files != set(manifest["tokenizer_hashes"]):
        raise ValueError("Checkpoint tokenizer file list has changed")
    for name, digest in manifest["tokenizer_hashes"].items():
        if hashlib.sha256((path / "tokenizer" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Checkpoint tokenizer has changed")
    actor = load_qwen_actor(config, spec, device, tokenizer_path=path / "tokenizer",
                            expected_identity=manifest["base_identity"])
    # Stage on CPU; don't retain a second set of actor weights on the accelerator.
    payload = torch.load(path / "trainable.pt", map_location="cpu", weights_only=True)
    actor_state = payload.pop("actor")
    parameters = dict(actor.named_parameters())
    expected = {name for name, param in parameters.items() if param.requires_grad}
    if set(actor_state) != expected:
        raise ValueError("Checkpoint trainable actor keys do not match exactly")
    with torch.no_grad():
        for name, value in actor_state.items():
            if value.shape != parameters[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid actor parameter: {name}")
            parameters[name].copy_(value)
    log_std = payload.pop("log_std").float()
    if log_std.shape != (spec.action_dim,) or not torch.isfinite(log_std).all():
        raise ValueError("Invalid log_std")
    return actor, log_std.to(device), manifest, payload


class ActorPolicy:
    """Actor-only deployment adapter with the same tanh action transform as MAPPO."""

    def __init__(self, actor, log_std):
        self.actor, self.log_std = actor, log_std

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        rows = np.asarray(observation, dtype=np.float32)
        if rows.ndim < 1 or rows.shape[-1] != self.actor.spec.obs_dim:
            raise ValueError("Observation dimension mismatch")
        mean = self.actor(self.actor.encode(rows.reshape(-1, rows.shape[-1])))
        raw = mean if deterministic else torch.distributions.Normal(
            mean, self.log_std.clamp(-5, 2).exp()).sample()
        return raw.tanh().cpu().numpy().reshape(*rows.shape[:-1], self.actor.spec.action_dim), None
