"""Local bounded-action behavior cloning, independent of language-model trainers."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from gym_pybullet_drones.learning.actor_checkpoint import load_actor, save_actor
from gym_pybullet_drones.learning.actors import ActorConfig, build_actor
from gym_pybullet_drones.learning.observation_prompt import ObservationSpec


class ActionDataset(Dataset):
    """Validated, action-before-step records with episode-level partitioning."""

    def __init__(self, directory):
        root = Path(directory)
        with (root / "dataset_manifest.json").open(encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest["schema_version"] != 1:
            raise ValueError("Unsupported dataset schema")
        self.spec = ObservationSpec(**self.manifest["observation_spec"])
        if (self.manifest["action_type"] != self.spec.action_type
                or self.manifest["action_dim"] != self.spec.action_dim
                or self.manifest["ctrl_freq"] != self.spec.ctrl_freq):
            raise ValueError("Dataset manifest disagrees with observation semantics")
        self.records = []
        seen = set()
        with (root / "samples.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                obs = np.asarray(record["observation"], dtype=np.float32)
                action = np.asarray(record["action"], dtype=np.float32)
                key = (record["episode_id"], record["step"], record["agent_index"])
                if key in seen or not 0 <= record["agent_index"] < 10 or record["step"] < 0:
                    raise ValueError("Duplicate sample or invalid step/agent index")
                seen.add(key)
                if obs.shape != (self.spec.obs_dim,) or action.shape != (self.spec.action_dim,):
                    raise ValueError("Dataset observation/action shape mismatch")
                if not np.isfinite(obs).all() or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                    raise ValueError("Nonfinite observation or invalid bounded action")
                self.records.append((record["episode_id"], torch.from_numpy(obs), torch.from_numpy(action)))
        if not self.records:
            raise ValueError("Empty dataset")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index][1:]

    def split(self, validation_fraction=0.2, seed=0):
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0,1)")
        episodes = sorted({record[0] for record in self.records})
        if len(episodes) < 2:
            raise ValueError("At least two distinct episodes are required")
        shuffled = np.random.default_rng(seed).permutation(episodes)
        count = min(len(episodes) - 1, max(1, round(len(episodes) * validation_fraction)))
        validation = set(shuffled[:count])
        train, val = [], []
        for index, record in enumerate(self.records):
            (val if record[0] in validation else train).append(index)
        return Subset(self, train), Subset(self, val)


def action_loss(actor, observations, actions):
    """One tanh only; ±1 expert labels remain finite without inverse tanh."""
    prediction = actor(actor.encode(observations)).tanh()
    return (prediction - actions.to(prediction.device)).square().mean(dim=0)


def train(config):
    """Run SFT only on explicit invocation, saving best validation and final actors."""
    allowed = {"dataset", "output", "actor", "actor_init", "device", "seed", "epochs",
               "batch_size", "microbatch_size", "validation_fraction", "max_grad_norm"}
    if set(config) - allowed:
        raise ValueError(f"Unknown SFT options: {sorted(set(config) - allowed)}")
    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    epochs, batch_size, micro = (config.get("epochs", 3), config.get("batch_size", 32),
                                 config.get("microbatch_size", 10))
    grad_norm = config.get("max_grad_norm", 0.5)
    if min(epochs, batch_size, micro, grad_norm) <= 0:
        raise ValueError("Training sizes and max_grad_norm must be positive")
    dataset = ActionDataset(config["dataset"])
    split_seed = dataset.manifest["episode_split_seed"]
    train_data, val_data = dataset.split(config.get("validation_fraction", 0.2), split_seed)
    device = config.get("device", "cpu")
    actor_config = ActorConfig(**config["actor"])
    if actor_config.actor_type != "qwen":
        raise ValueError("This SFT entry point requires a Qwen actor")
    if config.get("actor_init"):
        actor, _, _, _ = load_actor(config["actor_init"], device, actor_config.model_path, dataset.spec)
        # Initialization restores weights/semantics, not the previous run's learning rates.
        actor.config.actor_backbone_lr = actor_config.actor_backbone_lr
        actor.config.actor_head_lr = actor_config.actor_head_lr
    else:
        actor = build_actor(actor_config, dataset.spec.obs_dim, dataset.spec.action_dim, 128, device, dataset.spec)
    optimizer = torch.optim.Adam(actor.parameter_groups(), eps=1e-5)
    parameters = [p for p in actor.parameters() if p.requires_grad]
    log_std = torch.full((dataset.spec.action_dim,), -0.5, device=device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=micro)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    history, best = [], float("inf")
    for epoch in range(epochs):
        total, count = np.zeros(dataset.spec.action_dim), 0
        for observations, actions in loader:
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(observations), micro):
                obs, labels = observations[start:start + micro], actions[start:start + micro]
                errors = action_loss(actor, obs, labels)
                (errors.mean() * len(obs) / len(observations)).backward()
                total += errors.detach().cpu().numpy() * len(obs)
                count += len(obs)
            torch.nn.utils.clip_grad_norm_(parameters, grad_norm)
            optimizer.step()
        validation, val_count = np.zeros(dataset.spec.action_dim), 0
        with torch.no_grad():
            for observations, actions in val_loader:
                validation += action_loss(actor, observations, actions).cpu().numpy() * len(observations)
                val_count += len(observations)
        record = dict(epoch=epoch + 1, train_mse=float((total / count).mean()),
                      train_per_dim=(total / count).tolist(), val_mse=float((validation / val_count).mean()),
                      val_per_dim=(validation / val_count).tolist())
        history.append(record)
        print(json.dumps(record))
        extra = dict(sft_config=config, epoch=epoch + 1, split_seed=split_seed,
                     validation_episodes=sorted({dataset.records[i][0] for i in val_data.indices}))
        training = dict(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                        shuffle_rng=generator.get_state())
        if record["val_mse"] < best:
            best = record["val_mse"]
            save_actor(output / "best_model", actor, log_std, "sft", extra, training=training)
    save_actor(output / "final_model", actor, log_std, "sft", extra, training=training)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON SFT configuration")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        train(json.load(stream))
