"""Feed-forward MAPPO with a shared local actor and centralized team critic.

Policy ratios are computed per agent, never over the joint action. The critic
estimates the common discounted team return from all agents' observations.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from gym_pybullet_drones.learning.actors import ActorConfig, build_actor, mlp


@dataclass
class MAPPOConfig:
    """Hyperparameters; rollout and batch sizes count joint environment steps."""

    rollout_steps: int = 512
    batch_size: int = 256
    epochs: int = 5
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    update_microbatch_steps: int | None = None

    def __post_init__(self):
        if self.update_microbatch_steps is not None and self.update_microbatch_steps < 1:
            raise ValueError("update_microbatch_steps must be positive")
        if min(self.rollout_steps, self.batch_size, self.epochs, self.hidden_size) < 1:
            raise ValueError("Rollout, batch, epoch and hidden sizes must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0 or self.clip_range <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Learning rate, clipping and gradient norm must be positive")


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    """Compute team advantages and returns without crossing episode boundaries.

    Parameters
    ----------
    rewards, values, next_values : ndarray
        One scalar per transition. Next values use the pre-reset observation.
    terminated, truncated : ndarray
        Boundary flags: only true terminations disable bootstrapping.
    gamma, gae_lambda : float
        Discount and advantage trace decay.

    Returns
    -------
    tuple[ndarray, ndarray]
        Advantages and lambda returns, each shaped like rewards.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    trace = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        trace = delta + gamma * gae_lambda * (1 - (terminated[t] or truncated[t])) * trace
        advantages[t] = trace
    return advantages, advantages + values


class MAPPO:
    """MAPPO for homogeneous agents with bounded continuous actions.

    Parameters
    ----------
    observation_space, action_space : gymnasium.spaces.Box
        Spaces shaped (num_agents, local_dimension); actions must be in [-1, 1].
    config : MAPPOConfig, optional
        Optimization settings.
    device : str, optional
        Torch device; CPU is efficient for these small networks.
    """

    def __init__(self, observation_space, action_space, config=None, device="cpu",
                 actor_config=None, observation_spec=None, actor=None):
        self.config = config or MAPPOConfig()
        if (len(observation_space.shape) != 2 or len(action_space.shape) != 2
                or observation_space.shape[0] != action_space.shape[0]
                or not np.all(action_space.low == -1) or not np.all(action_space.high == 1)):
            raise ValueError("Expected (agents, features) observations and actions bounded by [-1, 1]")
        self.num_agents, self.obs_dim = observation_space.shape
        self.action_dim = action_space.shape[1]
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.actor_config = actor_config or ActorConfig()
        self.observation_spec = observation_spec
        self.actor = actor if actor is not None else build_actor(
            self.actor_config, self.obs_dim, self.action_dim, self.config.hidden_size,
            self.device, observation_spec)
        self.actor.eval()
        self.critic = nn.Sequential(*mlp(self.num_agents * self.obs_dim, 1,
                                         self.config.hidden_size, 1.0)).to(self.device)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5, device=self.device))
        if self.actor_config.actor_type == "qwen":
            parameters = self.actor.parameter_groups()
            parameters[1]["params"].append(self.log_std)
            parameters.append(dict(params=list(self.critic.parameters()), lr=self.config.learning_rate))
        else:
            parameters = list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std]
        self.optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate, eps=1e-5)
        self.num_timesteps = 0
        self.metadata = {}

    def _tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _distribution(self, obs, encoded=None):
        inputs = self.actor.encode(obs.reshape(-1, self.obs_dim)) if encoded is None else encoded
        means = self.actor(inputs).float().reshape(*obs.shape[:-1], self.action_dim)
        return Normal(means, self.log_std.clamp(-5, 2).exp())

    @staticmethod
    def _log_prob(distribution, raw_action):
        # Stable tanh Jacobian; retaining raw actions avoids inverse-tanh errors.
        correction = 2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        """Return actions using only local rows; a single drone row also works."""
        # Actors own preprocessing: Qwen constructs prompts on CPU; MLP moves to its device.
        obs = observation if isinstance(observation, torch.Tensor) else np.asarray(observation, dtype=np.float32)
        if obs.ndim < 1 or obs.shape[-1] != self.obs_dim:
            raise ValueError("Observation feature dimension does not match the policy")
        distribution = self._distribution(obs)
        raw = distribution.mean if deterministic else distribution.sample()
        return raw.tanh().cpu().numpy(), None

    @torch.no_grad()
    def _value(self, observation):
        return self.critic(self._tensor(observation).flatten()).item()

    def _collect_rollout(self, env, obs, steps):
        observations = np.empty((steps, self.num_agents, self.obs_dim), dtype=np.float32)
        actions = np.empty((steps, self.num_agents, self.action_dim), dtype=np.float32)
        log_probs = np.empty((steps, self.num_agents), dtype=np.float32)
        rewards, values, next_values = np.zeros((3, steps), dtype=np.float32)
        terminated, truncated = np.zeros((2, steps), dtype=bool)
        value = self._value(obs)
        tokens = [] if self.actor_config.actor_type == "qwen" else None
        for t in range(steps):
            observations[t] = obs
            values[t] = value
            with torch.no_grad():
                encoded = self.actor.encode(obs)
                if tokens is not None:
                    tokens.append(encoded)
                distribution = self._distribution(obs, encoded)
                raw = distribution.sample()
                actions[t] = raw.cpu().numpy()
                log_probs[t] = self._log_prob(distribution, raw).cpu().numpy()
            obs, reward, terminated[t], truncated[t], _ = env.step(raw.tanh().cpu().numpy())
            # Shared reward normalized so its scale does not grow with team size.
            rewards[t] = reward / self.num_agents
            next_values[t] = 0 if terminated[t] else self._value(obs)
            if terminated[t] or truncated[t]:
                obs, _ = env.reset()
                if t + 1 < steps:
                    value = self._value(obs)
            else:
                value = next_values[t]
        advantages, returns = compute_gae(rewards, values, next_values, terminated,
                                         truncated, self.config.gamma, self.config.gae_lambda)
        self._last_rollout_reward = float(rewards.mean())
        batch = tuple(self._tensor(x) for x in
                      (observations, actions, log_probs, values, advantages, returns))
        if tokens is not None:
            batch += (tokens,)
        return obs, batch

    def _update(self, batch):
        observations, actions, old_log_probs, old_values, advantages, returns = batch[:6]
        tokens = batch[6] if len(batch) == 7 else None
        # A one-step tail has no variance; centering it would erase its signal.
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        cfg = self.config
        actor_parameters = [p for p in self.actor.parameters() if p.requires_grad] + [self.log_std]
        losses = []
        for _ in range(cfg.epochs):
            indices = self.rng.permutation(len(observations))
            for start in range(0, len(indices), cfg.batch_size):
                joint = indices[start:start + cfg.batch_size]
                micro = cfg.update_microbatch_steps or (1 if tokens is not None else len(joint))
                self.optimizer.zero_grad(set_to_none=True)
                metrics = np.zeros(3)
                for offset in range(0, len(joint), micro):
                    selected = joint[offset:offset + micro]
                    idx = torch.as_tensor(selected, device=self.device)
                    encoded = [s for t in selected for s in tokens[t]] if tokens is not None else None
                    distribution = self._distribution(observations[idx], encoded)
                    log_probs = self._log_prob(distribution, actions[idx])
                    ratio = (log_probs - old_log_probs[idx]).exp()
                    advantage = advantages[idx, None]
                    policy_loss = -torch.minimum(ratio * advantage,
                                                ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * advantage).mean()
                    values = self.critic(observations[idx].flatten(start_dim=1)).squeeze(-1)
                    clipped = old_values[idx] + (values - old_values[idx]).clamp(-cfg.clip_range, cfg.clip_range)
                    value_loss = 0.5 * torch.maximum((values - returns[idx]).square(),
                                                     (clipped - returns[idx]).square()).mean()
                    entropy = -self._log_prob(distribution, distribution.rsample()).mean()
                    weight = len(selected) / len(joint)
                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                    (loss * weight).backward()
                    metrics += weight * np.array([policy_loss.item(), value_loss.item(), entropy.item()])
                # The critic's error scale must not shrink the actor's gradient.
                nn.utils.clip_grad_norm_(actor_parameters, cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                losses.append(metrics)
        return dict(zip(("policy_loss", "value_loss", "entropy"), np.mean(losses, axis=0)))

    def learn(self, env, total_timesteps, callback=None):
        """Train for exactly this many additional joint environment steps.

        The optional callback receives (model, metrics) after each update.
        Callers own and close the environment. Repeated calls start a fresh
        episode while retaining the networks and optimizer state.
        """
        if total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if (env.observation_space.shape != (self.num_agents, self.obs_dim)
                or env.action_space.shape != (self.num_agents, self.action_dim)):
            raise ValueError("Environment spaces do not match the model")
        if self.observation_spec is not None:
            from gym_pybullet_drones.learning.observation_prompt import ObservationSpec
            if ObservationSpec.from_env(env) != self.observation_spec:
                raise ValueError("Environment observation/action semantics differ from checkpoint")
        obs, _ = env.reset(seed=self.config.seed if self.num_timesteps == 0 else None)
        remaining = total_timesteps
        while remaining:
            steps = min(self.config.rollout_steps, remaining)
            obs, batch = self._collect_rollout(env, obs, steps)
            metrics = self._update(batch)
            metrics["rollout_reward"] = self._last_rollout_reward
            self.num_timesteps += steps
            remaining -= steps
            if callback is not None:
                callback(self, metrics)
        return self

    def save(self, path, metadata=None):
        """Save training state, retaining existing metadata unless replaced."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if metadata is not None:
            self.metadata = dict(metadata)
        training = dict(optimizer=self.optimizer.state_dict(), rng=self.rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)
        if self.actor_config.actor_type == "qwen":
            from gym_pybullet_drones.learning.actor_checkpoint import save_actor
            save_actor(path, self.actor, self.log_std, "mappo",
                       extra=dict(config=asdict(self.config), num_agents=self.num_agents,
                                  num_timesteps=self.num_timesteps, metadata=self.metadata),
                       critic=self.critic, training=training)
            return
        torch.save({"config": asdict(self.config), "num_agents": self.num_agents,
                    "obs_dim": self.obs_dim, "action_dim": self.action_dim,
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "log_std": self.log_std.detach().cpu(),
                    "num_timesteps": self.num_timesteps,
                    "metadata": self.metadata, **training}, path)

    def _restore_training_state(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        # Legacy MLP checkpoints contain optimizer state but no RNG state.
        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]
            torch.set_rng_state(state["torch_rng"].cpu())
            if self.device.type == "cuda":
                cuda_rng = state["cuda_rng"]
                if isinstance(cuda_rng, list):
                    # Legacy checkpoints require the original logical device index.
                    index = self.device.index if self.device.index is not None else torch.cuda.current_device()
                    cuda_rng = cuda_rng[index] if cuda_rng else None
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng.cpu(), self.device)

    @classmethod
    def load(cls, path, device="cpu", resume=True, model_path=None):
        """Load a checkpoint and return the model plus environment metadata."""
        from gymnasium import spaces

        if Path(path).is_dir():
            from gym_pybullet_drones.learning.actor_checkpoint import load_actor
            actor, log_std, manifest, payload = load_actor(path, device, model_path)
            if manifest["stage"] != "mappo":
                raise ValueError("Use actor_init for SFT checkpoints, not MAPPO.load")
            spec = actor.spec
            obs_space = spaces.Box(-np.inf, np.inf, (manifest["num_agents"], spec.obs_dim), dtype=np.float32)
            act_space = spaces.Box(-1, 1, (manifest["num_agents"], spec.action_dim), dtype=np.float32)
            model = cls(obs_space, act_space, MAPPOConfig(**manifest["config"]), device,
                        actor.config, spec, actor)
            model.critic.load_state_dict(payload["critic"])
            with torch.no_grad():
                model.log_std.copy_(log_std)
            model.num_timesteps = manifest["num_timesteps"]
            model.metadata = manifest["metadata"]
            if resume:
                state = torch.load(Path(path) / "optimizer.pt", map_location=device, weights_only=True)
                model._restore_training_state(state)
            return model, dict(model.metadata)

        data = torch.load(path, map_location=device, weights_only=True)
        obs_space = spaces.Box(-np.inf, np.inf, (data["num_agents"], data["obs_dim"]), dtype=np.float32)
        act_space = spaces.Box(-1, 1, (data["num_agents"], data["action_dim"]), dtype=np.float32)
        model = cls(obs_space, act_space, MAPPOConfig(**data["config"]), device)
        model.actor.load_state_dict(data["actor"])
        model.critic.load_state_dict(data["critic"])
        with torch.no_grad():
            model.log_std.copy_(data["log_std"])
        if resume:
            model._restore_training_state(data)
        model.num_timesteps = data["num_timesteps"]
        model.metadata = data["metadata"]
        return model, dict(model.metadata)
