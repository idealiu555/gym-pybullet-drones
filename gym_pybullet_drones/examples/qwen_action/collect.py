"""Collect candidate VEL expert trajectories; no model or training is involved."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from gym_pybullet_drones.learning.observation_prompt import ObservationSpec


def expert_action(observation, speed_limit, kp=1.0, kd=0.5):
    """Map world-frame PD velocity to the environment's direction/speed action."""
    if speed_limit <= 0 or kp <= 0 or kd < 0:
        raise ValueError("Invalid expert gains or speed limit")
    rows = np.asarray(observation)
    velocity = kp * rows[..., -3:] - kd * rows[..., 6:9]
    norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
    direction = np.divide(velocity, norm, out=np.zeros_like(velocity), where=norm > 0)
    return np.concatenate((direction, np.minimum(norm / speed_limit, 1)), axis=-1).astype(np.float32)


def collect(output="results/qwen_data", episodes=20, seed=0, kp=1.0, kd=0.5):
    """Write observations before stepping and keep failure episodes in the data."""
    from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary, NUM_DRONES
    from gym_pybullet_drones.utils.enums import ActionType

    if episodes < 2:
        raise ValueError("At least two episodes are needed for a held-out split")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    summaries = []
    spec = None
    with (root / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for episode in range(episodes):
            initial = np.zeros((NUM_DRONES, 3))
            initial[:, 0] = np.arange(NUM_DRONES) * 0.6
            initial[:, :2] += rng.uniform(-0.04, 0.04, (NUM_DRONES, 2))
            initial[:, 2] = rng.uniform(0.2, 0.35, NUM_DRONES)
            targets = initial + rng.uniform(-0.1, 0.1, (NUM_DRONES, 3))
            targets[:, 2] = rng.uniform(0.6, 1.0, NUM_DRONES)
            env = MultiHoverAviary(initial_xyzs=initial, target_positions=targets,
                                   act=ActionType.VEL, gui=False)
            try:
                spec = ObservationSpec.from_env(env)
                obs, _ = env.reset(seed=seed + episode)
                step = 0
                while True:
                    action = expert_action(obs, env.SPEED_LIMIT, kp, kd)
                    for agent in range(NUM_DRONES):
                        record = dict(episode_id=episode, step=step, agent_index=agent,
                                      observation=obs[agent].tolist(), action=action[agent].tolist())
                        stream.write(json.dumps(record) + "\n")
                    obs, _, terminated, truncated, info = env.step(action)
                    step += 1
                    if terminated or truncated:
                        summaries.append(dict(episode_id=episode, steps=step, failed=bool(terminated),
                                              success=bool(info["is_success"]),
                                              initial_xyzs=initial.tolist(), targets=targets.tolist()))
                        break
            finally:
                env.close()
    manifest = dict(schema_version=1, observation_spec=asdict(spec), action_type="vel", action_dim=4,
                    ctrl_freq=spec.ctrl_freq, physics="pyb", pyb_freq=240,
                    expert_version="world_pd_vel_v1", kp=kp, kd=kd, episode_split_seed=seed,
                    randomization=dict(x_spacing=0.6, xy_jitter=0.04, initial_z=[0.2, 0.35],
                                       goal_xy_offset=0.1, goal_z=[0.6, 1.0], disturbances="none"),
                    episodes=summaries)
    with (root / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/qwen_data")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--kd", type=float, default=0.5)
    collect(**vars(parser.parse_args()))
