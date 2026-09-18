"""Train single-drone PPO or cooperative MAPPO, with deterministic evaluation."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.learning import MAPPO, MAPPOConfig
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import str2bool


def evaluate(model, env, seed=42):
    """Evaluate one full episode, including continued hovering after arrival.

    Returns
    -------
    dict
        Episode return; multi-agent environments additionally report final
        success, worst distance in meters, and consecutive hover seconds.
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            result = {"reward": total_reward}
            if "is_success" in info:
                result.update(success=bool(info["is_success"]) and not terminated,
                              max_distance=float(np.max(info["distance"])),
                              hover_time=float(info["hover_time"]))
            return result


class _EvaluationCallback(BaseCallback):
    """Adapt the shared evaluation function to SB3's callback interface."""

    def __init__(self, evaluate_step):
        super().__init__()
        self.evaluate_step = evaluate_step

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # The previous rollout's optimizer updates have now finished.
        self.evaluate_step(self.model, {})


def run(multiagent=False, output_folder="results", gui=True, plot=True,
        colab=False, record_video=False, local=True, num_drones=2,
        total_timesteps=None, act=None, seed=0, eval_freq=10000,
        rollout_steps=512, batch_size=256, epochs=5, device="cpu",
        target_positions=None):
    """Train and save best/final models; return the run directory.

    Parameters
    ----------
    multiagent : bool
        Select MAPPO with local actors and a global critic; otherwise use PPO.
    act : ActionType or str, optional
        Defaults to full 3D velocity control for MAPPO, vertical RPM for PPO.
    total_timesteps : int, optional
        Joint environment steps. Defaults to 1,000,000 (128 if local=False).
    target_positions : array_like, optional
        Individual MAPPO targets, shaped (num_drones, 3), in meters.
    """
    total_timesteps = total_timesteps if total_timesteps is not None else (1000000 if local else 128)
    if total_timesteps < 1 or eval_freq < 1:
        raise ValueError("total_timesteps and eval_freq must be positive")
    action_type = ActionType(act) if act is not None else (ActionType.VEL if multiagent else ActionType.ONE_D_RPM)
    folder = Path(output_folder) / datetime.now().strftime("save-%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(obs=ObservationType.KIN, act=action_type)
    env_class = MultiHoverAviary if multiagent else HoverAviary
    if multiagent:
        env_kwargs.update(num_drones=num_drones, target_positions=target_positions)
    train_env = env_class(**env_kwargs)
    eval_env = None
    previous_threads = torch.get_num_threads()
    try:
        # Avoid thread-pool overhead on these small fully connected networks.
        torch.set_num_threads(1)
        eval_env = env_class(**env_kwargs)
        metadata = {"act": action_type.value}
        if multiagent:
            metadata.update(num_drones=num_drones, target_positions=train_env.TARGET_POS.tolist(),
                            initial_xyzs=train_env.INIT_XYZS.tolist(),
                            ctrl_freq=train_env.CTRL_FREQ, pyb_freq=train_env.PYB_FREQ,
                            episode_len_sec=train_env.EPISODE_LEN_SEC, hold_time=train_env.HOLD_TIME)
            model = MAPPO(train_env.observation_space, train_env.action_space,
                          MAPPOConfig(rollout_steps=rollout_steps, batch_size=batch_size,
                                      epochs=epochs, seed=seed), device=device)
        else:
            model = PPO("MlpPolicy", train_env, n_steps=rollout_steps,
                        batch_size=batch_size, n_epochs=epochs, seed=seed, device=device)
        extension = "pt" if multiagent else "zip"
        records = []
        best_reward = -np.inf
        last_eval = 0

        def save(model, name):
            path = folder / f"{name}.{extension}"
            if multiagent:
                model.save(path, metadata=metadata)
            else:
                model.save(path)

        def evaluate_step(model, metrics):
            if model.num_timesteps - last_eval < eval_freq:
                return
            record_evaluation(model, metrics)

        def record_evaluation(model, metrics):
            nonlocal best_reward, last_eval
            result = evaluate(model, eval_env, seed=seed + 1000)
            last_eval = model.num_timesteps
            records.append(dict(timesteps=last_eval, **result))
            if result["reward"] > best_reward:
                best_reward = result["reward"]
                save(model, "best_model")
            print(f"steps={last_eval} evaluation={result} training={metrics}")

        if multiagent:
            model.learn(train_env, total_timesteps, callback=evaluate_step)
        else:
            model.learn(total_timesteps, callback=_EvaluationCallback(evaluate_step))
        if last_eval != model.num_timesteps:
            record_evaluation(model, {})
        save(model, "final_model")
        np.savez(folder / "evaluations.npz",
                 **{key: np.asarray([row[key] for row in records]) for key in records[0]})
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()
        torch.set_num_threads(previous_threads)
    print(f"Models and evaluations saved to {folder}")
    if gui or plot or record_video:
        from gym_pybullet_drones.examples.play import play

        play(str(folder / f"best_model.{extension}"), multiagent=multiagent,
             gui=gui, plot=plot, record_video=record_video, act=action_type,
             output_folder=output_folder, colab=colab)
    return str(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiagent", type=str2bool, default=False)
    parser.add_argument("--num_drones", type=int, default=2)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--gui", type=str2bool, default=True)
    parser.add_argument("--plot", type=str2bool, default=True)
    parser.add_argument("--record_video", type=str2bool, default=False)
    parser.add_argument("--output_folder", default="results")
    parser.add_argument("--colab", type=str2bool, default=False)
    parser.add_argument("--act", choices=[a.value for a in ActionType])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    run(**vars(parser.parse_args()))
