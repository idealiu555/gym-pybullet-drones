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
from gym_pybullet_drones.learning.actors import ActorConfig
from gym_pybullet_drones.learning.observation_prompt import ObservationSpec
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
        colab=False, record_video=False, local=True,
        total_timesteps=None, act=None, seed=0, eval_freq=10000,
        rollout_steps=512, batch_size=256, epochs=5, device="cpu",
        target_positions=None, actor_type="mlp", model_path=None, actor_init=None,
        resume=None, update_microbatch_steps=None, backbone_dtype="float32"):
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
        Individual MAPPO targets, shaped (10, 3), in meters.
    """
    total_timesteps = total_timesteps if total_timesteps is not None else (1000000 if local else 128)
    if actor_init and resume:
        raise ValueError("actor_init and resume are mutually exclusive")
    if not multiagent and (actor_type != "mlp" or model_path or actor_init or resume
                           or update_microbatch_steps is not None or backbone_dtype != "float32"):
        raise ValueError("Qwen and MAPPO options require multiagent=True")
    if actor_init and actor_type != "qwen":
        raise ValueError("actor_init requires actor_type=qwen")
    if actor_type == "mlp" and not resume and (model_path or backbone_dtype != "float32"):
        raise ValueError("Model path and dtype apply only to Qwen")
    if total_timesteps < 1 or eval_freq < 1:
        raise ValueError("total_timesteps and eval_freq must be positive")
    action_type = ActionType(act) if act is not None else (ActionType.VEL if multiagent else ActionType.ONE_D_RPM)
    folder = Path(output_folder) / datetime.now().strftime("save-%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(obs=ObservationType.KIN, act=action_type)
    env_class = MultiHoverAviary if multiagent else HoverAviary
    if multiagent:
        env_kwargs["target_positions"] = target_positions
    restored_model = None
    if resume:
        if target_positions is not None or act is not None:
            raise ValueError("Resume restores environment settings; do not override act or targets")
        restored_model, saved_metadata = MAPPO.load(resume, device, model_path=model_path)
        env_kwargs = dict(saved_metadata, obs=ObservationType.KIN)
        action_type = env_kwargs["act"] = ActionType(env_kwargs["act"])
    train_env = env_class(**env_kwargs)
    eval_env = None
    previous_threads = torch.get_num_threads()
    try:
        # Avoid thread-pool overhead on these small fully connected networks.
        if actor_type == "mlp" and not (restored_model and restored_model.actor_config.actor_type == "qwen"):
            torch.set_num_threads(1)
        eval_env = env_class(**env_kwargs)
        metadata = {"act": action_type.value}
        if multiagent:
            metadata.update(target_positions=train_env.TARGET_POS.tolist(),
                            initial_xyzs=train_env.INIT_XYZS.tolist(),
                            ctrl_freq=train_env.CTRL_FREQ, pyb_freq=train_env.PYB_FREQ,
                            episode_len_sec=train_env.EPISODE_LEN_SEC, hold_time=train_env.HOLD_TIME)
            if restored_model is not None:
                model = restored_model
            else:
                spec = ObservationSpec.from_env(train_env) if actor_type == "qwen" else None
                actor_config = ActorConfig(actor_type=actor_type, model_path=model_path,
                                           backbone_dtype=backbone_dtype)
                actor = None
                if actor_init:
                    from gym_pybullet_drones.learning.actor_checkpoint import load_actor
                    actor, initial_std, _, _ = load_actor(actor_init, device, model_path, spec)
                    actor_config = actor.config
                model = MAPPO(train_env.observation_space, train_env.action_space,
                              MAPPOConfig(rollout_steps=rollout_steps, batch_size=batch_size,
                                          epochs=epochs, seed=seed,
                                          update_microbatch_steps=update_microbatch_steps),
                              device=device, actor_config=actor_config,
                              observation_spec=spec, actor=actor)
                if actor_init:
                    with torch.no_grad():
                        model.log_std.copy_(initial_std)
        else:
            model = PPO("MlpPolicy", train_env, n_steps=rollout_steps,
                        batch_size=batch_size, n_epochs=epochs, seed=seed, device=device)
        extension = ("" if model.actor_config.actor_type == "qwen" else ".pt") if multiagent else ".zip"
        records = []
        best_reward = -np.inf
        last_eval = 0

        def save(model, name):
            path = folder / f"{name}{extension}"
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

        # Playback reloads the best checkpoint; release the training actor and Adam first.
        used_cuda = model.device.type == "cuda"
        model = restored_model = actor = None
        if used_cuda:
            torch.cuda.empty_cache()
        play(str(folder / f"best_model{extension}"), multiagent=multiagent,
             gui=gui, plot=plot, record_video=record_video, act=action_type,
             output_folder=output_folder, colab=colab, device=device)
    return str(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiagent", type=str2bool, default=False)
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
    parser.add_argument("--actor_type", choices=["mlp", "qwen"], default="mlp")
    parser.add_argument("--model_path")
    parser.add_argument("--actor_init")
    parser.add_argument("--resume")
    parser.add_argument("--update_microbatch_steps", type=int)
    parser.add_argument("--backbone_dtype", choices=["float32", "bfloat16"], default="float32")
    run(**vars(parser.parse_args()))
