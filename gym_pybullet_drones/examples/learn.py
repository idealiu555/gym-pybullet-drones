"""Train single-drone PPO or cooperative MAPPO, with deterministic evaluation."""

import argparse
from dataclasses import asdict
from datetime import datetime
from os import getpid
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


class _TrainingCallback(BaseCallback):
    """Report completed PPO updates through the shared training callback."""

    def __init__(self, report):
        super().__init__()
        self.report = report

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # The previous rollout's optimizer updates have now finished.
        self._report_update()

    def _on_training_end(self):
        self._report_update()

    def _report_update(self):
        metrics = {name: value for name, value in self.model.logger.name_to_value.items()
                   if name.startswith("train/")}
        self.report(self.model, metrics)


class _GpuProcessMetrics:
    """Read memory and SM utilization for this training process only."""

    def __init__(self, device):
        self.nvml = None
        if device.type == "cuda":
            import pynvml

            try:
                pynvml.nvmlInit()
                index = device.index if device.index is not None else torch.cuda.current_device()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except pynvml.NVMLError:
                return
            self.nvml = pynvml
            self.pid = getpid()
            self.last_seen_time = 0

    def read(self):
        if self.nvml is None:
            return {}
        metrics = {}
        try:
            processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            process = next((item for item in processes if item.pid == self.pid), None)
            if process is not None and process.usedGpuMemory is not None:
                metrics["system/gpu_process_memory_mib"] = process.usedGpuMemory / 2**20
            samples = self.nvml.nvmlDeviceGetProcessUtilization(self.handle, self.last_seen_time)
        except self.nvml.NVMLError:
            self.nvml = None
            return metrics
        if samples:
            self.last_seen_time = max(sample.timeStamp for sample in samples)
        sample = next((item for item in reversed(samples) if item.pid == self.pid), None)
        if sample is not None:
            metrics["system/gpu_process_sm_utilization"] = sample.smUtil
        return metrics


def run(multiagent=False, output_folder="results", gui=True, plot=True,
        colab=False, record_video=False, local=True,
        total_timesteps=None, act=None, seed=0, eval_freq=10000,
        rollout_steps=512, batch_size=256, epochs=5, device="cpu",
        target_positions=None, actor_type="mlp", model_path=None, actor_init=None,
        resume=None, update_microbatch_steps=None, backbone_dtype="float32",
        swanlab_project="gym-pybullet-drones", swanlab_workspace=None,
        swanlab_mode="online"):
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
    if total_timesteps < 1 or eval_freq < 0:
        raise ValueError("total_timesteps must be positive and eval_freq cannot be negative")
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
    swanlab_run = None
    previous_threads = torch.get_num_threads()
    try:
        # Avoid thread-pool overhead on these small fully connected networks.
        if actor_type == "mlp" and not (restored_model and restored_model.actor_config.actor_type == "qwen"):
            torch.set_num_threads(1)
        if eval_freq:
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
        gpu_metrics = _GpuProcessMetrics(model.device)
        import swanlab

        training_config = dict(multiagent=multiagent, actor_type=actor_type,
                               device=str(model.device), total_timesteps=total_timesteps,
                               rollout_steps=rollout_steps, batch_size=batch_size,
                               epochs=epochs, eval_freq=eval_freq, seed=seed)
        if multiagent:
            training_config.update(asdict(model.config))
            training_config.update(asdict(model.actor_config))
        swanlab_run = swanlab.init(
            project=swanlab_project,
            workspace=swanlab_workspace or None,
            experiment_name=f"{folder.parent.name}-{folder.name}",
            config=training_config,
            logdir=str(folder / "swanlab"),
            mode=swanlab_mode,
        )
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

        def report(model, metrics):
            payload = {(name if "/" in name else f"train/{name}"): float(value)
                       for name, value in metrics.items()}
            payload.update(gpu_metrics.read())
            if payload:
                swanlab_run.log(payload, step=model.num_timesteps)
            if eval_freq and model.num_timesteps - last_eval >= eval_freq:
                record_evaluation(model, metrics)

        def record_evaluation(model, metrics):
            nonlocal best_reward, last_eval
            result = evaluate(model, eval_env, seed=seed + 1000)
            last_eval = model.num_timesteps
            records.append(dict(timesteps=last_eval, **result))
            swanlab_run.log({f"eval/{name}": float(value) for name, value in result.items()},
                            step=last_eval)
            if result["reward"] > best_reward:
                best_reward = result["reward"]
                save(model, "best_model")
            print(f"steps={last_eval} evaluation={result} training={metrics}")

        if multiagent:
            model.learn(train_env, total_timesteps, callback=report)
        else:
            model.learn(total_timesteps, callback=_TrainingCallback(report))
        if eval_freq and last_eval != model.num_timesteps:
            record_evaluation(model, {})
        save(model, "final_model")
        if records:
            np.savez(folder / "evaluations.npz",
                     **{key: np.asarray([row[key] for row in records]) for key in records[0]})
    except BaseException as error:
        if swanlab_run is not None:
            swanlab_run.finish(state="crashed", error=str(error))
        raise
    else:
        swanlab_run.finish(state="success", error=None)
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()
        torch.set_num_threads(previous_threads)
    print(f"Models and evaluations saved to {folder}")
    if gui or plot or record_video:
        from gym_pybullet_drones.examples.play import play

        # Playback reloads a checkpoint; release the training actor and Adam first.
        used_cuda = model.device.type == "cuda"
        model = restored_model = actor = None
        if used_cuda:
            torch.cuda.empty_cache()
        model_name = "best_model" if records else "final_model"
        play(str(folder / f"{model_name}{extension}"), multiagent=multiagent,
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
    parser.add_argument("--swanlab_project", default="gym-pybullet-drones")
    parser.add_argument("--swanlab_workspace")
    parser.add_argument("--swanlab_mode", choices=["online", "local", "offline", "disabled"], default="online")
    run(**vars(parser.parse_args()))
