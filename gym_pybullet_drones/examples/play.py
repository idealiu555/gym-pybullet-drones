"""Replay a saved PPO or MAPPO policy."""

import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.learning import MAPPO
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.Logger import Logger


def play(model_path="results/best_model.zip", multiagent=False, gui=True,
         plot=True, record_video=False, act=ActionType.ONE_D_RPM,
         output_folder="results", colab=False):
    """Replay one episode; MAPPO checkpoints restore their environment settings."""
    if multiagent:
        model, metadata = MAPPO.load(model_path)
        metadata["act"] = ActionType(metadata["act"])
        env = MultiHoverAviary(gui=gui, record=record_video, **metadata)
    else:
        model = PPO.load(model_path)
        env = HoverAviary(gui=gui, record=record_video, obs=ObservationType.KIN,
                         act=ActionType(act))
    try:
        logger = Logger(logging_freq_hz=env.CTRL_FREQ, num_drones=env.NUM_DRONES,
                        output_folder=output_folder, colab=colab) if plot else None
        obs, _ = env.reset(seed=42)
        start = time.time()
        i = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if logger is not None:
                for drone in range(env.NUM_DRONES):
                    logger.log(drone=drone, timestamp=(i + 1) / env.CTRL_FREQ,
                               state=env._getDroneStateVector(drone),
                               control=np.zeros(12))
            if gui:
                env.render()
                sync(i, start, env.CTRL_TIMESTEP)
            if terminated or truncated:
                break
            i += 1
        print(f"Episode complete: terminated={terminated}, truncated={truncated}, info={info}")
        if logger is not None:
            logger.plot()
        return info
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="results/best_model.zip")
    parser.add_argument("--multiagent", type=str2bool, default=False)
    parser.add_argument("--gui", type=str2bool, default=True)
    parser.add_argument("--plot", type=str2bool, default=True)
    parser.add_argument("--record_video", type=str2bool, default=False)
    parser.add_argument("--act", choices=[a.value for a in ActionType], default="one_d_rpm")
    play(**vars(parser.parse_args()))
