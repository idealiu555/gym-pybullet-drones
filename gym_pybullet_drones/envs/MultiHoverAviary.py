import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class MultiHoverAviary(BaseRLAviary):
    """Cooperative flight to individual targets followed by sustained hovering.

    Kinematic observations contain own state, own action history, and own
    target displacement (meters). Rewards are summed across drones. Success
    requires every drone to hover for hold_time seconds and does not end
    the episode: the policy must keep hovering until the time limit.
    """

    def __init__(self, drone_model=DroneModel.CF2X, num_drones=2,
                 neighbourhood_radius=np.inf, initial_xyzs=None, initial_rpys=None,
                 physics=Physics.PYB, pyb_freq=240, ctrl_freq=30, gui=False,
                 record=False, obs=ObservationType.KIN, act=ActionType.RPM,
                 target_positions=None, episode_len_sec=8, hold_time=1.0):
        """Initialize the cooperative task and base simulation settings.

        Parameters
        ----------
        target_positions : array_like, optional
            Individual XYZ targets in meters, shaped (num_drones, 3).
        episode_len_sec : float, optional
            Simulation time limit in seconds.
        hold_time : float, optional
            Required consecutive simultaneous hover duration in seconds.
        """
        if num_drones < 1 or not 0 < hold_time <= episode_len_sec:
            raise ValueError("Require num_drones >= 1 and 0 < hold_time <= episode_len_sec")
        if target_positions is not None:
            target_positions = np.asarray(target_positions, dtype=float)
            if (target_positions.shape != (num_drones, 3)
                    or not np.isfinite(target_positions).all()
                    or np.any(target_positions[:, 2] <= 0)):
                raise ValueError("target_positions must be finite (num_drones, 3) with z > 0")
        self.EPISODE_LEN_SEC = episode_len_sec
        self.HOLD_TIME = hold_time
        self._hover_steps = 0
        if initial_xyzs is not None:
            initial_xyzs = np.asarray(initial_xyzs, dtype=float)
        super().__init__(drone_model=drone_model, num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs, initial_rpys=initial_rpys,
                         physics=physics, pyb_freq=pyb_freq, ctrl_freq=ctrl_freq,
                         gui=gui, record=record, obs=obs, act=act)
        self.TARGET_POS = (self.INIT_XYZS + np.array([
            [0, 0, 1 / (i + 1)] for i in range(num_drones)
        ]) if target_positions is None else target_positions.copy())

    def _observationSpace(self):
        space = super()._observationSpace()
        if self.OBS_TYPE in (ObservationType.KIN, ObservationType.ALL):
            kin = space if self.OBS_TYPE == ObservationType.KIN else space["kin"]
            goal_bounds = np.full((self.NUM_DRONES, 3), np.inf, dtype=np.float32)
            kin = spaces.Box(np.hstack([kin.low, -goal_bounds]),
                             np.hstack([kin.high, goal_bounds]), dtype=np.float32)
            if self.OBS_TYPE == ObservationType.KIN:
                return kin
            space["kin"] = kin
        return space

    def _computeObs(self):
        obs = super()._computeObs()
        if self.OBS_TYPE in (ObservationType.KIN, ObservationType.ALL):
            kin = obs if self.OBS_TYPE == ObservationType.KIN else obs["kin"]
            kin = np.hstack([kin, self.TARGET_POS - self.pos]).astype(np.float32)
            if self.OBS_TYPE == ObservationType.KIN:
                return kin
            obs["kin"] = kin
        return obs

    def reset(self, seed=None, options=None):
        """Reset the consecutive hover counter and base simulation state."""
        self._hover_steps = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        """Advance all drones and measure consecutive simultaneous hovering."""
        obs, reward, terminated, truncated, info = super().step(action)
        self._hover_steps = self._hover_steps + 1 if info["hovering"].all() and not terminated else 0
        info["hover_time"] = self._hover_steps / self.CTRL_FREQ
        info["is_success"] = info["hover_time"] >= self.HOLD_TIME
        return obs, reward, terminated, truncated, info

    def _hovering(self):
        return ((np.linalg.norm(self.TARGET_POS - self.pos, axis=1) <= 0.05)
                & (np.linalg.norm(self.vel, axis=1) <= 0.1)
                & (np.linalg.norm(self.ang_v, axis=1) <= 0.2)
                & (np.max(np.abs(self.rpy[:, :2]), axis=1) <= 0.1))

    def _computeReward(self):
        # Positive dense rewards favor earlier arrival and continued survival.
        # Velocity cost and the hover bonus discourage flying through targets.
        distance = np.linalg.norm(self.TARGET_POS - self.pos, axis=1)
        rewards = np.exp(-2 * distance - 0.25 * np.sum(self.vel**2, axis=1))
        rewards += 0.5 * self._hovering()
        if self._computeTerminated():
            rewards -= 5.0
        return float(rewards.sum())

    def _computeTerminated(self):
        # Failure is terminal; unlike a time limit, it must not bootstrap.
        return bool(np.any(np.linalg.norm(self.pos - self.TARGET_POS, axis=1) > 3.0)
                    or np.any(self.pos[:, 2] < 0.02)
                    or np.any(np.abs(self.rpy[:, :2]) > 0.4))

    def _computeTruncated(self):
        # BaseAviary increments its counter AFTER computing this flag.
        return (self.step_counter + self.PYB_STEPS_PER_CTRL) / self.PYB_FREQ >= self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {"distance": np.linalg.norm(self.TARGET_POS - self.pos, axis=1),
                "hovering": self._hovering(),
                "hover_time": self._hover_steps / self.CTRL_FREQ,
                "is_success": self._hover_steps / self.CTRL_FREQ >= self.HOLD_TIME}
