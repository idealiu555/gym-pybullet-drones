"""Regression tests for CTDE, timeout semantics and the training entry point."""

from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from gym_pybullet_drones.learning import MAPPO, MAPPOConfig
from gym_pybullet_drones.learning.mappo import compute_gae


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    policy = MAPPO(spaces.Box(-np.inf, np.inf, (2, 3)),
                   spaces.Box(-1, 1, (2, 2)),
                   MAPPOConfig(rollout_steps=4, batch_size=2, epochs=2, hidden_size=16))
    yield policy
    torch.set_num_threads(previous)


def test_gae_bootstraps_timeout_but_stops_trace():
    advantages, returns = compute_gae(
        np.array([1., 2., 100.]), np.array([0.5, 1., 3.]), np.array([1., 10., 99.]),
        np.array([False, False, True]), np.array([False, True, False]), 0.9, 0.8)
    # At timeout: 2 + .9 * 10 - 1 = 10; the following episode must not leak in.
    np.testing.assert_allclose(advantages, [8.6, 10., 97.], rtol=1e-6)
    np.testing.assert_allclose(returns, [9.1, 11., 100.], rtol=1e-6)


def test_actor_is_local_and_critic_is_global(model):
    obs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    original, _ = model.predict(obs)
    obs[1] *= -10
    changed, _ = model.predict(obs)
    np.testing.assert_array_equal(original[0], changed[0])
    single, _ = model.predict(obs[0])
    np.testing.assert_allclose(original[0], single, atol=1e-7)
    state = torch.ones(6, requires_grad=True)
    model.critic(state).backward()
    assert state.grad[:3].abs().sum() > 0
    assert state.grad[3:].abs().sum() > 0


def test_squashed_actions_and_log_probs_are_finite(model):
    from torch.distributions import Independent, TransformedDistribution, TanhTransform

    obs = torch.zeros((2, 3))
    distribution = model._distribution(obs)
    raw = torch.tensor([[0.2, -0.4], [0.6, -0.8]])
    reference = Independent(TransformedDistribution(distribution, [TanhTransform()]), 1)
    torch.testing.assert_close(model._log_prob(distribution, raw), reference.log_prob(raw.tanh()))
    assert torch.isfinite(model._log_prob(distribution, raw * 1000)).all()
    action, _ = model.predict(np.zeros((2, 3)), deterministic=False)
    assert action.shape == (2, 2)
    assert np.isfinite(action).all() and (np.abs(action) <= 1).all()


def test_rollout_uses_terminal_observation_before_reset(model):
    class TimeoutEnv:
        def step(self, action):
            return np.full((2, 3), 10.), 2., False, True, {}

        def reset(self):
            return np.zeros((2, 3)), {}

    model._value = lambda obs: float(obs[0, 0])
    _, batch = model._collect_rollout(TimeoutEnv(), np.zeros((2, 3)), 2)
    np.testing.assert_allclose(batch[-1].numpy(), [10.9, 10.9])


def test_update_and_checkpoint(model, tmp_path):
    class SimpleEnv:
        observation_space = spaces.Box(-np.inf, np.inf, (2, 3))
        action_space = spaces.Box(-1, 1, (2, 2))

        def reset(self, seed=None):
            self.t = 0
            return np.zeros((2, 3), dtype=np.float32), {}

        def step(self, action):
            self.t += 1
            return np.full((2, 3), self.t, dtype=np.float32), float(self.t), False, self.t == 3, {}

    actor_before = [p.detach().clone() for p in model.actor.parameters()]
    critic_before = [p.detach().clone() for p in model.critic.parameters()]
    metrics = []
    model.learn(SimpleEnv(), 7, callback=lambda _, item: metrics.append(item))
    assert model.num_timesteps == 7
    assert all(np.isfinite(list(item.values())).all() for item in metrics)
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, model.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, model.critic.parameters()))
    path = tmp_path / "model.pt"
    model.save(path, metadata={"act": "vel"})
    restored, metadata = MAPPO.load(path)
    assert metadata == {"act": "vel"}
    assert restored.num_timesteps == 7
    np.testing.assert_array_equal(model.predict(np.ones((2, 3)))[0],
                                  restored.predict(np.ones((2, 3)))[0])
    restored.learn(SimpleEnv(), 1)
    assert restored.num_timesteps == 8
    restored.save(path)
    _, metadata = MAPPO.load(path)
    assert metadata == {"act": "vel"}


def test_single_step_update_keeps_policy_gradient(model):
    model.config.entropy_coef = 0
    obs = torch.ones((1, 2, 3))
    with torch.no_grad():
        distribution = model._distribution(obs)
        raw = distribution.mean + 0.5
        log_probs = model._log_prob(distribution, raw)
        values = model.critic(obs.flatten(start_dim=1)).squeeze(-1)
    before = [p.detach().clone() for p in model.actor.parameters()]
    model._update((obs, raw, log_probs, values, torch.ones(1), values + 1))
    assert any(not torch.equal(a, b) for a, b in zip(before, model.actor.parameters()))


@pytest.mark.parametrize("multiagent", [False, True])
def test_rl_reset_reproduces_pid_trajectory(multiagent):
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
    from gym_pybullet_drones.utils.enums import ActionType

    env = (MultiHoverAviary if multiagent else HoverAviary)(act=ActionType.VEL)
    try:
        initial, _ = env.reset(seed=0)
        action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        for _ in range(10):
            first, *_ = env.step(action)
        reset, _ = env.reset(seed=0)
        np.testing.assert_array_equal(initial, reset)
        for _ in range(10):
            second, *_ = env.step(action)
        np.testing.assert_allclose(first, second, atol=1e-6)
    finally:
        env.close()


def test_single_agent_evaluation_omits_unavailable_metrics():
    from types import SimpleNamespace

    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.examples.learn import evaluate
    from gym_pybullet_drones.utils.enums import ActionType

    env = HoverAviary(initial_xyzs=np.array([[0., 0., 1.]]), act=ActionType.ONE_D_RPM)
    policy = SimpleNamespace(predict=lambda obs, deterministic: (np.zeros((1, 1)), None))
    try:
        result = evaluate(policy, env)
        assert result == {"reward": pytest.approx(2.)}
    finally:
        env.close()


def test_ppo_final_evaluation_follows_update(monkeypatch):
    from gym_pybullet_drones.examples import learn

    def evaluate_updates(model, env, seed):
        return {"reward": float(model._n_updates), "success": False,
                "max_distance": 0., "hover_time": 0.}

    monkeypatch.setattr(learn, "evaluate", evaluate_updates)
    folder = Path(learn.run(gui=False, plot=False, output_folder="tmp",
                            total_timesteps=16, rollout_steps=8, batch_size=4,
                            epochs=1, eval_freq=8))
    with np.load(folder / "evaluations.npz") as data:
        assert data["timesteps"].tolist() == [8, 16]
        assert data["reward"].tolist() == [1., 2.]
    best = learn.PPO.load(folder / "best_model.zip")
    final = learn.PPO.load(folder / "final_model.zip")
    for a, b in zip(best.policy.parameters(), final.policy.parameters()):
        torch.testing.assert_close(a, b)


def test_single_agent_playback_reaches_environment_boundary(monkeypatch):
    from types import SimpleNamespace

    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.examples import play

    instances = []

    class TrackedHover(HoverAviary):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ended = False
            instances.append(self)

        def step(self, action):
            transition = super().step(action)
            self.ended = transition[2] or transition[3]
            return transition

    policy = SimpleNamespace(predict=lambda obs, deterministic: (np.zeros((1, 1)), None))
    monkeypatch.setattr(play.PPO, "load", lambda path: policy)
    monkeypatch.setattr(play, "HoverAviary", TrackedHover)
    play.play(gui=False, plot=False)
    assert instances[0].ended


def test_multihover_goal_reset_and_time_limit():
    from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
    from gym_pybullet_drones.utils.enums import ActionType

    env = MultiHoverAviary(num_drones=3, act=ActionType.VEL,
                          episode_len_sec=0.1, hold_time=0.1)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        np.testing.assert_allclose(obs[:, -3:], env.TARGET_POS - env.pos)
        for step in range(3):
            _, _, terminated, truncated, _ = env.step(np.zeros((3, 4), dtype=np.float32))
            assert not terminated
            assert truncated == (step == 2)
        env.step(np.full((3, 4), 0.1, dtype=np.float32))
        reset_obs, info = env.reset(seed=0)
        np.testing.assert_array_equal(obs, reset_obs)
        assert info["hover_time"] == 0
        assert all(np.all(controller.integral_pos_e == 0) for controller in env.ctrl)
    finally:
        env.close()


def test_hover_requires_all_drones_and_continues_after_success():
    import pybullet as p
    from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
    from gym_pybullet_drones.utils.enums import ActionType

    initial = np.array([[0., 0., 1.], [0.5, 0., 1.]])
    env = MultiHoverAviary(initial_xyzs=initial, target_positions=initial,
                          act=ActionType.ONE_D_RPM, hold_time=0.1)
    try:
        env.reset()
        for step in range(4):
            _, reward, terminated, truncated, info = env.step(np.zeros((2, 1)))
            assert not terminated and not truncated
            assert info["is_success"] == (step >= 2)
        assert reward == pytest.approx(3.)
        p.resetBaseVelocity(env.DRONE_IDS[1], linearVelocity=[0, 0, 1], physicsClientId=env.CLIENT)
        _, _, _, _, info = env.step(np.zeros((2, 1)))
        assert not info["is_success"] and info["hover_time"] == 0
        p.resetBasePositionAndOrientation(env.DRONE_IDS[1], [0.5, 0, 0.01],
                                         [0, 0, 0, 1], physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        assert env._computeTerminated()
        assert not env._computeTruncated()
        assert env._computeReward() < 0
    finally:
        env.close()


def test_mappo_training_and_playback():
    from gym_pybullet_drones.examples.learn import run
    from gym_pybullet_drones.examples.play import play

    folder = Path(run(multiagent=True, num_drones=3, gui=False, plot=False,
                      output_folder="tmp", total_timesteps=17, rollout_steps=8,
                      batch_size=4, epochs=1, eval_freq=8))
    assert (folder / "best_model.pt").is_file()
    assert (folder / "final_model.pt").is_file()
    with np.load(folder / "evaluations.npz") as data:
        assert data["timesteps"].tolist() == [8, 16, 17]
        assert np.isfinite(data["reward"]).all()
    result = play(str(folder / "final_model.pt"), multiagent=True, gui=False, plot=False)
    assert result["distance"].shape == (3,)
