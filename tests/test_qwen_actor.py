"""Tiny CPU substitutes only: no Transformers, downloaded weights or flight training."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from gym_pybullet_drones.learning.actors import ActorConfig
from gym_pybullet_drones.learning.observation_prompt import ObservationSpec, ObservationPromptEncoder
from gym_pybullet_drones.learning.qwen_actor import QwenActor
from gym_pybullet_drones.learning.mappo import MAPPO, MAPPOConfig
from gym_pybullet_drones.learning.actor_checkpoint import ActorPolicy, load_actor, save_actor
from gym_pybullet_drones.examples.qwen_action.collect import expert_action
from gym_pybullet_drones.examples.qwen_action.sft import ActionDataset, action_loss


class TinyTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __init__(self):
        self.calls = 0

    def __call__(self, prompts, **kwargs):
        self.calls += 1
        assert kwargs == dict(padding=False, truncation=False, add_special_tokens=True)
        return {"input_ids": [[ord(c) % 127 + 1 for c in p] for p in prompts]}

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text('{}', encoding="utf-8")


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embed_tokens = nn.Embedding(128, 8)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Dropout(.5))
                                     for _ in range(4)])
        self.norm = nn.LayerNorm(8)
        self.calls = 0

    def forward(self, input_ids, attention_mask, position_ids, use_cache, return_dict):
        self.calls += 1
        assert not use_cache and return_dict
        assert torch.equal(position_ids, (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0))
        hidden = self.embed_tokens(input_ids)
        # Causal mixing makes the last hidden depend on the full local observation.
        hidden = (hidden * attention_mask[..., None]).cumsum(1) / (position_ids[..., None] + 1)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_actor(spec=None, backbone_dtype="float32"):
    torch.manual_seed(42)
    return QwenActor(TinyBackbone().to(getattr(torch, backbone_dtype)), TinyTokenizer(),
                     ActorConfig(actor_type="qwen", backbone_dtype=backbone_dtype),
                     spec or ObservationSpec(), {"fake": "test-only"})


def tiny_mappo(actor=None, micro=1):
    actor = actor or tiny_actor()
    spec = actor.spec
    return MAPPO(spaces.Box(-np.inf, np.inf, (10, spec.obs_dim)),
                 spaces.Box(-1, 1, (10, spec.action_dim)),
                 MAPPOConfig(epochs=1, batch_size=3, hidden_size=8, entropy_coef=0,
                             update_microbatch_steps=micro),
                 actor_config=actor.config, observation_spec=spec, actor=actor)


@pytest.mark.parametrize("kind,dim", [("vel", 4), ("rpm", 4), ("pid", 3),
                                        ("one_d_rpm", 1), ("one_d_pid", 1)])
def test_prompt_complete_and_dynamic_head(kind, dim):
    spec = ObservationSpec(action_type=kind, action_dim=dim)
    actor = tiny_actor(spec)
    rows = np.zeros((10, spec.obs_dim), dtype=np.float32)
    rows[:, 0] = -.00001
    rows[:, 12:-3] = np.arange(spec.history_length * dim) / 100
    prompts = actor.encoder.prompts(rows)
    assert "-0.000" not in prompts[0]
    assert prompts[0].count(";") >= spec.history_length - 1
    tokens = actor.encode(rows)
    assert actor.encoder.tokenizer.calls == 1
    assert all(t.dtype == torch.int32 and t.device.type == "cpu" for t in tokens)
    assert actor(tokens).shape == (10, dim)
    assert actor.backbone.calls == 1


def test_prompt_limits_and_invalid_input():
    tokenizer = TinyTokenizer()
    spec = ObservationSpec()
    encoder = ObservationPromptEncoder(spec, tokenizer)
    obs = np.zeros((1, spec.obs_dim))
    with pytest.raises(ValueError, match="NaN"):
        encoder(obs + np.nan)
    with pytest.raises(ValueError, match="characters"):
        ObservationPromptEncoder(spec, tokenizer, max_chars=10)(obs)
    with pytest.raises(ValueError, match="token length"):
        ObservationPromptEncoder(spec, tokenizer, max_tokens=10)(obs)
    with pytest.raises(ValueError, match="Expected"):
        encoder(obs[:, :-1])


def test_padding_locality_and_trainable_suffix():
    actor = tiny_actor()
    actor.train()
    assert not actor.training and not actor.backbone.training
    rows = np.zeros((10, actor.spec.obs_dim), dtype=np.float32)
    rows[1] = -1
    tokens = actor.encode(rows)
    batch = actor(tokens)
    individual = torch.cat([actor([t]) for t in tokens])
    torch.testing.assert_close(batch, individual, atol=1e-7, rtol=1e-6)
    old = batch[0].detach().clone()
    tokens[1] = torch.tensor([0, 1, 2], dtype=torch.int32)  # valid pad ID must not be masked away
    torch.testing.assert_close(actor(tokens)[0], old)
    loss = action_loss(actor, rows, torch.ones((10, 4)))
    loss.mean().backward()
    for name, parameter in actor.named_parameters():
        trainable = name.startswith(("score.", "backbone.layers.2.", "backbone.layers.3."))
        assert parameter.requires_grad == trainable
        assert (parameter.grad is not None) == trainable
    assert any(p.grad.abs().sum() > 0 for p in actor.backbone.layers[-1].parameters())


def test_rollout_cached_tokens_ratio_and_microbatch():
    model = tiny_mappo()
    class FakeEnv:
        def step(self, action):
            assert action.shape == (10, 4)
            return np.ones((10, 75), dtype=np.float32), 1., False, False, {}

    _, batch = model._collect_rollout(FakeEnv(), np.zeros((10, 75)), 3)
    assert model.actor.backbone.calls == 3
    calls = model.actor.encoder.tokenizer.calls
    obs, raw, old_log, *_ = batch
    with torch.no_grad():
        distribution = model._distribution(obs, [s for step in batch[6] for s in step])
        torch.testing.assert_close(model._log_prob(distribution, raw), old_log, atol=1e-6, rtol=1e-6)
    steps = []
    handle = model.optimizer.register_step_post_hook(lambda *args: steps.append(1))
    model._update(batch)
    handle.remove()
    assert len(steps) == 1
    assert model.actor.encoder.tokenizer.calls == calls
    assert model.actor.backbone.calls == 7


def test_microbatch_weighting_matches_full_batch():
    first, second = tiny_mappo(micro=2), tiny_mappo(micro=3)
    observations = torch.zeros((3, 10, 75))
    observations[1] = .5
    with torch.no_grad():
        dist = first._distribution(observations)
        raw = dist.mean + .3
        log = first._log_prob(dist, raw)
        values = first.critic(observations.flatten(start_dim=1)).squeeze(-1)
    tokens = [first.actor.encode(o) for o in observations]
    batch = (observations, raw, log, values, torch.tensor([1., 2., 4.]), values + 1, tokens)
    first._update(batch)
    second._update(batch)
    for a, b in zip(first.actor.parameters(), second.actor.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(first.log_std, second.log_std)


def install_fake_loader(monkeypatch):
    def loader(config, spec, device, tokenizer_path=None, expected_identity=None):
        actor = tiny_actor(spec, config.backbone_dtype)
        actor.config = config
        assert expected_identity == actor.base_identity
        return actor
    monkeypatch.setattr("gym_pybullet_drones.learning.qwen_actor.load_qwen_actor", loader)


@pytest.mark.parametrize("backbone_dtype", ["float32", "bfloat16"])
def test_sft_mappo_inference_checkpoint_transfer(tmp_path, monkeypatch, backbone_dtype):
    install_fake_loader(monkeypatch)
    actor = tiny_actor(backbone_dtype=backbone_dtype)
    rows = np.zeros((10, 75), dtype=np.float32)
    optimizer = torch.optim.Adam(actor.parameter_groups())
    action_loss(actor, rows, torch.ones((10, 4))).mean().backward()
    optimizer.step()  # synthetic backward only, no real model training
    std = torch.full((4,), -.5)
    save_actor(tmp_path / "sft", actor, std, "sft")
    loaded, loaded_std, manifest, remaining = load_actor(tmp_path / "sft")
    assert remaining == {}
    assert manifest["stage"] == "sft"
    model = tiny_mappo(loaded)
    np.testing.assert_allclose(model.predict(rows)[0], ActorPolicy(actor, std).predict(rows)[0])
    np.testing.assert_allclose(model.predict(rows)[0], ActorPolicy(loaded, loaded_std).predict(rows)[0])
    model.save(tmp_path / "rl", {"act": "vel"})
    _, _, _, remaining = load_actor(tmp_path / "rl")
    assert set(remaining) == {"critic"}
    assert all(p.device.type == "cpu" for p in remaining["critic"].values())
    restored, _ = MAPPO.load(tmp_path / "rl")
    np.testing.assert_array_equal(restored.predict(rows)[0], model.predict(rows)[0])
    assert restored.rng.bit_generator.state == model.rng.bit_generator.state
    payload = torch.load(tmp_path / "rl" / "trainable.pt", weights_only=True)
    assert all(p.dtype == torch.float32 for p in payload["actor"].values())
    with pytest.raises(ValueError, match="semantics"):
        load_actor(tmp_path / "sft", expected_spec=ObservationSpec(action_type="rpm"))
    payload = torch.load(tmp_path / "sft" / "trainable.pt", weights_only=True)
    payload["actor"].pop(next(iter(payload["actor"])))
    torch.save(payload, tmp_path / "sft" / "trainable.pt")
    with pytest.raises(ValueError, match="keys"):
        load_actor(tmp_path / "sft")


def test_dataset_split_and_expert_mapping(tmp_path):
    spec = ObservationSpec()
    records = [dict(episode_id=ep, step=0, agent_index=i, observation=[0.] * 75,
                    action=[0., 0., 0., 1.]) for ep in range(4) for i in range(10)]
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(dict(
        schema_version=1, observation_spec=asdict(spec), episode_split_seed=42,
        action_type="vel", action_dim=4, ctrl_freq=30)), encoding="utf-8")
    (tmp_path / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    data = ActionDataset(tmp_path)
    train, val = data.split(seed=42)
    assert {data.records[i][0] for i in train.indices}.isdisjoint({data.records[i][0] for i in val.indices})
    observations = np.zeros((10, 75))
    observations[:, -3:] = [1., 2., 3.]
    action = expert_action(observations, .2)
    reconstructed = .2 * action[:, :3] * action[:, 3:]
    np.testing.assert_allclose(reconstructed, np.tile(np.array([1, 2, 3]) / np.sqrt(14) * .2, (10, 1)), rtol=1e-6)
    np.testing.assert_array_equal(expert_action(np.zeros((10, 75)), .2), np.zeros((10, 4)))


def test_checkpoint_fingerprint_and_shape_errors(tmp_path, monkeypatch):
    from gym_pybullet_drones.learning.actor_checkpoint import base_fingerprint
    from gym_pybullet_drones.learning.qwen_actor import load_qwen_actor

    with pytest.raises(ValueError, match="local"):
        load_qwen_actor(ActorConfig(actor_type="qwen"), ObservationSpec(), "cpu")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{}', encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"fake weights, not a real model")
    identity = base_fingerprint(base)
    (base / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        load_qwen_actor(ActorConfig(actor_type="qwen", model_path=str(base)),
                        ObservationSpec(), "cpu", expected_identity=identity)
    install_fake_loader(monkeypatch)
    checkpoint = tmp_path / "checkpoint"
    save_actor(checkpoint, tiny_actor(), torch.zeros(4), "sft")
    payload = torch.load(checkpoint / "trainable.pt", weights_only=True)
    payload["actor"]["score.6.weight"] = torch.zeros((1, 8))
    torch.save(payload, checkpoint / "trainable.pt")
    with pytest.raises(ValueError, match="score.6.weight"):
        load_actor(checkpoint)


def test_actual_loader_contract_without_transformers(tmp_path, monkeypatch):
    import sys
    from gym_pybullet_drones.learning.qwen_actor import load_qwen_actor

    (tmp_path / "config.json").write_text('{}', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fake")
    backbone = TinyBackbone()
    backbone.config.hidden_size = 1024
    backbone.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(24)])
    calls = []

    def from_pretrained(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=SimpleNamespace(language_model=backbone)), {}

    fake = SimpleNamespace(__version__="5.3.0",
                           Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=from_pretrained),
                           AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: TinyTokenizer()))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    actor = load_qwen_actor(ActorConfig(actor_type="qwen", model_path=str(tmp_path)), ObservationSpec(), "cpu")
    assert actor.backbone is backbone
    assert calls == [dict(local_files_only=True, use_safetensors=True, output_loading_info=True,
                          dtype=torch.float32, attn_implementation="eager")]
    assert not backbone.layers[21].weight.requires_grad
    assert backbone.layers[22].weight.requires_grad


@pytest.mark.parametrize("initialization", ["fresh", "checkpoint_defaults", "checkpoint_overrides"])
def test_sft_entry_with_tiny_synthetic_data(tmp_path, monkeypatch, initialization):
    from gym_pybullet_drones.examples.qwen_action import sft

    root = tmp_path / "data"
    root.mkdir()
    spec = ObservationSpec()
    (root / "dataset_manifest.json").write_text(json.dumps(dict(
        schema_version=1, observation_spec=asdict(spec), episode_split_seed=42,
        action_type="vel", action_dim=4, ctrl_freq=30)), encoding="utf-8")
    records = [dict(episode_id=ep, step=0, agent_index=i, observation=[0.] * 75,
                    action=[0., 0., 0., 1.]) for ep in range(2) for i in range(3)]
    (root / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    monkeypatch.setattr(sft, "build_actor", lambda *args: tiny_actor())
    install_fake_loader(monkeypatch)
    config = dict(dataset=str(root), output=str(tmp_path / "sft"), epochs=1,
                  batch_size=3, microbatch_size=2, actor={"actor_type": "qwen"})
    expected_rates = [1e-5, 1e-4]
    if initialization != "fresh":
        source = tiny_actor()
        source.config.actor_backbone_lr = 3e-4
        source.config.actor_head_lr = 4e-3
        source.config.max_prompt_tokens = 900
        checkpoint = tmp_path / "initial"
        save_actor(checkpoint, source, torch.full((4,), -.5), "sft")
        config["actor_init"] = str(checkpoint)
        if initialization == "checkpoint_overrides":
            expected_rates = [1e-6, 2e-5]
            config["actor"].update(actor_backbone_lr=expected_rates[0], actor_head_lr=expected_rates[1])

    adam = torch.optim.Adam
    created = []

    def inspect_optimizer(groups, **kwargs):
        optimizer = adam(groups, **kwargs)
        assert [group["lr"] for group in optimizer.param_groups] == expected_rates
        assert not optimizer.state
        created.append(optimizer)
        return optimizer

    monkeypatch.setattr(sft.torch.optim, "Adam", inspect_optimizer)
    output = sft.train(config)
    assert len(created) == 1
    actor, std, manifest, _ = load_actor(output / "final_model")
    assert manifest["stage"] == "sft" and manifest["epoch"] == 1
    for name, expected in zip(("actor_backbone_lr", "actor_head_lr"), expected_rates):
        assert manifest["actor_config"][name] == expected
    state = torch.load(output / "final_model" / "optimizer.pt", weights_only=True)
    assert [group["lr"] for group in state["optimizer"]["param_groups"]] == expected_rates
    assert actor.spec == spec
    if initialization != "fresh":
        assert actor.config.max_prompt_tokens == 900
        assert manifest["base_identity"] == source.base_identity
    torch.testing.assert_close(std, torch.full((4,), -.5))
    assert (output / "best_model" / "manifest.json").exists()


def test_legacy_mlp_checkpoint(tmp_path):
    model = MAPPO(spaces.Box(-np.inf, np.inf, (2, 3)), spaces.Box(-1, 1, (2, 2)))
    path = tmp_path / "legacy.pt"
    model.save(path)
    payload = torch.load(path, weights_only=True)
    del payload["config"]["update_microbatch_steps"]
    for key in ("rng", "torch_rng", "cuda_rng"):
        del payload[key]
    assert set(payload["actor"]) == {f"{i}.{p}" for i in (0, 2, 4) for p in ("weight", "bias")}
    torch.save(payload, path)
    restored, _ = MAPPO.load(path)
    np.testing.assert_array_equal(restored.predict(np.zeros((2, 3)))[0], model.predict(np.zeros((2, 3)))[0])


def test_collection_records_pre_step_observation(tmp_path, monkeypatch):
    from importlib import import_module
    from gym_pybullet_drones.examples.qwen_action.collect import collect
    from gym_pybullet_drones.utils.enums import ActionType

    class FakeEnv:
        ACT_TYPE = ActionType.VEL
        CTRL_FREQ = 30
        ACTION_BUFFER_SIZE = 15
        SPEED_LIMIT = .2
        observation_space = spaces.Box(-np.inf, np.inf, (10, 75))
        action_space = spaces.Box(-1, 1, (10, 4))

        def __init__(self, **kwargs):
            pass

        def reset(self, seed):
            rows = np.zeros((10, 75))
            rows[:, -1] = 1
            return rows, {}

        def step(self, action):
            np.testing.assert_array_equal(action, np.tile([0., 0., 1., 1.], (10, 1)))
            return np.ones((10, 75)), 0., True, False, {"is_success": False}

        def close(self):
            pass

    monkeypatch.setattr(import_module("gym_pybullet_drones.envs.MultiHoverAviary"), "MultiHoverAviary", FakeEnv)
    root = collect(str(tmp_path / "data"), episodes=2)
    with (root / "samples.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 20
    assert all(r["observation"][12:-3] == [0.] * 60 for r in records)
    with (root / "dataset_manifest.json").open() as stream:
        assert all(e["failed"] for e in json.load(stream)["episodes"])


def test_training_option_validation():
    from gym_pybullet_drones.examples.learn import run

    with pytest.raises(ValueError, match="multiagent"):
        run(actor_type="qwen")
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(multiagent=True, actor_type="qwen", actor_init="sft", resume="rl")


@pytest.mark.parametrize("kind", ["mlp", "qwen"])
def test_predict_leaves_preprocessing_to_actor(monkeypatch, kind):
    model = tiny_mappo() if kind == "qwen" else MAPPO(
        spaces.Box(-np.inf, np.inf, (10, 75)), spaces.Box(-1, 1, (10, 4)))
    rows = np.zeros((10, 75), dtype=np.float32)
    expected = model.predict(rows)[0]

    def unexpected_transfer(*args):
        pytest.fail("predict must not transfer observations before actor preprocessing")

    monkeypatch.setattr(model, "_tensor", unexpected_transfer)
    for observation in (rows, rows.tolist(), torch.from_numpy(rows)):
        np.testing.assert_array_equal(model.predict(observation)[0], expected)
    np.testing.assert_allclose(model.predict(rows[0])[0], expected[0], atol=1e-7)
    with pytest.raises(ValueError, match="dimension"):
        model.predict(0.)


def test_bf16_forward_uses_compute_copies_and_fp32_gradients():
    actor = tiny_actor(backbone_dtype="bfloat16")
    masters = dict(actor.backbone.named_parameters())
    observed = []

    def inspect_weights(module, args):
        observed.append(dict(module.named_parameters()))
        assert all(p.dtype == torch.bfloat16 for p in module.parameters())

    handle = actor.backbone.register_forward_pre_hook(inspect_weights)
    tokens = actor.encode(np.zeros((10, 75), dtype=np.float32))
    output = actor(tokens)
    handle.remove()
    assert actor.backbone.calls == 1 and len(observed) == 1
    assert output.dtype == torch.float32
    output.square().mean().backward()
    for name, p in actor.backbone.named_parameters():
        assert p is masters[name]
        if p.requires_grad:
            assert p.dtype == p.grad.dtype == torch.float32
            assert torch.isfinite(p.grad).all()
            assert observed[0][name] is not p
        else:
            assert p.dtype == torch.bfloat16 and p.grad is None
    assert any(p.grad.abs().sum() > 0 for p in actor.backbone.layers[-1].parameters())


@pytest.mark.parametrize("stage", ["sft", "mappo"])
def test_bf16_small_adam_updates_accumulate_and_resume(tmp_path, monkeypatch, stage):
    actor = tiny_actor(backbone_dtype="bfloat16")
    model = tiny_mappo(actor)
    optimizer = (torch.optim.Adam(actor.parameter_groups(), eps=1e-5)
                 if stage == "sft" else model.optimizer)
    parameter = actor.backbone.layers[-1][0].weight
    with torch.no_grad():
        parameter.fill_(0.02)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.019), atol=1e-7, rtol=0)
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    install_fake_loader(monkeypatch)
    path = tmp_path / stage
    if stage == "mappo":
        model.save(path)
        restored, _ = MAPPO.load(path)
        loaded_actor, loaded_optimizer = restored.actor, restored.optimizer
    else:
        save_actor(path, actor, model.log_std, "sft", training={"optimizer": optimizer.state_dict()})
        loaded_actor, _, _, _ = load_actor(path)
        loaded_optimizer = torch.optim.Adam(loaded_actor.parameter_groups(), eps=1e-5)
        state = torch.load(path / "optimizer.pt", weights_only=True)
        loaded_optimizer.load_state_dict(state["optimizer"])
    loaded = loaded_actor.backbone.layers[-1][0].weight
    torch.testing.assert_close(loaded, parameter, atol=0, rtol=0)
    assert loaded_optimizer.state[loaded]["exp_avg"].dtype == torch.float32
    for opt, p in ((optimizer, parameter), (loaded_optimizer, loaded)):
        opt.zero_grad(set_to_none=True)
        p.grad = torch.ones_like(p)
        opt.step()
    torch.testing.assert_close(loaded, parameter, atol=0, rtol=0)
