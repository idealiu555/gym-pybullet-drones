"""Versioned, lossless-in-layout text encoding of local KIN observations."""

from dataclasses import dataclass

import numpy as np
import torch


ACTION_DESCRIPTIONS = {
    "vel": (4, "xyz direction and speed magnitude"),
    "rpm": (4, "four motor offsets from hover RPM"),
    "pid": (3, "world xyz target position in m"),
    "one_d_rpm": (1, "shared motor offset from hover RPM"),
    "one_d_pid": (1, "vertical target displacement in units of 0.1m"),
}


@dataclass(frozen=True)
class ObservationSpec:
    """Local field layout; frequency is in Hz, history is oldest first."""

    action_type: str = "vel"
    action_dim: int = 4
    history_length: int = 15
    ctrl_freq: int = 30
    schema_version: str = "kin_v1"
    precision: int = 3

    def __post_init__(self):
        if self.action_type not in ACTION_DESCRIPTIONS:
            raise ValueError("Unknown action type")
        if self.action_dim != ACTION_DESCRIPTIONS[self.action_type][0]:
            raise ValueError("Action dimension does not match action type")
        if self.ctrl_freq < 2 or self.history_length != self.ctrl_freq // 2:
            raise ValueError("History must cover the environment's half-second buffer")
        if self.schema_version != "kin_v1" or self.precision != 3:
            raise ValueError("Only kin_v1 with three decimal places is supported")

    @property
    def obs_dim(self):
        return 15 + self.history_length * self.action_dim

    @classmethod
    def from_env(cls, env):
        spec = cls(env.ACT_TYPE.value, env.action_space.shape[-1],
                   env.ACTION_BUFFER_SIZE, env.CTRL_FREQ)
        if env.observation_space.shape[-1] != spec.obs_dim:
            raise ValueError("Qwen requires local KIN observations including goal delta")
        return spec


class ObservationPromptEncoder:
    """Encode complete observations; never truncate or infer masks from IDs."""

    def __init__(self, spec, tokenizer, max_tokens=1024, max_chars=1024):
        self.spec = spec
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer requires a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token

    def prompts(self, observations):
        rows = np.asarray(observations, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.spec.obs_dim:
            raise ValueError(f"Expected [batch, {self.spec.obs_dim}] observations")
        if not np.isfinite(rows).all():
            raise ValueError("Observations contain NaN or infinity")

        def numbers(values):
            parts = [f"{v:.3f}" for v in values]
            return ",".join("0.000" if v == "-0.000" else v for v in parts)

        prompts = []
        for index, row in enumerate(rows):
            history = row[12:-3].reshape(self.spec.history_length, self.spec.action_dim)
            text = (
                "Fly to goal quickly and hover. World xyz; p/g m; rpy rad; v m/s; w rad/s.\n"
                f"Action {self.spec.action_type}: {ACTION_DESCRIPTIONS[self.spec.action_type][1]}; [-1,1].\n"
                f"p=[{numbers(row[:3])}];rpy=[{numbers(row[3:6])}]\n"
                f"v=[{numbers(row[6:9])}];w=[{numbers(row[9:12])}]\n"
                f"g=[{numbers(row[-3:])}]\n"
                "History oldest first: h=[" + ";".join(numbers(h) for h in history)
                + "]\nAction features:"
            )
            if len(text) > self.max_chars:
                raise ValueError(f"Sample {index}: {len(text)} characters exceeds {self.max_chars}")
            prompts.append(text)
        if not prompts:
            raise ValueError("Empty observation batch")
        return prompts

    def __call__(self, observations):
        encoded = self.tokenizer(self.prompts(observations), padding=False,
                                 truncation=False, add_special_tokens=True)
        result = []
        for index, ids in enumerate(encoded["input_ids"]):
            if not 0 < len(ids) <= self.max_tokens:
                raise ValueError(f"Sample {index}: token length {len(ids)} outside [1,{self.max_tokens}]")
            result.append(torch.tensor(ids, dtype=torch.int32))
        return result
