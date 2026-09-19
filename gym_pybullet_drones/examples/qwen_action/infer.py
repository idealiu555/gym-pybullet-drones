"""Run one batched numeric-action prediction from a local actor checkpoint."""

import argparse
import json

from gym_pybullet_drones.learning.actor_checkpoint import ActorPolicy, load_actor


def infer(checkpoint, observations, device="cpu", model_path=None):
    actor, log_std, _, _ = load_actor(checkpoint, device, model_path)
    with open(observations, encoding="utf-8") as stream:
        rows = json.load(stream)
    actions, _ = ActorPolicy(actor, log_std).predict(rows)
    print(json.dumps(actions.tolist()))
    return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", required=True, help="JSON array [batch, obs_dim]")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_path", help="Relocated, fingerprint-identical base model")
    infer(**vars(parser.parse_args()))
