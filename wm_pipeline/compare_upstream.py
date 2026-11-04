"""
Step 9: compare our runner with upstream's eval, with the same checkpoint on the same env.

UpstreamShadow      runs upstream's agent.policy next to our encoder and planner, on the same frames and seeds, and
                    raises at the first step where they differ. CARLA's randomness can't affect this check.
episode_metrics     the per-episode numbers upstream's eval.py logs, as one metrics.jsonl line, so that upstream's own
                    eval_stats.py computes the results of both runs.
run_comparison      drives episodes with our pipeline, doing both.
"""

import copy
import json
import pathlib
from typing import Dict, List

import numpy as np

from .actions import Action
from .encoder import LatentState
from .observation import BEV_KEY, Observation
from .runner import EpisodeLog, run_episode

# eval_stats.py reads these per-episode values: stats/sum_* and stats/mean_* (eval_stats.py:91-106).
SUM_KEYS = ("travel_distance", "destination_reached", "time_exceeded", "is_collision", "out_of_lane")
MEAN_KEYS = ("ttc", "wpt_dis", "speed_norm")


def upstream_policy_batch(obs: Observation, is_first: bool) -> Dict[str, np.ndarray]:
    """
    What upstream's BatchEnv hands agent.policy(): every observation key, with a batch dimension of 1.

    Camera and collision are zeros because Observation doesn't carry them. Upstream's policy doesn't read them: its
    encoder only takes the BEV (encoder.cnn_keys in tasks.yaml:132).
    """
    return {
        BEV_KEY: obs.bev[None],
        "camera": np.zeros((1, 128, 128, 3), np.uint8),
        "collision": np.zeros((1, 1), np.float32),
        "reward": np.zeros(1, np.float32),
        "is_first": np.array([is_first]),
        "is_last": np.array([False]),
        "is_terminal": np.array([False]),
    }


class UpstreamShadow:
    """
    Stands in for both the encoder and the planner in run_episode: it runs ours, runs upstream's policy on the same
    inputs, and raises at the first difference. Our action is the one that gets executed; upstream's is only compared.

    Pass `rng` to run_episode as its seed: it is a copy of upstream's generator, so both sides get the same seeds.
    """

    def __init__(self, agent, encoder, planner):
        self._agent = agent
        self._encoder = encoder
        self._planner = planner
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(agent.rng.bit_generator.state)
        # Kept across episodes, as upstream's Driver keeps it (driver.py:49). is_first resets it inside the policy.
        self._upstream_state = None
        self._upstream_action = None
        self._step = 0
        self.n_checked = 0

    def encode(self, obs: Observation, x_prev=None, a_prev=None, seed=None) -> LatentState:
        outs, self._upstream_state = self._agent.policy(upstream_policy_batch(obs, is_first=x_prev is None), self._upstream_state, mode="eval")
        (upstream_latent, _), _, _ = self._upstream_state
        self._upstream_action = int(np.argmax(outs["action"][0]))  # what embodied.wrappers.OneHotAction does

        x = self._encoder.encode(obs, x_prev, a_prev, seed=seed)
        for name in ("deter", "stoch", "logit"):
            if not np.array_equal(getattr(x, name), upstream_latent[name][0]):
                raise RuntimeError(f"our latent differs from upstream's at step {self._step} of the comparison: {name}")
        return x

    def act(self, x: LatentState, seed=None) -> Action:
        u_nom = self._planner.act(x, seed=seed)
        if u_nom.index != self._upstream_action:
            raise RuntimeError(f"our action differs from upstream's at step {self._step} of the comparison: {u_nom.index} vs {self._upstream_action}")
        self._step += 1
        self.n_checked += 1
        return u_nom


class RecordingEnv:
    """Passes everything through to `env` and keeps the current episode's info dicts, which EpisodeLog doesn't."""

    def __init__(self, env):
        self._env = env
        self.infos: List[Dict] = []

    def reset(self):
        self.infos = []
        return self._env.reset()

    def step(self, action):
        result = self._env.step(action)
        self.infos.append(result[3])
        return result

    def route_waypoints(self):
        return self._env.route_waypoints()

    def ground_truth(self):
        return self._env.ground_truth()


def episode_metrics(log: EpisodeLog, infos: List[Dict]) -> Dict[str, float]:
    """One episode's metrics.jsonl values, computed the way eval.py:55-85 does: over the steps after the reset, in float32."""
    if len(infos) != log.length:
        raise ValueError(f"expected one info per step: {len(infos)} infos, {log.length} steps")

    def column(key):
        return np.asarray([info[key] for info in infos], dtype=np.float32)  # flags are (1,) bool arrays, so this is (T, 1)

    metrics = {"episode/length": log.length, "episode/score": log.total_reward}
    metrics.update({f"stats/sum_{key}": float(column(key).sum()) for key in SUM_KEYS})
    metrics.update({f"stats/mean_{key}": float(column(key).mean()) for key in MEAN_KEYS})
    return metrics


def run_comparison(env, agent, encoder, planner, metrics_path: pathlib.Path, steps: int) -> int:
    """
    Drive whole episodes with our pipeline until at least `steps` env steps have run, checking every action against
    upstream's. Each finished episode is written straight away, so stopping early keeps its results.

    :return: how many steps were checked.
    """
    shadow = UpstreamShadow(agent, encoder, planner)
    recording = RecordingEnv(env)
    metrics_path = pathlib.Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    total, n_episodes = 0, 0
    # Mode "x" refuses an existing file: appending to an earlier run would make eval_stats.py silently mix the two.
    with metrics_path.open("x") as f:
        while total < steps:
            log = run_episode(recording, shadow, shadow, seed=shadow.rng)
            total += log.length
            n_episodes += 1
            f.write(json.dumps({"step": total, **episode_metrics(log, recording.infos)}) + "\n")
            f.flush()
            print(f"episode {n_episodes}: {log.length} steps, score {log.total_reward:.1f}, {log.terminal_reason}; {total}/{steps} steps; actions matching upstream: {shadow.n_checked}", flush=True)
    return shadow.n_checked
