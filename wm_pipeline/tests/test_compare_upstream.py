"""Step 9 tests: the comparison with upstream, on the mock env with a random xsmall agent. No CARLA needed.

The `agent`, `encoder` and `planner` fixtures come from conftest.py.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wm_pipeline.actions import Action
from wm_pipeline.compare_upstream import MEAN_KEYS, SUM_KEYS, RecordingEnv, UpstreamShadow, episode_metrics, run_comparison
from wm_pipeline.mock_env import MockCarDreamerEnv
from wm_pipeline.runner import run_episode

# The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
pytestmark = pytest.mark.filterwarnings("ignore:.*truncated to dtype int32.*")

REPO_ROOT = Path(__file__).resolve().parents[2]


class EnvWithStatsInfo(MockCarDreamerEnv):
    """The mock env, plus the info values upstream's eval logs, which the real env provides (carla_wpt_env.py:117-132)."""

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, {**info, "travel_distance": 0.4, "ttc": 2.0, "wpt_dis": 0.5, "speed_norm": 3.0}


class WrongActionPlanner:
    """Always proposes something other than what the real planner chose, to prove the shadow notices."""

    def __init__(self, inner):
        self.inner = inner

    def act(self, x, seed=None):
        u = self.inner.act(x, seed=seed)
        return Action.from_index((u.index + 1) % 15)


class WrongSeedEncoder:
    """Samples its latent with a different seed from upstream's, to prove the shadow notices."""

    def __init__(self, inner):
        self.inner = inner

    def encode(self, obs, x_prev=None, a_prev=None, seed=None):
        return self.inner.encode(obs, x_prev, a_prev, seed=seed + 1)


def test_shadow_matches_upstream_across_episodes(agent, encoder, planner):
    # Two episodes in a row: the second starts with upstream's state carried over and reset by is_first, as in its
    # real eval loop, while ours starts from x_prev = None.
    env = MockCarDreamerEnv(seed=0, episode_length=4)
    shadow = UpstreamShadow(agent, encoder, planner)
    first = run_episode(env, shadow, shadow, seed=shadow.rng)
    second = run_episode(env, shadow, shadow, seed=shadow.rng)
    assert first.length == second.length == 4
    assert shadow.n_checked == 8


def test_shadow_catches_a_different_action(agent, encoder, planner):
    shadow = UpstreamShadow(agent, encoder, WrongActionPlanner(planner))
    with pytest.raises(RuntimeError, match="action differs"):
        run_episode(MockCarDreamerEnv(seed=0, episode_length=4), shadow, shadow, seed=shadow.rng)


def test_shadow_catches_a_different_latent(agent, encoder, planner):
    shadow = UpstreamShadow(agent, WrongSeedEncoder(encoder), planner)
    with pytest.raises(RuntimeError, match="latent differs"):
        run_episode(MockCarDreamerEnv(seed=0, episode_length=4), shadow, shadow, seed=shadow.rng)


def test_shadow_does_not_change_what_our_pipeline_chooses(agent, encoder, planner):
    # Upstream only watches: the same seeds with and without it must give the same actions.
    shadow = UpstreamShadow(agent, encoder, planner)
    seeds = np.random.default_rng()
    seeds.bit_generator.state = copy.deepcopy(shadow.rng.bit_generator.state)
    with_shadow = run_episode(MockCarDreamerEnv(seed=0, episode_length=4), shadow, shadow, seed=shadow.rng)
    without = run_episode(MockCarDreamerEnv(seed=0, episode_length=4), encoder, planner, seed=seeds)
    assert [step.u_nom.index for step in with_shadow.steps] == [step.u_nom.index for step in without.steps]


def test_recording_env_keeps_only_the_current_episode():
    env = RecordingEnv(MockCarDreamerEnv(seed=0, episode_length=3))
    env.reset()
    env.step(7)
    env.step(7)
    assert len(env.infos) == 2
    assert env.infos[0]["timesteps"] == 1
    assert env.route_waypoints() and env.ground_truth().timestep == 2
    env.reset()
    assert env.infos == []


def test_episode_metrics_sums_and_averages_like_upstream():
    infos = [
        {"travel_distance": 0.5, "ttc": 1.0, "wpt_dis": 0.2, "speed_norm": 2.0, "destination_reached": np.array([False]), "time_exceeded": np.array([False]), "is_collision": np.array([False]), "out_of_lane": np.array([False])},
        {"travel_distance": 0.25, "ttc": 3.0, "wpt_dis": 0.4, "speed_norm": 4.0, "destination_reached": np.array([True]), "time_exceeded": np.array([False]), "is_collision": np.array([False]), "out_of_lane": np.array([False])},
    ]
    metrics = episode_metrics(SimpleNamespace(length=2, total_reward=9.5), infos)
    assert metrics["episode/length"] == 2
    assert metrics["episode/score"] == 9.5
    assert metrics["stats/sum_travel_distance"] == 0.75
    assert metrics["stats/sum_destination_reached"] == 1.0
    assert metrics["stats/sum_is_collision"] == 0.0
    assert metrics["stats/mean_ttc"] == 2.0
    assert metrics["stats/mean_speed_norm"] == 3.0
    assert metrics["stats/mean_wpt_dis"] == pytest.approx(0.3)
    assert {f"stats/sum_{k}" for k in SUM_KEYS} | {f"stats/mean_{k}" for k in MEAN_KEYS} <= set(metrics)


def test_episode_metrics_rejects_a_mismatched_episode():
    with pytest.raises(ValueError, match="one info per step"):
        episode_metrics(SimpleNamespace(length=3, total_reward=0.0), [])


def _load_upstream_eval_stats(monkeypatch):
    # Load the real eval_stats.py, so this fails if our file stops being what it reads. Only tabulate, which prints the
    # table and isn't in the laptop's environment, is replaced.
    fake_tabulate = SimpleNamespace(tabulate=lambda table, **kwargs: "\n".join(f"{name}: {value}" for name, value in table))
    monkeypatch.setitem(sys.modules, "tabulate", fake_tabulate)
    spec = importlib.util.spec_from_file_location("upstream_eval_stats", REPO_ROOT / "dreamerv3" / "eval_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_comparison_writes_what_upstream_eval_stats_reads(agent, encoder, planner, tmp_path, monkeypatch, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    n_checked = run_comparison(EnvWithStatsInfo(seed=0, episode_length=4), agent, encoder, planner, metrics_path, steps=12)
    assert n_checked == 12
    lines = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(lines) == 3
    assert [line["step"] for line in lines] == [4, 8, 12]

    capsys.readouterr()
    _load_upstream_eval_stats(monkeypatch).main(SimpleNamespace(logdir=str(tmp_path), distance=None))
    table = capsys.readouterr().out
    assert "Number of Episodes: 3" in table
    assert "Avg. Speed Norm: 3.00" in table
    # The mock env's episodes end on time_exceeded, never destination_reached.
    assert "Success Rate: 0.00%" in table


def test_run_comparison_refuses_to_mix_with_an_earlier_run(agent, encoder, planner, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text("{}\n")
    with pytest.raises(FileExistsError):
        run_comparison(EnvWithStatsInfo(seed=0, episode_length=4), agent, encoder, planner, metrics_path, steps=4)
    assert metrics_path.read_text() == "{}\n"
