"""Fixtures shared by several test files. pytest finds this file by name and injects them by argument name."""

import numpy as np
import pytest

from wm_pipeline.dreamer_agent import CPU_OVERRIDES, build_agent, make_config
from wm_pipeline.encoder import DreamerEncoder
from wm_pipeline.mock_env import MockCarDreamerEnv
from wm_pipeline.observation import make_observation
from wm_pipeline.planner import DreamerPlanner


@pytest.fixture(scope="session")
def agent():
    # A randomly initialized xsmall DreamerV3 agent on the CPU. Building it compiles upstream's training graph
    # (a few seconds), so the whole test run shares one.
    return build_agent(make_config("xsmall", CPU_OVERRIDES))


@pytest.fixture(scope="session")
def encoder(agent):
    # Shared so its jitted function compiles once; tests that need reproducibility pass explicit seeds.
    return DreamerEncoder(agent)


@pytest.fixture(scope="session")
def planner(agent):
    return DreamerPlanner(agent)


@pytest.fixture(scope="session")
def episode():
    # Three consecutive Stage 1 observations from the mock env.
    env = MockCarDreamerEnv(seed=0)
    raw = env.reset()
    observations = [make_observation(raw, env.route_waypoints())]
    for _ in range(2):
        raw, _, _, _ = env.step(7)
        observations.append(make_observation(raw, env.route_waypoints()))
    return observations


@pytest.fixture(scope="session")
def upstream_batch():
    """Builds what upstream's BatchEnv hands agent.policy(): every observation key, with a batch dimension of 1."""

    def build(obs, is_first):
        return {
            "birdeye_wpt": obs.bev[None],
            "camera": np.zeros((1, 128, 128, 3), np.uint8),
            "collision": np.zeros((1, 1), np.float32),
            "reward": np.zeros(1, np.float32),
            "is_first": np.array([is_first]),
            "is_last": np.array([False]),
            "is_terminal": np.array([False]),
        }

    return build
