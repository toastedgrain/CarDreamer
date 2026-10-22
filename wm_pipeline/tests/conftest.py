"""Fixtures shared by several test files. pytest finds this file by name and injects them by argument name."""

import pytest

from wm_pipeline.dreamer_agent import CPU_OVERRIDES, build_agent, make_config
from wm_pipeline.mock_env import MockCarDreamerEnv
from wm_pipeline.observation import make_observation


@pytest.fixture(scope="session")
def agent():
    # A randomly initialized xsmall DreamerV3 agent on the CPU. Building it compiles upstream's training graph
    # (a few seconds), so the whole test run shares one.
    return build_agent(make_config("xsmall", CPU_OVERRIDES))


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
