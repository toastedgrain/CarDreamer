"""Step 6 tests: the safety-filter slot and the runner loop, on the mock env with a random xsmall agent.

The `agent`, `encoder`, `planner` and `upstream_batch` fixtures come from conftest.py.
"""

import copy

import numpy as np
import pytest

from wm_pipeline.actions import Action
from wm_pipeline.mock_env import MockCarDreamerEnv
from wm_pipeline.observation import make_observation
from wm_pipeline.runner import run_episode
from wm_pipeline.safety_filter import identity_filter

# The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
pytestmark = pytest.mark.filterwarnings("ignore:.*truncated to dtype int32.*")

BRAKE = Action.from_index(2)


class RecordingEnv(MockCarDreamerEnv):
    """The mock env, also remembering every action it was asked to execute."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.executed = []

    def step(self, action):
        self.executed.append(action)
        return super().step(action)


class RecordingEncoder:
    """Wraps the real encoder and remembers the prev_action it receives at each step."""

    def __init__(self, inner):
        self.inner = inner
        self.prev_actions = []

    def encode(self, obs, prev=None, prev_action=None, seed=None):
        self.prev_actions.append(prev_action)
        return self.inner.encode(obs, prev, prev_action, seed=seed)


def always_brake(u_nom, gt):
    """A stand-in for a CBF that intervenes at every step."""
    return BRAKE


def test_identity_filter_returns_the_nominal_action_unchanged():
    env = MockCarDreamerEnv(seed=0)
    env.reset()
    u_nom = Action.from_index(10)
    assert identity_filter(u_nom, env.ground_truth()) is u_nom


def test_episode_with_the_identity_filter_runs_to_the_end(encoder, planner):
    env = RecordingEnv(seed=0, episode_length=5)
    log = run_episode(env, encoder, planner)

    assert log.length == 5 and [step.t for step in log.steps] == list(range(5))
    assert all(step.u_safe is step.u_nom for step in log.steps) and log.n_filtered == 0
    assert env.executed == [step.u_nom.index for step in log.steps]  # the env executed exactly the planner's actions
    assert [step.done for step in log.steps] == [False, False, False, False, True]
    assert log.terminal_reason == "time_exceeded"
    assert log.total_reward == 0.0  # the mock env's reward is always 0
    assert [step.gt.timestep for step in log.steps] == list(range(5))  # gt is the state before each action


def test_filter_slot_changes_what_the_env_and_the_world_model_see(encoder, planner):
    env = RecordingEnv(seed=0, episode_length=4)
    spy = RecordingEncoder(encoder)
    log = run_episode(env, spy, planner, safety_filter=always_brake)

    assert env.executed == [BRAKE.index] * 4  # the env executes u_safe...
    assert all(step.u_safe == BRAKE for step in log.steps)
    assert spy.prev_actions[0] is None  # (first step: no previous action)
    for prev_action in spy.prev_actions[1:]:  # ...and the world model is told u_safe, not u_nom
        np.testing.assert_array_equal(prev_action, BRAKE.one_hot)
    assert log.n_filtered == sum(step.u_nom != BRAKE for step in log.steps)


def test_max_steps_cuts_the_episode_short(encoder, planner):
    log = run_episode(MockCarDreamerEnv(seed=0, episode_length=50), encoder, planner, max_steps=3)
    assert log.length == 3
    assert not log.steps[-1].done and log.terminal_reason is None


def test_same_seed_gives_the_same_episode(encoder, planner):
    a = run_episode(MockCarDreamerEnv(seed=0, episode_length=6), encoder, planner, seed=4)
    b = run_episode(MockCarDreamerEnv(seed=0, episode_length=6), encoder, planner, seed=4)
    assert [step.u_nom for step in a.steps] == [step.u_nom for step in b.steps]


def test_runner_reproduces_upstream_policy_over_a_whole_episode(agent, encoder, planner, upstream_batch):
    """With the identity filter, our runner must choose step for step exactly what upstream's policy loop chooses."""
    # Our runner draws its per-step seeds from a copy of upstream's generator, so both sides get the same seeds.
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(agent.rng.bit_generator.state)
    log = run_episode(MockCarDreamerEnv(seed=0, episode_length=5), encoder, planner, seed=rng)

    # The same episode again, driven the way upstream's eval drives it.
    env = MockCarDreamerEnv(seed=0, episode_length=5)
    raw, state, upstream_actions, done = env.reset(), None, [], False
    while not done:
        obs = make_observation(raw, env.route_waypoints())
        outs, state = agent.policy(upstream_batch(obs, is_first=not upstream_actions), state, mode="eval")
        index = int(np.argmax(outs["action"][0]))  # what embodied.wrappers.OneHotAction does
        upstream_actions.append(index)
        raw, _, done, _ = env.step(index)

    assert [step.u_nom.index for step in log.steps] == upstream_actions
