"""Step 9 tests: the real-env adapter, on a fake with the attributes the real CarDreamer env has. No CARLA needed.

The `encoder` and `planner` fixtures come from conftest.py. The real env is checked by scripts/check_carla_env.py.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wm_pipeline.carla_env import CarDreamerEnvAdapter, make_carla_env
from wm_pipeline.observation import make_observation
from wm_pipeline.runner import run_episode

# The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
pytestmark = pytest.mark.filterwarnings("ignore:.*truncated to dtype int32.*")

EGO_ID, CAR_ID = 24, 31
RIGHT_TURN_ACTION = SimpleNamespace(discrete=True, discrete_acc=[-2.0, 0.0, 2.0], discrete_steer=[-0.9, -0.3, 0.0, 0.3, 0.9])
DEFAULT_ACTION = SimpleNamespace(discrete=True, discrete_acc=[-2.0, 0.0, 2.0], discrete_steer=[-0.6, -0.2, 0.0, 0.2, 0.6])
ROUTE = [(-33.0, -135.1, 0.0), (-32.2, -135.1, 0.0)]


class FakeActor:
    def __init__(self, actor_id, x, y, yaw, vx, vy):
        self.id = actor_id
        self.transform = SimpleNamespace(location=SimpleNamespace(x=x, y=y), rotation=SimpleNamespace(yaw=yaw))
        self.polygon = [[x + 1, y + 1], [x + 1, y - 1], [x - 1, y - 1], [x - 1, y + 1]]  # a list of lists, like poly.tolist()
        self._velocity = SimpleNamespace(x=vx, y=vy)

    def get_velocity(self):
        return self._velocity


class FakeWorld:
    """The part of WorldManager the adapter reads."""

    def __init__(self, actors):
        self.actors = actors

    @property
    def actor_polygons(self):
        return {actor.id: actor.polygon for actor in self.actors}

    @property
    def actor_transforms(self):
        return {actor.id: actor.transform for actor in self.actors}


class FakeBaseEnv:
    """The part of CarlaRightTurnEnv the adapter reads. A second car appears at step 1, as the flow spawns it."""

    def __init__(self, action=RIGHT_TURN_ACTION, eval_mode=False, episode_length=3):
        self._config = SimpleNamespace(eval=eval_mode, action=action)
        self._ego = FakeActor(EGO_ID, -33.8, -135.1, 0.0, 1.5, 0.0)
        self._world = FakeWorld([self._ego])
        self._episode_length = episode_length
        self._time_step = 0
        self.executed = []

    @property
    def unwrapped(self):
        return self

    def get_ego_vehicle(self):
        return self._ego

    def get_state(self):
        return {"ego_waypoints": ROUTE, "timesteps": self._time_step}

    def reset(self):
        self._time_step = 0
        self._world.actors = [self._ego]
        return self._obs()

    def step(self, action):
        self.executed.append(action)
        self._time_step += 1
        self._world.actors = [self._ego, FakeActor(CAR_ID, -3.4, -151.2, 90.0, 0.0, 4.0)]
        done = self._time_step >= self._episode_length
        # Terminal flags as 1-element bool arrays, like carla_base_env.py:134.
        info = {"destination_reached": np.array([done]), "is_collision": np.array([False])}
        return self._obs(), 1.0, done, info

    def _obs(self):
        return {
            "birdeye_wpt": np.zeros((128, 128, 3), np.uint8),
            "camera": np.zeros((128, 128, 3), np.uint8),
            "collision": np.zeros((1,), np.float32),
        }


class FakeGymWrapper:
    """gym.make's wrapping: reset() and step() go through it, but the env's own methods need .unwrapped."""

    def __init__(self, base):
        self._inner = base

    @property
    def unwrapped(self):
        return self._inner

    def reset(self):
        return self._inner.reset()

    def step(self, action):
        return self._inner.step(action)


def _adapter(**kwargs):
    base = FakeBaseEnv(**kwargs)
    return CarDreamerEnvAdapter(FakeGymWrapper(base)), base


def test_reset_and_step_pass_through_unchanged():
    adapter, base = _adapter()
    adapter.reset()
    obs, reward, done, info = adapter.step(12)
    assert base.executed == [12]
    assert obs["birdeye_wpt"].shape == (128, 128, 3)
    assert reward == 1.0
    assert done is False
    assert set(info) == {"destination_reached", "is_collision"}


def test_route_waypoints_come_from_the_env_state():
    adapter, _ = _adapter()
    adapter.reset()
    assert adapter.route_waypoints() == ROUTE
    assert make_observation(adapter.reset(), adapter.route_waypoints()).instruction.waypoints.shape == (2, 3)


def test_ground_truth_describes_the_ego():
    adapter, _ = _adapter()
    adapter.reset()
    gt = adapter.ground_truth()
    assert gt.timestep == 0
    assert gt.ego.actor_id == EGO_ID
    np.testing.assert_array_equal(gt.ego.position, [-33.8, -135.1])
    assert gt.ego.yaw_deg == 0.0
    np.testing.assert_array_equal(gt.ego.velocity, [1.5, 0.0])
    assert gt.ego.speed == 1.5
    assert gt.ego.polygon.shape == (4, 2) and gt.ego.polygon.dtype == np.float64


def test_ground_truth_keeps_other_cars_apart_from_the_ego_as_they_spawn():
    adapter, _ = _adapter()
    adapter.reset()
    assert adapter.ground_truth().others == ()
    adapter.step(12)
    gt = adapter.ground_truth()
    assert gt.timestep == 1
    assert [car.actor_id for car in gt.others] == [CAR_ID]
    np.testing.assert_array_equal(gt.others[0].position, [-3.4, -151.2])
    assert gt.others[0].yaw_deg == 90.0
    np.testing.assert_array_equal(gt.others[0].velocity, [0.0, 4.0])


def test_reading_before_reset_fails_clearly():
    adapter, _ = _adapter()
    with pytest.raises(RuntimeError, match="reset"):
        adapter.route_waypoints()
    with pytest.raises(RuntimeError, match="reset"):
        adapter.ground_truth()


def test_rejects_a_task_with_other_steer_values():
    # common.yaml's default set, which left turn and lane merge use: our planner would label those actions wrongly.
    with pytest.raises(ValueError, match="right-turn"):
        _adapter(action=DEFAULT_ACTION)


def test_rejects_eval_mode_because_it_renames_the_info_keys():
    with pytest.raises(ValueError, match="env.eval"):
        _adapter(eval_mode=True)


def test_make_carla_env_passes_the_port_and_ignores_the_callers_command_line(monkeypatch):
    calls = []

    def create_task(task_name, argv):
        calls.append((task_name, argv))
        return FakeGymWrapper(FakeBaseEnv()), SimpleNamespace()

    monkeypatch.setitem(sys.modules, "car_dreamer", SimpleNamespace(create_task=create_task))
    monkeypatch.setattr(sys, "argv", ["some_script.py", "--checkpoint", "x.ckpt"])
    assert isinstance(make_carla_env("carla_right_turn_hard", carla_port=2005), CarDreamerEnvAdapter)
    assert calls == [("carla_right_turn_hard", ["--env.world.carla_port", "2005"])]


def test_runner_drives_an_episode_through_the_adapter(encoder, planner):
    adapter, base = _adapter(episode_length=3)
    log = run_episode(adapter, encoder, planner, seed=0)
    assert log.length == 3
    assert log.total_reward == 3.0
    assert log.terminal_reason == "destination_reached"
    assert base.executed == [step.u_safe.env_action for step in log.steps]
    # The state each step recorded is from before its action: the flow car only exists from step 1.
    assert [len(step.gt.others) for step in log.steps] == [0, 1, 1]
