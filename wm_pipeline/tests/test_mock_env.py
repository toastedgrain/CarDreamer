"""Step 4b tests: the mock env behaves like CarDreamer's right-turn env at the interface our pipeline uses."""

import numpy as np
import pytest

from wm_pipeline.mock_env import DT, EGO_ID, MockCarDreamerEnv, box_polygon
from wm_pipeline.observation import make_observation

# Keys, shapes and dtypes of the real carla_right_turn_* observation (notes/codebase_walkthrough.md section 4.5).
REAL_OBS_FORMAT = {
    "birdeye_wpt": ((128, 128, 3), np.uint8),
    "camera": ((128, 128, 3), np.uint8),
    "collision": ((1,), np.float32),
}
TERMINAL_FLAGS = ["is_collision", "time_exceeded", "out_of_lane", "destination_reached"]


def _check_obs_format(obs):
    assert set(obs) == set(REAL_OBS_FORMAT)
    for key, (shape, dtype) in REAL_OBS_FORMAT.items():
        assert obs[key].shape == shape, key
        assert obs[key].dtype == dtype, key


def test_reset_and_step_return_the_real_env_formats():
    env = MockCarDreamerEnv(seed=0)
    _check_obs_format(env.reset())

    result = env.step(7)
    assert len(result) == 4  # old gym API: (obs, reward, done, info)
    obs, reward, done, info = result
    _check_obs_format(obs)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    for flag in TERMINAL_FLAGS:
        assert info[flag].shape == (1,) and info[flag].dtype == np.bool_, flag
    assert info["timesteps"] == 1
    assert info["action"] == 7
    assert info["ego_waypoints"] == env.route_waypoints()


def test_every_step_produces_a_valid_stage1_observation_and_ground_truth():
    env = MockCarDreamerEnv(seed=0, episode_length=10)
    raw, done = env.reset(), False
    while True:
        obs = make_observation(raw, env.route_waypoints())  # raises if any shape or dtype is wrong
        gt = env.ground_truth()
        assert obs.instruction.waypoints.shape[1] == 3
        assert gt.ego.actor_id == EGO_ID and len(gt.others) == 3
        if done:
            break
        raw, _, done, _ = env.step(7)


def test_route_only_keeps_waypoints_ahead_and_shrinks():
    env = MockCarDreamerEnv(seed=0, episode_length=50)
    env.reset()
    lengths = [len(env.route_waypoints())]
    for _ in range(50):
        env.step(7)
        ego_x = env.ground_truth().ego.position[0]
        assert all(wp[0] > ego_x for wp in env.route_waypoints())
        lengths.append(len(env.route_waypoints()))
    assert lengths[0] == 60  # the planner's cap
    assert lengths == sorted(lengths, reverse=True)  # never grows
    assert lengths[-1] < 60  # 50 steps x 0.4 m = 20 m driven


def test_box_polygon_matches_upstream_corner_formula():
    # world_manager.py:295-301: local corners [[l, w], [l, -w], [-l, -w], [-l, w]], rotated by yaw, shifted to (x, y).
    np.testing.assert_allclose(box_polygon(0.0, 0.0, 0.0), [[2.4, 1.0], [2.4, -1.0], [-2.4, -1.0], [-2.4, 1.0]])
    # Yaw 90: the car points along +y, so the front corners (local x = +2.4) end up at y = 5 + 2.4.
    np.testing.assert_allclose(box_polygon(10.0, 5.0, 90.0), [[9.0, 7.4], [11.0, 7.4], [11.0, 2.6], [9.0, 2.6]], atol=1e-12)


def test_other_cars_move_at_constant_velocity():
    env = MockCarDreamerEnv(seed=0)
    env.reset()
    before = env.ground_truth()
    env.step(7)
    after = env.ground_truth()
    for b, a in zip(before.others, after.others):
        np.testing.assert_allclose(a.position, b.position + b.velocity * DT)


def test_episode_ends_after_episode_length_steps():
    env = MockCarDreamerEnv(seed=0, episode_length=5)
    env.reset()
    dones = [env.step(7)[2] for _ in range(5)]
    assert dones == [False, False, False, False, True]


def test_last_step_reports_time_exceeded():
    env = MockCarDreamerEnv(seed=0, episode_length=1)
    env.reset()
    _, _, done, info = env.step(7)
    assert done and info["terminal"]
    assert info["time_exceeded"][0]
    assert info["episode_timesteps"] == 1


def test_stepping_before_reset_or_after_done_raises():
    env = MockCarDreamerEnv(seed=0, episode_length=1)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(7)
    env.reset()
    env.step(7)
    with pytest.raises(RuntimeError, match="episode is over"):
        env.step(7)
    env.reset()  # a new episode starts cleanly
    assert env.ground_truth().timestep == 0


def test_bad_action_is_rejected_without_advancing_time():
    env = MockCarDreamerEnv(seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(15)
    assert env.ground_truth().timestep == 0


def test_continuous_mode_takes_acc_and_steer():
    env = MockCarDreamerEnv(seed=0, discrete=False)
    env.reset()
    env.step(np.array([1.0, -0.3]))
    with pytest.raises(ValueError):
        env.step(7)


def test_same_seed_gives_the_same_episode():
    a, b = MockCarDreamerEnv(seed=3), MockCarDreamerEnv(seed=3)
    np.testing.assert_array_equal(a.reset()["birdeye_wpt"], b.reset()["birdeye_wpt"])
    np.testing.assert_array_equal(a.step(7)[0]["birdeye_wpt"], b.step(7)[0]["birdeye_wpt"])
    for va, vb in zip(a.ground_truth().others, b.ground_truth().others):
        np.testing.assert_array_equal(va.position, vb.position)


def test_different_seeds_give_different_episodes():
    a, b = MockCarDreamerEnv(seed=0), MockCarDreamerEnv(seed=1)
    assert not np.array_equal(a.reset()["birdeye_wpt"], b.reset()["birdeye_wpt"])
