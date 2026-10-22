"""Step 4a tests: Stage 1 data structures accept CarDreamer-shaped data and reject anything else."""

import dataclasses

import numpy as np
import pytest

from wm_pipeline.observation import GroundTruthState, Observation, RouteInstruction, VehicleState, make_observation


def _raw_obs():
    # Same keys, shapes and dtypes as CarDreamer's right-turn env returns.
    return {
        "birdeye_wpt": np.zeros((128, 128, 3), dtype=np.uint8),
        "camera": np.zeros((128, 128, 3), dtype=np.uint8),
        "collision": np.zeros((1,), dtype=np.float32),
    }


def _vehicle(actor_id, x=0.0, y=0.0, vx=0.0, vy=0.0):
    corners = np.array([[x + 2.4, y + 1.0], [x + 2.4, y - 1.0], [x - 2.4, y - 1.0], [x - 2.4, y + 1.0]])
    return VehicleState(actor_id=actor_id, position=np.array([x, y]), yaw_deg=0.0, velocity=np.array([vx, vy]), polygon=corners)


# --- Observation / RouteInstruction -------------------------------------------------------------


def test_make_observation_keeps_only_bev_and_route():
    raw = _raw_obs()
    # CarDreamer's planner gives a list of (x, y, yaw) tuples.
    obs = make_observation(raw, [(-33.8, -135.1, 0.0), (-33.0, -135.1, 0.0), (-32.2, -135.0, 2.5)])

    assert obs.bev is raw["birdeye_wpt"]
    assert obs.bev.shape == (128, 128, 3) and obs.bev.dtype == np.uint8
    assert obs.instruction.waypoints.shape == (3, 3)
    assert obs.instruction.waypoints.dtype == np.float64
    assert not hasattr(obs, "camera")  # BEV only: the world model never sees the camera


def test_empty_route_is_allowed():
    # Near the destination the route runs out; reset/step must still produce a valid observation.
    obs = make_observation(_raw_obs(), [])
    assert obs.instruction.waypoints.shape == (0, 3)


def test_missing_bev_key_raises():
    raw = _raw_obs()
    del raw["birdeye_wpt"]
    with pytest.raises(KeyError, match="birdeye_wpt"):
        make_observation(raw, [])


def test_bev_with_wrong_dtype_raises():
    raw = _raw_obs()
    raw["birdeye_wpt"] = raw["birdeye_wpt"].astype(np.float32)
    with pytest.raises(TypeError, match="uint8"):
        make_observation(raw, [])


def test_bev_with_wrong_shape_raises():
    raw = _raw_obs()
    raw["birdeye_wpt"] = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        make_observation(raw, [])


def test_waypoints_without_yaw_raise_instead_of_being_reshaped():
    # Three (x, y) pairs have 6 numbers, which a careless reshape would turn into two (x, y, yaw) rows.
    with pytest.raises(ValueError, match="shape"):
        RouteInstruction.from_waypoints([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])


def test_more_than_60_waypoints_raise():
    with pytest.raises(ValueError, match="at most 60"):
        RouteInstruction.from_waypoints(np.zeros((61, 3)))


def test_observation_is_frozen():
    obs = make_observation(_raw_obs(), [])
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.bev = np.zeros((128, 128, 3), dtype=np.uint8)


# --- Ground truth ----------------------------------------------------------------------------------


def test_vehicle_speed_is_norm_of_velocity():
    assert _vehicle(1, vx=3.0, vy=4.0).speed == pytest.approx(5.0)


def test_vehicle_polygon_must_have_four_corners():
    with pytest.raises(ValueError, match="polygon"):
        VehicleState(actor_id=1, position=np.zeros(2), yaw_deg=0.0, velocity=np.zeros(2), polygon=np.zeros((3, 2)))


def test_ground_truth_holds_ego_and_others():
    gt = GroundTruthState(timestep=7, ego=_vehicle(1), others=(_vehicle(2, x=10.0), _vehicle(3, x=20.0)))
    assert gt.ego.actor_id == 1
    assert [v.actor_id for v in gt.others] == [2, 3]


def test_ground_truth_rejects_ego_in_others():
    with pytest.raises(ValueError, match="ego"):
        GroundTruthState(timestep=0, ego=_vehicle(1), others=(_vehicle(1), _vehicle(2)))
