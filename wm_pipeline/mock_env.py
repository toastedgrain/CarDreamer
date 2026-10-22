"""
A stand-in for CarDreamer's right-turn env that runs on a laptop: no CARLA, no GPU.

It copies the interface our pipeline uses, not the driving:
- reset() -> obs and step(action) -> (obs, reward, done, info): the old gym 4-tuple CarDreamer uses.
- obs has the same keys, shapes and dtypes as carla_right_turn_*: {"birdeye_wpt", "camera", "collision"}.
- info uses upstream's key names for the fields our pipeline reads.
- route_waypoints() and ground_truth() are the two extra methods our runner needs. In Phase B a thin wrapper
  gives the real CarDreamer env the same two methods.

Deliberately minimal: the images are seeded random noise, the ego slides along a straight route at constant
speed whatever the action is, the other cars drive at constant velocity, nothing collides, the reward is
always 0, and the episode ends after `episode_length` steps. It tests plumbing, shapes and episode handling,
not driving quality.
"""

from typing import Dict, List, Tuple

import numpy as np

from .actions import ActionConfig, decode_action
from .observation import BEV_KEY, BEV_SHAPE, MAX_WAYPOINTS, GroundTruthState, VehicleState

CAMERA_SHAPE = (128, 128, 3)  # common.yaml camera.shape
DT = 0.1  # seconds per step: CarDreamer's fixed_delta_seconds
EGO_ID = 1
EGO_SPEED = 4.0  # m/s: the right-turn tasks' desired_speed
WAYPOINT_SPACING = 0.8  # m: FixedPathPlanner's sampling radius
HALF_LENGTH, HALF_WIDTH = 2.4, 1.0  # m: bounding-box half extents of a typical car


def box_polygon(x: float, y: float, yaw_deg: float) -> np.ndarray:
    """(4, 2) bounding-box corners in world x, y, built like WorldManager._get_actor_polygons (world_manager.py:279-304)."""
    yaw = np.deg2rad(yaw_deg)
    local = np.array([[HALF_LENGTH, HALF_WIDTH], [HALF_LENGTH, -HALF_WIDTH], [-HALF_LENGTH, -HALF_WIDTH], [-HALF_LENGTH, HALF_WIDTH]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.array([x, y])


class MockCarDreamerEnv:
    """Minimal fake of CarDreamer's right-turn env. Same seed -> same episode."""

    def __init__(self, seed: int = 0, episode_length: int = 50, n_others: int = 3, discrete: bool = True):
        if episode_length < 1:
            raise ValueError(f"episode_length must be at least 1, got {episode_length}")
        self.action_config = ActionConfig(discrete=discrete)
        self._episode_length = episode_length
        self._n_others = n_others
        self._rng = np.random.default_rng(seed)
        # Straight route along +x (yaw 0): 60 waypoints, 0.8 m apart, from x = 0.8 m to x = 48 m.
        self._route = [(float(x), 0.0, 0.0) for x in WAYPOINT_SPACING * np.arange(1, MAX_WAYPOINTS + 1)]
        self._timestep = None  # None until reset() is called
        self._done = False

    def reset(self) -> Dict[str, np.ndarray]:
        self._timestep = 0
        self._done = False
        self._ego_x = 0.0
        # Cross-traffic ahead of the ego, driving along +y (yaw 90 in CARLA's frame) at a constant speed.
        self._others_pos = np.column_stack([self._rng.uniform(10.0, 40.0, self._n_others), self._rng.uniform(-20.0, -5.0, self._n_others)])
        self._others_vel = np.column_stack([np.zeros(self._n_others), self._rng.uniform(3.0, 5.0, self._n_others)])
        return self._observe()

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        self._require_reset()
        if self._done:
            # The real env would keep stepping; stopping here makes a runner that forgets to reset fail loudly.
            raise RuntimeError("episode is over; call reset() before stepping again")
        decode_action(action, self.action_config)  # rejects bad actions like the real env would; the motion ignores it

        self._timestep += 1
        self._ego_x += EGO_SPEED * DT
        self._others_pos = self._others_pos + self._others_vel * DT
        self._done = self._timestep >= self._episode_length

        info = {
            "timesteps": self._timestep,
            "ego_waypoints": self.route_waypoints(),
            # Same format as upstream's terminal flags: 1-element bool arrays (carla_base_env.py:134).
            "is_collision": np.array([False]),
            "time_exceeded": np.array([self._done]),
            "out_of_lane": np.array([False]),
            "destination_reached": np.array([False]),
            "terminal": self._done,
            "action": action,
        }
        if self._done:
            info["episode_timesteps"] = self._timestep
        reward = 0.0
        return self._observe(), reward, self._done, info

    def route_waypoints(self) -> List[Tuple[float, float, float]]:
        """Waypoints still ahead of the ego, nearest first: the format of CarDreamer's get_state()["ego_waypoints"]."""
        self._require_reset()
        return [wp for wp in self._route if wp[0] > self._ego_x][:MAX_WAYPOINTS]

    def ground_truth(self) -> GroundTruthState:
        """Exact state of the ego and the other cars at the current timestep."""
        self._require_reset()
        ego = VehicleState(
            actor_id=EGO_ID,
            position=np.array([self._ego_x, 0.0]),
            yaw_deg=0.0,
            velocity=np.array([EGO_SPEED, 0.0]),
            polygon=box_polygon(self._ego_x, 0.0, 0.0),
        )
        others = tuple(
            VehicleState(actor_id=EGO_ID + 1 + i, position=pos.copy(), yaw_deg=90.0, velocity=vel.copy(), polygon=box_polygon(pos[0], pos[1], 90.0))
            for i, (pos, vel) in enumerate(zip(self._others_pos, self._others_vel))
        )
        return GroundTruthState(timestep=self._timestep, ego=ego, others=others)

    def _observe(self) -> Dict[str, np.ndarray]:
        # Random noise in the real shapes and dtypes: nothing downstream may depend on image content yet.
        return {
            BEV_KEY: self._rng.integers(0, 256, size=BEV_SHAPE, dtype=np.uint8),
            "camera": self._rng.integers(0, 256, size=CAMERA_SHAPE, dtype=np.uint8),
            "collision": np.zeros((1,), dtype=np.float32),
        }

    def _require_reset(self) -> None:
        if self._timestep is None:
            raise RuntimeError("call reset() first")
