"""
Stage 1: the observation O_t, the instruction l, and the ground-truth state.

Stages 2-3 (encoder, planner) only ever receive an `Observation` = (O_t, l), exactly the information
CarDreamer's world model uses. `GroundTruthState` is the simulator's exact state of every car: it travels
next to the observation for logging now and for the CBF safety filter later, but is never shown to the
world model. Keeping them apart keeps the "with vs. without CBF" comparison clean.

Shapes follow CarDreamer's carla_right_turn_* tasks (see notes/codebase_walkthrough.md):
- BEV `birdeye_wpt`: (128, 128, 3) uint8 RGB, ego at row 80 / col 64 facing up, 0.25 m per pixel.
- Route waypoints: (N, 3) float64 rows of (x [m], y [m], yaw [deg]) in CARLA's world frame, N <= 60.

Images are uint8 like the env produces them; all geometry is float64 because CARLA reports Python floats
and no network consumes these arrays directly.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, Union

import numpy as np

BEV_KEY = "birdeye_wpt"
BEV_SHAPE = (128, 128, 3)
MAX_WAYPOINTS = 60  # CarDreamer's BasePlanner never returns more than 60 waypoints


def _check_array(name: str, value, shape: Tuple, dtype) -> None:
    """Raise a clear error if `value` is not an array of this shape and dtype. `None` in `shape` means any size."""
    # Explicit raises instead of `assert`: asserts are skipped when Python runs with -O.
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(value).__name__}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}, got {value.dtype}")
    if value.ndim != len(shape) or any(want is not None and want != got for want, got in zip(shape, value.shape)):
        raise ValueError(f"{name} must have shape {shape} (None = any size), got {value.shape}")


@dataclass(frozen=True)
class RouteInstruction:
    """
    The instruction l: the route the ego should follow, as CarDreamer's route planner gives it.

    Its own class so the instruction stays swappable: a text instruction would be a sibling class,
    without changing anything else in the pipeline.
    """

    waypoints: np.ndarray  # (N, 3) float64: x [m], y [m], yaw [deg]; world frame; nearest first

    def __post_init__(self):
        _check_array("waypoints", self.waypoints, (None, 3), np.float64)
        if len(self.waypoints) > MAX_WAYPOINTS:
            raise ValueError(f"expected at most {MAX_WAYPOINTS} waypoints, got {len(self.waypoints)}")

    @classmethod
    def from_waypoints(cls, waypoints: Union[Sequence[Sequence[float]], np.ndarray]) -> "RouteInstruction":
        """Build from CarDreamer's format: a list of (x, y, yaw) tuples (or an (N, 3) array)."""
        if len(waypoints) == 0:
            # np.asarray([]) has shape (0,), so an empty route needs its shape spelled out.
            return cls(np.zeros((0, 3), dtype=np.float64))
        # No reshape here on purpose: a reshape would silently turn bad input such as (x, y) pairs into (N, 3).
        return cls(np.asarray(waypoints, dtype=np.float64))


@dataclass(frozen=True)
class Observation:
    """What Stages 2-3 see at time t: O_t (the BEV image) and l (the route)."""

    bev: np.ndarray  # (128, 128, 3) uint8, RGB; CarDreamer's obs["birdeye_wpt"]
    instruction: RouteInstruction

    def __post_init__(self):
        _check_array("bev", self.bev, BEV_SHAPE, np.uint8)
        if not isinstance(self.instruction, RouteInstruction):
            raise TypeError(f"instruction must be a RouteInstruction, got {type(self.instruction).__name__}")


def make_observation(raw_obs: Mapping[str, np.ndarray], route_waypoints) -> Observation:
    """
    Build (O_t, l) from what the env provides.

    :param raw_obs: the dict returned by env.reset() / env.step(), e.g. {"birdeye_wpt", "camera", "collision"}.
        Only the BEV is kept: CarDreamer's world model never sees the camera or the collision sensor.
    :param route_waypoints: the env's current route, a list of (x, y, yaw) tuples or an (N, 3) array.
        It comes from the env rather than from `info` because CarDreamer's reset() returns no info.
    """
    if BEV_KEY not in raw_obs:
        raise KeyError(f"raw observation has no '{BEV_KEY}' (keys: {sorted(raw_obs)})")
    return Observation(bev=raw_obs[BEV_KEY], instruction=RouteInstruction.from_waypoints(route_waypoints))


@dataclass(frozen=True)
class VehicleState:
    """Ground truth for one car, in CARLA's world frame."""

    actor_id: int
    position: np.ndarray  # (2,) float64: x, y [m]
    yaw_deg: float  # heading [deg], CARLA convention
    velocity: np.ndarray  # (2,) float64: vx, vy [m/s]
    polygon: np.ndarray  # (4, 2) float64: bounding-box corners x, y [m], same format as WorldManager.actor_polygons

    def __post_init__(self):
        _check_array("position", self.position, (2,), np.float64)
        _check_array("velocity", self.velocity, (2,), np.float64)
        _check_array("polygon", self.polygon, (4, 2), np.float64)

    @property
    def speed(self) -> float:
        """Speed [m/s], the norm of the velocity."""
        return float(np.linalg.norm(self.velocity))


@dataclass(frozen=True)
class GroundTruthState:
    """The simulator's exact state at one timestep. Logged now; input to the CBF later; never seen by the world model."""

    timestep: int
    ego: VehicleState
    others: Tuple[VehicleState, ...]  # every other car; how many changes as cross-traffic spawns and despawns

    def __post_init__(self):
        if not isinstance(self.others, tuple):
            raise TypeError(f"others must be a tuple of VehicleState, got {type(self.others).__name__}")
        # Upstream's WorldManager actor lists include the ego, so it is easy to pass it in by mistake.
        if any(v.actor_id == self.ego.actor_id for v in self.others):
            raise ValueError(f"others must not include the ego (actor_id {self.ego.actor_id}); filter it out by id")
