"""
The real CarDreamer env behind the same interface as MockCarDreamerEnv, so runner.py can drive CARLA unchanged.

reset() and step() pass straight through, so the env behaves exactly as it does under upstream's eval. The adapter
only reads from the env, to provide the two methods our runner needs and upstream doesn't have:
- route_waypoints(): the route, from get_state()["ego_waypoints"] (carla_wpt_env.py:43-44).
- ground_truth(): every car's exact state, from WorldManager (world_manager.py:265-356).

Per the package rule, `car_dreamer` is imported only inside make_carla_env(). Everything else works on any object
with the same attributes, which is how the tests run without CARLA.
"""

from typing import Dict, List, Tuple

import numpy as np

from .actions import ActionConfig
from .observation import GroundTruthState, VehicleState


class CarDreamerEnvAdapter:
    """A CarDreamer env from create_task(), plus route_waypoints() and ground_truth(). It never changes the env."""

    def __init__(self, env):
        self._env = env
        self._base = env.unwrapped  # gym.make may add wrappers; the state we read lives on the env itself
        config = self._base._config
        # With eval on, upstream renames every info key to eval_* (carla_base_env.py:157-159), and the runner would
        # silently find no terminal flags.
        if config.eval:
            raise ValueError("env.eval must be False: it renames the info keys the runner reads")
        # This task's action values, read from its merged config: right turn overrides the steer set to +-0.9/+-0.3
        # (tasks.yaml:122-123), while left turn, lane merge and navigation keep common.yaml's +-0.6/+-0.2. Give them
        # to DreamerPlanner: the env executes the index either way, but with another task's values u_nom's logged
        # (acc, steer) would be wrong, and so would a CBF reading them.
        self.action_config = ActionConfig(
            discrete=bool(config.action.discrete),
            discrete_acc=tuple(config.action.discrete_acc),
            discrete_steer=tuple(config.action.discrete_steer),
        )
        self._was_reset = False

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self._env.reset()
        self._was_reset = True
        return obs

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        return self._env.step(action)

    def route_waypoints(self) -> List[Tuple[float, float, float]]:
        """Waypoints still ahead of the ego, nearest first: (x, y, yaw) tuples, at most 60."""
        self._require_reset()
        return self._base.get_state()["ego_waypoints"]

    def ground_truth(self) -> GroundTruthState:
        """Exact state of the ego and the other cars at the current timestep."""
        self._require_reset()
        world = self._base._world
        # These two are cached per step and filled when the BEV is drawn (birdeye_renderer.py:161-167), so the ground
        # truth describes the same snapshot as the BEV. Velocities aren't cached, but in synchronous mode CARLA only
        # changes them on a tick. Upstream spawns and destroys cars before drawing (carla_base_env.py:142-147), so a
        # missing id below would mean that order changed, and a KeyError is the right response.
        polygons, transforms = world.actor_polygons, world.actor_transforms
        vehicles = {}
        for actor in world.actors:
            transform, velocity = transforms[actor.id], actor.get_velocity()
            vehicles[actor.id] = VehicleState(
                actor_id=actor.id,
                position=np.array([transform.location.x, transform.location.y], dtype=np.float64),
                yaw_deg=float(transform.rotation.yaw),
                velocity=np.array([velocity.x, velocity.y], dtype=np.float64),
                polygon=np.array(polygons[actor.id], dtype=np.float64),
            )
        ego = vehicles.pop(self._base.get_ego_vehicle().id)  # WorldManager's actor list includes the ego
        return GroundTruthState(timestep=self._base.get_state()["timesteps"], ego=ego, others=tuple(vehicles.values()))

    def _require_reset(self) -> None:
        if not self._was_reset:
            raise RuntimeError("call reset() first")


def make_carla_env(task_name: str = "carla_right_turn_hard", carla_port: int = 2000) -> CarDreamerEnvAdapter:
    """Create a task on a CARLA server that is already running. Needs the `carla` and `car_dreamer` packages."""
    import car_dreamer

    # An explicit argv, so create_task can't pick up flags meant for the script that called this.
    env, _ = car_dreamer.create_task(task_name, ["--env.world.carla_port", str(carla_port)])
    return CarDreamerEnvAdapter(env)
