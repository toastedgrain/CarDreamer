"""
Check the real-env adapter on a running CARLA server, before any model is involved. No JAX, no checkpoint.

    cd ~/projects/CarDreamer
    PYTHONPATH=. python wm_pipeline/scripts/check_carla_env.py --task carla_lane_merge

It resets the task (default carla_right_turn_hard) and holds the accelerator with the wheel straight for 20 steps
(2 s), printing what each adapter method returns and asserting what must be true of it. The car's driving here means
nothing.
"""

import argparse

import numpy as np

from wm_pipeline.actions import decode_action
from wm_pipeline.carla_env import make_carla_env
from wm_pipeline.observation import make_observation

ACCELERATE_STRAIGHT = 12  # index 12 = 5 * 2 + 2: discrete_acc[2] = +2, discrete_steer[2] = 0 in every task's steer set
# Each task's lane_start_point in tasks.yaml (the x, y part). Navigation spawns the ego at a random point instead.
START_XY = {
    "carla_right_turn_hard": (-33.8, -135.1),
    "carla_left_turn_hard": (6.0, -101.0),
    "carla_lane_merge": (49.4, 141.4),
    "carla_navigation": None,
}
N_STEPS = 20


def show(env, raw, label):
    """Print and check one observation + ground truth. Returns the ground truth."""
    obs = make_observation(raw, env.route_waypoints())  # raises if the shapes or dtypes are wrong
    gt = env.ground_truth()
    print(f"--- {label} (timestep {gt.timestep})")
    print(f"BEV    {obs.bev.shape} {obs.bev.dtype}, min {obs.bev.min()}, max {obs.bev.max()}")
    print(f"route  {obs.instruction.waypoints.shape} {obs.instruction.waypoints.dtype}, nearest {obs.instruction.waypoints[0].round(1)}")
    print(f"ego    id {gt.ego.actor_id}, position {gt.ego.position.round(2)}, yaw {gt.ego.yaw_deg:.1f}, speed {gt.ego.speed:.2f} m/s")
    print(f"others {len(gt.others)}: {[tuple(car.position.round(1)) for car in gt.others]}")

    assert obs.bev.max() > 0, "the BEV is blank"
    assert len(obs.instruction.waypoints) > 0, "the route is empty"
    # WorldManager builds the outline around the car's position, so its centre must be the position we report.
    assert np.allclose(gt.ego.polygon.mean(axis=0), gt.ego.position, atol=1e-3), "the ego's outline isn't around its position"
    return gt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(START_XY), default="carla_right_turn_hard")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    env = make_carla_env(args.task, carla_port=args.port)
    print(f"task {args.task}: steer values {env.action_config.discrete_steer}")
    print(f"action {ACCELERATE_STRAIGHT} decodes to (acc, steer) = {decode_action(ACCELERATE_STRAIGHT, env.action_config)}")

    start = show(env, env.reset(), "after reset")
    assert start.timestep == 0
    start_xy = START_XY[args.task]
    if start_xy is not None:
        assert np.allclose(start.ego.position, start_xy, atol=1.0), f"the ego should start near {start_xy}"

    for _ in range(N_STEPS):
        raw, reward, done, info = env.step(ACCELERATE_STRAIGHT)
        if done:
            break
    end = show(env, raw, "after accelerating")
    print(f"reward {reward:.3f}, done {done}, keys in info: {sorted(info)}")

    assert end.timestep > start.timestep
    assert end.ego.speed > 0.1, "the ego should be moving after holding the accelerator"
    assert np.linalg.norm(end.ego.position - start.ego.position) > 0.1, "the ego should have moved"
    assert end.ego.actor_id == start.ego.actor_id
    assert "destination_reached" in info, "the runner reads the terminal flags from info"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
