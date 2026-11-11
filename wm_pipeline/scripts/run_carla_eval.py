"""
Run our stage 1-3 pipeline on real CARLA with a pretrained checkpoint and compare it with upstream's eval (Step 9 for
right turn hard; the same run for left turn hard, lane merge and navigation). Needs a running CARLA server.

    cd ~/projects/CarDreamer
    PYTHONPATH=. python wm_pipeline/scripts/run_carla_eval.py --task carla_lane_merge --steps 500 --logdir ./logdir/runner_test_lane_merge
    PYTHONPATH=. python wm_pipeline/scripts/run_carla_eval.py --task carla_lane_merge --logdir ./logdir/runner_lane_merge
    python dreamerv3/eval_stats.py --logdir ./logdir/runner_lane_merge

--task picks the env and, unless --checkpoint and --sha256 say otherwise, the task's pretrained checkpoint in
~/checkpoints/cardreamer/ and the SHA-256 Hugging Face publishes for it (dreamer_agent.PRETRAINED_CHECKPOINTS).
The other defaults match upstream's eval: 50,000 steps, with the `small` model in float16 on the GPU.
It stops at the first step where our latent or action differs from upstream's. Ctrl+C is safe: finished episodes are saved.
"""

import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path

from wm_pipeline.carla_env import make_carla_env
from wm_pipeline.compare_upstream import run_comparison
from wm_pipeline.dreamer_agent import (
    CHECKPOINT_DIR,
    PRETRAINED_CHECKPOINTS,
    avoid_rtx5080_compile_crash,
    build_agent,
    load_checkpoint,
    make_config,
)
from wm_pipeline.encoder import DreamerEncoder
from wm_pipeline.planner import DreamerPlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", type=Path, required=True, help="where to write metrics.jsonl; must not already have one")
    parser.add_argument("--task", choices=sorted(PRETRAINED_CHECKPOINTS), default="carla_right_turn_hard")
    parser.add_argument("--checkpoint", type=Path, help="default: the task's file in ~/checkpoints/cardreamer/")
    parser.add_argument("--sha256", help="default: the SHA-256 Hugging Face publishes for the task's checkpoint")
    parser.add_argument("--steps", type=int, default=50000, help="upstream's eval.py runs 50,000")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    file_name, published_sha256 = PRETRAINED_CHECKPOINTS[args.task]
    checkpoint = (args.checkpoint or CHECKPOINT_DIR / file_name).expanduser()
    # Checked before connecting to CARLA, which spends a while loading the map.
    if not checkpoint.is_file():
        parser.error(f"no checkpoint at {checkpoint}: download {file_name} first, or pass --checkpoint")

    avoid_rtx5080_compile_crash()
    # The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    env = make_carla_env(args.task, carla_port=args.port)
    print(f"task {args.task}: steer values {env.action_config.discrete_steer}", flush=True)
    agent = build_agent(make_config("small"))  # no overrides: upstream's defaults, GPU and float16
    extras = load_checkpoint(agent, checkpoint, sha256=args.sha256 or published_sha256)
    print(f"checkpoint ok: SHA-256 matches, all entries fit the agent, trained for {extras.get('step')} steps", flush=True)

    planner = DreamerPlanner(agent, action_config=env.action_config)
    n_checked = run_comparison(env, agent, DreamerEncoder(agent), planner, args.logdir / "metrics.jsonl", args.steps)
    print(f"DONE: all {n_checked} latents and actions matched upstream's. Now run:")
    print(f"python dreamerv3/eval_stats.py --logdir {args.logdir}")


if __name__ == "__main__":
    # CarDreamer's env starts its monitor web server in a non-daemon thread that never ends
    # (car_dreamer/toolkit/monitor/monitor.py:17), so a normal exit waits on it forever, after success and errors
    # alike, and eval_and_push.sh would never reach the next task. So end the process directly, keeping the exit code.
    try:
        main()
        code = 0
    except SystemExit as e:  # argparse errors and --help
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:  # includes Ctrl+C
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # metrics.jsonl is already closed: run_comparison writes it inside a `with` block
