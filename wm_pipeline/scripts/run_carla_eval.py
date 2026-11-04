"""
Step 9: run our stage 1-3 pipeline on real CARLA and compare it with upstream's eval. Needs a running CARLA server.

    cd ~/projects/CarDreamer
    PYTHONPATH=. python wm_pipeline/scripts/run_carla_eval.py --steps 500 --logdir ./logdir/runner_test
    PYTHONPATH=. python wm_pipeline/scripts/run_carla_eval.py --logdir ./logdir/runner_right_turn_hard
    python dreamerv3/eval_stats.py --logdir ./logdir/runner_right_turn_hard

The defaults match upstream's eval: 50,000 steps on carla_right_turn_hard, with the `small` model in float16 on the GPU.
It stops at the first step where our latent or action differs from upstream's. Ctrl+C is safe: finished episodes are saved.
"""

import argparse
import os
import warnings
from pathlib import Path

from wm_pipeline.carla_env import make_carla_env
from wm_pipeline.compare_upstream import run_comparison
from wm_pipeline.dreamer_agent import RIGHT_TURN_HARD_SHA256, build_agent, load_checkpoint, make_config
from wm_pipeline.encoder import DreamerEncoder
from wm_pipeline.planner import DreamerPlanner

TRITON_OFF = "--xla_gpu_enable_triton_gemm=false"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", type=Path, required=True, help="where to write metrics.jsonl; must not already have one")
    parser.add_argument("--checkpoint", type=Path, default=Path("~/checkpoints/cardreamer/right_turn_hard.ckpt"))
    parser.add_argument("--sha256", default=RIGHT_TURN_HARD_SHA256)
    parser.add_argument("--steps", type=int, default=50000, help="upstream's eval.py runs 50,000")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    # The RTX 5080 crashes compiling a float16 model without this (CLAUDE.md section 4). XLA reads it when JAX first
    # uses the GPU, which is inside build_agent, so setting it here is early enough.
    flags = os.environ.get("XLA_FLAGS", "")
    if TRITON_OFF not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} {TRITON_OFF}".strip()
    # The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    env = make_carla_env("carla_right_turn_hard", carla_port=args.port)
    agent = build_agent(make_config("small"))  # no overrides: upstream's defaults, GPU and float16
    extras = load_checkpoint(agent, args.checkpoint.expanduser(), sha256=args.sha256)
    print(f"checkpoint ok: SHA-256 matches, all entries fit the agent, trained for {extras.get('step')} steps", flush=True)

    n_checked = run_comparison(env, agent, DreamerEncoder(agent), DreamerPlanner(agent), args.logdir / "metrics.jsonl", args.steps)
    print(f"DONE: all {n_checked} latents and actions matched upstream's. Now run:")
    print(f"python dreamerv3/eval_stats.py --logdir {args.logdir}")


if __name__ == "__main__":
    main()
