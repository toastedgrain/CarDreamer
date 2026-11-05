"""
Step 10: the real BEV vs the world model's imagined BEV over a short horizon, on CARLA (like Fig. 4 of the paper).
Needs a running CARLA server.

    cd ~/projects/CarDreamer
    PYTHONPATH=. python wm_pipeline/scripts/real_vs_imagined.py --out /mnt/c/Users/Researcher/real_vs_imagined.png

It drives one episode of carla_right_turn_hard with our pipeline and the pretrained checkpoint. From each start step,
the world model imagines the next 3 s under the actions the car actually took, without seeing any more frames.
In the figure, each pair of rows is one start step: the real BEV on top and the imagined one below, every 0.3 s from
0.3 s to 3.0 s after the start.
"""

import argparse
import warnings
from pathlib import Path

from wm_pipeline.carla_env import make_carla_env
from wm_pipeline.dreamer_agent import RIGHT_TURN_HARD_SHA256, avoid_rtx5080_compile_crash, build_agent, load_checkpoint, make_config
from wm_pipeline.encoder import DreamerEncoder
from wm_pipeline.figures import real_vs_imagined_grid, write_png
from wm_pipeline.planner import DreamerPlanner
from wm_pipeline.real_vs_imagined import RecordingEncoder, compare_with_imagination
from wm_pipeline.runner import run_episode

SECONDS_PER_STEP = 0.1  # fixed_delta_seconds in common.yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the PNG to write")
    parser.add_argument("--checkpoint", type=Path, default=Path("~/checkpoints/cardreamer/right_turn_hard.ckpt"))
    parser.add_argument("--sha256", default=RIGHT_TURN_HARD_SHA256)
    parser.add_argument("--starts", type=int, nargs="+", default=[20, 60, 100], help="the steps to imagine from")
    parser.add_argument("--horizon", type=int, default=30, help="steps to imagine: 30 = 3 s")
    parser.add_argument("--every", type=int, default=3, help="show every n-th imagined step in the figure")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    avoid_rtx5080_compile_crash()
    # The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    env = make_carla_env("carla_right_turn_hard", carla_port=args.port)
    agent = build_agent(make_config("small"))  # no overrides: upstream's defaults, GPU and float16
    load_checkpoint(agent, args.checkpoint.expanduser(), sha256=args.sha256)
    recorder, planner = RecordingEncoder(DreamerEncoder(agent)), DreamerPlanner(agent)

    log = run_episode(env, recorder, planner, seed=0)
    print(f"episode: {log.length} steps, {log.terminal_reason}", flush=True)

    comparisons = []
    for start in args.starts:
        if start + args.horizon >= log.length:
            print(f"start {start}: skipped, the episode ended too soon")
            continue
        comparison = compare_with_imagination(log, recorder, planner, start, args.horizon, seed=0)
        comparisons.append(comparison)
        shown = range(args.every - 1, args.horizon, args.every)
        print(f"start {start} ({start * SECONDS_PER_STEP:.1f} s into the episode): mean pixel difference from the real BEV, 0-255")
        print("  after    " + "  ".join(f"{(k + 1) * SECONDS_PER_STEP:5.1f}s" for k in shown))
        print("  model    " + "  ".join(f"{comparison.model_error[k]:6.1f}" for k in shown))
        print("  no-move  " + "  ".join(f"{comparison.no_change_error[k]:6.1f}" for k in shown))
    if not comparisons:
        raise SystemExit("the episode was too short for every start step; try smaller --starts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_png(args.out, real_vs_imagined_grid(comparisons, args.every))
    print(f"saved {args.out}: starts {[c.start for c in comparisons]}, real on top and imagined below in each pair of rows")


if __name__ == "__main__":
    main()
