"""
Check a pretrained CarDreamer checkpoint on the laptop CPU and save a "dream" figure. No CARLA, no GPU.

    cd ~/projects/CarDreamer
    PYTHONPATH=. .venv/bin/python wm_pipeline/scripts/check_checkpoint.py \
        --checkpoint ~/checkpoints/cardreamer/right_turn_hard.ckpt --out notes/figures/right_turn_hard_dream.png

What it checks:
  1. the file's SHA-256 matches the one Hugging Face publishes (integrity);
  2. it loads through load_checkpoint's restricted unpickler, so it holds only arrays and optimizer states;
  3. every entry fits our 128x128 `small` agent built with upstream's float16 precision (same architecture/config);
  4. our runner drives a mock episode with the trained weights;
  5. what the trained actor does at the start of an episode, against a random-weight actor;
  6. a 3 s "dream": the actor drives inside the world model from its learned start state (act, imagine one
     step, repeat), saved as a PNG. --compare-random adds the same with random weights as a second row (noise,
     which makes the file ~620 KB; upstream's pre-commit hook rejects new files over 500 KB, so the figure kept
     in notes/figures/ is the trained row only).

This folder has no __init__.py on purpose: the smoke test imports every package module, and must not run this.
"""

import argparse
import collections
import struct
import time
import warnings
import zlib
from pathlib import Path

import jax
import numpy as np

from wm_pipeline.actions import Action
from wm_pipeline.dreamer_agent import build_agent, import_dreamerv3, load_checkpoint, make_config, run_on_agent_weights
from wm_pipeline.encoder import DreamerEncoder, LatentState
from wm_pipeline.mock_env import MockCarDreamerEnv
from wm_pipeline.planner import DreamerPlanner
from wm_pipeline.runner import run_episode

RIGHT_TURN_HARD_SHA256 = "1338b236014f1fca126678cafd058d939adac35910463b1833a37991d3351bbd"
# The checkpoints were trained with upstream's default precision float16 (see load_checkpoint), so build that way.
CHECKPOINT_OVERRIDES = {"jax.platform": "cpu", "jax.precision": "float16"}
ACC_NAMES = {-2.0: "brake", 0.0: "coast", 2.0: "gas"}
STEER_NAMES = {-0.9: "hard right", -0.3: "right", 0.0: "straight", 0.3: "left", 0.9: "hard left"}


def describe(index: int) -> str:
    u = Action.from_index(index)
    return f"{index} ({ACC_NAMES[u.acc]}, {STEER_NAMES[u.steer]})"


def initial_latent(agent) -> LatentState:
    """The model's learned episode-start state (dreamerv3/nets.py:58-64)."""
    _, _, nj = import_dreamerv3()
    model = agent.agent
    fn = nj.jit(nj.pure(lambda: model.wm.rssm.initial(1)), device=agent.policy_devices[0])
    state = run_on_agent_weights(agent, fn, (), seed=0)
    return LatentState(**{k: state[k][0] for k in ("deter", "stoch", "logit")})


def dream(agent, steps: int, seed: int):
    """Let the actor drive inside the world model from the learned start: act, imagine one step, repeat."""
    planner = DreamerPlanner(agent, rng=np.random.default_rng(seed))
    x, frames, actions = initial_latent(agent), [], []
    for _ in range(steps):
        u = planner.act(x)
        step = planner.imagine(x, [u])
        x = step.latents[0]
        frames.append(step.bev[0])
        actions.append(u)
    return frames, actions


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal RGB PNG writer (standard library only, so no image package is needed)."""
    height, width, _ = rgb.shape
    rows = b"".join(b"\x00" + np.ascontiguousarray(rgb[y]).tobytes() for y in range(height))

    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", default=RIGHT_TURN_HARD_SHA256)
    parser.add_argument("--out", type=Path, required=True, help="where to save the dream PNG")
    parser.add_argument("--compare-random", action="store_true", help="add a second PNG row dreamed with random weights")
    args = parser.parse_args()
    # The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    start = time.time()
    trained = build_agent(make_config("small", CHECKPOINT_OVERRIDES))
    random_init = trained.save()
    extras = load_checkpoint(trained, args.checkpoint.expanduser(), sha256=args.sha256)
    print(f"[1-3] SHA-256 ok, restricted unpickle ok, all {len(random_init)} entries fit the 128x128 small agent "
          f"(float16). Loaded in {time.time() - start:.1f}s; training step {extras.get('step')}")

    loaded = trained.save()
    weight_keys = [k for k in loaded if "_opt/" not in k]
    leaves = lambda tree: [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]
    changed = sum(any(not np.array_equal(a, b) for a, b in zip(leaves(loaded[k]), leaves(random_init[k]))) for k in weight_keys)
    print(f"      {changed}/{len(weight_keys)} weight entries differ from the random initialization")

    log = run_episode(MockCarDreamerEnv(seed=0, episode_length=5), DreamerEncoder(trained), DreamerPlanner(trained))
    print(f"[4] runner on the mock env with trained weights: {log.length} steps, actions {[s.u_nom.index for s in log.steps]}")

    random_agent = build_agent(make_config("small", CHECKPOINT_OVERRIDES))  # never loaded: random weights
    rows = []
    for name, agent in (("trained", trained), ("random", random_agent)):
        planner, x0 = DreamerPlanner(agent), initial_latent(agent)
        counts = collections.Counter(planner.act(x0, seed=s).index for s in range(200))
        top = ", ".join(f"{describe(i)}: {n}" for i, n in counts.most_common(4))
        print(f"[5] {name:7s} actor at the episode start, 200 samples: {len(counts)} distinct actions; most common {top}")

        frames, actions = dream(agent, steps=30, seed=0)
        rows.append(np.concatenate(frames[2::3], axis=1))  # every 3rd frame: t = 0.3 s, 0.6 s, ..., 3.0 s
        print(f"[6] {name:7s} 3 s dream: gas on {sum(a.acc > 0 for a in actions)}/30 steps, "
              f"steering right on {sum(a.steer < 0 for a in actions)}/30")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_png(args.out, np.concatenate(rows if args.compare_random else rows[:1], axis=0))
    layout = "top row trained, bottom row random weights" if args.compare_random else "trained weights"
    print(f"saved {args.out} ({layout}; every 3rd imagined frame over 3 s)")


if __name__ == "__main__":
    main()
