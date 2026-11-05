# notes/

Our own notes for the World Model + CBF project (AICPS lab, UC Irvine).
Nothing in this folder is upstream CarDreamer code; the project brief is in `../CLAUDE.md`.

| File | What it covers | Written in step |
|---|---|---|
| `README.md` | this index + the step log below | 1 |
| `codebase_walkthrough.md` | one `step()` of `carla_right_turn_hard`: config → env → observer → BEV → reward → termination | 2 |
| `stages_1_3.md` | each stage 1–3 module: inputs/outputs/shapes, and how it maps to (O_t, l), E, D, R, π, C, u_nom; measured shapes and parameter counts; open questions for the mentor | 7 |
| `figures/right_turn_hard_dream.png` | the trained world model "dreaming" 3 s of the right turn while its actor drives (every 3rd imagined frame) | Phase B prep |

## Repo layout and git
- The clone lives in WSL at `~/projects/CarDreamer` (Linux filesystem: fast for Python, and OneDrive never touches `.git`).
- `master` = upstream CarDreamer, never committed to. All our work goes on `dev`.
- Remotes: `origin` = our fork (`toastedgrain/CarDreamer`), `upstream` = `ucd-dare/CarDreamer`.
  Pulling upstream fixes later: `git fetch upstream && git merge upstream/master` (on `dev`).

## Step log

### Step 1: repo setup (2026-10-21)
- Cloned `ucd-dare/CarDreamer` at upstream commit `1601324`, created the `dev` branch, added `notes/` and the project `CLAUDE.md`.
- Verify: `git branch` shows `* dev`; `git log --oneline -2` shows our commit on top of `1601324`.

### Step 2: code walkthrough (2026-10-21)
- Wrote `codebase_walkthrough.md`: one `step()` of `carla_right_turn_hard` traced through config → env → observer → BEV → reward → termination, with `file:line` references, plus a preview of where the DreamerV3 agent plugs in.
- Corrected facts in `../CLAUDE.md` that the code contradicted: right-turn steer set, BEV layers, waypoint format, collision lag, TTC meaning, and eval sampling.
- Verify: spot-check a few `file:line` references in the note; its last section has commands that run without CARLA.

### Step 3: package skeleton (2026-10-21)
- Created `wm_pipeline/`, our own package, separate from upstream. It has a README with the package rules and the planned modules for Steps 4–6, and `tests/test_smoke.py`, which checks that importing `wm_pipeline` never imports `carla` or `car_dreamer` (Phase A has no CARLA).
- Phase A Python environment: `.venv` in the repo root (already ignored by upstream's `.gitignore`), with Python 3.10.12, numpy 1.24.3 (upstream's pin) and pytest 9.1.1. Setup commands are in `wm_pipeline/README.md`.
- Verify: `.venv/bin/pytest wm_pipeline -v` gives 1 passed, and `git diff master --stat` lists only `notes/`, `wm_pipeline/` and `CLAUDE.md`.

### Step 4a: Stage 1 data structures (2026-10-21)
- `wm_pipeline/observation.py` defines what Stage 1 hands to the rest of the pipeline:
  - `Observation`: O_t = `bev` (128, 128, 3) uint8, plus l = `instruction`.
  - `RouteInstruction`: `waypoints` (N, 3) float64 rows of (x, y, yaw°), with N ≤ 60.
  - `VehicleState`: one car's position, yaw, velocity and 4-corner polygon.
  - `GroundTruthState`: `timestep`, `ego`, and `others` (never including the ego).
- `make_observation(raw_obs, route_waypoints)` keeps only the BEV (decision: BEV only, no camera) and the route.
- Every class checks its shapes and dtypes when it's created. Ground truth stays separate from `Observation`, so the world model never sees it.
- Verify: `.venv/bin/pytest wm_pipeline -v` gives 13 passed.

### Step 4b: minimal mock env + action decoding (2026-10-21)
- `wm_pipeline/actions.py`: `decode_action` turns an action into the `(acc, steer)` command with the same formula as `CarlaBaseEnv.get_vehicle_control`. The defaults are the right-turn values: index 0–14 → acc ∈ {−2, 0, 2} × steer ∈ {−0.9, −0.3, 0, 0.3, 0.9}. Continuous `[acc, steer]` is clipped to [−3, 3] × [−1, 1], where upstream clipping happens in DreamerV3's wrapper / CARLA.
- `wm_pipeline/mock_env.py`: `MockCarDreamerEnv` has the same `reset()` / `step()` 4-tuple and the same obs keys, shapes and dtypes as the real right-turn env, plus `route_waypoints()` and `ground_truth()`. It's deliberately minimal:
  - random-noise images;
  - the ego slides along a straight 48 m route at 4 m/s, whatever the action is;
  - 3 cross-traffic cars move at constant velocity;
  - nothing collides, and the reward is always 0;
  - the episode ends after `episode_length` steps;
  - it's seeded, so the same seed gives the same episode;
  - stepping before `reset()` or after the episode is over raises an error.
- The smoke test now imports every module in the package, so any module with a top-level `import carla` fails it.
- Verify: `.venv/bin/pytest wm_pipeline -v` gives 40 passed.

### Step 5: Stage 2 encoder (2026-10-21)
- New packages, all at upstream's pins and installed from `wm_pipeline/requirements-phase-a.txt`: jax/jaxlib 0.4.34 (CPU), optax, tensorflow-probability, tensorflow-cpu 2.13, ruamel.yaml, cloudpickle.
  TensorFlow is only there because importing DreamerV3's `embodied` package imports it.
- `wm_pipeline/dreamer_agent.py`: `make_config(size, overrides)` and `build_agent(config)` build a DreamerV3 agent the way `dreamerv3/eval.py` does, with the same observation and action spaces as the right-turn task.
- `wm_pipeline/encoder.py`:
  - `LatentState` is x_t, made of `deter` (GRU memory: 512 for small, 64 for xsmall), `stoch` (32 × 32, each row a one-hot sample) and `logit` (32 × 32, the distribution `stoch` was sampled from). `features()` gives deter + flattened stoch, which is what the heads read.
  - `DreamerEncoder.encode(obs, prev, prev_action)` runs upstream's own `preprocess` → CNN encoder → `rssm.obs_step` on the agent's weights, using upstream's `nj.pure` + `nj.jit` mechanism. `prev=None` means the first step of an episode.
- Key test: over two steps, with the same weights, BEV and random keys, `encode()` gives **bit-for-bit the same latent** as upstream's `agent.policy(..., mode="eval")`. I also confirmed the check fails when the random keys differ.
- Verify: `.venv/bin/pytest wm_pipeline -v` gives 47 passed in about 20 s.

### Step 6: Stage 3 planner, safety-filter slot and runner (2026-10-21)
- `Action` (in `actions.py`) is one command: `index`, `acc`, `steer`; `one_hot`, the agent's format, which is fed back to the encoder; and `env_action`, what `env.step()` takes. It covers continuous mode too, for the later CBF.
- `planner.py`, `DreamerPlanner`:
  - `act(x_t, seed)` samples u_nom from upstream's actor, exactly as eval mode does.
  - `imagine(x_t, actions, seed)` rolls the RSSM forward under given actions with no observations and decodes `bev` (H, 128, 128, 3) uint8 and `reward` (H,). It's the recipe of upstream's `WorldModel.report()` open-loop videos.
- Random keys: upstream's single policy call uses keys from one per-step seed in this order: prior sample, posterior sample, action sample. Our encoder and planner share the step seed, and the planner skips the encoder's two keys.
- `safety_filter.py`: `identity_filter(u_nom, gt) -> u_nom`, the slot the CBF will replace.
- `runner.py`: `run_episode(env, encoder, planner, safety_filter, seed, max_steps)` runs Stage 1 → 2 → 3 → filter → `env.step`. It feeds **u_safe**, the action actually executed, back to the encoder, and returns an `EpisodeLog` of `StepRecord`s (gt, u_nom, u_safe, reward, done, terminal flags).
- Key tests:
  - encoder + planner pick **exactly** upstream's eval action (and fail if the key skip is removed);
  - over a whole mock episode the runner chooses exactly what upstream's `agent.policy` loop chooses;
  - one imagined step gives exactly the encoder's next `deter`;
  - an "always brake" filter changes both what the env executes and what the world model is told.
- Verify: `.venv/bin/pytest wm_pipeline -v` gives 68 passed in about 20 s.

### Step 7: Stages 1–3 write-up (2026-10-21)
- Wrote `stages_1_3.md`. It contains:
  - the symbol map (O_t, l, gt, E, x_t, D, R, π, C, u_nom, u_safe, with what's used while driving);
  - one section per module;
  - a measured shapes and parameter-count sheet for `xsmall` and `small`;
  - the annotated control step;
  - what's verified and what isn't;
  - open questions for the mentor.
- Measured: `small` with the 128×128 BEV has 30.5M trainable parameters, not ~18M. 17.9M is the same preset at 64×64. Corrected in `../CLAUDE.md`.
- Phase A (Steps 1–7) is complete. Next is Phase B on the GPU machine.

### Phase B prep: the pretrained checkpoint on the laptop (2026-10-21)
- Downloaded `CarDreamer_checkpoints/right_turn_hard.ckpt` (371.0 MB) from `huggingface.co/ucd-dare/CarDreamer` to WSL `~/checkpoints/cardreamer/`, outside the repo. Its SHA-256 matches the published `1338b236…3351bbd`.
- Upstream checkpoints are **pickles**, and plain `pickle.load` can run code hidden in the file. `load_checkpoint` (in `dreamer_agent.py`) uses a restricted unpickler that only builds numpy arrays and optax optimizer-state NamedTuples. A safe inspection showed the file contains exactly those: `step` = 461,400, `replay` = None, and 100 arrays + 3 optimizer states.
- The checkpoint was trained in **float16**: upstream's float16 optimizer adds loss-scaling state. So the agent must be built with `jax.precision: float16`. Built that way, all 103 entries fit our 128×128 `small` agent exactly (30.5M trainable).
- `wm_pipeline/scripts/check_checkpoint.py` reruns every check and saves `figures/right_turn_hard_dream.png`:
  - the trained actor accelerates at the episode start;
  - in a 3 s dream, the model imagines the road, the ego car, the route turning right at an intersection, and cross traffic.
- Verify: `.venv/bin/pytest wm_pipeline -v` gives 74 passed (6 new loader tests, no download needed). With the checkpoint downloaded: `PYTHONPATH=. .venv/bin/python wm_pipeline/scripts/check_checkpoint.py --checkpoint ~/checkpoints/cardreamer/right_turn_hard.ckpt --out /tmp/dream.png`.

### Step 8: upstream eval of the pretrained checkpoint on the lab machine (2026-11-03)
- Ran upstream's `dreamerv3/eval.py` on `carla_right_turn_hard` with `right_turn_hard.ckpt`, on the lab PC (RTX 5080, 16 GB). The full setup and commands are in `../CLAUDE.md` section 4.
- Result over 289 episodes (50,000 steps, about 15 minutes):

  | Metric | Ours | Paper |
  |---|---|---|
  | Success | 100.00% | 99.58% |
  | Collision | 0.00% | 0.42% |
  | Avg speed (m/s) | 2.95 ± 0.02 | 2.92 |

  Also: out of lane 0%, time exceeded 0, avg episode length 171.3 steps, avg score 345.93.
- How to read it:
  - The results are consistent with the paper, but not better than it. If the true failure rate were the paper's 0.42%, there would still be about a 30% chance of 0 failures in 289 episodes. Zero failures only tells us the true rate is probably below about 1%.
  - The `±` from `eval_stats.py` is the standard error over 3 batches of episodes, so `0.00% ± 0.00%` means no batch had a failure, not that there's no uncertainty.
  - "Avg. Speed Over 4 Ratio" (73.75%) is mean speed ÷ 4 m/s (`eval_stats.py:65`), not the share of time spent above 4 m/s.
- Differences from upstream's setup: Triton matmul off (`--xla_gpu_enable_triton_gemm=false`, needed to avoid a compile crash), CARLA's Windows build with Python in WSL2, TensorFlow CPU build. No side-by-side run on upstream's exact setup was done, so their effect isn't measured.
- The metrics are on the lab machine in `logdir/eval_right_turn_hard/metrics.jsonl` (gitignored).
- Verify, on the lab machine from the repo root: `python dreamerv3/eval_stats.py --logdir ./logdir/eval_right_turn_hard`.
- Not done yet: Step 9 (our runner with the same checkpoint) and Step 10 (real vs imagined BEV).

### Step 9: our runner on CARLA, compared with upstream's eval (2026-11-05)
- `wm_pipeline/scripts/run_carla_eval.py` drove 50,000 steps of `carla_right_turn_hard` with our stages 1–3: the CARLA adapter, `Observation`, `DreamerEncoder`, `DreamerPlanner`, the identity safety filter, then `env.step`. It used `right_turn_hard.ckpt`, float16, on the lab machine's GPU.
- **Check 1, the same action at every step:** upstream's `agent.policy` ran alongside on the same frames with the same seeds. Every latent (deter, stoch, logit) was bit-identical and every action identical. The script stops at the first difference, and it reached its 50,000-step target, so no step differed. A 646-step test run before it also matched at every step.
- **Check 2, the same results:** both runs are scored by upstream's own `eval_stats.py`.

  | Metric | Upstream eval (Step 8) | Our pipeline | Paper |
  |---|---|---|---|
  | Episodes | 289 | 296 | — |
  | Success | 100.00% ± 0.00% | 99.66% ± 0.34% | 99.58% |
  | Collision | 0.00% ± 0.00% | 0.34% ± 0.34% (1 episode) | 0.42% |
  | Out of lane | 0.00% | 0.00% | — |
  | Avg speed (m/s) | 2.95 ± 0.02 | 2.98 ± 0.02 | 2.92 |
  | Avg episode length (steps) | 171.30 ± 1.83 | 169.16 ± 1.72 | — |
  | Avg score | 345.93 ± 0.11 | 343.52 ± 1.76 | — |
- How to read it:
  - **The one collision doesn't come from our wrapper.** Check 1 shows our pipeline chose exactly what upstream's policy chooses from the same frame. The two runs saw different traffic: the gap between cross-traffic cars is random (`np.random.uniform` in `carla_wpt_fixed_env.py`), and so is CARLA's traffic manager.
  - **1 collision in 296 episodes is consistent with 0 in 289.** If both runs had the same true rate, a split like this would be common. It is also close to the paper's 0.42%. The lower average score comes mostly from that episode: the collision penalty, and no destination bonus.
  - **Speed and episode length differ by about one standard error**, which is normal run-to-run variation. The `±` is the standard error over 3 batches of episodes (see Step 8).
- The metrics are on the lab machine in `logdir/runner_right_turn_hard/metrics.jsonl` (gitignored).
- Verify, on the lab machine from the repo root:
  - `PYTHONPATH=. python wm_pipeline/scripts/run_carla_eval.py --logdir <new dir>` should end with `DONE: all ... matched`.
  - `python dreamerv3/eval_stats.py --logdir <that dir>` gives the table.
  - On the laptop, `.venv/bin/pytest wm_pipeline -q` gives 92 passed.
- Next: Step 10, real vs imagined BEV.
