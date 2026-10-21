# CLAUDE.md — World Model + CBF Safety for Autonomous Driving (Stages 1–3)

This file gives Claude Code the context for this project. Read it fully before doing anything.

---

## 1. Who I am and how to work with me

- I'm an undergraduate researcher in the AICPS lab at UC Irvine, working under my mentor Ratnangshu Das.
- **My mentor will ask me to explain every part of this implementation in a meeting.** He explicitly does not want Claude to write the whole thing end to end without me understanding it. So:
  - Work in **small steps**. One module or one change at a time.
  - **Before** writing code for a step, explain in plain language what you're about to build and why, and what the inputs/outputs are (including tensor shapes).
  - **After** each step, give me a short summary I can paste into my notes, and tell me how to verify it.
  - Prefer simple, readable code over clever code. Add comments that explain *why*, not just *what*.
  - Never make large, sweeping changes across many files in one go. Ask before touching the upstream CarDreamer / DreamerV3 code.
  - If something is ambiguous (see Section 8), **ask me** instead of guessing.
- Commit after each completed step with a clear message, on a `dev` branch (never commit to `main`).

---

## 2. The overall project (for context — most of it is NOT in scope yet)

We are building a safe autonomous driving pipeline in the CARLA simulator:

```
Camera/observation + instruction (O_t, l)
        │
        ▼
Encoder  E(O_t, l) → x_t          (latent state)
        │
        ▼
World model planning               (imagine futures, propose an action)
        │  u_nom  (nominal action)
        ▼
CBF safety filter                  (control barrier function, minimally corrects u_nom)
        │  u_safe
        ▼
CARLA ego vehicle executes u_safe → new observation → repeat
```

- The world model follows the standard agent decomposition **W = (E, D, R, π, C)**:
  - **E** Encoder: observation → latent state
  - **D** Dynamics: predicts next latent state from current state + action
  - **R** Reward: predicts reward from latent state
  - **π** Policy (actor): chooses actions from latent state
  - **C** Critic: estimates long-term value (used to train π)
- The **CBF safety layer** (later stage) is modeled on the VLSA/AEGIS paper (arXiv 2512.11891): a plug-and-play layer after the planner that solves
  `u_safe = argmin ||u − u_nom||²  s.t.  ḣ(x) ≥ −α(h(x))`.
  Vehicle dynamics for the CBF will be omnidirectional first, then Ackermann (kinematic bicycle).

**This week's scope is ONLY stages 1–3: observation + instruction, encoder, and world model planning (outputting u_nom).** Do not implement the CBF yet, but design the code so it can be inserted later (see Section 6).

---

## 3. The base work: CarDreamer

My mentor said to replicate **CarDreamer** (https://github.com/ucd-dare/CarDreamer, paper arXiv 2405.09111).

Why CarDreamer: it uses a **bird's-eye-view (BEV)** observation instead of realistic camera images. With camera images, a world model must handle appearance, lighting, shadows, occlusions, perspective, and scene geometry. With BEV, it can focus directly on road layout, vehicle positions, waypoints, and how the scene evolves in response to driving actions. BEV also makes the later CBF easy, because obstacle positions come straight from the map / simulator ground truth.

### What CarDreamer is
- A platform for **world-model-based RL** in CARLA. Not a new world model — it packages **DreamerV3** (also DreamerV2, Planning2Explore) with driving tasks, rewards, BEV rendering, and evaluation.
- Experiments in the paper use a **small DreamerV3 (~18M params)**: 512 GRU units, CNN depth multiplier 32, 2-layer MLPs.
- The tasks and the world model communicate **only through the Gym interface** (`reset()`, `step(action)`), which is what lets us insert components between them.

### Repository map
```
CarDreamer/
├── car_dreamer/                     # environment side (tasks)
│   ├── __init__.py                  # create_task(name), load_task_configs(name)
│   ├── carla_base_env.py            # CarlaBaseEnv: step(), action decoding, termination
│   ├── carla_wpt_env.py             # waypoint-following base: reward + terminal conditions + TTC
│   ├── carla_right_turn_env.py, carla_left_turn_env.py, ...   # individual tasks
│   ├── configs/common.yaml          # shared defaults: action space, BEV settings, sensors
│   ├── configs/tasks.yaml           # per-task overrides (town, routes, traffic density)
│   └── toolkit/
│       ├── carla_manager/           # WorldManager: spawns/tracks actors; actor_transforms, actor_polygons, actor_actions
│       ├── observer/                # Observer + handlers (birdeye_handler, sensor_handlers, message_handler)
│       ├── planner/                 # route planners → ego waypoints
│       └── monitor/                 # live visualization server (port = carla_port + 7000)
├── dreamerv3/                       # agent side (JAX implementation)
│   ├── agent.py                     # Agent; WorldModel (E, D, R + decoder); ImagActorCritic (π); VFunction (C)
│   ├── nets.py                      # RSSM, encoder/decoder networks
│   ├── train.py / eval.py           # entry points; eval_stats.py computes metrics
│   └── dreamerv3.yaml               # model sizes (xsmall/small/medium/...)
├── train_dm3.sh / eval_dm3.sh       # launch CARLA (off-screen, 10 FPS) + script; auto-restart on crash
```

### Key facts from the code
- **Simulation rate:** 10 Hz (`fixed_delta_seconds: 0.1`).
- **Observations** (per task, `observation.enabled`): usually `[camera, collision, birdeye_wpt]`. Traffic-light/stop-sign tasks use `birdeye_with_traffic_lights`.
- **BEV (`birdeye_wpt`):** 128×128×3 uint8 RGB image, 32 m range, ego offset 12, renders `roadmap, waypoints, background_waypoints, ego_vehicle, background_vehicles` (configurable entities, observability `full | recursive_fov | fov`). `background_waypoints` (orange) are other cars' planned paths: the paper's "intention sharing". It is the **only** input to the world model (`dreamerv3.encoder.cnn_keys: birdeye_wpt`); camera and collision are never encoded.
- **Action space (default discrete):** 15 actions = acceleration ∈ {−2, 0, +2} × steer ∈ {−0.6, −0.2, 0, 0.2, 0.6}. **Right-turn tasks override steer to {−0.9, −0.3, 0, 0.3, 0.9}** (`tasks.yaml:122-123`); always check the merged task config, not `common.yaml`. Decoded in `CarlaBaseEnv.get_vehicle_control`:
  `acc = discrete_acc[action // n_steer]`, `steer = discrete_steer[action % n_steer]`; acc > 0 → throttle = acc/3, else brake = −acc/3; CARLA steer = −steer, so a **negative action steer turns right**.
- **Route waypoints** (`info["ego_waypoints"]`) are ≤ 60 tuples `(x, y, yaw_degrees)`, 0.8 m apart; the third value is heading, not z.
  Continuous mode exists (`env.action.discrete: False`, acc ∈ [−3, 3], steer ∈ [−1, 1]). Note "steer" is a **normalized command**, not a physical wheel angle.
- **Reward** (`carla_wpt_env.py`): waypoint bonus + speed term (track desired speed along route, penalize perpendicular speed) − collision penalty (scaled by speed) − out-of-lane penalty − time penalty + destination bonus.
- **Termination:** collision, time limit, out of lane, destination reached. `step()` checks termination *before* refreshing the observation, so collision termination reads the previous step's sensor (one-step lag). Don't reorder this in our runner, or results stop matching upstream.
- **TTC (time-to-collision)** is already computed and logged each step, but it only sees vehicles ahead in the ego's own lane, and it returns **0.0 when there is no such vehicle**. Don't read `ttc == 0` as "about to crash".
- **Ground-truth obstacle info** is available from `WorldManager` (`actor_transforms`, `actor_polygons`, `actor_actions`). This is what the CBF will use later.

### Paper results to reproduce (DreamerV3 small, full observability, intention sharing)
| Task | Success | Collision | Avg speed (m/s) |
|---|---|---|---|
| Right turn simple / medium / hard | 99.21% / 98.11% / 99.58% | 0.00% / 1.89% / 0.42% | 3.19 / 2.99 / 2.92 |
| Left turn simple / medium / hard | 100% / 97.58% / 92.36% | 0% / 2.42% / 7.64% | 3.21 / 3.04 / 2.97 |
| Overtake | 100% | 0% | 3.12 |
| Four lane | 96.83% | 3.17% | 3.47 |
| Navigation | 80.95% | 7.58% | 3.97 |
| Lane merge | 90.61% | 9.39% | 5.17 |
| Roundabout | 96.00% | 4.00% | 3.58 |
| Traffic lights / Stop sign | 92.65% / 97.94% | 0% / 0% | 2.26 / 1.80 |

Pretrained checkpoints for every task: https://huggingface.co/ucd-dare/CarDreamer/tree/main
Recommended first reproduction target: **`carla_right_turn_hard`** (also used for the paper's observability and intention-sharing experiments).

---

## 4. Compute and environment constraints (important)

- **Right now I only have a Windows laptop with no NVIDIA GPU.** I work inside **WSL2 (Ubuntu)** with VS Code.
- **CARLA 0.9.15 and JAX-CUDA training/eval require an NVIDIA GPU.** The paper used a single RTX 4090; README says 10–20 GB GPU memory + ~3 GB for CARLA.
- **My mentor will give me GPU compute later this week** (probably SSH to a lab machine).
- Therefore, plan the work in two phases:
  - **Phase A — laptop (no GPU, no CARLA server):** read and document the code, design the module structure and interfaces, write the stage 1–3 code, and test it with **mock data** (e.g. synthetic 128×128 BEV arrays, a fake env that implements the same Gym interface). Nothing in Phase A may require a running CARLA server or a GPU.
  - **Phase B — GPU machine:** install CARLA 0.9.15 + CarDreamer + DreamerV3 deps, evaluate the pretrained `carla_right_turn_hard` checkpoint, compare with the paper, then run our stage 1–3 pipeline against the real environment.
- Upstream install (for Phase B, from the CarDreamer README):
  ```bash
  export CARLA_ROOT="</path/to/carla>"
  export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
  conda create python=3.10 --name cardreamer && conda activate cardreamer
  pip install flit && flit install --symlink
  cd dreamerv3 && pip install -r requirements.txt
  pip install "jax[cuda12_pip]==0.4.34" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  ```
- Eval command shape: `bash eval_dm3.sh <carla_port> <gpu_id> <checkpoint_path> --task carla_right_turn_hard ...`

---

## 5. Scope for this week: Stages 1–3

Goal: a clean, understandable pipeline that goes from **observation + instruction → latent state x_t → nominal action u_nom**, built on CarDreamer + DreamerV3, structured so the CBF can be inserted afterward.

### Stage 1 — Observation + instruction `(O_t, l)`
- **O_t** = the BEV image (`birdeye_wpt`, 128×128×3), optionally plus camera.
- **l (instruction)**: CarDreamer has **no natural-language instruction**. The "instruction" in CarDreamer is the **route**: ego waypoints from the route planner, rendered into the BEV (and, in some tasks, a destination one-hot `dest` / neighbor command `message` from `message_handler.py`).
- Build a small, explicit **observation module** that returns a typed structure, e.g.
  `Observation(bev: np.ndarray[128,128,3], waypoints: np.ndarray[N,3], instruction: ...)`,
  so it's obvious what O_t and l are. Keep the instruction representation swappable (route waypoints now; possibly text later — see Section 8).
- Also expose the **ground-truth state** we'll need later for the CBF (ego position, heading, speed; other vehicles' positions/polygons from `WorldManager`) — just pass it through and log it, don't use it yet.

### Stage 2 — Encoder `E(O_t, l) → x_t`
- Use **DreamerV3's encoder + RSSM posterior** as the encoder: observation → latent state (deterministic `deter` + stochastic `stoch`).
- Wrap it in a clear interface, e.g. `encode(obs, prev_state, prev_action) -> LatentState`, and document the latent shapes for the `small` config.
- Don't modify the DreamerV3 internals; wrap them.

### Stage 3 — World model planning → `u_nom`
- Use DreamerV3's trained **actor (π)** on the latent state to produce the nominal action `u_nom` (this is how CarDreamer drives).
- Also expose an **`imagine(latent, actions, horizon)`** helper using the world model's dynamics (D), returning predicted latent states (and decoded BEV frames for visualization). This shows the "world model" part explicitly and will be needed to predict future states for the CBF.
- Output of this stage: `u_nom` (both the raw discrete index and the decoded `(acc, steer)`), plus the latent state.

### Out of scope this week
- The CBF safety filter itself.
- Training a world model from scratch (use pretrained checkpoints first).
- Changing rewards, tasks, or the DreamerV3 architecture.
- Real camera-based (non-BEV) world models.

---

## 6. Design requirement: leave a clean insertion point for the CBF

The CBF will later sit **between the policy output and `env.step()`**. Structure the control loop so that is a one-line change, e.g.

```python
obs = observe(env)                       # Stage 1
x_t = encoder.encode(obs, ...)           # Stage 2
u_nom = planner.act(x_t)                 # Stage 3
u_safe = safety_filter(u_nom, gt_state)  # Stage 4 (later) — identity pass-through for now
obs, reward, done, info = env.step(u_safe)
```

- Implement `safety_filter` now as an **identity pass-through** (`return u_nom`), with logging of `u_nom` and `u_safe`.
- Prefer implementing the loop as a **Gym wrapper or a standalone runner script** rather than editing upstream files, so we can switch the safety layer on/off for "with vs. without CBF" comparisons.
- Note for later: the default action space is **discrete (15 actions)**, but a CBF QP works on continuous actions. Keep the code able to use either `env.action.discrete: True` or `False`.

---

## 7. Suggested step-by-step plan (confirm with me before starting each step)

**Phase A (laptop)**
1. Clone CarDreamer into my fork, create `dev` branch, add a `notes/` folder.
2. Walk me through the code path of one `step()` for `carla_right_turn_hard`: config → env → observer → BEV → reward → termination. Produce `notes/codebase_walkthrough.md`.
3. Create our own package folder (e.g. `wm_pipeline/`) — do not mix our code into upstream folders.
4. Stage 1: observation module + a **mock env** with the same Gym interface (random/synthetic 128×128 BEV, fake waypoints, fake ego/other-vehicle state). Unit tests.
5. Stage 2: encoder wrapper interface. On the laptop, test with a tiny randomly initialized model (`xsmall` config) on CPU, if JAX CPU works; otherwise stub it and document the expected shapes.
6. Stage 3: planner wrapper (`act`, `imagine`) + identity `safety_filter` + the runner loop. Test end-to-end against the mock env.
7. Write `notes/stages_1_3.md`: what each module does, inputs/outputs/shapes, and how it maps to (O_t, l), E, D, R, π, C, u_nom.

**Phase B (GPU machine)**
8. Install CARLA 0.9.15 + CarDreamer + DreamerV3; run the upstream eval on the pretrained `carla_right_turn_hard` checkpoint and compare to the paper (success 99.58%, collision 0.42%, speed 2.92 m/s).
9. Run our stage 1–3 runner with the same checkpoint and confirm it reproduces the upstream eval numbers (our wrapper must not change behavior).
10. Save visualizations: real BEV vs. imagined BEV over a short horizon (like Fig. 4 of the paper).

### Verification checks to always include
- Shapes and dtypes printed/asserted at every module boundary.
- Our runner with `safety_filter = identity` must give the **same** results as upstream eval (same checkpoint, same seed) — this proves our wrappers don't change behavior.
  Caveat: upstream eval *samples* actions (`dreamerv3/agent.py:56-58`), and the env has unseeded randomness, so episodes won't match one-for-one. Proposed check (to confirm with mentor): identical actions for the same recorded observations + RNG key, plus matching success/collision/speed rates. See `notes/codebase_walkthrough.md` §8.
- Imagined vs. real BEV comparison over a short horizon as a sanity check of the world model.

---

## 8. Open questions to raise with me / my mentor (do not guess)

1. **What is the instruction `l` for driving?** CarDreamer uses route waypoints (rendered into BEV) — is that enough, or should we add a text instruction (e.g. "turn right at the intersection") encoded by a language model?
2. **Planner:** use DreamerV3's learned actor (π) as the planner (CarDreamer's approach), or a search-based planner (e.g. CEM over imagined rollouts, like DINO-WM) on top of the world model?
3. **Action space:** stay with discrete (15 actions) for reproduction, then switch to continuous for the CBF?
4. **Which task(s)** to reproduce first — `carla_right_turn_hard` only, or also left turn hard / lane merge (highest collision rates, best for showing CBF gains later)?
5. **Camera input:** BEV only for now, or also camera?

---

## 9. Background papers (for reference)
- CarDreamer — arXiv 2405.09111 (the base work)
- VLSA / AEGIS — arXiv 2512.11891 (the CBF safety-layer design we'll follow later)
- DreamerV3 — arXiv 2301.04104 (the world model backbone)
- PlaNet — arXiv 1811.04551 (origin of the RSSM latent dynamics model)
- DINO-WM — arXiv 2411.04983 (alternative: frozen pretrained encoder + CEM planning)
