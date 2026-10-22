# Stages 1–3: what each module does, and how it maps to (O_t, l), E, D, R, π, C, u_nom

This note describes `wm_pipeline/` as of the end of Step 6 (commit `7906985`).
- Our own code is referred to by function name. Upstream code is referred to by `file:line` (upstream commit `1601324`).
- Every shape and parameter count in §3 was **measured** on the laptop CPU, by building `xsmall` and `small` agents with random weights. None of them are worked out by hand.

---

## 0. Summary

We take CarDreamer's bird's-eye-view image O_t, which already has the route l drawn into it, and pass it through
two parts of DreamerV3:
- the **encoder E**, which turns it into a latent state x_t;
- the **actor π**, which turns x_t into a nominal action u_nom (one of 15 discrete commands).

A **safety-filter slot** sits between u_nom and the simulator. Today it passes u_nom through unchanged. Later, the
CBF will minimally correct it into u_safe, using the simulator's ground-truth state.

We **wrap upstream's code instead of changing it**. On the laptop (random weights, mock env), tests show our pipeline
picks bit-for-bit the same actions as upstream's own eval loop.

The world model can also **imagine** the future under any action sequence (dynamics D plus decoder). That's what a
CBF or a search-based planner would use to look ahead.

```
env (mock now, CARLA in Phase B)                                   wm_pipeline module
 ├─ raw obs ── make_observation ──► Observation (O_t, l)             observation.py
 │                                   bev (128,128,3) uint8, route (N≤60, 3)
 ├─ ground_truth() ───────────────► GroundTruthState gt_t ─────┐     observation.py
 │                                                             │
 │      DreamerEncoder.encode   (E)  ──► x_t = LatentState     │     encoder.py
 │                                       deter (512,), stoch (32,32), logit (32,32)
 │      DreamerPlanner.act      (π)  ──► u_nom = Action        │     planner.py
 │                                       index 0–14, (acc, steer), one_hot (15,)
 │      safety_filter(u_nom, gt_t)   ──► u_safe   ◄────────────┘     safety_filter.py (identity today)
 └─◄─── env.step(u_safe.env_action) ◄── run_episode() repeats this   runner.py
```

---

## 1. Symbol map

| symbol | what it is here | our code | upstream code | used while driving? |
|---|---|---|---|---|
| **O_t** | BEV image `obs["birdeye_wpt"]`: (128, 128, 3) uint8 RGB, ego at row 80 / col 64 facing up, 0.25 m per pixel | `Observation.bev` | `toolkit/observer/handlers/birdeye_handler.py:36-81` | yes: the **only** input the world model sees |
| **l** | the route: ≤ 60 waypoints `(x, y, yaw°)`, 0.8 m apart. It reaches the model only as the blue line drawn into O_t | `RouteInstruction` | `toolkit/planner/base_planner.py:98-121`; drawn at `birdeye_renderer.py:142-156` | yes, through O_t |
| gt | the simulator's exact state of the ego and the other cars | `GroundTruthState`, `VehicleState` | `world_manager.py:279-356` (`actor_polygons`, `actor_transforms`) | not by the model; the CBF will use it |
| **E** | CNN encoder + RSSM posterior: (O_t, x_{t−1}, a_{t−1}) → x_t | `DreamerEncoder.encode` | `dreamerv3/agent.py:49-52`; CNN `nets.py:211-260, 332-374`; posterior `nets.py:105-123` | yes |
| **x_t** | latent state: `deter` (memory), `stoch` (sample), `logit` (its distribution) | `LatentState` | RSSM state dict, `nets.py:44-66` | yes |
| **D** | dynamics (prior): (x_t, a_t) → x_{t+1} via a GRU and a sampled `stoch` | inside `encode` (posterior starts from the prior); `DreamerPlanner.imagine` | `img_step` `nets.py:125-144`; `imagine` `nets.py:85-94` | yes, one step inside E every step |
| **R** | reward head: x → predicted reward | `Imagination.reward` | `agent.py:139`; used for training in `behaviors.py:12` | **no**: training only (plus our `imagine` readout) |
| **π** | actor: x_t → distribution over the 15 actions. Eval **samples** it | `DreamerPlanner.act` | `agent.py:278-292`; eval sampling `agent.py:54-58` | yes |
| **C** | critic (`VFunction`): x → expected future return | **not wrapped** | `agent.py:359-398`; used in `ImagActorCritic.train`, `agent.py:294-313` | **no**: only to train π |
| **u_nom** | the action π sampled | `Action` | one-hot → index like `embodied/core/wrappers.py:140-146` | yes |
| **u_safe** | the action actually executed | output of `safety_filter`, an `Action` | (none: our addition) | yes |
| decoder (not in W) | x → reconstructed BEV | `Imagination.bev` | `agent.py:138`; `nets.py:387-426` | no: training loss and our visualizations |
| `cont` head (not in W) | x → probability the episode continues | not used | `agent.py:140` | no |

**Three relations worth being able to explain**

1. **E contains one step of D.** The posterior first runs the prior, `img_step`. That gives the new memory `deter_t`
   from `stoch_{t−1}` and `a_{t−1}` alone. Only then does the image refine `stoch_t`. So the frame never changes
   `deter`; it changes the distribution over `stoch` (`nets.py:116-122`).
   Test: `test_first_imagined_memory_equals_the_encoders_next_memory` shows imagining one step and encoding the real
   next frame give **exactly** the same `deter`, whichever frame comes next.
2. **R and C only matter in training.** While driving, π alone picks the action. R and C are *why* π is good:
   DreamerV3 trains π on imagined rollouts scored by R and C.
3. **l has no input of its own.** CarDreamer gives the model the route only as pixels in O_t. If the project adds a
   text instruction, `encode` is where it would plug in.

---

## 2. Module by module

### `observation.py`: Stage 1, (O_t, l) and ground truth

| | name | shape / dtype |
|---|---|---|
| in | `raw_obs` from `env.reset()` / `env.step()` | `birdeye_wpt` (128,128,3) uint8, `camera` (128,128,3) uint8, `collision` (1,) float32 |
| in | `route_waypoints` from the env | list of `(x, y, yaw°)` or (N, 3) array |
| out | `make_observation(...)` → `Observation` | `bev` (128,128,3) uint8; `instruction.waypoints` (N, 3) float64, 0 ≤ N ≤ 60 |
| out | `GroundTruthState` (built by the env) | `timestep` int; `ego` and `others` are `VehicleState`s: `position` (2,), `yaw_deg`, `velocity` (2,), `polygon` (4, 2), all float64 |

**Decisions:**
- **BEV only.** The camera is dropped because the world model never sees it.
- **Ground truth is kept apart from `Observation`**, so the model can't see it.
- Everything is **frozen** and checks its shapes and dtypes when created.
- **Waypoints come from the env, not `info`**, because CarDreamer's `reset()` returns no `info`.

**Tests:** `test_observation.py` (12).

### `mock_env.py`: a laptop stand-in for the right-turn env

`MockCarDreamerEnv(seed, episode_length=50, n_others=3, discrete=True)` provides:
- `reset() → obs` and `step(a) → (obs, reward, done, info)`;
- `route_waypoints()` and `ground_truth()`.

It copies the real env's **interface**: the same obs keys, shapes and dtypes, and upstream's terminal flags as 1-element bool arrays. It does **not** copy the driving:
- the images are noise;
- the ego slides 0.4 m per step along a straight 48 m route, whatever the action;
- 3 cross-traffic cars drive at constant velocity;
- nothing collides, and the reward is 0;
- the episode ends after `episode_length` steps.

It's seeded, and stepping before `reset()` or after the episode is over raises an error.

In Phase B, the real env needs a thin adapter that provides the same two extra methods: `route_waypoints()` from `env.unwrapped.get_state()["ego_waypoints"]`, and `ground_truth()` from the `WorldManager`.

**Tests:** `test_mock_env.py` (12).

### `actions.py`: the action space, and the `Action` type for u_nom / u_safe

- `decode_action(a, config) → (acc, steer)` uses the formula from `carla_base_env.py:105-125`:
  index = 5·i_acc + i_steer, with acc ∈ {−2, 0, +2} and steer ∈ {−0.9, −0.3, 0, 0.3, 0.9}.
  - **Right-turn tasks override the default steer set** (`tasks.yaml:122-123`).
  - acc ±2 means a 2/3 brake or throttle pedal; 0 means coasting.
  - A **negative steer turns right** (CARLA steer = −steer).
- `Action(index, acc, steer)` can only be built with consistent fields. It provides:
  - `one_hot`: (15,) float32, the agent's own format, which is fed back to the encoder;
  - `env_action`: the index, or `[acc, steer]` in continuous mode.

**Tests:** `test_actions.py` (20).

### `dreamer_agent.py`: building upstream's agent

- `make_config(size, overrides)` = `dreamerv3.yaml` defaults + a size + `encoder/decoder.cnn_keys = "birdeye_wpt"`, as in `eval.py:109-118`. `CPU_OVERRIDES` sets CPU + float32 for the laptop; upstream uses GPU + float16.
- `build_agent(config)` creates the right-turn spaces of `from_gym.py:27-48` plus the 15-way one-hot action of `wrappers.py:126-138`.
- `run_on_agent_weights(...)` is shared by the encoder and planner. It:
  - moves data explicitly, because upstream sets JAX's transfer guard to "disallow" (`jaxagent.py:148`);
  - turns a seed into a random key the way upstream does;
  - fails if a call created weights instead of using the agent's.
- `draw_seed(rng)` draws a step seed the way `jaxagent.py:208-210` does.
- `load_checkpoint(agent, path, sha256)` loads upstream's pickle checkpoints **safely**. Its restricted unpickler
  only builds numpy arrays and optax optimizer states (NamedTuples), so nothing else in the file can run. It
  checks every entry against the agent and then calls upstream's `JAXAgent.load`.
  - CarDreamer's checkpoints were trained in float16, so the agent must be built with `jax.precision: float16`.
  - `scripts/check_checkpoint.py` runs the full check on a real checkpoint.

### `encoder.py`: **E**

| | name | shape / dtype (small; xsmall in brackets) |
|---|---|---|
| in | `obs` | `Observation` |
| in | `prev`, `prev_action` | `LatentState` and (15,) float32 one-hot, or both `None` on an episode's first step |
| in | `seed` | per-step int (optional) |
| inside | CNN embedding | (8192,) = 4×4×512   [(1024,) = 4×4×64] |
| out | `LatentState` | `deter` (512,) [(64,)], `stoch` (32, 32) with one-hot rows, `logit` (32, 32); `features()` (1536,) [(1088,)] |

**What runs**, exactly upstream's `agent.py:49-52`:
1. `preprocess`: pixels ÷ 255.
2. The CNN: 5 stride-2 convolutions, 128 → 64 → 32 → 16 → 8 → 4 px, with channels 32 → 512.
3. `obs_step`: the prior `img_step` (GRU) gives `deter_t`, then the posterior logits from `[deter_t, embed]` → sampled `stoch_t`.

**Episode start:** on the first step, the RSSM starts from its learned initial state (`nets.py:110-115`). The placeholders we pass use the agent's compute precision, so float16 in Phase B will match upstream.

**dtype:** float32 on the laptop, float16 in upstream's GPU config.

**Tests:** `test_encoder.py` (8), including a bit-for-bit match with upstream's policy latent over two steps.

### `planner.py`: **π** (`act`), and **D** + **R** + decoder (`imagine`)

| | name | shape / dtype |
|---|---|---|
| in | `act(x_t, seed)` | `LatentState` |
| out | u_nom | `Action` (index 0–14). The actor reads `features()` (1536 for small) and outputs 15 logits with a 1% uniform mix (`dreamerv3.yaml:170-180, 221`) |
| in | `imagine(x_t, actions, seed)` | `LatentState` and H `Action`s |
| out | `Imagination` | `latents`: H `LatentState`s = x_{t+1} … x_{t+H}; `bev` (H, 128, 128, 3) uint8; `reward` (H,) float32 |

- **`act`** samples, like eval mode (`agent.py:54-58`). That makes u_nom a *sample*, not the single most likely action.
- **`imagine`** uses no observations: it's `rssm.imagine` (`nets.py:85-94`), then the decoder's image (×255) and R. That's the recipe of upstream's open-loop videos in `WorldModel.report()` (`agent.py:231-245`).

**Tests:** `test_planner.py` (9), including a bit-for-bit match with upstream's action over two steps.

### `safety_filter.py`: the CBF slot

`SafetyFilter = (u_nom: Action, gt: GroundTruthState) → Action`. `identity_filter` returns u_nom. The CBF replaces
this one function; nothing else in the loop changes (CLAUDE.md §6).

### `runner.py`: the loop

`run_episode(env, encoder, planner, safety_filter=identity_filter, seed=0, max_steps=None) → EpisodeLog`
- `seed` can also be a generator, for example a copy of upstream's `agent.rng` when reproducing upstream's eval.
- `EpisodeLog.steps` holds one `StepRecord` per step: `t`, `gt`, `u_nom`, `u_safe`, `reward`, `done` and the terminal `flags`.
- `EpisodeLog` adds the summaries `length`, `total_reward`, `terminal_reason` and `n_filtered`.

**Tests:** `test_runner.py` (6), including that a whole mock episode matches upstream's `agent.policy` loop action for action.

---

## 3. Shapes cheat sheet (measured)

| quantity | `xsmall` (our tests) | `small` (paper, checkpoints) |
|---|---|---|
| config: `rssm.deter` / `units` / `encoder.cnn_depth` / actor layers × units | 64 / 64 / 4 / 1 × 64 | 512 / 512 / 32 / 2 × 512 |
| BEV input | (128, 128, 3) uint8 | (128, 128, 3) uint8 |
| CNN embedding | (1024,) | (8192,) |
| `deter` | (64,) | (512,) |
| `stoch`, `logit` | (32, 32) | (32, 32) |
| `features()`: what π, R, C and the decoder read | (1088,) | (1536,) |
| action, one-hot | (15,) | (15,) |
| `imagine`, H steps | `bev` (H, 128, 128, 3) uint8, `reward` (H,) float32 | same |
| latent dtype | float32 on CPU | float32 on CPU; **float16** in upstream's GPU config |
| build time on the laptop CPU | ~4 s | ~3–6 s |

**Trainable parameters** (counted by upstream's optimizers):

| part | `xsmall` | `small` |
|---|---|---|
| encoder CNN (`wm/enc`) | 43,960 | 2,788,800 |
| RSSM, the dynamics (`wm/rssm`) | 298,752 | 7,880,704 |
| decoder (`wm/dec`) | 1,158,971 | 15,378,883 |
| reward head R (`wm/rew`) | 86,335 | 1,181,439 |
| `cont` head (`wm/cont`) | 69,825 | 1,051,137 |
| **world model total** | **1,657,843** | **28,280,963** |
| actor π | 70,735 | 1,058,319 |
| critic C* | 86,335 | 1,181,439 |
| **all trainable** | **1,814,913** | **30,520,721** |

\* The critic also keeps a "slow" target copy of the same size, which is updated by averaging rather than by the optimizer.

**About "~18M" in CLAUDE.md.** Building the same `small` preset with a 64×64 image measures **17,932,177**
parameters. So ~18M is DreamerV3's standard "small" size for 64×64 inputs. CarDreamer's 128×128 BEV adds one more
CNN stage to both the encoder and decoder, and enlarges two layers:
- the decoder's input layer, 1536 → 4×4×512, which is **12.6M** on its own;
- the RSSM's observation layer, (512 + 8192) → 512, which is **4.5M**.

That brings `small` to **30.5M**. CLAUDE.md is corrected accordingly.

---

## 4. One control step, annotated (`run_episode`)

```python
obs    = make_observation(raw, env.route_waypoints())   # Stage 1: O_t (BEV) and l (route)
gt     = env.ground_truth()                              # Stage 1: ground truth, for the filter only
seed   = draw_seed(rng)                                  # ONE seed per step, like upstream's policy call
x      = encoder.encode(obs, x_prev, a_prev, seed=seed)  # E   uses key 1 (prior) and key 2 (posterior)
u_nom  = planner.act(x, seed=seed)                       # π   skips keys 1-2, samples the action with key 3
u_safe = safety_filter(u_nom, gt)                        # CBF slot (identity today)
raw, reward, done, info = env.step(u_safe.env_action)
x_prev, a_prev = x, u_safe.one_hot                       # tell the world model what was EXECUTED
```

**Why share one seed?** Upstream does E and π inside one function call. That call takes its random keys, in order,
from a single per-step seed: the prior sample, then the posterior sample, then the action sample. Our encoder and
planner are separate calls, so they share the step seed and take the same keys in the same order.

Removing the planner's two skipped keys makes the actions differ from upstream's (checked by hand in Step 6).
`wm_pipeline/README.md` rule 4 records this coupling.

**Why feed back u_safe?** The world model's memory must describe what really happened. Once the CBF changes an action,
telling the model u_nom would make its latent state drift away from reality.

---

## 5. What is verified, and what isn't yet

**Verified on the laptop** (random weights, mock env, 68 tests in about 20 s):
- shapes and dtypes are checked at every module boundary;
- `encode` = upstream's policy latent, bit for bit over two steps. With different random keys the check fails;
- `encode` + `act` = upstream's eval action, bit for bit over two steps. Without the key skip it fails;
- `run_episode` = upstream's `agent.policy` loop, action for action over a whole mock episode;
- one imagined step gives exactly the encoder's next `deter`;
- the filter slot changes both what the env executes and what the world model is told.

**Verified with the real `right_turn_hard` checkpoint on the laptop CPU** (float16, `scripts/check_checkpoint.py`):
- the SHA-256 matches Hugging Face's, and the file holds only arrays and optimizer states;
- **all 103 entries fit our 128×128 `small` agent exactly**: same architecture and config, 30.5M trainable parameters, trained for 461,400 steps;
- our runner drives with the trained weights;
- at the episode start, the trained actor nearly always accelerates (193 of 200 samples are "gas"; the random actor spreads over all 15 actions);
- letting the actor drive inside the world model for 3 s produces a coherent BEV "dream": road, red ego car, and the blue route turning right at an approaching intersection, with green cross traffic appearing. See `figures/right_turn_hard_dream.png`. With random weights the same procedure gives noise.

**Not verified yet:**
- **The real weights on a GPU.** float16 numerics on a GPU haven't been run, and the dream above is qualitative.
- **The real CARLA env**, and the Phase B adapter that gives it `route_waypoints()` / `ground_truth()`.
- **Upstream's paper numbers.** Step 8 checks whether they reproduce, and Step 9 checks our runner against upstream's eval on CARLA.
- **Imagination quality.** With random weights the decoded frames are noise; Step 10 compares real and imagined BEV using the checkpoint.

---

## 6. Open questions for the mentor

1. **Instruction l.** Is the route drawn into the BEV enough, or should we add a text instruction? If we add one, it would go into `encode`.
2. **Planner.** Keep the actor π (CarDreamer's approach, and it matches upstream exactly), or use a search-based planner such as CEM over `imagine()`? `imagine` is ready for either.
3. **Discrete vs continuous for the CBF.** The world model was trained on one-hot actions, so a continuous u_safe can't be fed back as-is (`Action.one_hot` raises). The options:
   - the CBF picks among the 15 discrete actions;
   - project the CBF's continuous answer to the nearest discrete action;
   - train a continuous-action model.
4. **Sampled or most-likely u_nom?** Upstream samples. Taking the most likely action would make the planner deterministic, which may be easier to reason about with a CBF.
5. **How Step 9 proves "our wrapper doesn't change behavior."**
   - A per-step action match on recorded observations works, as shown here.
   - Whole CARLA episodes won't match one for one, because the env has unseeded randomness, so compare success, collision and speed **rates** over many episodes.
6. **Safety metric.** Upstream's TTC is 0 when no car is ahead, and it ignores other lanes, including the cross traffic in the right turn. Should we measure the minimum distance from the ground-truth polygons instead?
7. ~~**Model size.**~~ **Answered:** `right_turn_hard.ckpt` is exactly the 128×128 `small` model (30.5M trainable), trained in float16. So the paper's "~18M" doesn't describe the released checkpoints.
