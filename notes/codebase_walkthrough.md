# Codebase walkthrough: one `step()` of `carla_right_turn_hard`

This note follows a single call to `env.step(action)` through CarDreamer, from config to termination.
It was written from the code at upstream commit `1601324`; every `file:line` is relative to the repo root.
Nothing here needs CARLA running. The config numbers were checked by loading the YAML files directly (see "Verify it yourself" at the end).

---

## 0. The whole step on one screen

```
action (int 0..14)                                   carla_base_env.py:140  step()
  │
  ├─ 1. apply_control(action)                        carla_wpt_env.py:46
  │      index → (acc, steer) → throttle/brake/steer → ego.apply_control()
  │
  ├─ 2. self._world.step()                           world_manager.py:91
  │      CARLA world.tick()  → physics advances exactly 0.1 s (10 Hz, synchronous mode)
  │      then the env's on_step() callback:
  │        CarlaRightTurnEnv.on_step   → delete cross-traffic cars that left the area
  │        CarlaWptEnv.on_step         → route planner.run_step() → self.waypoints
  │        CarlaWptFixedEnv.on_step    → maybe spawn a new cross-traffic car
  │
  ├─ 3. env_state = get_state()                      carla_wpt_env.py:43
  │      {"ego_waypoints": [...], "timesteps": t}
  │
  ├─ 4. _is_terminal()                               carla_base_env.py:127
  │      collision / time limit / out of lane / destination reached
  │
  ├─ 5. obs = observer.get_observation(env_state)    observer.py:71
  │      {"camera": 128×128×3, "collision": (1,), "birdeye_wpt": 128×128×3}
  │
  ├─ 6. reward()                                     carla_wpt_env.py:54
  │      waypoint + speed − collision − out-of-lane + destination − time
  │
  └─ 7. return (obs, reward, done, info)             carla_base_env.py:163   (old gym 4-tuple API)
```

The order matters: **termination (4) is checked before the observation (5) is refreshed.** See §4.4 for the consequence.

---

## 1. Config: how the name `carla_right_turn_hard` becomes settings

**Loading** (`car_dreamer/__init__.py:30-50`, `load_task_configs`):
1. Read `car_dreamer/configs/common.yaml` (shared defaults for every task).
2. Read `car_dreamer/configs/tasks.yaml` and take the `carla_right_turn_hard` entry.
3. Merge them with `Config.update` (`car_dreamer/toolkit/config.py:102-127`). Both are flattened to dotted keys
   like `env.action.discrete_steer`, and the task's values overwrite the defaults.
   Values are cast to the type of the old value; for example, an int can't silently become a fractional float.

`create_task` (`__init__.py:53-67`) then applies command-line overrides (`--env.world.carla_port 2000` etc., via
`toolkit/flags.py`) and calls `gym.make(config.env.name, config=config.env)`. Only the `env` part of the config goes to the env.
Env names are registered automatically from filenames (`__init__.py:70-86`): `carla_right_turn_env.py` → `CarlaRightTurnEnv-v0`.

**Inheritance inside `tasks.yaml`** (YAML anchors):
```
carla_wpt  (tasks.yaml:1-16)            reward + terminal settings shared by waypoint tasks
   └─ carla_right_turn_simple (117-140) Town03, the route, steer set, flow spawn point
        ├─ carla_right_turn_medium (142-145)   + flow distance 8–16 m
        └─ carla_right_turn_hard   (147-150)   + flow distance 6–8 m
```
**simple, medium and hard differ only in how densely cross-traffic is spawned.** `simple` sets no `min_flow_dist`,
so it spawns **no cross-traffic at all** (`carla_wpt_fixed_env.py:53`).

**Effective settings for `carla_right_turn_hard`** (printed from the merged config):

| Setting | Value | Where it comes from |
|---|---|---|
| env class | `CarlaRightTurnEnv-v0` | tasks.yaml:121 |
| map / sim step | Town03, `fixed_delta_seconds` 0.1 s (10 Hz) | tasks.yaml:120, common.yaml:8 |
| observations | `camera`, `collision`, `birdeye_wpt` | tasks.yaml:124 |
| action space | discrete, acc ∈ {−2, 0, +2}, **steer ∈ {−0.9, −0.3, 0, 0.3, 0.9}** | common.yaml:165-169, **overridden** at tasks.yaml:122-123 |
| desired speed | 4 m/s | tasks.yaml:4 |
| reward scales | waypoint 2.0, speed 0.5, collision 30, out_of_lane 3.0, time 0, destination 20 | tasks.yaml:5-13 |
| termination | time_limit 500 steps, out_lane_thres 3 m | tasks.yaml:14-16 |
| ego start | x=−33.8, y=−135.1, yaw 0° | tasks.yaml:126 |
| ego route | (−33.8, −135.1) → (−5.0, −110.3), following the road | tasks.yaml:127-128 |
| cross-traffic | spawned at (−3.4, −151.2), yaw 90°, 6–8 m apart | tasks.yaml:129, 149-150 |
| world model input | `encoder.cnn_keys: "birdeye_wpt"` (only the BEV) | tasks.yaml:132 |

> The steer set in our CLAUDE.md (±0.6, ±0.2) is the `common.yaml` default. **All right-turn tasks use ±0.9, ±0.3.**

---

## 2. Who implements what (class stack)

```
gym.Env
 └─ CarlaBaseEnv        carla_base_env.py     step()/reset() skeleton, action decoding, terminal bookkeeping
     └─ CarlaWptEnv     carla_wpt_env.py      waypoint-following: get_state, reward, terminal conditions
         └─ CarlaWptFixedEnv  carla_wpt_fixed_env.py  fixed route + spawning the car flow
             └─ CarlaRightTurnEnv  carla_right_turn_env.py  only removes flow cars that left the junction
```

The `toolkit/` helpers the env owns (created in `CarlaBaseEnv.__init__`, `carla_base_env.py:13-23`):
- `WorldManager` (`toolkit/carla_manager/world_manager.py`) connects to CARLA, ticks the world, spawns actors and
  tracks them. It also answers ground-truth questions: `actor_transforms`, `actor_polygons`, `actor_actions`.
- `Observer` (`toolkit/observer/observer.py`) is one *handler* per enabled observation key.
- `EnvMonitorOpenCV` is the live display (only for watching; it doesn't affect the agent).

Hook pattern: the env registers its `on_reset` and `on_step` methods as callbacks on the WorldManager (`carla_base_env.py:18-19`).
The WorldManager calls them at the right moment, i.e. right after the simulator ticks.

---

## 3. Before the first step: `reset()` (`carla_base_env.py:92-103`)

1. `observer.destroy()` removes the old sensors.
2. `world.reset()` (`world_manager.py:76-89`): destroy every actor it spawned, turn synchronous mode **off**, call `on_reset`, turn synchronous mode **on**, then sleep 1 s.
   - `CarlaWptFixedEnv.on_reset` (`carla_wpt_fixed_env.py:26-46`): spawns the ego at `lane_start_point`, builds a
     `FixedPathPlanner` along `ego_path`, computes the first waypoints, and prepares an empty car-flow queue.
   - The ego is a random `vehicle.audi*` model (`world_manager.py:144-149`), so **the ego's size can change between episodes**.
3. `observer.reset(ego)` attaches the camera and collision sensors to the new ego (`sensor_handlers.py:63-65`).
4. `_time_step = 0`, and the first observation is returned.

---

## 4. One `step(action)`, in order

### 4.1 Action → vehicle control (`carla_base_env.py:105-125`)
With 5 steer values, `acc = discrete_acc[action // 5]` and `steer = discrete_steer[action % 5]`.

| index | acc | steer | throttle | brake | CARLA steer |
|---|---|---|---|---|---|
| 0–4 | −2 (brake) | −0.9, −0.3, 0, 0.3, 0.9 | 0 | 0.667 | +0.9 … −0.9 |
| 5–9 | 0 (coast) | same | 0 | 0 | +0.9 … −0.9 |
| 10–14 | +2 (gas) | same | 0.667 | 0 | +0.9 … −0.9 |

- `throttle = acc/3` and `brake = −acc/3`, each clipped to [0, 1]. So "±2" means 2/3 pedal, not 2 m/s².
- **The env flips the sign: CARLA steer = −steer.** CARLA's positive steer turns right, so in this action space a
  **negative steer turns right** (indices 0, 1, 5, 6, 10, 11).
- CARLA steer is a normalized command in [−1, 1] (fraction of the max wheel angle), not an angle in radians.
- Continuous mode (`action.discrete: False`) skips the table and uses `action = [acc, steer]` directly (`carla_base_env.py:114-116`).

`CarlaWptEnv.apply_control` (`carla_wpt_env.py:46-48`) sends this control to the ego. It takes effect during the next tick.

### 4.2 The simulator advances 0.1 s (`world_manager.py:91-95`)
`world.tick()` advances physics by exactly `fixed_delta_seconds` = 0.1 s, because synchronous mode is on.
Then the env's `on_step` chain runs (each class calls `super().on_step()`):

1. `CarlaRightTurnEnv.on_step` (`carla_right_turn_env.py:12-19`): if the oldest flow car has left the area
   (y > −81.2, x < −38.4, or x > 31.6), destroy it.
2. `CarlaWptEnv.on_step` (`carla_wpt_env.py:50-52`): `planner.run_step()` updates `self.waypoints` and `num_completed` (§6).
3. `CarlaWptFixedEnv.on_step` (`carla_wpt_fixed_env.py:48-65`): decides whether to spawn the next flow car. It draws
   `flow_dist ~ Uniform(6, 8)` m **anew every step** and spawns when the newest flow car is at least that far from the spawn point.
   - Flow cars come from `try_spawn_aggresive_actor` (`world_manager.py:232-252`): they're on autopilot and **ignore
     traffic lights and other vehicles 100%**, so they won't brake for the ego. The right turn means merging into
     a dense stream of cars that don't yield.

### 4.3 Environment state (`carla_wpt_env.py:43-44`)
`{"ego_waypoints": self.waypoints, "timesteps": self._time_step}`. This is what the BEV renderer uses to draw the route.

### 4.4 Termination (`carla_base_env.py:127-138`, conditions at `carla_wpt_env.py:137-146`)

| Condition | Rule |
|---|---|
| `is_collision` | `self.obs["collision"][0] > 0` |
| `time_exceeded` | `_time_step > 500` (after 501 steps = 50.1 s) |
| `out_of_lane` | distance from ego to the **nearest remaining waypoint** > 3 m |
| `destination_reached` | 3 or fewer waypoints left in the route |

`done` is True if any of them is True. Each flag is returned in `info` as a 1-element bool array.

> **One-step collision lag.** `_is_terminal()` runs *before* `self.obs` is refreshed (`carla_base_env.py:146-147`), so
> `is_collision` reads the **previous** step's collision sensor. The step on which a collision first appears in the
> observation gets the collision penalty but `done=False`. The **next** step ends the episode, and gets the penalty again,
> because the collision value never resets (§5.2). This comes from reading the code; confirm it in Phase B by logging
> `obs["collision"]`, `reward` and `done` around a crash.

### 4.5 Observation (`observer.py:71-79`)
The observer asks each enabled handler for its piece and merges the dicts:

| key | shape, dtype | handler | used by the world model? |
|---|---|---|---|
| `birdeye_wpt` | (128, 128, 3) uint8 | `BirdeyeHandler` (§5.1) | **yes**, the only input |
| `camera` | (128, 128, 3) uint8 | `CameraHandler` (`sensor_handlers.py:68-77`) | no, only logged as video |
| `collision` | (1,) float32 | `CollisionHandler` (`sensor_handlers.py:128-135`) | no, only used by the env |

The DreamerV3 defaults use no keys (`dreamerv3/dreamerv3.yaml:79-82`, `cnn_keys: "none"`). The task config switches on
only `birdeye_wpt`, and no vector (`mlp_keys`) inputs. So **O_t, as far as the world model is concerned, is just the BEV image.**

### 4.6 Reward (`carla_wpt_env.py:54-132`)
Details in §7. Note it runs *after* the observation refresh, so the collision term uses the current sensor value.

### 4.7 Return (`carla_base_env.py:150-163`)
`info` merges everything computed this step:
- `ego_waypoints`, `timesteps`
- the four terminal flags, plus `terminal` and `episode_timesteps`
- reward diagnostics: `num_completed`, `num_obsolete`, `travel_distance`, `ego_x`, `ego_y`, `speed_parallel`,
  `speed_perpendicular`, `speed_norm`, `wpt_dis`, `r_waypoints`, `r_speed`, `r_collision`, `r_out_of_lane`, `ttc`
- `action`

`info` does **not** include other vehicles or the ego's heading. For the CBF we'll read those from the WorldManager (§9).

---

## 5. Observations in detail

### 5.1 The BEV image `birdeye_wpt`
**Geometry** (`birdeye_handler.py:19-21`, `birdeye_renderer.py:344-352`, `birdeye_handler.py:72-76`):
- It's drawn at 512×512 px covering 32 m × 32 m (16 px/m), then shrunk to **128×128 (4 px/m, so 1 px = 0.25 m)**.
- The image is **rotated so the ego always faces up**. The ego sits at pixel (row 80, col 64) of the 128×128 image,
  so it sees **20 m ahead, 12 m behind, and 16 m to each side** (`ego_offset: 12`).

**Layers**, drawn in this order (later layers paint over earlier ones; `common.yaml:89-96`, renderer `birdeye_renderer.py:66-73`).
The arrays are **RGB** (`renderer/constants.py:40-73`):

| order | entity | looks like | source |
|---|---|---|---|
| 1 | `roadmap` | grey road surface + lane markings, pre-rendered once per map | `birdeye_renderer.py:75-81` |
| 2 | `waypoints` | **blue (0,0,255)** thick line from the ego along its route = the "instruction" | `:142-156` |
| 3 | `background_waypoints` | **orange (252,175,62)** lines = other cars' planned paths from the Traffic Manager (**intention sharing**) | `:174-201` |
| 4 | `ego_vehicle` | **red (255,0,0)** box | `:83-87` |
| 5 | `background_vehicles` | **green (0,255,0)** boxes | `:158-172` |

- Path lines are 15 px wide at 512 px, which is about 0.94 m.
- Path points already behind the car aren't drawn (`:389-398`).
- Vehicles more than 3 m above or below the ego (e.g. on a bridge) are skipped (`handlers/utils.py:189-190`).
- `observability: full` means every spawned actor is drawn (`birdeye_handler.py:47-53`). `fov` and `recursive_fov` would hide some of them.
- The roadmap layer also serves as "clearing" the canvas: `_clear_surface` is a no-op (`birdeye_renderer.py:61-64`), so `roadmap` must stay first in the list.

### 5.2 Collision sensor
A CARLA collision sensor attached to the ego writes `‖normal impulse‖` into the handler whenever a collision event fires
(`sensor_handlers.py:132-135`). It's **never reset during the episode**: after the first hit it stays > 0 until the next `reset()`.

---

## 6. The route planner (where the "instruction" l comes from)

`FixedPathPlanner` (`toolkit/planner/fixed_path_planner.py`) builds the route once, on the first `run_step()`:
- For the right turn, the single segment (−33.8, −135.1) → (−5.0, −110.3) has `use_road_waypoints = True`, so CARLA's
  `GlobalRoutePlanner` traces it **along the road lanes**, with a waypoint every 0.8 m (`fixed_path_planner.py:20, 28-39`).
- Each waypoint is a tuple **`(x, y, yaw_degrees)`**. The third value is **heading, not z** (`base_planner.py:84-96`).

Every step, `run_step()` (`base_planner.py:98-113`, `123-146`):
- finds the waypoint closest to the ego and drops every waypoint before it;
- counts waypoints within **0.5 m** of the ego as "completed" (`num_completed`), which feeds the waypoint reward;
- returns at most **60** waypoints (about 48 m of route ahead).

So `self.waypoints` (`info["ego_waypoints"]`) is a list of up to 60 `(x, y, yaw)` tuples in world coordinates, and the
same route is drawn into the BEV as the blue line.

---

## 7. The reward, term by term (`carla_wpt_env.py:54-132`)

Let `wp` be the next waypoint, `d̂` its heading direction, `v` the ego's 2-D velocity:
- `v∥ = v·d̂` is the speed along the route;
- `lat` is the ego's perpendicular distance from the line through `wp`;
- `v⊥` is the speed toward/away from that line.

| term | formula (hard) | meaning |
|---|---|---|
| waypoint | +2.0 if any waypoint was within 0.5 m this step | progress along the route |
| speed | 0.5 · (4 − \|v∥ − 4\| − 2·min(v⊥, 0.5)) | best (+2.0) at exactly 4 m/s along the route; 0 when stopped or at 8 m/s |
| collision | −30 · ‖v‖ if the collision sensor > 0 | worse the faster you hit |
| out of lane | −3.0 · (lat − 0.5) if lat > 0.5 m | drifting off the route line |
| destination | +20 if ≤ 3 waypoints left | arrival bonus |
| time | −0 per step | disabled for this task |

Two different distances are in play:
- the reward's out-of-lane term uses the **lateral** distance `lat`;
- the out-of-lane **termination** uses the **straight-line** distance to the nearest waypoint (> 3 m).

**TTC** is computed here too (`toolkit/carla_manager/utils.py:87-195`), only for logging. Things to know before using it as a safety metric:
- It only considers vehicles in the **same road and lane** as the ego, and in front of it (`utils.py:110-123`).
- It uses distance ÷ the *magnitude* of the relative velocity, not the closing speed (`utils.py:164-172`).
- **It returns 0.0 when there's no vehicle ahead** (`utils.py:195`). So `ttc = 0` means "nothing to measure", not "imminent crash".
  Cross-traffic in another lane is invisible to it.

---

## 8. Where the agent plugs in (preview of Stages 2–3)

Upstream eval (`dreamerv3/eval.py`) builds this chain around the gym env:
```
create_task(...)                       eval.py:116
 → FromGym                             eval.py:134   adds reward / is_first / is_last / is_terminal to the obs dict
 → InfoWrapper, OneHotAction, ...      eval.py:14-34 agent outputs a one-hot vector of 15; wrapper takes argmax → index
 → BatchEnv                            eval.py:136
loop: action = agent.policy(obs, state, mode="eval")   eval.py:102
```
Inside `Agent.policy` (`dreamerv3/agent.py:47-69`), which is exactly our E → x_t → π chain:
```
embed  = wm.encoder(obs)                                     # CNN on birdeye_wpt          (Stage 2: E)
latent = wm.rssm.obs_step(prev_latent, prev_action, embed, is_first)   # posterior x_t    (Stage 2)
dist   = task_behavior.policy(latent)                        # actor π(a | x_t)           (Stage 3)
action = dist.sample(seed=rng)            # eval mode SAMPLES, it doesn't take the most likely action   (agent.py:56-58)
```
The one-hot → index conversion is `embodied/core/wrappers.py:126-146`.

> **Implication for Step 9 ("our runner must reproduce upstream eval").** Eval samples actions with a JAX RNG, and the env
> itself is random (flow spacing via `np.random`, random car models, CARLA sensor callbacks arriving on their own thread).
> So even two upstream runs shouldn't be expected to produce identical episodes (not yet tested). Realistic checks are:
> - **(a)** feed the *same* recorded observations and the same RNG key to upstream `agent.policy` and to our wrappers, and require identical actions;
> - **(b)** compare success / collision / speed **rates** over many episodes.
>
> To agree with the mentor before Step 9.

---

## 9. What this means for our Stage 1 (pointers)

| our name | where it lives in CarDreamer |
|---|---|
| O_t (world-model input) | `obs["birdeye_wpt"]`, (128, 128, 3) uint8, RGB, ego-centred, ego facing up |
| l (instruction) | route waypoints `info["ego_waypoints"]`: list of ≤ 60 `(x, y, yaw_deg)`; already drawn into O_t as the blue line |
| ego ground truth | `env.unwrapped.ego` (a `carla.Actor`): `get_transform()`, `get_velocity()` |
| other vehicles' ground truth | `env.unwrapped._world.actor_transforms` / `actor_polygons` (4 corners in world x, y) / `actor_actions`. These **include the ego**, so filter by `ego.id` |
| safety metrics already logged | `is_collision`, `out_of_lane`, `ttc` (with the caveats in §7) |

Everything in the right column needs a live CARLA server. The Phase A mock env will fake these with the same shapes.

---

## 10. Gotchas found while reading (short list)

1. Right-turn steer set is **±0.9, ±0.3, 0**, not ±0.6, ±0.2 (tasks.yaml:122-123).
2. **Negative action steer = turn right** (CARLA steer = −steer).
3. Collision termination lags the observation by **one step**, and the collision penalty can land twice (§4.4).
4. `ttc == 0` means **no vehicle ahead**, and cross-traffic in other lanes is ignored (§7).
5. Out-of-lane *reward* uses lateral distance; out-of-lane *termination* uses straight-line distance to the nearest waypoint.
6. Waypoint tuples are `(x, y, yaw°)`, with no z.
7. `birdeye_wpt` also draws other cars' planned paths (orange, "intention sharing"); our CLAUDE.md entity list missed this.
8. `camera` and `collision` are in the observation, but the world model never sees them.
9. Flow cars ignore the ego completely; `simple` has no flow cars at all.
10. Eval-mode actions are sampled, so exact episode-by-episode reproduction isn't expected (§8).
11. The ego's car model (and size) is random among `vehicle.audi*` each episode, which matters for CBF safety margins later.

---

## Verify it yourself

Everything below runs on the laptop (no CARLA). From `~/projects/CarDreamer` in WSL:

```bash
# The step() order, termination before observation:
sed -n 140,163p car_dreamer/carla_base_env.py
# Right-turn steer override and flow distances:
sed -n 117,150p car_dreamer/configs/tasks.yaml
# Eval mode samples the action:
sed -n 47,69p dreamerv3/agent.py
```

To print the merged config and the 15-action table without importing `car_dreamer` (which needs `carla`), load just
`toolkit/config.py` with PyYAML: build `Config(common.yaml)`, then `.update(tasks.yaml["carla_right_turn_hard"])`, and print `config.env`.
The same check confirms that `carla_right_turn_simple` has no `min_flow_dist`.
