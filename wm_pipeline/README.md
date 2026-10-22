# wm_pipeline

Our Stage 1–3 code: observation + instruction (O_t, l) → encoder E → latent x_t → actor π → nominal action u_nom,
with a safety-filter slot before `env.step()` where the CBF will go later. The project brief is `../CLAUDE.md`;
the upstream code we build on is explained in `../notes/codebase_walkthrough.md`.

Nothing in this folder is upstream CarDreamer code, and upstream folders (`car_dreamer/`, `dreamerv3/`) stay untouched.

## Rules for this package

1. **Importing `wm_pipeline` must never import `carla` or `car_dreamer`.** Both need a CARLA install, and Phase A
   runs on a laptop without one. Code that talks to the real env imports them inside the function that needs them.
   `tests/test_smoke.py` enforces this by importing every module of the package in a fresh process.
2. **No edits to upstream files.** If something seems to need one, ask first.
3. **Assert shapes and dtypes at every module boundary.**

## Modules (filled in one step at a time)

| file | what it does | step | status |
|---|---|---|---|
| `observation.py` | Stage 1: `Observation` (O_t = BEV, l = route waypoints) + ground-truth state for the CBF | 4a | done |
| `mock_env.py` | minimal fake of the right-turn env: same `reset`/`step` interface and obs format, plus `route_waypoints()` and `ground_truth()`; no CARLA | 4b | done |
| `actions.py` | action index / `[acc, steer]` → `(acc, steer)`, same formula as `CarlaBaseEnv.get_vehicle_control` | 4b | done |
| `dreamer_agent.py` | builds a DreamerV3 agent with the config and spaces upstream's `eval.py` uses (random weights now, checkpoint in Phase B) | 5 | done |
| `encoder.py` | Stage 2: E(O_t, l) → x_t via DreamerV3 encoder + RSSM posterior; `LatentState` (deter, stoch, logit) | 5 | done |
| `planner.py` | Stage 3: `act` (actor π → u_nom) and `imagine` (dynamics D rollouts) | 6 | planned |
| `safety_filter.py` | u_nom → u_safe; identity now, CBF later | 6 | planned |
| `runner.py` | the control loop that wires the stages to `env.step()` | 6 | planned |

## Setup (in WSL, once per machine)

```bash
sudo apt install -y python3.10-venv        # Ubuntu's Python ships without venv/pip
cd ~/projects/CarDreamer
python3 -m venv .venv                       # .venv is already ignored by upstream's .gitignore
.venv/bin/pip install -r wm_pipeline/requirements-phase-a.txt
```

Every version in `requirements-phase-a.txt` is upstream's own pin (`pyproject.toml`, `dreamerv3/requirements.txt`),
with the CPU builds of jaxlib and TensorFlow. TensorFlow does no math here: it is only needed because importing
DreamerV3's `embodied` package imports it. The venv is about 1.7 GB.

## Run the tests

```bash
cd ~/projects/CarDreamer && .venv/bin/pytest wm_pipeline -v
```

About 20 s on the laptop; most of it is building one random `xsmall` DreamerV3 agent for the encoder tests.

`tests/__init__.py` is intentionally empty: it makes pytest put the repo root on `sys.path`, so `import wm_pipeline`
works when you run plain `pytest` from the repo root.
