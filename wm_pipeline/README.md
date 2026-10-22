# wm_pipeline

Our Stage 1–3 code: observation + instruction (O_t, l) → encoder E → latent x_t → actor π → nominal action u_nom,
with a safety-filter slot before `env.step()` where the CBF will go later. The project brief is `../CLAUDE.md`;
the upstream code we build on is explained in `../notes/codebase_walkthrough.md`.

Nothing in this folder is upstream CarDreamer code, and upstream folders (`car_dreamer/`, `dreamerv3/`) stay untouched.

## Rules for this package

1. **Importing `wm_pipeline` must never import `carla` or `car_dreamer`.** Both need a CARLA install, and Phase A
   runs on a laptop without one. Code that talks to the real env imports them inside the function that needs them.
   `tests/test_smoke.py` enforces this.
2. **No edits to upstream files.** If something seems to need one, ask first.
3. **Assert shapes and dtypes at every module boundary.**

## Modules (filled in one step at a time)

| file | what it does | step | status |
|---|---|---|---|
| `observation.py` | Stage 1: `Observation` (O_t = BEV, l = route waypoints) + ground-truth state for the CBF | 4a | done |
| `mock_env.py` | fake env with CarDreamer's gym interface, so everything runs without CARLA | 4b | planned |
| `actions.py` | action index / `[acc, steer]` → `(acc, steer)`, same formula as `CarlaBaseEnv.get_vehicle_control` | 4b | planned |
| `encoder.py` | Stage 2: E(O_t, l) → x_t via DreamerV3 encoder + RSSM posterior | 5 | planned |
| `planner.py` | Stage 3: `act` (actor π → u_nom) and `imagine` (dynamics D rollouts) | 6 | planned |
| `safety_filter.py` | u_nom → u_safe; identity now, CBF later | 6 | planned |
| `runner.py` | the control loop that wires the stages to `env.step()` | 6 | planned |

## Setup (in WSL, once per machine)

```bash
sudo apt install -y python3.10-venv        # Ubuntu's Python ships without venv/pip
cd ~/projects/CarDreamer
python3 -m venv .venv                       # .venv is already ignored by upstream's .gitignore
.venv/bin/pip install numpy==1.24.3 pytest==9.1.1
```

`numpy==1.24.3` matches upstream's pin (`pyproject.toml`, `dreamerv3/requirements.txt`), so Phase A tests run on the
same numpy as the GPU machine. JAX gets added in Step 5.

## Run the tests

```bash
cd ~/projects/CarDreamer && .venv/bin/pytest wm_pipeline -v
```

`tests/__init__.py` is intentionally empty: it makes pytest put the repo root on `sys.path`, so `import wm_pipeline`
works when you run plain `pytest` from the repo root.
