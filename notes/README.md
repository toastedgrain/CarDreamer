# notes/

Our own notes for the World Model + CBF project (AICPS lab, UC Irvine).
Nothing in this folder is upstream CarDreamer code; the project brief is in `../CLAUDE.md`.

| File | What it covers | Written in step |
|---|---|---|
| `README.md` | this index + the step log below | 1 |
| `codebase_walkthrough.md` | one `step()` of `carla_right_turn_hard`: config → env → observer → BEV → reward → termination | 2 |
| `stages_1_3.md` | each stage 1–3 module: inputs/outputs/shapes, and how it maps to (O_t, l), E, D, R, π, C, u_nom | 7 |

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
