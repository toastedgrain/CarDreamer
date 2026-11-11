#!/usr/bin/env bash
# Evaluate our pipeline on several CARLA tasks, one after another, and commit and push each task's results as soon
# as it finishes, so they show up on GitHub while nobody is at the lab PC. Start CARLA first, then:
#
#   cd ~/projects/CarDreamer
#   bash wm_pipeline/scripts/eval_and_push.sh --steps 500 --tag smoke --no-push   # quick check, nothing pushed
#   bash wm_pipeline/scripts/eval_and_push.sh                                      # the full runs
#
# Options: --tasks "a b c"  (default: carla_left_turn_hard carla_lane_merge carla_navigation)
#          --steps N        (default 50000, upstream's eval length)    --tag NAME (default: today's date)
#          --port N         (default 2000)                             --no-push  (commit only)
#
# Each task's results go to notes/results/<tag>/<task>/:
#   metrics.jsonl    one line per episode, what upstream's eval_stats.py reads
#   eval_stats.txt   eval_stats.py's summary (success, collisions, speed, ...)
#   run_tail.txt     the last 200 lines the run printed (the whole log stays in logdir/; .txt because .gitignore
#                    ignores *.log, and `git add` would skip it without a word)
#
# Pushing is written for the lab PC, whose Ubuntu has no internet: the commit goes to the Windows bare mirror
# (C:\Users\Researcher\CarDreamer.git), and Windows' git.exe pushes it to GitHub by URL. Never `git.exe push origin`
# from that mirror: it is a `clone --mirror`, so that would also delete every GitHub branch the mirror lacks.
# Everything is checked before the first run starts, including a dry-run push, so a GitHub sign-in happens now
# rather than hours later. A failed task still gets its log committed and pushed, then the loop stops.
#
# For testing elsewhere, PYTHON, WIN_GIT, MIRROR, MIRROR_WIN and GITHUB_URL can be set in the environment.

set -u

TASKS="carla_left_turn_hard carla_lane_merge carla_navigation"
STEPS=50000
TAG=$(date +%F)
PORT=2000
PUSH=1
PYTHON=${PYTHON:-.venv/bin/python}
WIN_GIT=${WIN_GIT:-git.exe}
MIRROR=${MIRROR:-/mnt/c/Users/Researcher/CarDreamer.git}
GITHUB_URL=${GITHUB_URL:-https://github.com/toastedgrain/CarDreamer.git}

usage() { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit "${1:-0}"; }
fail() { echo "STOPPED: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --tasks) TASKS=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    -h | --help) usage ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done

# ---- Checks, all before the first run: each run is long, so anything that would fail later should fail now. ----
[ -d wm_pipeline ] && [ -d dreamerv3 ] || fail "run this from the repo root (~/projects/CarDreamer)"
[ "$(git branch --show-current)" = dev ] || fail "not on the dev branch"
git diff --cached --quiet || fail "there are staged changes; the automatic commits must hold only results"
[ -n "$(git config user.name)" ] && [ -n "$(git config user.email)" ] ||
  fail "git doesn't know who you are: run  git config --global user.name ...  and  git config --global user.email ...  once"
[ ! -e "notes/results/$TAG" ] || fail "notes/results/$TAG already exists: pick another --tag"
for task in $TASKS; do
  [ ! -e "logdir/runner_${TAG}_${task}" ] || fail "logdir/runner_${TAG}_${task} already exists: pick another --tag"
done

# Every task's checkpoint, so a missing download shows up now and not after the first task's run.
"$PYTHON" -c '
import sys
from wm_pipeline.dreamer_agent import CHECKPOINT_DIR, PRETRAINED_CHECKPOINTS
for task in sys.argv[1:]:
    if task not in PRETRAINED_CHECKPOINTS:
        sys.exit(f"unknown task {task}; known: {sorted(PRETRAINED_CHECKPOINTS)}")
    path = CHECKPOINT_DIR.expanduser() / PRETRAINED_CHECKPOINTS[task][0]
    if not path.is_file():
        sys.exit(f"no checkpoint at {path}: download it first")
' $TASKS || fail "the task and checkpoint check failed (see above)"

"$PYTHON" -c '
import sys, carla
client = carla.Client("127.0.0.1", int(sys.argv[1]))
client.set_timeout(10.0)
print("CARLA", client.get_server_version(), "is up")
' "$PORT" || fail "can't reach CARLA on port $PORT: start CarlaUE4.exe first"

if [ "$PUSH" = 1 ]; then
  MIRROR_WIN=${MIRROR_WIN:-$(wslpath -w "$MIRROR")}
  git fetch -q "$MIRROR" dev || fail "can't read the mirror at $MIRROR"
  git merge-base --is-ancestor FETCH_HEAD HEAD || fail "the mirror's dev has commits this checkout doesn't: pull first"
  echo "Checking that Windows git can push to GitHub (sign in if a window asks)..."
  "$WIN_GIT" -C "$MIRROR_WIN" push --dry-run "$GITHUB_URL" dev ||
    fail "Windows git can't push to $GITHUB_URL: not signed in, or the mirror is behind GitHub (fetch it, then pull)"
fi

# ---- Runs ----
commit_and_push() {  # $1 = task, $2 = the run's exit status
  local subject="Eval results: $1, $STEPS steps ($TAG)"
  [ "$2" = 0 ] || subject="$subject, run stopped with an error"
  git add "notes/results/$TAG/$1"
  git commit -q -m "$subject" || { echo "WARNING: nothing to commit for $1"; return 0; }
  [ "$PUSH" = 1 ] || return 0
  # Into the mirror with Linux git, then from the mirror to GitHub with Windows git (Ubuntu here has no internet).
  if git push -q "$MIRROR" HEAD:dev && "$WIN_GIT" -C "$MIRROR_WIN" push -q "$GITHUB_URL" dev; then
    echo "pushed: $subject"
  else
    echo "WARNING: push failed; the results are committed here, so push them later"
  fi
}

mkdir -p logdir
for task in $TASKS; do
  logdir="logdir/runner_${TAG}_${task}"
  out="notes/results/$TAG/$task"
  mkdir -p "$out"
  echo "===== $task: $STEPS steps; results will be in $out"
  PYTHONPATH=. "$PYTHON" -u wm_pipeline/scripts/run_carla_eval.py --task "$task" --steps "$STEPS" --port "$PORT" \
    --logdir "$logdir" 2>&1 | tee "$logdir.log"
  status=${PIPESTATUS[0]}
  tail -n 200 "$logdir.log" > "$out/run_tail.txt"
  if [ -s "$logdir/metrics.jsonl" ]; then  # missing if the run failed before finishing an episode
    cp "$logdir/metrics.jsonl" "$out/"
    "$PYTHON" dreamerv3/eval_stats.py --logdir "$logdir" > "$out/eval_stats.txt" 2>&1
    cat "$out/eval_stats.txt"
  fi
  commit_and_push "$task" "$status"
  [ "$status" = 0 ] || fail "$task's run exited with status $status; its whole log is $logdir.log"
done
echo "ALL DONE: results in notes/results/$TAG/"
