"""Step 10 tests: recording an episode and lining the real BEVs up with the imagined ones, on the mock env with a random
xsmall agent; plus the figure, PNG and XLA-flag helpers. The `encoder` and `planner` fixtures come from conftest.py.
"""

import os
import struct
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from wm_pipeline.dreamer_agent import TRITON_OFF, avoid_rtx5080_compile_crash
from wm_pipeline.figures import BLOCK_GAP, ROW_GAP, real_vs_imagined_grid, write_png
from wm_pipeline.mock_env import MockCarDreamerEnv
from wm_pipeline.real_vs_imagined import RecordingEncoder, compare_with_imagination
from wm_pipeline.runner import run_episode

# The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
pytestmark = pytest.mark.filterwarnings("ignore:.*truncated to dtype int32.*")

EPISODE_LENGTH = 12


class SpyPlanner:
    """The real planner, also remembering what imagine() was given."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def act(self, x, seed=None):
        return self.inner.act(x, seed=seed)

    def imagine(self, x, actions, seed=None):
        self.calls.append((x, list(actions)))
        return self.inner.imagine(x, actions, seed=seed)


class PerfectPlanner:
    """Imagines exactly the real frames that followed, which only works if the comparison lines up the right ones."""

    def __init__(self, recorder):
        self.recorder = recorder

    def imagine(self, x, actions, seed=None):
        start = next(t for t, latent in enumerate(self.recorder.latents) if latent is x)
        return SimpleNamespace(bev=np.stack(self.recorder.bevs[start + 1 : start + 1 + len(actions)]))


@pytest.fixture(scope="module")
def recorded(encoder, planner):
    recorder = RecordingEncoder(encoder)
    log = run_episode(MockCarDreamerEnv(seed=0, episode_length=EPISODE_LENGTH), recorder, planner, seed=0)
    return log, recorder


def test_recorder_keeps_one_frame_and_latent_per_step_of_the_current_episode(encoder, planner):
    recorder = RecordingEncoder(encoder)
    log = run_episode(MockCarDreamerEnv(seed=0, episode_length=6), recorder, planner, seed=0)
    assert len(recorder.bevs) == len(recorder.latents) == log.length == 6
    assert recorder.bevs[0].shape == (128, 128, 3) and recorder.bevs[0].dtype == np.uint8
    run_episode(MockCarDreamerEnv(seed=1, episode_length=4), recorder, planner, seed=0)
    assert len(recorder.bevs) == len(recorder.latents) == 4


def test_comparison_feeds_the_executed_actions_and_the_real_frames_that_followed(recorded, planner):
    log, recorder = recorded
    spy = SpyPlanner(planner)
    comparison = compare_with_imagination(log, recorder, spy, start=3, horizon=6, seed=0)

    (x, actions), = spy.calls
    assert x is recorder.latents[3]
    assert actions == [log.steps[t].u_safe for t in range(3, 9)]
    np.testing.assert_array_equal(comparison.real, np.stack(recorder.bevs[4:10]))
    assert comparison.real.shape == comparison.imagined.shape == (6, 128, 128, 3)
    assert comparison.imagined.dtype == np.uint8
    assert comparison.model_error.shape == comparison.no_change_error.shape == (6,)
    expected = np.abs(recorder.bevs[4].astype(np.int16) - recorder.bevs[3].astype(np.int16)).mean()
    assert comparison.no_change_error[0] == pytest.approx(expected)


def test_a_perfect_imagination_has_zero_error(recorded):
    # The mock's frames are random noise, so comparing against any other frame would give a large error.
    log, recorder = recorded
    comparison = compare_with_imagination(log, recorder, PerfectPlanner(recorder), start=2, horizon=5)
    np.testing.assert_array_equal(comparison.model_error, np.zeros(5))
    assert (comparison.no_change_error > 0).all()


def test_the_horizon_must_end_before_the_episode_does(recorded, planner):
    log, recorder = recorded
    compare_with_imagination(log, recorder, PerfectPlanner(recorder), start=EPISODE_LENGTH - 7, horizon=6)  # last frame fits
    with pytest.raises(ValueError, match="don't fit"):
        compare_with_imagination(log, recorder, planner, start=EPISODE_LENGTH - 6, horizon=6)
    with pytest.raises(ValueError, match="don't fit"):
        compare_with_imagination(log, recorder, planner, start=-1, horizon=6)


def test_rejects_a_recording_of_another_episode(recorded, planner):
    log, _ = recorded
    other = RecordingEncoder(None)
    other.bevs, other.latents = [np.zeros((128, 128, 3), np.uint8)] * 3, [None] * 3
    with pytest.raises(ValueError, match="same episode"):
        compare_with_imagination(log, other, planner, start=0, horizon=1)


def test_grid_puts_real_above_imagined_for_each_start():
    rng = np.random.default_rng(0)
    frames = lambda: rng.integers(0, 256, size=(6, 128, 128, 3), dtype=np.uint8)
    comparisons = [SimpleNamespace(real=frames(), imagined=frames()) for _ in range(2)]
    grid = real_vs_imagined_grid(comparisons, every=3)  # shows steps 2 and 5 of each horizon
    block = 2 * 128 + ROW_GAP
    assert grid.shape == (2 * block + BLOCK_GAP, 2 * 128, 3)
    np.testing.assert_array_equal(grid[:128, :128], comparisons[0].real[2])
    np.testing.assert_array_equal(grid[128 + ROW_GAP : block, 128:], comparisons[0].imagined[5])
    np.testing.assert_array_equal(grid[block + BLOCK_GAP : block + BLOCK_GAP + 128, :128], comparisons[1].real[2])
    assert (grid[block : block + BLOCK_GAP] == 255).all()


def test_write_png_round_trips(tmp_path):
    image = np.random.default_rng(0).integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    path = tmp_path / "x.png"
    write_png(path, image)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (7, 5)
    idat_length = struct.unpack(">I", data[33:37])[0]
    rows = zlib.decompress(data[41 : 41 + idat_length])
    decoded = np.frombuffer(rows, np.uint8).reshape(5, 1 + 7 * 3)[:, 1:].reshape(5, 7, 3)  # drop each row's filter byte
    np.testing.assert_array_equal(decoded, image)
    with pytest.raises(ValueError, match="uint8"):
        write_png(path, image.astype(np.float32))


def test_rtx5080_flag_is_added_once_and_keeps_other_flags(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_some_other_flag=true")
    avoid_rtx5080_compile_crash()
    avoid_rtx5080_compile_crash()
    assert os.environ["XLA_FLAGS"] == f"--xla_gpu_some_other_flag=true {TRITON_OFF}"
    monkeypatch.delenv("XLA_FLAGS")
    avoid_rtx5080_compile_crash()
    assert os.environ["XLA_FLAGS"] == TRITON_OFF
