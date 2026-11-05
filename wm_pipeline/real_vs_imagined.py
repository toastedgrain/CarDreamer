"""
Step 10: compare the world model's imagined BEV with the real one, as a sanity check of its dynamics D.

RecordingEncoder           wraps the encoder during an episode and keeps each BEV it saw and each x_t it returned.
compare_with_imagination   from x_t at one step, imagines H steps under the actions the car actually took, and
                           lines the imagined BEVs up with the real BEVs that followed them.
"""

from dataclasses import dataclass

import numpy as np

from .encoder import LatentState
from .observation import BEV_SHAPE, Observation
from .runner import EpisodeLog


class RecordingEncoder:
    """Stands in for the encoder in run_episode, and keeps this episode's BEVs and latents (index t = step t)."""

    def __init__(self, inner):
        self._inner = inner
        self.bevs = []
        self.latents = []

    def encode(self, obs: Observation, x_prev=None, a_prev=None, seed=None) -> LatentState:
        if x_prev is None:  # the first step of an episode
            self.bevs, self.latents = [], []
        x = self._inner.encode(obs, x_prev, a_prev, seed=seed)
        self.bevs.append(obs.bev)
        self.latents.append(x)
        return x


@dataclass(frozen=True)
class ImaginationComparison:
    """Real vs imagined over H steps from one start step. Index k is the frame after the (k+1)-th action."""

    start: int  # the step whose x_t the imagination starts from
    real: np.ndarray  # (H, 128, 128, 3) uint8: the BEVs the env produced
    imagined: np.ndarray  # (H, 128, 128, 3) uint8: the decoder's BEVs of the predicted x_{t+1} ... x_{t+H}
    model_error: np.ndarray  # (H,) float64: mean |real - imagined| per pixel, 0-255
    no_change_error: np.ndarray  # (H,) float64: mean |real - BEV at the start step|: a model that predicts no motion


def _mean_abs_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # int16 so the uint8 subtraction can't wrap around.
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=(1, 2, 3))


def compare_with_imagination(log: EpisodeLog, recorder: RecordingEncoder, planner, start: int, horizon: int, seed: int = 0) -> ImaginationComparison:
    """
    Imagine `horizon` steps from step `start` of a finished episode, and line them up with what really happened.

    The model gets the actions that were executed (u_safe), as the world model must always be told, and no frames
    after `start`: everything after that comes from its dynamics alone.
    """
    if len(recorder.bevs) != log.length:
        raise ValueError(f"the recorder has {len(recorder.bevs)} frames for a {log.length}-step episode: record the same episode")
    # The real frame after the last imagined action is at index start + horizon, and the episode's last frame is at
    # log.length - 1: run_episode stops before encoding the terminal frame.
    if start < 0 or start + horizon >= log.length:
        raise ValueError(f"steps {start}..{start + horizon} don't fit in a {log.length}-step episode")

    actions = [log.steps[t].u_safe for t in range(start, start + horizon)]
    imagination = planner.imagine(recorder.latents[start], actions, seed=seed)
    real = np.stack(recorder.bevs[start + 1 : start + horizon + 1])
    if real.shape != imagination.bev.shape or real.shape[1:] != BEV_SHAPE:
        raise ValueError(f"real {real.shape} and imagined {imagination.bev.shape} BEVs should both be ({horizon}, 128, 128, 3)")
    return ImaginationComparison(
        start=start,
        real=real,
        imagined=imagination.bev,
        model_error=_mean_abs_difference(real, imagination.bev),
        no_change_error=_mean_abs_difference(real, recorder.bevs[start][None]),
    )
