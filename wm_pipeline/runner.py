"""
The control loop that wires Stages 1-3 and the safety-filter slot to an env (CLAUDE.md section 6).

One step:
    obs    = make_observation(raw, env.route_waypoints())    Stage 1: (O_t, l)
    gt     = env.ground_truth()                              Stage 1: ground truth, passed through
    x_t    = encoder.encode(obs, x_prev, a_prev, seed)       Stage 2: E
    u_nom  = planner.act(x_t, seed)                          Stage 3: pi
    u_safe = safety_filter(u_nom, gt)                        CBF slot: the identity for now
    raw, reward, done, info = env.step(u_safe.env_action)

`env` can be anything with reset(), step(), route_waypoints() and ground_truth(): MockCarDreamerEnv now, a thin
adapter around the real CarDreamer env in Phase B.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .actions import Action
from .dreamer_agent import draw_seed
from .observation import GroundTruthState, make_observation
from .safety_filter import SafetyFilter, identity_filter

TERMINAL_FLAGS = ("is_collision", "time_exceeded", "out_of_lane", "destination_reached")


@dataclass(frozen=True)
class StepRecord:
    """What happened at one control step."""

    t: int  # step index within the episode, from 0
    gt: GroundTruthState  # the state the safety filter saw, before the action
    u_nom: Action  # what the planner proposed
    u_safe: Action  # what the safety filter let through and the env executed
    reward: float
    done: bool
    flags: Dict[str, bool]  # upstream's terminal flags after the action: is_collision, time_exceeded, ...

    @property
    def filtered(self) -> bool:
        """True if the safety filter changed the action."""
        return self.u_safe != self.u_nom


@dataclass(frozen=True)
class EpisodeLog:
    """Every step of one episode, plus the summaries a with/without-CBF comparison needs."""

    steps: Tuple[StepRecord, ...]

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_reward(self) -> float:
        return float(sum(step.reward for step in self.steps))

    @property
    def terminal_reason(self) -> Optional[str]:
        """The terminal condition that ended the episode, or None if max_steps cut it short."""
        if not self.steps or not self.steps[-1].done:
            return None
        return next((flag for flag in TERMINAL_FLAGS if self.steps[-1].flags.get(flag)), "unknown")

    @property
    def n_filtered(self) -> int:
        """How many steps the safety filter changed the action."""
        return sum(step.filtered for step in self.steps)


def run_episode(
    env,
    encoder,
    planner,
    safety_filter: SafetyFilter = identity_filter,
    seed: Union[int, np.random.Generator] = 0,
    max_steps: Optional[int] = None,
) -> EpisodeLog:
    """
    Drive one episode with Stages 1-3 and the safety filter, and record every step.

    :param seed: an int, or a generator to draw the per-step seeds from (for example a copy of upstream's agent.rng
        when reproducing upstream's eval). Each step draws one seed, shared by the encoder and the planner, just as
        upstream draws one per policy call.
    :param max_steps: stop after this many steps even if the episode has not ended.
    """
    rng = np.random.default_rng(seed)  # returns `seed` itself if it is already a generator
    raw = env.reset()
    x_prev, a_prev = None, None
    steps = []
    for t in itertools.count():
        if max_steps is not None and t >= max_steps:
            break
        obs = make_observation(raw, env.route_waypoints())  # Stage 1
        gt = env.ground_truth()
        step_seed = draw_seed(rng)
        x = encoder.encode(obs, x_prev, a_prev, seed=step_seed)  # Stage 2
        u_nom = planner.act(x, seed=step_seed)  # Stage 3
        u_safe = safety_filter(u_nom, gt)  # CBF slot
        raw, reward, done, info = env.step(u_safe.env_action)

        flags = {flag: bool(np.any(info[flag])) for flag in TERMINAL_FLAGS if flag in info}
        steps.append(StepRecord(t=t, gt=gt, u_nom=u_nom, u_safe=u_safe, reward=float(reward), done=bool(done), flags=flags))
        # The world model must be told the action that was actually executed, not the one it proposed.
        x_prev, a_prev = x, u_safe.one_hot
        if done:
            break
    return EpisodeLog(steps=tuple(steps))
