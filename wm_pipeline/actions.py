"""
Action decoding: an action from the planner -> the (acc, steer) command CarDreamer executes.

Mirrors CarlaBaseEnv.get_vehicle_control (car_dreamer/carla_base_env.py:105-125):
- discrete:   index a in [0, 15)  ->  acc = discrete_acc[a // 5], steer = discrete_steer[a % 5]
- continuous: [acc, steer]         ->  clipped to [-3, 3] x [-1, 1]

Units, as in CarDreamer: `acc` is a pedal command (acc / 3 becomes throttle or brake), not m/s^2, and
`steer` is a normalized command in [-1, 1] that the env negates for CARLA, so a negative steer turns right.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

# carla_right_turn_* values: tasks.yaml:122-123 overrides common.yaml's default steer set (+-0.6, +-0.2).
RIGHT_TURN_DISCRETE_ACC = (-2.0, 0.0, 2.0)
RIGHT_TURN_DISCRETE_STEER = (-0.9, -0.3, 0.0, 0.3, 0.9)


@dataclass(frozen=True)
class ActionConfig:
    """The action part of CarDreamer's env config (`env.action`), with right-turn defaults."""

    discrete: bool = True
    discrete_acc: Tuple[float, ...] = RIGHT_TURN_DISCRETE_ACC
    discrete_steer: Tuple[float, ...] = RIGHT_TURN_DISCRETE_STEER
    continuous_acc: Tuple[float, float] = (-3.0, 3.0)
    continuous_steer: Tuple[float, float] = (-1.0, 1.0)

    @property
    def n_actions(self) -> int:
        """Number of discrete actions: 3 acc values x 5 steer values = 15."""
        return len(self.discrete_acc) * len(self.discrete_steer)


def decode_action(action, config: ActionConfig = ActionConfig()) -> Tuple[float, float]:
    """Return the (acc, steer) command that `action` stands for under `config`."""
    if config.discrete:
        # bool is a subclass of int in Python, so True would otherwise pass as action 1.
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"discrete action must be an integer index, got {action!r}")
        if not 0 <= action < config.n_actions:
            raise ValueError(f"discrete action must be in [0, {config.n_actions}), got {action}")
        n_steer = len(config.discrete_steer)
        return config.discrete_acc[action // n_steer], config.discrete_steer[action % n_steer]

    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,):
        raise ValueError(f"continuous action must have shape (2,) = [acc, steer], got {action.shape}")
    # The env itself does not clip; DreamerV3's ClipAction wrapper and CARLA's own limits do.
    # Clipping here reports the command that actually gets executed.
    acc = float(np.clip(action[0], *config.continuous_acc))
    steer = float(np.clip(action[1], *config.continuous_steer))
    return acc, steer


@dataclass(frozen=True)
class Action:
    """
    One driving command: u_nom from the planner, or u_safe from the safety filter.

    Discrete mode: `index` in [0, 15) together with its decoded (acc, steer).
    Continuous mode: `index` is None and (acc, steer) is the command itself.
    Build one with Action.from_index() or Action.from_continuous(), which keep the fields consistent.
    """

    index: Optional[int]
    acc: float
    steer: float
    config: ActionConfig = ActionConfig()

    def __post_init__(self):
        if self.index is None:
            if self.config.discrete:
                raise ValueError("a discrete-mode action needs an index")
            expected = decode_action(np.array([self.acc, self.steer]), self.config)
        else:
            expected = decode_action(self.index, self.config)
        if (self.acc, self.steer) != expected:
            raise ValueError(f"(acc, steer) = {(self.acc, self.steer)} does not match the decoded command {expected}")

    @classmethod
    def from_index(cls, index: int, config: ActionConfig = ActionConfig()) -> "Action":
        acc, steer = decode_action(index, config)
        return cls(index=int(index), acc=acc, steer=steer, config=config)

    @classmethod
    def from_continuous(cls, acc: float, steer: float, config: ActionConfig = ActionConfig(discrete=False)) -> "Action":
        acc, steer = decode_action(np.array([acc, steer]), config)
        return cls(index=None, acc=acc, steer=steer, config=config)

    @property
    def one_hot(self) -> np.ndarray:
        """(15,) float32 with a 1 at `index`: the agent's own action format, which the encoder needs back."""
        if self.index is None:
            raise ValueError("a continuous action has no one-hot form")
        vector = np.zeros(self.config.n_actions, np.float32)
        vector[self.index] = 1.0
        return vector

    @property
    def env_action(self) -> Union[int, np.ndarray]:
        """What env.step() takes: the index in discrete mode, [acc, steer] in continuous mode."""
        return self.index if self.index is not None else np.array([self.acc, self.steer])
