"""
Action decoding: an action from the planner -> the (acc, steer) command CarDreamer executes.

Mirrors CarlaBaseEnv.get_vehicle_control (car_dreamer/carla_base_env.py:105-125):
- discrete:   index a in [0, 15)  ->  acc = discrete_acc[a // 5], steer = discrete_steer[a % 5]
- continuous: [acc, steer]         ->  clipped to [-3, 3] x [-1, 1]

Units, as in CarDreamer: `acc` is a pedal command (acc / 3 becomes throttle or brake), not m/s^2, and
`steer` is a normalized command in [-1, 1] that the env negates for CARLA, so a negative steer turns right.
"""

from dataclasses import dataclass
from typing import Tuple

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
