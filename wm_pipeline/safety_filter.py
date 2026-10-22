"""
The safety-filter slot between the planner and env.step() (CLAUDE.md section 6).

A safety filter maps the planner's nominal action and the simulator's ground truth to the action that is actually
executed: u_safe = safety_filter(u_nom, gt). The CBF stage will solve

    u_safe = argmin ||u - u_nom||^2   subject to   h_dot(x) >= -alpha(h(x)),

reading the state x from `gt`. Until then the filter is the identity, so the pipeline drives exactly like upstream's
agent: the "without CBF" baseline.
"""

from typing import Callable

from .actions import Action
from .observation import GroundTruthState

SafetyFilter = Callable[[Action, GroundTruthState], Action]


def identity_filter(u_nom: Action, gt: GroundTruthState) -> Action:
    """Pass u_nom through unchanged."""
    return u_nom
