"""Step 4b tests: action decoding matches CarlaBaseEnv.get_vehicle_control for the right-turn tasks."""

import numpy as np
import pytest

from wm_pipeline.actions import ActionConfig, decode_action

CONTINUOUS = ActionConfig(discrete=False)


def test_there_are_15_discrete_actions():
    assert ActionConfig().n_actions == 15


@pytest.mark.parametrize(
    "index, expected",
    [
        # Rows from the action table in notes/codebase_walkthrough.md section 4.1.
        (0, (-2.0, -0.9)),  # brake, hard right
        (2, (-2.0, 0.0)),  # brake, straight
        (7, (0.0, 0.0)),  # coast, straight
        (10, (2.0, -0.9)),  # gas, hard right
        (14, (2.0, 0.9)),  # gas, hard left
    ],
)
def test_discrete_index_decodes_like_upstream(index, expected):
    assert decode_action(index) == expected


def test_numpy_integer_index_is_accepted():
    # The planner will hand over numpy integers, not Python ints.
    assert decode_action(np.int64(7)) == (0.0, 0.0)


@pytest.mark.parametrize("index", [-1, 15])
def test_out_of_range_index_raises(index):
    with pytest.raises(ValueError, match="in \\[0, 15\\)"):
        decode_action(index)


@pytest.mark.parametrize("bad", [3.0, True, np.array([0.0, 1.0])])
def test_non_integer_discrete_action_raises(bad):
    with pytest.raises(TypeError, match="integer index"):
        decode_action(bad)


def test_continuous_action_passes_through_in_range():
    assert decode_action(np.array([1.5, -0.2]), CONTINUOUS) == (1.5, -0.2)


def test_continuous_action_is_clipped_to_its_range():
    assert decode_action([5.0, -2.0], CONTINUOUS) == (3.0, -1.0)


def test_continuous_action_must_be_acc_and_steer():
    with pytest.raises(ValueError, match="shape \\(2,\\)"):
        decode_action(7, CONTINUOUS)
