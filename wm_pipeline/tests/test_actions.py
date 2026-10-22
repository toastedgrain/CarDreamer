"""Step 4b tests: action decoding matches CarlaBaseEnv.get_vehicle_control for the right-turn tasks."""

import dataclasses

import numpy as np
import pytest

from wm_pipeline.actions import Action, ActionConfig, decode_action

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


# --- Action -----------------------------------------------------------------------------------------


def test_action_from_index_has_all_three_forms():
    u = Action.from_index(np.int64(10))  # gas, hard right
    assert (u.index, u.acc, u.steer) == (10, 2.0, -0.9)
    assert u.env_action == 10
    assert u.one_hot.shape == (15,) and u.one_hot.dtype == np.float32
    assert u.one_hot.argmax() == 10 and u.one_hot.sum() == 1.0


def test_action_from_bad_index_raises():
    with pytest.raises(ValueError):
        Action.from_index(15)


def test_action_with_inconsistent_fields_raises():
    with pytest.raises(ValueError, match="does not match"):
        Action(index=10, acc=0.0, steer=0.0)


def test_continuous_action_is_clipped_and_has_no_one_hot():
    u = Action.from_continuous(5.0, -0.3)
    assert (u.index, u.acc, u.steer) == (None, 3.0, -0.3)
    np.testing.assert_array_equal(u.env_action, [3.0, -0.3])
    with pytest.raises(ValueError, match="no one-hot"):
        u.one_hot


def test_actions_compare_by_value_and_are_frozen():
    assert Action.from_index(7) == Action.from_index(7)
    with pytest.raises(dataclasses.FrozenInstanceError):
        Action.from_index(7).index = 3
