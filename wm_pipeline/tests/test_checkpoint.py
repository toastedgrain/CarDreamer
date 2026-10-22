"""Tests for load_checkpoint, using checkpoints written in upstream's format. No download needed.

Upstream writes checkpoints with embodied.basics.pack, which is pickle.dumps({"agent": agent.save(), ...}).
"""

import os
import pickle

import jax
import numpy as np
import pytest

from wm_pipeline.dreamer_agent import CPU_OVERRIDES, build_agent, load_checkpoint, make_config

# The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
pytestmark = pytest.mark.filterwarnings("ignore:.*truncated to dtype int32.*")


@pytest.fixture(scope="module")
def target():
    # A separate agent to load into, so these tests never change the weights the other test files share.
    return build_agent(make_config("xsmall", CPU_OVERRIDES))


def _write(path, data):
    path.write_bytes(pickle.dumps(data))
    return path


def _plus_one(entry):
    """A changed copy of one checkpoint entry (float arrays + 1), so a successful load is visible."""
    # ones_like keeps the dtype: `a + 1` would turn 0-d float32 arrays into float64 under numpy 1.x promotion.
    return jax.tree_util.tree_map(lambda a: a + np.ones_like(a) if np.issubdtype(np.asarray(a).dtype, np.floating) else a, entry)


def _assert_same(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_round_trip_in_upstream_format(agent, target, tmp_path):
    weights = {key: _plus_one(entry) for key, entry in agent.save().items()}
    path = _write(tmp_path / "model.ckpt", {"agent": weights, "step": 5, "replay": None, "_timestamp": 0.0})

    extras = load_checkpoint(target, path)

    assert extras == {"step": 5, "replay": None, "_timestamp": 0.0}
    loaded = target.save()
    assert set(loaded) == set(weights)
    for key in weights:
        _assert_same(loaded[key], weights[key])


def test_matching_sha256_is_accepted_and_a_wrong_one_refused(agent, target, tmp_path):
    import hashlib

    path = _write(tmp_path / "model.ckpt", {"agent": agent.save()})
    load_checkpoint(target, path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(target, path, sha256="0" * 64)


def test_a_pickle_that_runs_a_command_is_refused_without_running_it(target, tmp_path):
    marker = tmp_path / "command_ran"

    class Malicious:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))  # plain pickle.load would run this shell command

    path = _write(tmp_path / "malicious.ckpt", {"agent": Malicious()})
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        load_checkpoint(target, path)
    assert not marker.exists()


def test_a_checkpoint_of_a_different_model_is_refused_and_nothing_changes(agent, target, tmp_path):
    weights = dict(agent.save())
    removed = sorted(weights)[0]
    del weights[removed]
    resized = next(k for k in sorted(weights) if k.endswith("/kernel"))
    weights[resized] = np.zeros((3, 3), np.float32)
    path = _write(tmp_path / "other_model.ckpt", {"agent": weights})

    before = target.save()
    with pytest.raises(ValueError, match="does not fit") as error:
        load_checkpoint(target, path)
    assert removed in str(error.value) and resized in str(error.value)
    for key, entry in before.items():
        _assert_same(target.save()[key], entry)


def test_an_optimizer_only_mismatch_points_at_the_precision(agent, target, tmp_path):
    # float16 training adds loss-scaling state like this (jaxutils.py:407-411). It is faked here because building
    # a float16 agent would switch upstream's global compute precision for every agent in this test process.
    weights = dict(agent.save())
    weights["agent/wm/model_opt/grad_scale/value"] = np.float32(1e4)
    path = _write(tmp_path / "float16_trained.ckpt", {"agent": weights})
    with pytest.raises(ValueError, match="Only optimizer state differs.*float16"):
        load_checkpoint(target, path)


def test_a_file_without_an_agent_entry_is_refused(target, tmp_path):
    path = _write(tmp_path / "empty.ckpt", {"step": 1})
    with pytest.raises(ValueError, match="no 'agent' entry"):
        load_checkpoint(target, path)
