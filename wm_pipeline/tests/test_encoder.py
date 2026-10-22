"""Step 5 tests: the Stage 2 encoder, on a randomly initialized xsmall DreamerV3 agent on the CPU.

The `agent` and `episode` fixtures come from conftest.py.
"""

import copy

import numpy as np
import pytest

from wm_pipeline.dreamer_agent import draw_seed
from wm_pipeline.encoder import DreamerEncoder

# The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
pytestmark = pytest.mark.filterwarnings("ignore:.*truncated to dtype int32.*")

XSMALL_SHAPES = {"deter": (64,), "stoch": (32, 32), "logit": (32, 32)}


def _one_hot(index, n=15):
    action = np.zeros(n, np.float32)
    action[index] = 1.0
    return action


def test_latent_shapes_and_dtypes(agent, episode):
    x = DreamerEncoder(agent).encode(episode[0])
    for name, shape in XSMALL_SHAPES.items():
        assert getattr(x, name).shape == shape, name
        assert getattr(x, name).dtype == np.float32, name
    assert x.features().shape == (64 + 32 * 32,)


def test_stoch_is_one_hot_per_row(agent, episode):
    x = DreamerEncoder(agent).encode(episode[0])
    assert set(np.unique(x.stoch)) <= {0.0, 1.0}
    np.testing.assert_array_equal(x.stoch.sum(axis=1), np.ones(32))


def test_same_seed_gives_the_same_latent(agent, episode):
    a = DreamerEncoder(agent, rng=np.random.default_rng(5)).encode(episode[0])
    b = DreamerEncoder(agent, rng=np.random.default_rng(5)).encode(episode[0])
    for name in XSMALL_SHAPES:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def test_explicit_seed_is_the_same_as_drawing_it_from_the_generator(agent, episode):
    # The runner passes one seed per step; it must mean exactly what the encoder's own generator would have drawn.
    from_generator = DreamerEncoder(agent, rng=np.random.default_rng(7)).encode(episode[0])
    explicit = DreamerEncoder(agent).encode(episode[0], seed=draw_seed(np.random.default_rng(7)))
    for name in XSMALL_SHAPES:
        np.testing.assert_array_equal(getattr(from_generator, name), getattr(explicit, name), err_msg=name)


def test_different_seeds_sample_different_stoch_from_the_same_distribution(agent, episode):
    a = DreamerEncoder(agent, rng=np.random.default_rng(1)).encode(episode[0])
    b = DreamerEncoder(agent, rng=np.random.default_rng(2)).encode(episode[0])
    np.testing.assert_array_equal(a.logit, b.logit)  # the posterior itself does not depend on the seed
    assert not np.array_equal(a.stoch, b.stoch)  # only the sample drawn from it does


def test_memory_carries_over_between_steps(agent, episode):
    encoder = DreamerEncoder(agent)
    x0 = encoder.encode(episode[0])
    x1 = encoder.encode(episode[1], prev=x0, prev_action=_one_hot(7))
    x1_as_first_step = encoder.encode(episode[1])  # the same frame, but as the start of an episode
    assert not np.allclose(x1.deter, x1_as_first_step.deter)


def test_prev_and_prev_action_must_come_together(agent, episode):
    encoder = DreamerEncoder(agent)
    x0 = encoder.encode(episode[0])
    with pytest.raises(ValueError, match="together"):
        encoder.encode(episode[1], prev=x0)
    with pytest.raises(ValueError, match="shape"):
        encoder.encode(episode[1], prev=x0, prev_action=np.zeros(3))


def test_matches_upstream_policy_latent_over_two_steps(agent, episode):
    """Same weights, same BEV, same random keys: encode() must give exactly the latent upstream's policy computes."""
    # Upstream's agent draws one random key per policy call from agent.rng (jaxagent.py:53, 208-210).
    # Our encoder gets a generator in the same state, so it draws the same keys.
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(agent.rng.bit_generator.state)
    encoder = DreamerEncoder(agent, rng=rng)

    upstream_state, ours_prev, prev_action = None, None, None
    for t, obs in enumerate(episode[:2]):
        # What upstream's BatchEnv hands the agent: every observation key, with a batch dimension of 1.
        batch = {
            "birdeye_wpt": obs.bev[None],
            "camera": np.zeros((1, 128, 128, 3), np.uint8),
            "collision": np.zeros((1, 1), np.float32),
            "reward": np.zeros(1, np.float32),
            "is_first": np.array([t == 0]),
            "is_last": np.array([False]),
            "is_terminal": np.array([False]),
        }
        _, upstream_state = agent.policy(batch, upstream_state, mode="eval")
        (upstream_latent, upstream_action), _, _ = upstream_state

        ours = encoder.encode(obs, ours_prev, prev_action)
        for name in XSMALL_SHAPES:
            np.testing.assert_array_equal(getattr(ours, name), upstream_latent[name][0], err_msg=f"step {t}: {name}")
        # Both sides continue with the action upstream's actor sampled, as the env would execute it.
        ours_prev, prev_action = ours, upstream_action[0]
