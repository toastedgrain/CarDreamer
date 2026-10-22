"""Step 6 tests: the Stage 3 planner (act, imagine) on a random xsmall agent. Shared fixtures come from conftest.py."""

import copy

import numpy as np
import pytest

from wm_pipeline.actions import Action
from wm_pipeline.dreamer_agent import draw_seed
from wm_pipeline.encoder import DreamerEncoder
from wm_pipeline.planner import DreamerPlanner

# The warning upstream's eval.py:11 silences: int64 seeds are truncated to int32 for JAX's random keys.
pytestmark = pytest.mark.filterwarnings("ignore:.*truncated to dtype int32.*")

BRAKE, COAST, GAS_RIGHT = Action.from_index(2), Action.from_index(7), Action.from_index(10)


@pytest.fixture(scope="module")
def encoder(agent):
    return DreamerEncoder(agent)


@pytest.fixture(scope="module")
def planner(agent):
    return DreamerPlanner(agent)


@pytest.fixture(scope="module")
def x0(encoder, episode):
    return encoder.encode(episode[0], seed=0)


# --- act ---------------------------------------------------------------------------------------------


def test_act_returns_a_valid_action(planner, x0):
    u = planner.act(x0, seed=0)
    assert isinstance(u, Action) and 0 <= u.index < 15
    assert u.one_hot.argmax() == u.index


def test_act_with_the_same_seed_gives_the_same_action(planner, x0):
    assert planner.act(x0, seed=3) == planner.act(x0, seed=3)


def test_act_samples_from_a_distribution(planner, x0):
    # A random-weight actor is close to uniform over the 15 actions, so 20 seeds will not all pick the same one.
    assert len({planner.act(x0, seed=s).index for s in range(20)}) > 1


def test_encoder_and_planner_match_upstream_policy_over_two_steps(agent, episode, encoder, planner, upstream_batch):
    """Same weights, observations and seeds: encode() + act() choose exactly the action upstream's policy chooses."""
    # Upstream draws one seed per policy call from agent.rng; a copy of that generator gives us the same seeds.
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(agent.rng.bit_generator.state)

    upstream_state, x_prev, a_prev = None, None, None
    for t, obs in enumerate(episode[:2]):
        upstream_outs, upstream_state = agent.policy(upstream_batch(obs, is_first=t == 0), upstream_state, mode="eval")
        seed = draw_seed(rng)  # the seed upstream drew for this call; encoder and planner share it
        x = encoder.encode(obs, x_prev, a_prev, seed=seed)
        u = planner.act(x, seed=seed)
        np.testing.assert_array_equal(u.one_hot, upstream_outs["action"][0], err_msg=f"step {t}")
        x_prev, a_prev = x, u.one_hot


# --- imagine -----------------------------------------------------------------------------------------


def test_imagine_shapes_and_dtypes(planner, x0):
    imagination = planner.imagine(x0, [COAST, COAST, BRAKE], seed=0)
    assert len(imagination.latents) == 3
    assert imagination.latents[0].deter.shape == (64,) and imagination.latents[0].stoch.shape == (32, 32)
    assert imagination.bev.shape == (3, 128, 128, 3) and imagination.bev.dtype == np.uint8
    assert imagination.reward.shape == (3,) and imagination.reward.dtype == np.float32


def test_imagine_with_the_same_seed_is_reproducible(planner, x0):
    a = planner.imagine(x0, [COAST] * 3, seed=1)
    b = planner.imagine(x0, [COAST] * 3, seed=1)
    np.testing.assert_array_equal(a.bev, b.bev)
    np.testing.assert_array_equal(a.reward, b.reward)
    for la, lb in zip(a.latents, b.latents):
        np.testing.assert_array_equal(la.deter, lb.deter)
        np.testing.assert_array_equal(la.stoch, lb.stoch)


def test_imagined_future_depends_on_the_actions(planner, x0):
    after_brake = planner.imagine(x0, [BRAKE], seed=1).latents[0]
    after_gas = planner.imagine(x0, [GAS_RIGHT], seed=1).latents[0]
    assert not np.allclose(after_brake.deter, after_gas.deter)


def test_first_imagined_memory_equals_the_encoders_next_memory(planner, encoder, episode, x0):
    """
    The memory deter_{t+1} is computed from x_t and a_t alone (the GRU in img_step); the next frame only refines
    stoch. So one imagined step and the encoded real next frame agree on deter, whatever that frame shows.
    """
    x1 = encoder.encode(episode[1], prev=x0, prev_action=COAST.one_hot, seed=2)
    x1_other_frame = encoder.encode(episode[2], prev=x0, prev_action=COAST.one_hot, seed=2)
    np.testing.assert_array_equal(x1.deter, x1_other_frame.deter)  # the frame does not change the memory...
    assert not np.array_equal(x1.logit, x1_other_frame.logit)  # ...it changes the posterior over stoch

    imagined = planner.imagine(x0, [COAST], seed=1).latents[0]
    np.testing.assert_array_equal(imagined.deter, x1.deter)


def test_imagine_needs_at_least_one_action(planner, x0):
    with pytest.raises(ValueError, match="at least one"):
        planner.imagine(x0, [])
