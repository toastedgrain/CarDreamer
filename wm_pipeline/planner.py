"""
Stage 3: world model planning -> the nominal action u_nom, plus open-loop imagination with the world model.

act(x_t)                The actor pi reads x_t and samples u_nom from its distribution over the 15 actions,
                        exactly as upstream's eval mode does (dreamerv3/agent.py:54-58, 278-292).
imagine(x_t, actions)   The dynamics D roll x_t forward under given actions with no observations; the decoder and
                        the reward head R read out each predicted state. The same recipe as upstream's open-loop
                        videos in WorldModel.report() (dreamerv3/agent.py:235-241).

Both run upstream's own modules on the agent's weights (nj.pure + nj.jit), without modifying upstream code.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .actions import Action, ActionConfig
from .dreamer_agent import draw_seed, import_dreamerv3, run_on_agent_weights
from .encoder import LatentState
from .observation import BEV_KEY, BEV_SHAPE

LATENT_KEYS = ("deter", "stoch", "logit")


@dataclass(frozen=True)
class Imagination:
    """An open-loop rollout of H steps from x_t: what the world model predicts would follow the given actions."""

    latents: Tuple[LatentState, ...]  # predicted x_{t+1}, ..., x_{t+H}
    bev: np.ndarray  # (H, 128, 128, 3) uint8: the decoder's BEV image of each predicted state
    reward: np.ndarray  # (H,) float32: the reward head's prediction for each predicted state

    def __post_init__(self):
        horizon = len(self.latents)
        if horizon < 1:
            raise ValueError("an imagination needs at least one step")
        if self.bev.shape != (horizon, *BEV_SHAPE) or self.bev.dtype != np.uint8:
            raise ValueError(f"bev must be ({horizon}, 128, 128, 3) uint8, got {self.bev.shape} {self.bev.dtype}")
        if self.reward.shape != (horizon,) or self.reward.dtype != np.float32:
            raise ValueError(f"reward must be ({horizon},) float32, got {self.reward.shape} {self.reward.dtype}")


class DreamerPlanner:
    """pi and D of a DreamerV3 agent: the nominal action, and imagined futures."""

    def __init__(self, agent, rng: Optional[np.random.Generator] = None):
        """
        :param agent: a dreamerv3.Agent, e.g. from dreamer_agent.build_agent().
        :param rng: where random seeds come from when act() or imagine() is not given one.
        """
        _, _, nj = import_dreamerv3()
        model = agent.agent  # the ninjax module whose methods upstream's policy calls
        if len(agent.policy_devices) != 1:
            raise NotImplementedError("DreamerPlanner supports a single policy device, like CarDreamer's eval")
        # Every pretrained CarDreamer checkpoint uses the 15 discrete actions; a continuous actor would need training.
        if not model.act_space.discrete or model.act_space.shape != (ActionConfig().n_actions,):
            raise NotImplementedError(f"DreamerPlanner needs the 15-action discrete space, got {model.act_space}")
        self._agent = agent
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._action_config = ActionConfig()

        def act_step(latent):
            # Upstream's policy call takes its random keys from one step seed in this order: key 1 for the prior
            # sample and key 2 for the posterior sample (both in the encoder), then key 3 for the action. Skipping
            # the first two makes the action sample use key 3, so with a shared seed we match upstream exactly.
            nj.rng()
            nj.rng()
            outs, _ = model.task_behavior.policy(latent, {})  # {"action": the actor's distribution}
            return outs["action"].sample(seed=nj.rng())

        def imagine_step(start, actions):
            prior = model.wm.rssm.imagine(actions, start)  # D: one img_step per action, no observations
            bev = model.wm.heads["decoder"](prior)[BEV_KEY].mode()  # in [0, 1], like the preprocessed input image
            reward = model.wm.heads["reward"](prior).mean()  # R
            return prior, bev, reward

        device = agent.policy_devices[0]
        self._act_step = nj.jit(nj.pure(act_step), device=device)
        self._imagine_step = nj.jit(nj.pure(imagine_step), device=device)

    def act(self, x: LatentState, seed: Optional[int] = None) -> Action:
        """
        Sample u_nom from the actor pi(a | x_t).

        :param seed: this step's random seed. Pass the one the encoder used for x_t to reproduce upstream's choice
            exactly. If None, a seed is drawn from `rng`.
        """
        if seed is None:
            seed = draw_seed(self._rng)
        one_hot = run_on_agent_weights(self._agent, self._act_step, (_batched(x),), seed)[0]
        # Like embodied.wrappers.OneHotAction: check the sample is one-hot, then take its argmax (wrappers.py:140-146).
        if one_hot.shape != (self._action_config.n_actions,) or one_hot.max() != 1.0 or one_hot.sum() != 1.0:
            raise RuntimeError(f"the actor's sample is not a one-hot action: {one_hot}")
        return Action.from_index(int(np.argmax(one_hot)), self._action_config)

    def imagine(self, x: LatentState, actions: Sequence[Action], seed: Optional[int] = None) -> Imagination:
        """
        Roll the world model forward from x_t under `actions` (H of them), without observations.

        latents[k] is the predicted state after executing actions[0..k], i.e. x_{t+k+1}, so bev[k] can be compared
        with the real BEV that follows the same actions.
        """
        if len(actions) == 0:
            raise ValueError("imagine() needs at least one action")
        if seed is None:
            seed = draw_seed(self._rng)
        one_hots = np.stack([u.one_hot for u in actions])[None]  # (1, H, 15)
        prior, bev, reward = run_on_agent_weights(self._agent, self._imagine_step, (_batched(x), one_hots), seed)
        latents = tuple(LatentState(**{k: prior[k][0, h] for k in LATENT_KEYS}) for h in range(len(actions)))
        # The decoder predicts the preprocessed image (pixels / 255), so scale back to uint8 pixels.
        frames = np.clip(np.round(bev[0] * 255.0), 0, 255).astype(np.uint8)
        return Imagination(latents=latents, bev=frames, reward=reward[0].astype(np.float32))


def _batched(x: LatentState) -> dict:
    """x_t as upstream's latent dict, with a batch dimension of 1."""
    return {k: getattr(x, k)[None] for k in LATENT_KEYS}
