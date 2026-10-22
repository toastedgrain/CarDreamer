"""
Stage 2: the encoder E(O_t, l) -> x_t, wrapping DreamerV3's CNN encoder and RSSM posterior.

It runs exactly what upstream's Agent.policy runs before it picks an action (dreamerv3/agent.py:49-52):

    obs    = agent.preprocess(obs)                                     # uint8 BEV -> floats in [0, 1]
    embed  = agent.wm.encoder(obs)                                     # CNN: BEV -> embedding vector
    x_t, _ = agent.wm.rssm.obs_step(x_prev, a_prev, embed, is_first)   # posterior latent state

and runs it the way upstream's JAXAgent does (nj.pure + nj.jit over the agent's weights, jaxagent.py:153-176),
so no upstream code is modified.

The instruction l has no separate input: in CarDreamer the route reaches the model only as the blue line drawn
into the BEV. If the project adds a text instruction, this is where it would be encoded.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dreamer_agent import draw_seed, import_dreamerv3, run_on_agent_weights
from .observation import BEV_KEY, Observation


@dataclass(frozen=True)
class LatentState:
    """
    x_t, the world model's state, without the batch dimension.

    Shapes for the `small` model (`xsmall` in brackets): deter (512,) [(64,)], stoch (32, 32), logit (32, 32).
    The dtype is the agent's compute precision: float16 in upstream's GPU config, float32 on the laptop.
    """

    deter: np.ndarray  # deterministic GRU memory h_t: carries the history of earlier frames and actions
    stoch: np.ndarray  # stochastic z_t: 32 categorical variables x 32 classes, each row one-hot (a sample)
    logit: np.ndarray  # logits of the posterior distribution that `stoch` was sampled from

    def __post_init__(self):
        for name in ("deter", "stoch", "logit"):
            if not isinstance(getattr(self, name), np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(getattr(self, name)).__name__}")
        if self.deter.ndim != 1:
            raise ValueError(f"deter must have shape (D,), got {self.deter.shape}")
        if self.stoch.ndim != 2 or self.logit.shape != self.stoch.shape:
            raise ValueError(f"stoch and logit must both have shape (S, C), got {self.stoch.shape} and {self.logit.shape}")
        dtypes = {self.deter.dtype, self.stoch.dtype, self.logit.dtype}
        if len(dtypes) != 1 or not np.issubdtype(self.deter.dtype, np.floating):
            raise TypeError(f"deter, stoch and logit must share one float dtype, got {sorted(map(str, dtypes))}")

    def features(self) -> np.ndarray:
        """What the actor, critic, reward and decoder heads read: deter, then the flattened stoch (nets.py:673-696)."""
        return np.concatenate([self.deter, self.stoch.reshape(-1)])


class DreamerEncoder:
    """E(O_t, l) -> x_t using a DreamerV3 agent's weights: random in Phase A tests, the checkpoint in Phase B."""

    def __init__(self, agent, rng: Optional[np.random.Generator] = None):
        """
        :param agent: a dreamerv3.Agent, e.g. from dreamer_agent.build_agent().
        :param rng: where random seeds come from when encode() is not given one. Upstream's agent draws its seeds
            the same way from its own generator (jaxagent.py:208-210).
        """
        _, _, nj = import_dreamerv3()
        from dreamerv3 import jaxutils

        if len(agent.policy_devices) != 1:
            raise NotImplementedError("DreamerEncoder supports a single policy device, like CarDreamer's eval")
        self._agent = agent
        self._rng = rng if rng is not None else np.random.default_rng(0)
        rssm_config = agent.agent.config.rssm
        self._latent_shapes = {
            "deter": (rssm_config.deter,),
            "stoch": (rssm_config.stoch, rssm_config.classes),
            "logit": (rssm_config.stoch, rssm_config.classes),
        }
        self._action_size = agent.agent.act_space.shape[0]
        # The precision upstream keeps latents in (set from config.jax.precision when the agent was built).
        self._compute_dtype = np.dtype(jaxutils.COMPUTE_DTYPE)
        model = agent.agent  # the ninjax module whose methods upstream's policy calls

        def encode_step(obs, prev_latent, prev_action):
            obs = model.preprocess(obs)
            embed = model.wm.encoder(obs)
            latent, _ = model.wm.rssm.obs_step(prev_latent, prev_action, embed, obs["is_first"])
            return latent

        self._encode_step = nj.jit(nj.pure(encode_step), device=agent.policy_devices[0])

    def encode(
        self,
        obs: Observation,
        prev: Optional[LatentState] = None,
        prev_action: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> LatentState:
        """
        :param obs: Stage 1 output (O_t, l).
        :param prev: x_{t-1}, or None on the first step of an episode. On the first step the RSSM ignores any
            previous state and starts from its learned initial state (nets.py:110-115).
        :param prev_action: the action executed after x_{t-1}, in the agent's own format: a one-hot vector of
            length 15 in discrete mode. Must be None exactly when `prev` is None.
        :param seed: this step's random seed. The runner gives the planner the same seed, so that together they
            make the same random choices as upstream's single policy call (see planner.py). If None, a seed is
            drawn from `rng`.
        """
        is_first = prev is None
        if is_first != (prev_action is None):
            raise ValueError("pass prev and prev_action together, or neither on the first step of an episode")
        if is_first:
            # Placeholders only: obs_step multiplies them by (1 - is_first) = 0 before use. They still need the
            # compute dtype, as upstream's initial state has, or the first step's arithmetic changes precision.
            prev = LatentState(**{k: np.zeros(s, self._compute_dtype) for k, s in self._latent_shapes.items()})
            prev_action = np.zeros(self._action_size, np.float32)
        prev_action = np.asarray(prev_action, dtype=np.float32)
        if prev_action.shape != (self._action_size,):
            raise ValueError(f"prev_action must have shape ({self._action_size},), got {prev_action.shape}")

        # Everything gets a batch dimension of 1, like the single env in upstream's BatchEnv.
        batch = {
            BEV_KEY: obs.bev[None],
            "is_first": np.array([is_first]),
            "is_terminal": np.array([False]),  # preprocess() needs it to compute `cont`; it does not affect x_t
        }
        prev_latent = {"deter": prev.deter[None], "stoch": prev.stoch[None], "logit": prev.logit[None]}
        if seed is None:
            seed = draw_seed(self._rng)
        latent = run_on_agent_weights(self._agent, self._encode_step, (batch, prev_latent, prev_action[None]), seed)
        return LatentState(deter=latent["deter"][0], stoch=latent["stoch"][0], logit=latent["logit"][0])
