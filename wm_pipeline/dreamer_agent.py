"""
Building a DreamerV3 agent the same way CarDreamer's dreamerv3/eval.py does, for Stages 2-3.

Phase A builds one with random weights (the `xsmall` size on CPU) to test shapes and wiring;
load_checkpoint() puts the pretrained CarDreamer weights into a `small` agent.

dreamerv3 is imported inside these functions rather than at the top of the module: importing it also
imports tensorflow (several seconds), and `import wm_pipeline` should stay light.
"""

import hashlib
import io
import pathlib
import pickle
import sys
import warnings
from typing import Optional

import jax
import numpy as np

from .actions import ActionConfig
from .observation import BEV_KEY, BEV_SHAPE

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# What tasks.yaml:131-134 sets for every right-turn task: the world model encodes and decodes only the BEV.
TASK_OVERRIDES = {"encoder.cnn_keys": BEV_KEY, "decoder.cnn_keys": BEV_KEY}

# Phase A runs on the laptop CPU in full precision. Upstream's defaults are jax.platform "gpu" and
# jax.precision "float16" (dreamerv3.yaml:13-16), which is what Phase B must use to match its results.
CPU_OVERRIDES = {"jax.platform": "cpu", "jax.precision": "float32"}


def import_dreamerv3():
    """Import upstream's dreamerv3 package and return (dreamerv3, embodied, ninjax)."""
    # dreamerv3 must be importable as a package from the repo root (some of its files use absolute
    # `from dreamerv3.embodied...` imports); its __init__ then adds dreamerv3/ to sys.path for `import embodied`.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import dreamerv3
    import embodied
    from dreamerv3 import ninjax

    return dreamerv3, embodied, ninjax


def make_config(size: str = "small", overrides: dict = None):
    """dreamerv3.yaml defaults + a model size + the right-turn task overrides, as in eval.py:109-118."""
    from ruamel import yaml

    dreamerv3, embodied, _ = import_dreamerv3()
    configs = yaml.YAML(typ="safe").load((pathlib.Path(dreamerv3.__file__).parent / "dreamerv3.yaml").read_text())
    config = embodied.Config(configs["defaults"])
    config = config.update(configs[size])
    return config.update({**TASK_OVERRIDES, **(overrides or {})})


def build_agent(config):
    """
    A DreamerV3 agent with the observation and action spaces upstream's eval gives it for the right-turn tasks
    (discrete actions). Its weights are randomly initialized; Phase B loads a checkpoint into it.
    """
    dreamerv3, embodied, _ = import_dreamerv3()
    # The spaces dreamerv3/embodied/envs/from_gym.py:27-48 produces from the right-turn env.
    obs_space = {
        BEV_KEY: embodied.Space(np.uint8, BEV_SHAPE),
        "camera": embodied.Space(np.uint8, (128, 128, 3)),
        "collision": embodied.Space(np.float32, (1,)),
        "reward": embodied.Space(np.float32),
        "is_first": embodied.Space(bool),
        "is_last": embodied.Space(bool),
        "is_terminal": embodied.Space(bool),
    }
    # embodied.wrappers.OneHotAction (wrappers.py:126-138) turns Discrete(15) into a one-hot vector space
    # and marks it discrete by setting this private flag; we do the same so the actor gets the same head.
    action_space = embodied.Space(np.float32, (ActionConfig().n_actions,), 0, 1)
    action_space._discrete = True
    act_space = {"action": action_space, "reset": embodied.Space(bool)}
    return dreamerv3.Agent(obs_space, act_space, embodied.Counter(), config)


# Security: upstream checkpoints are pickles (embodied/core/basics.py:91-95), and plain pickle.load calls whatever
# Python object the file names, i.e. it can run arbitrary code. This unpickler only hands out numpy's array builders
# and optax's optimizer-state classes after checking they are NamedTuples (plain data: nothing runs when one is
# built). Any other name stops loading before it is called. Inspecting right_turn_hard.ckpt showed exactly these.
_NUMPY_ARRAY_BUILDERS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _NUMPY_ARRAY_BUILDERS:
            return super().find_class(module, name)
        if module.startswith("optax.") and name.endswith("State"):
            cls = super().find_class(module, name)
            if isinstance(cls, type) and issubclass(cls, tuple) and hasattr(cls, "_fields"):
                return cls
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}: a checkpoint should only hold arrays and optax optimizer states")


def load_checkpoint(agent, path, sha256: Optional[str] = None) -> dict:
    """
    Put an upstream checkpoint's weights into `agent`, as eval.py does with checkpoint.load(path, keys=["agent"])
    (dreamerv3/eval.py:94-97), but without plain pickle and only if every entry fits the agent.

    The agent must be built with the precision the checkpoint was trained with: CarDreamer's checkpoints use
    upstream's default jax.precision float16, which gives the optimizer extra loss-scaling state
    (dreamerv3/jaxutils.py:407-411) that a float32 agent does not have.

    :param sha256: if given, the file's SHA-256 must match (Hugging Face publishes it for every checkpoint).
    :return: the checkpoint's other top-level entries, e.g. {"step": 461400, ...}, for logging.
    """
    raw = pathlib.Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256.lower():
        raise ValueError(f"SHA-256 of {path} does not match the expected {sha256}")
    data = _CheckpointUnpickler(io.BytesIO(raw)).load()
    if not isinstance(data, dict) or "agent" not in data:
        raise ValueError(f"{path} has no 'agent' entry; top level is {type(data).__name__}")

    saved = data["agent"]
    current = agent.save()  # upstream's own export of the agent's weights and optimizer state (jaxagent.py:100-107)
    problems = [f"missing from checkpoint: {k}" for k in sorted(set(current) - set(saved))]
    problems += [f"not in this agent: {k}" for k in sorted(set(saved) - set(current))]
    for key in sorted(set(current) & set(saved)):
        if jax.tree_util.tree_structure(current[key]) != jax.tree_util.tree_structure(saved[key]):
            problems.append(f"different structure: {key}")
            continue
        for mine, theirs in zip(jax.tree_util.tree_leaves(current[key]), jax.tree_util.tree_leaves(saved[key])):
            if np.shape(mine) != np.shape(theirs) or np.asarray(mine).dtype != np.asarray(theirs).dtype:
                problems.append(f"{key}: agent has {np.shape(mine)} {np.asarray(mine).dtype}, checkpoint has {np.shape(theirs)} {np.asarray(theirs).dtype}")
                break
    if problems:
        shown = "\n  ".join(problems[:10]) + (f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else "")
        if all("_opt/" in problem for problem in problems):
            shown += (
                "\n  Only optimizer state differs, so the weights fit: build the agent with the checkpoint's"
                " jax.precision (float16 for CarDreamer's checkpoints)."
            )
        raise ValueError(f"checkpoint {path} does not fit this agent (wrong size or config?):\n  {shown}")

    agent.load(saved)  # upstream's JAXAgent.load (jaxagent.py:109-114)
    return {k: v for k, v in data.items() if k != "agent"}


def draw_seed(rng: np.random.Generator) -> np.int64:
    """One random seed, drawn the way upstream's agent draws one per policy call (jaxagent.py:208-210)."""
    return rng.integers(2**63 - 1)


def run_on_agent_weights(agent, jitted_fn, inputs: tuple, seed):
    """
    Run a function built with nj.jit(nj.pure(...)) on the agent's policy weights; return its output as numpy.

    Shared by the encoder and the planner. It moves data explicitly (upstream disallows implicit transfers,
    jaxagent.py:148), turns the seed into a random key the way upstream does, and fails if the call created
    weights instead of using the agent's.
    """
    device = agent.policy_devices[0]
    varibs = agent.varibs if agent.single_device else agent.policy_varibs
    inputs = jax.device_put(inputs, device)
    with warnings.catch_warnings():
        # The warning upstream's eval.py:11 silences: the int64 seed is truncated to int32 for the random key.
        warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
        rng = jax.device_put(np.int64(seed), device)
    out, state = jitted_fn(varibs, rng, *inputs)
    # nj.jit only creates weights that do not exist yet. If any were created, this call ran on fresh random
    # weights instead of the agent's (e.g. a module path changed), so fail loudly.
    created = set(state) - set(varibs)
    if created:
        raise RuntimeError(f"the call created {len(created)} new weights instead of using the agent's, e.g. {sorted(created)[:3]}")
    return jax.device_get(out)
