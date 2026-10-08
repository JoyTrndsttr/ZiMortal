"""Observation-only, history-consistent Monte Carlo cash-Q teacher."""

import math
import random
from dataclasses import dataclass

import numpy as np

from zimortal.belief.sampling import sample_world
from zimortal.engine import RuleEngine

from .runtime import choose, choose_batch


@dataclass(frozen=True)
class RolloutTargets:
    q_cash: tuple
    standard_errors: tuple
    policy: tuple
    rollouts_per_action: int
    sampling_attempts: int
    paired_standard_errors: tuple = ()


def soft_target(values, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive cash units")
    q = np.asarray(values, np.float64)
    q = (q - q.max()) / temperature
    p = np.exp(q)
    return tuple(float(x) for x in p / p.sum())


def teacher(
    obs,
    *,
    rollouts=64,
    seed=0,
    temperature=10.0,
    model=None,
    max_attempts=2000,
    batched=False,
    reference_index=0,
):
    if rollouts < 2 or not obs.legal_actions:
        raise ValueError("two or more rollouts and legal actions required")
    if not 0 <= reference_index < len(obs.legal_actions):
        raise ValueError("invalid reference action")
    outcomes = [[] for _ in obs.legal_actions]
    worlds, candidates = [], []
    attempts = 0
    engine = RuleEngine()
    for i in range(rollouts):
        world, used = sample_world(obs, seed + i * 104729, max_attempts)
        attempts += used
        # Same hidden particle and random stream for all candidate actions.
        for j, action in enumerate(obs.legal_actions):
            state = engine.step(world, action)
            rng = random.Random(seed + i)
            if batched and model is not None:
                worlds.append(state)
                candidates.append(j)
                continue
            for _ in range(1000):
                if state.terminal:
                    break
                acts = engine.legal_actions(state)
                visible = engine.observation(state, acts[0].player)
                state = engine.step(
                    state, choose(visible, rng, model, policy="model" if model else "teacher")
                )
            else:
                raise RuntimeError("rollout did not terminate")
            state.validate()
            outcomes[j].append(state.settlement.payments[obs.player] if state.settlement else 0)
    if worlds:
        for _ in range(1000):
            waiting, observations = [], []
            active = False
            for index, state in enumerate(worlds):
                if state.terminal:
                    continue
                active = True
                acts = engine.legal_actions(state)
                if len(acts) == 1:
                    worlds[index] = engine.step(state, acts[0])
                else:
                    waiting.append(index)
                    observations.append(engine.observation(state, acts[0].player))
            if not active:
                break
            for index, action in zip(waiting, choose_batch(model, observations), strict=True):
                worlds[index] = engine.step(worlds[index], action)
        else:
            raise RuntimeError("batched rollout did not terminate")
        for state, j in zip(worlds, candidates, strict=True):
            state.validate()
            outcomes[j].append(state.settlement.payments[obs.player] if state.settlement else 0)
    q = tuple(float(np.mean(x)) for x in outcomes)
    se = tuple(float(np.std(x, ddof=1) / math.sqrt(rollouts)) for x in outcomes)
    paired = tuple(
        float(
            np.std(np.asarray(x) - np.asarray(outcomes[reference_index]), ddof=1)
            / math.sqrt(rollouts)
        )
        for x in outcomes
    )
    return RolloutTargets(q, se, soft_target(q, temperature), rollouts, attempts, paired)
