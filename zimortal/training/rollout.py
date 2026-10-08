"""Observation-only, history-consistent Monte Carlo cash-Q teacher."""

import math
import random
from dataclasses import dataclass

import numpy as np

from zimortal.belief.sampling import sample_world
from zimortal.engine import RuleClarificationRequired, RuleEngine

from .runtime import choose, choose_batch


@dataclass(frozen=True)
class RolloutTargets:
    q_cash: tuple
    standard_errors: tuple
    policy: tuple
    rollouts_per_action: int
    sampling_attempts: int
    paired_standard_errors: tuple = ()
    outcomes_cash: tuple = ()


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
    particle_offset=0,
    world_sampler=None,
    keep_outcomes=False,
):
    if rollouts < 2 or not obs.legal_actions:
        raise ValueError("two or more rollouts and legal actions required")
    if not 0 <= reference_index < len(obs.legal_actions):
        raise ValueError("invalid reference action")
    outcomes = [[] for _ in obs.legal_actions]
    worlds, candidates, roots = [], [], []
    attempts = 0
    engine = RuleEngine()
    for i in range(rollouts):
        particle_seed = seed + (particle_offset + i) * 104729
        world, used = (
            world_sampler.sample(particle_seed, max_attempts)
            if world_sampler is not None
            else sample_world(obs, particle_seed, max_attempts)
        )
        attempts += used
        roots.append(world)
        # Same hidden particle and random stream for all candidate actions.
        for j, action in enumerate(obs.legal_actions):
            state = engine.step(world, action)
            rng = random.Random(seed + particle_offset + i)
            if batched and model is not None:
                worlds.append(state)
                candidates.append(j)
                continue
            for _ in range(1000):
                if state.terminal:
                    break
                try:
                    acts = engine.legal_actions(state)
                except RuleClarificationRequired as exc:
                    exc.rollout_context = {
                        "particle": particle_offset + i,
                        "action_index": j,
                        "root_world": world.serialize(),
                        "state": state.serialize(),
                    }
                    raise
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
                try:
                    acts = engine.legal_actions(state)
                except RuleClarificationRequired as exc:
                    exc.rollout_context = {
                        "particle": particle_offset + index // len(obs.legal_actions),
                        "action_index": candidates[index],
                        "root_world": roots[index // len(obs.legal_actions)].serialize(),
                        "state": state.serialize(),
                    }
                    raise
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
    raw = tuple(tuple(x[i] for x in outcomes) for i in range(rollouts)) if keep_outcomes else ()
    return RolloutTargets(q, se, soft_target(q, temperature), rollouts, attempts, paired, raw)
