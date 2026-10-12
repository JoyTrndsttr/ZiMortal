"""Optimization must preserve sampling law and avoid hidden-world action choice."""

import random
from dataclasses import replace

import numpy as np
import pytest

from zimortal.belief.conditional import HistorySampler
from zimortal.model.encoding import _huxi_features, encode_observation
from zimortal.training.branch_teacher import (
    compact_evidence,
    diagnostic_observation,
    reference_summary,
)
from zimortal.training.runtime import _action_features, _visible_features


def sequential_allocation(sampler, rng):
    # Independent reference for the original weighted conditional draw.
    hands = [list(sampler.obs.hand) if p == sampler.obs.player else [] for p in range(3)]
    deck, left = [], sampler.sizes
    for tile in range(20):
        ticket = rng.randrange(sampler.ways(tile, *left))
        for a, b, d, weight in sampler.options[tile]:
            mass = weight * sampler.ways(tile + 1, left[0] - a, left[1] - b)
            if ticket < mass:
                hands[sampler.opponents[0]].extend([tile] * a)
                hands[sampler.opponents[1]].extend([tile] * b)
                deck.extend([tile] * d)
                left = left[0] - a, left[1] - b
                break
            ticket -= mass
        else:
            raise AssertionError("invalid reference draw")
    rng.shuffle(deck)
    return hands, deck


@pytest.mark.parametrize("case,step", [("split-001", None), ("split-002", None), ("split-002", 2)])
def test_conditional_cdf_preserves_exact_random_stream(case, step):
    sampler = HistorySampler(diagnostic_observation(case, step=step))
    for seed in range(100):
        original, optimized = random.Random(seed), random.Random(seed)
        assert sequential_allocation(sampler, original) == sampler.allocation(optimized)
        assert original.getstate() == optimized.getstate()
    assert sampler.transitions.cache_info().currsize <= 8192


def test_encoding_cache_readonly_history_and_version_isolation():
    obs = diagnostic_observation("split-001")
    for version in ("legacy", "huxi"):
        assert np.array_equal(_visible_features(obs, version), encode_observation(obs, version))
    with pytest.raises(ValueError):
        _visible_features(obs, "huxi")[0, 0] = 0
    actions = _action_features(obs.legal_actions, obs.player)
    assert not actions.flags.writeable
    changed = replace(obs, history=())
    assert not np.array_equal(_visible_features(obs, "huxi"), _visible_features(changed, "huxi"))
    before = _huxi_features.cache_info().hits
    encode_observation(changed, "huxi")
    assert _huxi_features.cache_info().hits > before


def test_pass_fixed_policy_does_not_choose_hidden_world_max():
    raw = []
    for particle in range(4):
        branches = []
        for action in (0, 1):
            branches.append(
                {
                    "first_index": 0,
                    "information_set": "same-visible-state",
                    "observation": {},
                    "second_action": {"tile": action},
                    "reference_selected": action == 0,
                    "cash": 100 if action == particle % 2 else -100,
                }
            )
        raw.append({"particle": particle, "branches": branches})
    result = reference_summary(raw, alpha=0.05, looks=1)
    assert result["reference_statistics"]["rows"][0]["mean_cash"] == 0
    assert result["reference_outcomes"] == [[100], [-100], [100], [-100]]
    evidence = compact_evidence(raw)
    assert len(evidence["information_sets"]) == 1
    assert all("observation" not in b for p in evidence["particles"] for b in p["branches"])
    raw[1]["branches"][0]["reference_selected"] = False
    raw[1]["branches"][1]["reference_selected"] = True
    with pytest.raises(ValueError, match="changed action"):
        reference_summary(raw, alpha=0.05, looks=1)


def test_pass_prefilter_preserves_accepted_particles_and_attempt_numbers():
    obs = diagnostic_observation("split-002")
    original, optimized = HistorySampler(obs), HistorySampler(obs)
    original.pass_checks = ()
    for seed in (712, 1000712, 1105441):
        a, attempts_a = original.sample(seed, 100000)
        b, attempts_b = optimized.sample(seed, 100000)
        assert a.serialize() == b.serialize()
        assert attempts_a == attempts_b


def test_discard_pass_prefilter_never_rejects_legal_public_histories():
    from zimortal.belief.conditional import discard_pass_checks, satisfies_discard_passes
    from zimortal.engine import RuleClarificationRequired, RuleEngine

    engine = RuleEngine()
    checked = 0
    for seed in range(30):
        state, rng = engine.new_game(seed, seed % 3), random.Random(seed)
        for _ in range(60):
            if state.terminal:
                break
            try:
                actions = engine.legal_actions(state)
                obs = engine.observation(state, actions[0].player)
                checks = discard_pass_checks(obs)
                assert satisfies_discard_passes([p.hand for p in state.players], checks)
                checked += bool(checks)
                state = engine.step(state, rng.choice(actions))
            except RuleClarificationRequired:
                break
    assert checked > 100
