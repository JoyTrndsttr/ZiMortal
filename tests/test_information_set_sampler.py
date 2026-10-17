"""No weights or hidden benchmark artifacts required; ordinary engine only."""

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from zimortal.belief.conditional import HistorySampler
from zimortal.belief.sampling import SamplingFailure
from zimortal.engine import ActionType as A
from zimortal.engine import RuleClarificationRequired, RuleEngine
from zimortal.training.information_set_sampler import (
    HiddenWorld,
    InformationSetSampler,
    PublicInformationState,
    SlotPrior,
    certify_world,
    enumerate_physical_worlds,
    observation_bytes,
    physical_pool,
    public_from_json,
)

FIXTURES = Path(__file__).parent / "fixtures/information-set"


def small(number):
    return public_from_json(json.loads((FIXTURES / f"small-{number}.json").read_text()))


def opening():
    engine = RuleEngine()
    state = engine.new_game(118)
    while state.phase == "opening":
        state = engine.step(state, engine.legal_actions(state)[-1])
    return PublicInformationState(engine.observation(state, 0))


@pytest.mark.parametrize("number", [82, 155])
def test_exhaustive_replay_support_complete_and_illegal_mass_zero(number):
    public = small(number)
    sampler = InformationSetSampler(public)
    legal, illegal = {}, 0
    for world in enumerate_physical_worlds(public):
        try:
            certify_world(public, world)
        except ValueError:
            illegal += 1
        else:
            legal[world.key] = world
            assert sampler.contains(world)  # No optimized proposal omission.
    assert legal and illegal
    sampled = Counter(sampler.sample(seed, 20000)[0].key for seed in range(512))
    assert set(sampled) == set(legal)
    # Existing HistorySampler support checked separately, not used as oracle.
    old = HistorySampler(public.observation)
    for seed in range(64):
        state, _ = old.sample(seed, 20000)
        assert HiddenWorld.from_state(state).key in legal


def test_toys_include_claim_and_mandatory_history_and_two_card_hand():
    public = small(82)
    assert {A.PENG, A.WEI, A.CHI} <= {a.kind for a in public.observation.history}
    assert 2 in [p.hand_count for p in small(155).observation.players]


def test_duplicate_weights_are_multinomial_not_uniform_count_vectors():
    # Four physical cards A,A,B,C split one/one/two. Analytical P(first=A)=1/2.
    prior = SlotPrior((2, 1, 1), (1, 1))
    rng = random.Random(713)
    counts = Counter(prior.draw(rng)[0][0] for _ in range(12000))
    assert counts[0] / 12000 == pytest.approx(0.5, abs=0.015)
    # Count vectors (A|A|BC) and (B|C|AA) have weights 2 and 1.
    assert (1, 1, 0, 2) in prior.options[0]
    assert (0, 0, 2, 1) in prior.options[0]
    assert prior.mass == 12  # Four ordered physical slots modulo wall ordering.


def test_physical_instance_id_permutation_invariance():
    public = opening()
    pool, _, sizes = physical_pool(public)
    cards = [(tile, copy) for tile, count in enumerate(pool) for copy in range(count)]
    shuffled_ids = [(tile, 1000 + (3 - copy)) for tile, copy in reversed(cards)]
    for labeled in (cards, shuffled_ids):
        counts = Counter(tile for tile, _ in labeled)
        collapsed = tuple(counts[t] for t in range(20))
        assert collapsed == pool
        prior = SlotPrior(collapsed, sizes)
        if labeled is cards:
            expected = [prior.draw(random.Random(seed)) for seed in range(32)]
        else:
            assert [prior.draw(random.Random(seed)) for seed in range(32)] == expected
    assert public.sha256 == PublicInformationState(public.observation).sha256


def test_paired_distinct_hidden_truth_has_identical_public_input_and_samples():
    public = opening()
    sampler = InformationSetSampler(public)
    worlds = [sampler.sample(seed)[0] for seed in (19, 29)]
    assert worlds[0].hands != worlds[1].hands
    engine = RuleEngine()
    full = [certify_world(public, w) for w in worlds]
    projected = [
        PublicInformationState(engine.observation(s, public.observation.player)) for s in full
    ]
    assert projected[0] == projected[1] == public
    a, b = [InformationSetSampler(p) for p in projected]
    assert [a.sample(seed) for seed in range(8)] == [b.sample(seed) for seed in range(8)]
    with pytest.raises(TypeError):
        InformationSetSampler(full[0])
    with pytest.raises(TypeError):
        PublicInformationState(full[0])


def test_soundness_bytes_hand_sizes_kans_and_reproducibility():
    public = opening()
    sampler = InformationSetSampler(public)
    for seed in range(32):
        world, attempts, rejects = sampler.sample(seed)
        assert attempts == 1 + sum(rejects.values())
        state = certify_world(public, world)
        assert observation_bytes(RuleEngine().observation(state, 0)) == observation_bytes(
            public.observation
        )
        assert all(
            p.kans == {t for t, n in Counter(p.hand).items() if n == 3} for p in state.players
        )  # Opening only: no later kan consumption.
    assert sampler.sample(1234) == sampler.sample(1234)


def test_reject_without_fallback_and_unknown_rule_is_fatal(monkeypatch):
    sampler = InformationSetSampler(opening())

    def reject(*args):
        raise ValueError("deliberately impossible")

    monkeypatch.setattr("zimortal.training.information_set_sampler.certify_world", reject)
    with pytest.raises(SamplingFailure, match="budget exhausted"):
        sampler.sample(1, 3)

    def unknown(*args):
        raise RuleClarificationRequired("unknown boundary")

    monkeypatch.setattr("zimortal.training.information_set_sampler.certify_world", unknown)
    with pytest.raises(RuleClarificationRequired):
        sampler.sample(1, 3)


def test_private_json_fields_and_invalid_physical_counts_rejected():
    public = opening()
    obs = public.observation
    with pytest.raises(ValueError, match="physical"):
        physical_pool(PublicInformationState(replace(obs, hand=(0,) * 5 + obs.hand)))
    with pytest.raises(TypeError):
        public_from_json({"root": "private full state"})


def test_elective_peng_pass_does_not_imply_absence_of_pair():
    engine = RuleEngine()
    state = engine.new_game(118)
    rng = random.Random(118)
    for _ in range(200):
        actions = engine.legal_actions(state)
        peng = next((a for a in actions if a.kind == A.PENG), None)
        if peng:
            state = engine.step(state, next(a for a in actions if a.kind == A.PASS))
            assert state.players[peng.player].hand.count(peng.tile) == 2
            public = PublicInformationState(engine.observation(state, (peng.player + 1) % 3))
            assert InformationSetSampler(public).contains(HiddenWorld.from_state(state))
            break
        if state.terminal:
            pytest.fail("fixed regression trace ended before PENG")
        state = engine.step(state, rng.choice(actions))
    else:
        pytest.fail("fixed regression trace has no PENG")
