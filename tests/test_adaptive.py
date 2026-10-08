"""Adaptive stopping, independent conditional particles and clone isolation."""

import copy
import random
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from zimortal.belief.conditional import HistorySampler, count_constraints
from zimortal.belief.sampling import observed_draws, reverse_hands
from zimortal.engine import ActionType, RuleEngine
from zimortal.model.network import PolicyValueNet
from zimortal.training.adaptive import AdaptiveConfig, empirical_bernstein, relations
from zimortal.training.rollout import teacher


def visible(seed=118, step=10):
    e = RuleEngine()
    s = e.new_game(seed, seed % 3)
    rng = random.Random(seed)
    for _ in range(step):
        if s.terminal:
            break
        s = e.step(s, rng.choice(e.legal_actions(s)))
    return e, s, e.observation(s, e.legal_actions(s)[0].player)


def test_clone_preserves_frozen_records_and_isolates_all_mutable_containers():
    e, s, _ = visible()
    old = s.serialize()
    c = s.clone()
    assert c.serialize() == copy.deepcopy(s).serialize()
    c.players[0].hand.append(0)
    c.players[1].melds.clear()
    c.players[2].passed_chi.add(19)
    c.players[0].passed_peng.add(18)
    c.players[0].kans.add(17)
    c.history.clear()
    c.deck.clear()
    c.river.clear()
    c.passed.add(1)
    c.hu_passed.add(2)
    assert s.serialize() == old
    e.step(s, e.legal_actions(s)[0])
    assert s.serialize() == old


def test_conditional_worlds_are_deterministic_and_replay_entire_observation():
    e, _, obs = visible()
    sampler = HistorySampler(obs)
    first, attempts = sampler.sample(77, 10000)
    second, n = sampler.sample(77, 10000)
    assert first.serialize() == second.serialize() and attempts == n
    assert e.observation(first, obs.player) == obs
    first.validate()
    assert not hasattr(obs, "deck")
    # Local constraints must agree with the authoritative reverse trace.
    allowed = count_constraints(obs)
    rng = random.Random(91)
    for _ in range(100):
        hands, _ = sampler.allocation(rng)
        restored = reverse_hands(obs, [h.copy() for h in hands], observed_draws(obs))
        for p in range(3):
            assert max(restored[p].count(t) for t in range(20)) <= 4
            assert all(hands[p].count(t) in allowed[p][t] for t in range(20))


def test_conditional_weighting_matches_known_hypergeometric_marginal():
    e = RuleEngine()
    s = e.new_game(118)
    obs = e.observation(s, 0)
    # Isolate unconditioned allocation weighting from opening no-quad evidence.
    obs = replace(obs, legal_actions=())
    sampler = HistorySampler(obs)
    rng = random.Random(9)
    tile = next(t for t in range(20) if obs.hand.count(t) == 0)
    count = 0
    for _ in range(4000):
        _, deck = sampler.allocation(rng)
        count += deck.count(tile)
    # 59 hidden physical slots, 19 deck slots, 4 copies. Unweighted count
    # vectors would have a different distribution and fail this comparison.
    assert count / 4000 == pytest.approx(4 * 19 / 59, abs=0.045)


def test_zero_sample_variance_does_not_certify_tie():
    low, high = empirical_bernstein(0, 0, 128, -100, 100, 0.001)
    assert low < -5 and high > 5
    _, _, obs = visible(step=1)
    while len(obs.legal_actions) == 1:
        _, _, obs = visible(step=len(obs.history) + 1)
    result = relations(np.zeros((128, len(obs.legal_actions))), obs, AdaptiveConfig(), 0)
    assert not result["resolved"]


def test_common_particle_offsets_match_one_uninterrupted_block():
    torch.set_num_threads(1)
    e = RuleEngine()
    s = e.new_game(118)
    while len(e.legal_actions(s)) == 1:
        s = e.step(s, e.legal_actions(s)[0])
    obs = e.observation(s, e.legal_actions(s)[0].player)
    model = PolicyValueNet(width=8, feature_version="huxi").eval()
    for parameter in model.parameters():
        parameter.data.zero_()
    sampler = HistorySampler(obs)
    full = teacher(
        obs,
        model=model,
        rollouts=4,
        seed=17,
        world_sampler=sampler,
        batched=True,
        keep_outcomes=True,
    )
    first = teacher(
        obs,
        model=model,
        rollouts=2,
        seed=17,
        world_sampler=sampler,
        batched=True,
        keep_outcomes=True,
    )
    last = teacher(
        obs,
        model=model,
        rollouts=2,
        seed=17,
        particle_offset=2,
        world_sampler=sampler,
        batched=True,
        keep_outcomes=True,
    )
    assert full.outcomes_cash == first.outcomes_cash + last.outcomes_cash


def test_tie_requires_equivalence_not_merely_interval_covering_zero(monkeypatch):
    from zimortal.training import adaptive

    _, _, obs = visible(step=1)
    a = obs.legal_actions[0]
    obs = replace(
        obs, legal_actions=(replace(a, kind=ActionType.DISCARD), replace(a, kind=ActionType.PASS))
    )
    monkeypatch.setattr(adaptive, "payoff_bounds", lambda _: (-10, 10))
    values = np.zeros((8192, 2))
    result = relations(values, obs, AdaptiveConfig(tie_cash=1), 0)
    assert result["resolved"] and result["relations"][0, 1] == 2
    values[:, 0] = 5
    result = relations(values, obs, AdaptiveConfig(tie_cash=1), 0)
    assert result["resolved"] and result["relations"][0, 1] == 1
    assert result["optimal"] == [0]


def test_adaptive_stopping_and_resume_use_two_certified_looks(monkeypatch):
    from types import SimpleNamespace

    from zimortal.training import adaptive

    _, _, obs = visible(step=1)
    a = obs.legal_actions[0]
    obs = replace(
        obs, legal_actions=(replace(a, kind=ActionType.DISCARD), replace(a, kind=ActionType.PASS))
    )
    monkeypatch.setattr(adaptive, "payoff_bounds", lambda _: (-10, 10))
    monkeypatch.setattr(adaptive, "HistorySampler", lambda _: object())
    offsets = []

    def simulated(_obs, **kwargs):
        offsets.append((kwargs["particle_offset"], kwargs["rollouts"]))
        return SimpleNamespace(outcomes_cash=((5, 0),) * kwargs["rollouts"])

    monkeypatch.setattr(adaptive, "teacher", simulated)
    config = AdaptiveConfig(minimum=1024, maximum=4096, chunk=32, tie_cash=1)
    first, _, _, status = adaptive.adaptive_rollout(obs, None, config, new_look_limit=1)
    assert status == "pending" and len(first) == 1024
    assert offsets[0][0] == 0 and offsets[-1][0] == 992
    offsets.clear()
    resumed, result, _, status = adaptive.adaptive_rollout(obs, None, config, initial=first)
    assert status == "qualified" and len(resumed) == 2048 and result["optimal"] == [0]
    assert offsets[0][0] == 1024
    full, _, _, status = adaptive.adaptive_rollout(obs, None, config)
    np.testing.assert_array_equal(resumed, full)
    assert status == "qualified"


def test_game_transitions_match_original_deepcopy_cloning(monkeypatch):
    from zimortal.engine import GameState

    e = RuleEngine()
    for seed in range(12):
        state = e.new_game(seed, seed % 3)
        rng = random.Random(seed)
        for _ in range(1000):
            if state.terminal:
                break
            acts = e.legal_actions(state)
            action = next((a for a in acts if a.kind == ActionType.HU), rng.choice(acts))
            before = state.serialize()
            fast = e.step(state, action)
            with monkeypatch.context() as patch:
                patch.setattr(GameState, "clone", copy.deepcopy)
                original = e.step(state, action)
            assert fast.serialize() == original.serialize()
            assert state.serialize() == before
            fast.validate()
            state = fast
        else:
            raise AssertionError("nonterminal game")


def test_visible_serialization_round_trip_has_no_private_opponent_hands():
    from zimortal.training.active import deserialize_observation, serialize_observation

    _, _, obs = visible()
    encoded = serialize_observation(obs)
    assert deserialize_observation(encoded) == obs
    assert "deck" not in encoded
    assert all("hand" not in player for player in encoded["players"])


def test_training_gate_rejects_small_or_unresolved_corpus():
    from zimortal.training.active import training_gate

    with pytest.raises(ValueError, match="5000"):
        training_gate({"records": {"train": [], "valid": []}}, target=4999)
    with pytest.raises(ValueError, match="not enough"):
        training_gate({"records": {"train": [], "valid": []}})

    def row(i):
        return {
            "input_hash": str(i),
            "seed": i,
            "qualified": True,
            "rollouts_per_action": 256,
            "scope": "top_set",
            "confidence_method": "alpha-spent simultaneous empirical Bernstein",
            "tags": ["chi_bi"],
            "confidence_index": i + 1,
            "alpha": 0.05 / ((i + 1) * (i + 2)),
        }

    manifest = {
        "records": {
            "train": [row(i) for i in range(5000)],
            "valid": [row(i) for i in range(6000, 6500)],
        },
        "config": {
            "quotas": {"chi_bi": 1000},
            "adaptive": {"alpha": 0.05, "minimum": 128, "scope": "top_set"},
        },
    }
    assert training_gate(manifest)["chi_bi"] == 5000
    manifest["records"]["train"][0]["qualified"] = False
    with pytest.raises(ValueError, match="uncertified"):
        training_gate(manifest)


def test_rule_derived_cash_bounds_contain_actual_game_settlements():
    from zimortal.training.adaptive import payoff_bounds

    e = RuleEngine()
    wins = 0
    for seed in range(50):
        s = e.new_game(seed, seed % 3)
        rng = random.Random(seed)
        observations = []
        while not s.terminal:
            a = e.legal_actions(s)
            observations.append(e.observation(s, a[0].player))
            selected = next((x for x in a if x.kind == ActionType.HU), rng.choice(a))
            s = e.step(s, selected)
        if s.settlement:
            wins += 1
            for obs in observations:
                low, high = payoff_bounds(obs)
                assert low <= s.settlement.payments[obs.player] <= high
    assert wins > 0


def test_payoff_bound_includes_upgrades_of_already_exposed_wei():
    from zimortal.engine import Meld, MeldType
    from zimortal.engine.observation import PublicPlayer
    from zimortal.engine.scoring import settle
    from zimortal.training.adaptive import payoff_bounds

    _, _, obs = visible(step=1)
    wei = tuple(Meld(MeldType.WEI, (t,) * 3) for t in range(12, 18))
    obs = replace(
        obs,
        player=0,
        hand=(18, 18),
        hu_disabled=False,
        players=(PublicPlayer(2, wei), PublicPlayer(20, ()), PublicPlayer(20, ())),
    )
    upgraded = tuple(Meld(MeldType.TI, (t,) * 4) for t in range(12, 18)) + (
        Meld(MeldType.PAIR, (18, 18)),
    )
    amount = settle(0, upgraded, ordinary_fan=1).payments[0]
    assert payoff_bounds(obs)[1] >= amount


def test_top_set_certifies_best_without_resolving_inferior_ties(monkeypatch):
    from zimortal.training import adaptive

    _, _, obs = visible(step=1)
    a = replace(obs.legal_actions[0], kind=ActionType.DISCARD)
    obs = replace(obs, legal_actions=tuple(replace(a, tile=i) for i in range(3)))
    monkeypatch.setattr(adaptive, "payoff_bounds", lambda _: (-10, 10))
    raw = np.zeros((1024, 3), dtype=np.int16)
    raw[:, 0] = 10
    raw[:512, 1] = 1
    top = relations(raw, obs, AdaptiveConfig(tie_cash=0.1), 0)
    assert top["resolved"] and top["optimal"] == [0]
    assert top["relations"][1, 2] == 0
    assert not relations(raw, obs, AdaptiveConfig(tie_cash=0.1, scope="full_order"), 0)["resolved"]


def test_chunked_cash_statistics_preserve_paired_variance():
    from zimortal.training.active import payment_statistics

    raw = np.random.default_rng(44).integers(-1000, 2000, (9001, 5), dtype=np.int16)
    mean, se, paired = payment_statistics(raw, 2)
    floating = raw.astype(np.float64)
    np.testing.assert_allclose(mean, floating.mean(0))
    np.testing.assert_allclose(se, floating.std(0, ddof=1) / np.sqrt(len(raw)))
    np.testing.assert_allclose(
        paired, (floating - floating[:, 2, None]).std(0, ddof=1) / np.sqrt(len(raw))
    )


@pytest.mark.parametrize("invalid", [float("nan"), 0.5, 32768, -32769])
def test_resume_rejects_lossy_particle_evidence(invalid):
    from zimortal.training.adaptive import adaptive_rollout

    _, _, obs = visible(step=1)
    raw = np.zeros((128, len(obs.legal_actions)))
    raw[0, 0] = invalid
    with pytest.raises(ValueError, match="lossless"):
        adaptive_rollout(obs, None, initial=raw)
