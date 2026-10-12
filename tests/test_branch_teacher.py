"""Public-only two-decision expansion, physical replay and paired cash evidence."""

import copy
import math
from collections import Counter

import numpy as np
import pytest
import torch

from zimortal.belief.conditional import HistorySampler
from zimortal.belief.sampling import SamplingFailure, reverse_hands
from zimortal.engine import ActionType as A
from zimortal.engine import (
    GameState,
    PendingTile,
    PlayerState,
    RuleClarificationRequired,
    RuleEngine,
)
from zimortal.engine import SourceType as S
from zimortal.model.network import PolicyValueNet
from zimortal.training import branch_teacher as bt


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    return PolicyValueNet("mlp", width=8, feature_version="huxi").eval()


@pytest.fixture
def peng_obs():
    return bt.diagnostic_observation("split-001")


def test_paired_cash_statistics_and_unresolved():
    x = np.array([[10, 13], [-10, -4], [30, 38], [0, 2]], dtype=float)
    result = bt.paired_statistics(x)
    d = x[:, 0] - x[:, 1]
    assert result["rows"][0]["paired_vs_best"] == pytest.approx(d.mean())
    assert result["rows"][0]["paired_standard_error"] == pytest.approx(d.std(ddof=1) / 2)
    assert result["rows"][1]["mean_cash"] == pytest.approx(x[:, 1].mean())
    assert result["rows"][1]["standard_error"] == pytest.approx(x[:, 1].std(ddof=1) / 2)
    noisy = bt.paired_statistics([[10, -20], [-10, 30]])
    assert not noisy["resolved"]
    identical = bt.paired_statistics([[0, 0], [0, 0]])
    assert not identical["resolved"]
    assert identical["status"] == "insufficient evidence"
    with pytest.raises(ValueError):
        bt.paired_statistics([[math.nan, 1], [0, 2]])


@pytest.mark.parametrize("case,kind", [("split-001", A.PENG), ("split-005", A.CHI)])
def test_chi_peng_expand_all_necessary_discards(case, kind, model):
    obs = bt.diagnostic_observation(case)
    world, _ = HistorySampler(obs).sample(712)
    engine = RuleEngine()
    action = next(a for a in obs.legal_actions if a.kind == kind)
    original = world.serialize()
    state = engine.step(world, action)
    (found,) = bt.advance_to_decision(engine, [state], obs.player, model)
    assert found.phase == "discard"
    actions = engine.legal_actions(found)
    assert len(actions) > 1 and all(a.kind == A.DISCARD and a.player == obs.player for a in actions)
    children = [engine.step(found, a) for a in actions]
    for child in children:
        child.validate()
    assert world.serialize() == original


def test_shared_particle_reproducibility_observation_only_and_conservation(
    peng_obs, model, monkeypatch
):
    world, _ = HistorySampler(peng_obs).sample(712)
    engine = RuleEngine()
    for first in peng_obs.legal_actions:
        child = engine.step(world, first)
        assert child.deck == world.deck
        assert all(
            child.players[p].hand == world.players[p].hand for p in range(3) if p != peng_obs.player
        )
    seen = []
    original_choose = bt.choose_batch

    def visible_only(network, observations, device="cpu"):
        from zimortal.engine.observation import Observation

        assert all(isinstance(obs, Observation) for obs in observations)
        assert all(not hasattr(obs, "deck") for obs in observations)
        seen.extend(observations)
        return original_choose(network, observations, device=device)

    monkeypatch.setattr(bt, "choose_batch", visible_only)
    before = copy.deepcopy(model.state_dict())
    a = bt.evaluate_two_step_branches(peng_obs, model=model, rollouts=2, minimum=2, seed=712)
    b = bt.evaluate_two_step_branches(peng_obs, model=model, rollouts=2, minimum=2, seed=712)
    assert a["valid"] and b["valid"] and seen
    assert a["raw_particles"] == b["raw_particles"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert len({p["world_hash"] for p in a["raw_particles"]}) == 2
    for particle in a["raw_particles"]:
        assert {r["first_index"] for r in particle["branches"]} == {0, 1}
        peng = [r for r in particle["branches"] if r["first_action"]["kind"] == "peng"]
        assert len({r["information_set"] for r in peng}) == 1
        assert {r["second_action"]["tile"] for r in peng} == {1, 2, 3, 4, 6, 9, 11, 12, 13, 15}
    with pytest.raises(TypeError):
        bt.evaluate_two_step_branches(RuleEngine().new_game(118), model=model)


def test_original_player_free_choice_and_forced_actions(model):
    engine = RuleEngine()
    state = engine.new_game(118, 0)
    result = bt.advance_to_decision(engine, [state], 2, model)[0]
    actions = engine.legal_actions(result)
    assert result.terminal or len(actions) > 1 and actions[0].player == 2
    assert len(result.history) > 0
    # Opening ti and draws are mandatory and must not be exposed as a2.
    state = next(
        engine.new_game(seed, 0)
        for seed in range(100)
        if engine.legal_actions(engine.new_game(seed, 0))[0].kind == A.TI
    )
    result = bt.advance_to_decision(engine, [state], 0, model)[0]
    assert result.history[0].kind == A.TI
    assert len(engine.legal_actions(result)) > 1
    assert engine.legal_actions(result)[0].kind == A.DISCARD
    result.validate()


def test_terminal_branch_and_complete_history_replay(peng_obs, model):
    class CapturingSampler:
        def __init__(self):
            self.base = HistorySampler(peng_obs)
            self.worlds = []

        def sample(self, seed, attempts):
            world, n = self.base.sample(seed, attempts)
            self.worlds.append(world)
            return world, n

    class RecordingEngine(RuleEngine):
        def __init__(self):
            super().__init__()
            self.terminals = []
            self.hu_roots = []

        def step(self, state, action):
            out = super().step(state, action)
            if out.terminal:
                self.terminals.append(out)
                if action.kind == A.HU:
                    self.hu_roots.append(state)
            return out

    sampler, engine = CapturingSampler(), RecordingEngine()
    result = bt.evaluate_two_step_branches(
        peng_obs, model=model, rollouts=2, minimum=2, seed=712, sampler=sampler, engine=engine
    )
    assert result["valid"] and engine.terminals
    initial = []
    for w in sampler.worlds:
        hands = reverse_hands(peng_obs, [p.hand.copy() for p in w.players], sampler.base.draws)
        initial.append(
            GameState(
                [PlayerState(h, kans={t for t, n in Counter(h).items() if n == 3}) for h in hands],
                w.deck + list(reversed(sampler.base.draws)),
                dealer=sampler.base.dealer,
                turn=sampler.base.dealer,
                phase="opening",
            )
        )
    for terminal in engine.terminals:
        terminal.validate()
        replayed = False
        for root in initial:
            try:
                rebuilt = RuleEngine().replay(root, terminal.history)
            except (ValueError, RuleClarificationRequired):
                continue
            if rebuilt.serialize() == terminal.serialize():
                replayed = True
                break
        assert replayed
    terminal = engine.terminals[0]
    assert bt.advance_to_decision(engine, [terminal], peng_obs.player, model) == [terminal]
    hand = [0, 0] + [t for t in (2, 3, 4, 5, 7, 8) for _ in range(3)]
    pool = Counter({t: 4 for t in range(20)}) - Counter(hand + [0])
    rest = sorted(pool.elements())
    hands = [hand, rest[:40:2], rest[1:40:2]]
    hu_root = GameState(
        [PlayerState(h, kans={t for t, n in Counter(h).items() if n == 3}) for h in hands],
        rest[40:],
        phase="respond",
        pending=PendingTile(0, 1, S.DRAW),
        turn=1,
    )
    hu_root.validate()
    hu_actions = RuleEngine().legal_actions(hu_root)
    hu_obs = RuleEngine().observation(hu_root, hu_actions[0].player)

    class FixedSampler:
        def sample(self, *_):
            return hu_root, 1

    direct = bt.evaluate_two_step_branches(
        hu_obs, model=model, rollouts=2, minimum=2, sampler=FixedSampler()
    )
    assert direct["valid"]
    immediate = [
        b for b in direct["raw_particles"][0]["branches"] if b["first_action"]["kind"] == "hu"
    ]
    assert len(immediate) == 1 and immediate[0]["second_action"] is None
    assert (
        immediate[0]["cash"]
        == engine.step(hu_root, hu_actions[0]).settlement.payments[hu_obs.player]
    )


@pytest.mark.parametrize(
    "failure", [SamplingFailure("no sample"), RuleClarificationRequired("unknown rule")]
)
def test_entire_root_fails_instead_of_zero_labels(peng_obs, model, failure):
    class BrokenSampler:
        def sample(self, *_):
            raise failure

    result = bt.evaluate_two_step_branches(
        peng_obs, model=model, rollouts=2, minimum=2, sampler=BrokenSampler()
    )
    assert not result["valid"] and not result["resolved"]
    assert result["status"] == "failed root; no label"
    assert result["completed_particles"] == 0


def test_future_undefined_rule_invalidates_complete_root(peng_obs, model):
    class BrokenEngine(RuleEngine):
        def step(self, *_):
            raise RuleClarificationRequired("undefined transition")

    result = bt.evaluate_two_step_branches(
        peng_obs, model=model, rollouts=2, minimum=2, engine=BrokenEngine()
    )
    assert not result["valid"] and result["rule_failures"] == 1


def test_dynamic_public_states_never_form_clairvoyant_max_label():
    branch = {
        "first_index": 0,
        "first_action": {"kind": "pass"},
        "second_action": {"kind": "discard"},
        "observation": {},
        "cash": 100,
    }
    raw = [
        {"particle": 0, "branches": [dict(branch, information_set="draw-one")]},
        {"particle": 1, "branches": [dict(branch, information_set="draw-two")]},
    ]
    summary = bt._summary(raw, alpha=0.05, looks=1)
    assert summary["dynamic_information_sets"]
    assert not summary["resolved"] and summary["root_statistics"] is None
