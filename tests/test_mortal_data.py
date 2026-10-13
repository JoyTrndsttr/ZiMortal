import json
from pathlib import Path

import numpy as np
import pytest
import torch

from zimortal.engine import RuleClarificationRequired, RuleEngine
from zimortal.engine.scoring import Settlement
from zimortal.model.network import PolicyValueNet
from zimortal.training.mortal_data import (
    audit_dataset,
    encode_visible,
    exploration,
    generate_games,
    load_rows,
    save_shard,
    specs,
    terminal_rows,
)


def test_exploration_probability_is_normalized_and_nucleus_retains_crossing():
    logits = np.log([0.6, 0.3, 0.1])
    p = exploration(logits, temperature=1, top_p=0.7)
    assert p.tolist() == pytest.approx([2 / 3, 1 / 3, 0])
    p = exploration(logits, temperature=1, top_p=0.1, epsilon=0.3)
    assert p.tolist() == pytest.approx([0.8, 0.1, 0.1])
    with pytest.raises(ValueError):
        exploration(logits, temperature=0)


def test_seat_dealer_cross_product():
    assignments = specs(1000000000, 9)
    assert {(d, s) for _, d, s in assignments} == {(d, s) for d in range(3) for s in range(3)}
    assert [s for _, _, s in assignments] == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_cash_labels_and_remaining_own_decisions():
    state = RuleEngine().new_game(118, 1)
    with pytest.raises(TypeError):
        encode_visible(state)
    state.phase = "terminal"
    state.winner = 0
    state.settlement = Settlement(0, 18, {}, 100, (200, -100, -100))
    for seat, cash in [(0, 2.0), (1, -1.0), (2, -1.0)]:
        rows = terminal_rows([{}, {}, {}], state, seed=118, seat=seat, game_id=1, version="test")
        assert [r["terminal_cash"] for r in rows] == [cash] * 3
        assert [r["steps_to_terminal"] for r in rows] == [2, 1, 0]
    state.winner = None
    state.settlement = None
    assert (
        terminal_rows([{}], state, seed=118, seat=0, game_id=1, version="test")[0]["terminal_cash"]
        == 0
    )


def test_complete_games_reproducible_replay_and_ragged_shard(tmp_path):
    torch.manual_seed(8)
    model = PolicyValueNet(feature_version="huxi").eval()
    assignments = specs(1100000000, 6)
    left, failures, _ = generate_games(model, assignments, slots=3)
    right, errors, _ = generate_games(model, assignments, slots=3)
    assert not failures and not errors and len(left) == 6
    a = {g["seed"]: g for g in left}
    b = {g["seed"]: g for g in right}
    engine = RuleEngine()
    for seed, g in a.items():
        assert g["state"] == b[seed]["state"]
        state = __import__("zimortal.engine", fromlist=["GameState"]).GameState.deserialize(
            g["state"]
        )
        state.validate()
        assert (
            engine.replay(engine.new_game(seed, g["dealer"]), state.history).serialize()
            == g["state"]
        )
        for row, other in zip(g["rows"], b[seed]["rows"], strict=True):
            assert np.array_equal(row["features"], other["features"])
            assert (
                row["chosen"] == other["chosen"]
                and row["behavior_logprob"] == other["behavior_logprob"]
            )
            assert len(row["actions"]) > 1
            assert row["terminal_cash"] == (
                state.settlement.payments[g["seat"]] / 100 if state.settlement else 0
            )
    shard = save_shard(tmp_path, 0, left)
    rows = load_rows(tmp_path, {"shards": [shard]})
    assert len(rows) == sum(len(g["rows"]) for g in left)
    assert all(r["behavior_probabilities"].sum() == pytest.approx(1) for r in rows)
    assert all(r["policy_version"] == "ZiMortal-v0" for r in rows)
    assert Path(tmp_path / shard["replays"]).exists()
    report = audit_dataset(tmp_path, {"shards": [shard]})
    assert report["games"] == 6 and report["visible_rows_verified"]


def test_undefined_rule_excludes_whole_game_and_all_partial_rows():
    class Broken(RuleEngine):
        def step(self, state, action):
            if len(state.history) >= 5:
                raise RuleClarificationRequired("audit test missing rule")
            return super().step(state, action)

    model = PolicyValueNet(feature_version="huxi").eval()
    games, failures, _ = generate_games(model, specs(1100000000, 3), engine=Broken())
    assert games == [] and len(failures) == 3
    assert all(f["status"] == "whole game excluded" and "rows" not in f for f in failures)
    assert all(json.loads(f["state"]) for f in failures)
