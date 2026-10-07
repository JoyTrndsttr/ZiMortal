"""Training contracts: hidden information, full candidates, and legal masks."""

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from zimortal.engine import Action, ActionType, RuleEngine
from zimortal.model.encoding import CHANNELS, encode_action, encode_observation
from zimortal.model.network import PolicyValueNet
from zimortal.training.data import make_puzzle, winning_tiles


def test_encoding_does_not_depend_on_hidden_tiles():
    engine = RuleEngine()
    state = engine.new_game(118)
    obs = engine.observation(state, 0)
    hidden = state.clone()
    hidden.deck.reverse()
    hidden.players[1].hand.reverse()
    # Exchange hidden tiles without changing publicly visible hand sizes.
    hidden.players[1].hand[0], hidden.deck[0] = hidden.deck[0], hidden.players[1].hand[0]
    np.testing.assert_array_equal(
        encode_observation(obs), encode_observation(engine.observation(hidden, 0))
    )
    assert encode_observation(obs).shape == (CHANNELS, 20)


def test_bi_group_partition_is_encoded():
    # Complete bi patterns participate in candidate signatures.
    a = Action(ActionType.CHI, 0, 0, chi=(0, 1, 2), bi=((0, 1, 2), (0, 10, 10)))
    # Use actual legal chi patterns for both signatures.
    b = replace(a, bi=((0, 0, 10), (10, 11, 12)))
    assert not np.array_equal(encode_action(a, 0), encode_action(b, 0))


def test_puzzle_handles_opponent_opening_ti():
    obs, _ = make_puzzle(1100059)
    assert obs.legal_actions and all(a.player == 0 for a in obs.legal_actions)


def test_masks_exclude_padding_and_gradients_are_finite():
    model = PolicyValueNet("resnet")
    obs, _ = make_puzzle(1100001)
    x = torch.tensor(encode_observation(obs)[None])
    a = torch.tensor(np.stack([encode_action(t, 0) for t in obs.legal_actions])[None])
    a = torch.cat((a, torch.zeros_like(a[:, :1])), dim=1)
    mask = torch.ones(a.shape[:2], dtype=torch.bool)
    mask[:, -1] = False
    logits, v, w = model(x, a, mask)
    assert float(logits[0, -1].detach()) < -1e8
    (logits[:, :-1].mean() + v.mean() + w.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_puzzle_physical_and_repeatable():
    for seed in range(40):
        a, _ = make_puzzle(seed)
        b, _ = make_puzzle(seed)
        assert a == b and a.legal_actions
        assert max(a.hand.count(t) for t in range(20)) <= 3


def test_waits_respect_four_copies():
    assert 0 not in winning_tiles((0, 0, 0, 0), (), ())


def test_teacher_marks_its_hypothetical_discard_public(monkeypatch):
    from collections import Counter

    from zimortal.training import data

    obs, _ = make_puzzle(1100001)
    engine = RuleEngine()
    state = engine.new_game(1)
    while state.phase != "discard":
        state = engine.step(state, engine.legal_actions(state)[-1])
    obs = engine.observation(state, state.turn)
    action = obs.legal_actions[0]
    obs = replace(obs, legal_actions=(action,))
    captured = []
    monkeypatch.setattr(
        data,
        "hand_quality",
        lambda hand, melds, kans, visible: captured.append(visible.copy()) or 0.0,
    )
    data.teacher_scores(obs)
    before = Counter(obs.river)
    for player in obs.players:
        before.update(t for m in player.melds for t in m.tiles)
    assert captured[0][action.tile] == before[action.tile] + 1


def test_disabled_player_has_no_readiness_label():
    from zimortal.training.data import example

    obs, _ = make_puzzle(1100001)
    obs = replace(obs, hu_disabled=True)
    row = example(obs)
    assert row[3] == 0 and not row[4].any()


def test_selfplay_records_only_legal_candidates_and_terminal_targets():
    from zimortal.training.reinforce import collect

    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = PolicyValueNet("mlp")
    rows, report = collect(model, model, 3, 33000)
    assert report["games"] == 3 and report["replay_verified"]
    assert rows
    for row, logprob, value, target in rows:
        assert 0 <= row[2] < len(row[1])
        assert np.isfinite(logprob) and logprob <= 0
        assert np.isfinite(value) and -0.5 <= target <= 1


def test_random_baseline_and_frozen_opponent_routes():
    from zimortal.training.runtime import tournament

    torch.set_num_threads(2)
    random_report = tournament(None, [7000], model_policy="random")
    assert random_report["games"] == 3
    model = PolicyValueNet("mlp")
    report = tournament(model, [7000], opponent="model", opponent_model=model)
    assert report["games"] == 3 and report["illegal_actions"] == 0


def test_numeric_shards_preserve_ragged_actions(tmp_path):
    from zimortal.training.data import example
    from zimortal.training.scale import read_shard, write_shard

    rows = [example(make_puzzle(s)[0]) for s in (1100000, 1100001)]
    path = tmp_path / "data.npz"
    write_shard(path, rows)
    restored = read_shard(path)
    for original, result in zip(rows, restored, strict=True):
        for i in (0, 1, 4):
            np.testing.assert_array_equal(original[i], result[i])
        assert original[2:4] == result[2:4]
    assert len(restored[0][1]) != len(restored[1][1])


def test_shards_reject_illegal_teacher_indices(tmp_path):
    from zimortal.training.data import example
    from zimortal.training.scale import read_shard, write_shard

    path = tmp_path / "bad.npz"
    write_shard(path, [example(make_puzzle(1100001)[0])])
    with np.load(path, allow_pickle=False) as source:
        fields = {k: source[k] for k in source.files}
    fields["policy"][0] = len(fields["actions"])
    np.savez_compressed(path, **fields)
    with pytest.raises(ValueError, match="teacher index"):
        read_shard(path)


def test_scale_validation_excludes_exact_training_inputs():
    from zimortal.training.data import example
    from zimortal.training.scale import deduplicate

    a = example(make_puzzle(1100000)[0])
    b = example(make_puzzle(1100001)[0])
    training, valid = deduplicate([a, a], [a, b, b])
    assert len(training) == 1 and len(valid) == 1
    np.testing.assert_array_equal(valid[0][0], b[0])
