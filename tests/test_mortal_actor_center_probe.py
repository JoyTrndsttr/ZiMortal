"""Protocol invariants and real ordinary-engine collection for center intervention."""

from collections import Counter

import numpy as np
import pytest
import torch
from test_mortal_deviation import BEHAVIOR, FixedQ, Parent

from zimortal.engine import Action, ActionType
from zimortal.model.encoding import encode_action
from zimortal.training.mortal_actor_center_probe import (
    OwnershipPolicy,
    matched_selection,
    ownership,
    probabilities,
    safety_stop,
    specs,
)
from zimortal.training.mortal_data import audit_dataset, save_shard
from zimortal.training.mortal_online import collect


def test_predeal_nine_strata_exact_quota_and_fresh_blocks():
    seen = set()
    for g in (1, 2, 3):
        train = specs(g)
        mask = ownership(train, f"train-{g}")
        assert ownership(train, f"train-{g}") == mask
        assert len(mask) == 5000 and sum(r["probe"] for r in mask) == 500
        quotas = Counter((r["dealer"], r["seat"]) for r in mask if r["probe"])
        assert [quotas[k] for k in sorted(quotas)] == [56] * 5 + [55] * 4
        for ss in (train, specs(g, evaluation=True)):
            seeds = {s[0] for s in ss}
            assert not seeds & seen
            seen |= seeds
        ev = ownership(specs(g, evaluation=True), f"eval-{g}", evaluation=True)
        assert sum(r["probe"] for r in ev) == 150
        for seed in {r["seed"] for r in ev}:
            assert len({r["probe"] for r in ev if r["seed"] == seed}) == 1
    hold = specs(holdout=True)
    assert not seen & {s[0] for s in hold}
    assert sum(r["probe"] for r in ownership(hold, "holdout", evaluation=True)) == 600


def test_epsilon_is_greedy_softmax_mixture_and_hu_is_greedy():
    score = np.array([1.0, 0.0, -0.5])
    softmax = np.exp(score / 0.5) / np.exp(score / 0.5).sum()
    np.testing.assert_allclose(probabilities(score, 0.005), 0.005 * softmax + [0.995, 0, 0])
    np.testing.assert_array_equal(probabilities(score, 0.005, hu=True), [1, 0, 0])
    np.testing.assert_array_equal(probabilities(score, 0), [1, 0, 0])
    with pytest.raises(ValueError):
        probabilities([np.inf], 0.005)


def test_hu_never_raw_and_nonprobe_uses_same_anchor():
    for family in ("discard", "hu"):
        aa = (
            [Action(ActionType.HU, 0), Action(ActionType.PASS, 0)]
            if family == "hu"
            else [Action(ActionType.DISCARD, 0, 0), Action(ActionType.DISCARD, 0, 1)]
        )
        actions = torch.from_numpy(np.stack([encode_action(a, 0) for a in aa]))[None]
        legal = torch.ones(1, 2, dtype=torch.bool)
        for flag in (False, True):
            mask = [{"seed": 1, "seat": 0, "probe": flag}]
            actors = {
                a: OwnershipPolicy(FixedQ(), Parent(), BEHAVIOR, mask, arm=a, training=True)
                for a in ("A", "B")
            }
            for a, actor in actors.items():
                actor.prepare([(1, 0, 0, 0)])
                actor(torch.zeros(1, 55, 20), actions, legal)
                assert actor.raw_center == [a == "B" and flag and family != "hu"]
                assert actor.last[3].all()  # No support filter.
                if family == "hu":
                    assert not actor.branches[0]
                    assert actor.actor_probabilities[0].max() == 1
            if not flag or family == "hu":
                np.testing.assert_array_equal(actors["A"].last[-1], actors["B"].last[-1])
                assert actors["A"].choices == actors["B"].choices


def test_matched_single_use_rows_and_continuous_rng():
    rngs = {a: np.random.default_rng(3) for a in ("A", "B")}
    first = matched_selection(1025, 800, rngs)
    assert all(len(v) == len(set(v)) == 768 for v in first.values())
    second = matched_selection(1025, 800, rngs)
    assert first != second
    assert all(max(v) < n for v, n in zip(first.values(), (1025, 800), strict=True))
    with pytest.raises(ValueError):
        matched_selection(255, 800, rngs)


def test_only_two_consecutive_extreme_deployment_losses_stop():
    def result(upper):
        return {"cash": {"B-D": {"seed_bootstrap_95": [-10, upper]}}}

    assert not safety_stop([result(-9)])
    assert not safety_stop([result(-6), result(-5)])
    assert safety_stop([result(-6), result(-5.01)])
    assert not safety_stop([result(-6), result(-4), result(-6)])


def test_real_trajectories_no_branch_labels_and_same_nonprobe_execution(tmp_path):
    game_specs = [(1699999900 + i, i % 3, i % 3, 0) for i in range(8)]
    mask = [{"seed": s[0], "seat": s[2], "probe": i % 2 == 0} for i, s in enumerate(game_specs)]
    evidence = {}
    for arm in ("A", "B"):
        actor = OwnershipPolicy(FixedQ(), Parent(), BEHAVIOR, mask, arm=arm, training=True)
        games, records, data = collect(
            actor.eval(), actor.parent, game_specs, tmp_path / arm, arm, training_rows=True
        )
        manifest = {"shards": [save_shard(tmp_path / arm, 0, data)]}
        assert audit_dataset(tmp_path / arm, manifest)["games"] == len(game_specs) == len(games)
        assert len(records) == sum(len(g["rows"]) for g in data)
        evidence[arm] = [
            (r["seed"], r["seat"], r["step"], r["chosen"], r["terminal_cash"])
            for r in records
            if not r["probe"]
        ]
        assert all(
            not r["raw_center"]
            for r in records
            if r["family"] == "hu/pass" or arm == "A" or not r["probe"]
        )
    assert evidence["A"] == evidence["B"]


def test_probe_switches_actual_greedy_center_without_support_guard():
    class PeakedParent(Parent):
        def forward(self, x, a, mask):
            logits, value, extra = super().forward(x, a, mask)
            logits[:, 0] = 8.0
            return logits, value, extra

    aa = [Action(ActionType.DISCARD, 0, tile=i) for i in range(2)]
    actions = torch.from_numpy(np.stack([encode_action(a, 0) for a in aa]))[None]
    legal = torch.ones(1, 2, dtype=torch.bool)
    mask = [{"seed": 10, "seat": 0, "probe": True}]
    for arm, expected in (("A", 0), ("B", 1), ("D", 1)):
        actor = OwnershipPolicy(FixedQ(), PeakedParent(), BEHAVIOR, mask, arm=arm)
        actor.prepare([(10, 0, 0, 0)])
        scores = actor(torch.zeros(1, 55, 20), actions, legal)[0]
        assert scores.argmax(-1).item() == expected
        assert actor.last[1][0, 1] < 0.1
        assert actor.last[3].all()
