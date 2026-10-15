import pytest

from zimortal.training.mortal_anchor_30k import BUDGETS, SEEDS, START, summarize


def test_only_two_fixed_policies_and_new_seed_range():
    assert BUDGETS == {"baseline": 0, "anchor-all": None}
    assert START == 1160000000 and SEEDS == 30000


def test_summary_requires_complete_pairs_and_counts_real_deviations():
    games = [{"seed": s, "seat": seat, "cash": 3} for s in (1, 2) for seat in range(3)]
    baseline = {(g["seed"], g["seat"]): {"cash": 1} for g in games}
    counts = dict.fromkeys(baseline, 0)
    counts[1, 0] = 1
    rows = [{"seed": 1, "seat": 0, "family": "discard"}]
    result = summarize(rows, games, baseline, counts, 10)
    assert result["primary"]["paired_delta"] == 2
    assert result["primary"]["seed_bootstrap_95"] == [2, 2]
    assert result["deviation_rate"] == .1
    assert result["count_histogram"] == {0: 5, 1: 1}
    assert result["family"]["discard"]["deviations"] == 1
    assert result["family"]["chi/pass"]["deviations"] == 0
    with pytest.raises(RuntimeError, match="incomplete"):
        summarize(rows, games[:-1], baseline, counts, 10)
    incomplete = {(1, 0): {"cash": 0}}
    with pytest.raises(RuntimeError, match="three seats"):
        summarize([], [games[0]], incomplete, {(1, 0): 0}, 1)
    with pytest.raises(RuntimeError, match="count mismatch"):
        summarize([], games, baseline, counts, 10)
