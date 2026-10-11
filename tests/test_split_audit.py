from collections import Counter

from zimortal.engine import Action
from zimortal.engine import ActionType as A
from zimortal.training.split_audit import diagnostics, plan


def counts(hand):
    counter = Counter(hand)
    return tuple(counter[t] for t in range(20))


def test_high_huxi_plans_do_not_double_count_overlap_or_unlock_kans():
    assert plan(counts([10, 11, 12, 16, 19]))[0] == 6
    assert plan(counts([10, 11, 11, 12, 16, 19]))[0] == 12
    assert plan(counts([10, 11, 11, 11, 12, 16, 19]))[0] == 0


def test_peng_breaks_big_123_but_new_peng_huxi_is_credited():
    report = diagnostics([0, 1, 2, 10, 11, 11, 12], Action(A.PENG, 0, 11))
    assert report["before_pattern_huxi"] == 9
    assert report["after_pattern_huxi"] == 3
    assert report["new_meld_huxi"] == 3 and report["static_huxi_delta"] == -3
    assert report["broken_patterns"] == [[10, 11, 12]]


def test_chi_meld_and_every_bi_are_credited_without_counting_incoming_as_hand():
    report = diagnostics([0, 0, 1, 1, 2, 10, 11], Action(A.CHI, 0, 2, (0, 1, 2), ((0, 1, 2),)))
    assert report["new_meld_huxi"] == 6
    assert report["static_huxi_delta"] == 3 and report["broken_patterns"] == []


def test_scoring_chi_conversion_and_board_are_distinguished():
    report = diagnostics([10, 11, 16, 19], Action(A.CHI, 0, 12, (10, 11, 12)))
    assert report["static_huxi_delta"] == 0
    assert report["broken_patterns"] == [[11, 16, 19]]
    assert not report["board"]
    board = diagnostics([10, 10, 11, 12], Action(A.CHI, 0, 0, (0, 10, 10)))
    assert board["board"] and board["static_huxi_delta"] == -6
    reverse = diagnostics([0, 0, 10], Action(A.CHI, 0, 10, (0, 10, 10)))
    assert reverse["mixed_pair"] and not reverse["board"]
