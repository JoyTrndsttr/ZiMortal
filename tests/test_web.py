import json

from zimortal.web.server import build_game


def test_review_trace_seed118_replay_and_visibility():
    trace = build_game(118)
    assert trace["replay_verified"]
    assert trace["frames"][-1]["terminal"]
    assert all(f["verified"] and not f["error"] for f in trace["frames"])
    assert [len(p["hand"]) for p in trace["frames"][0]["players"]] == [21, 20, 20]
    assert all(f["note"] for f in trace["frames"])
    assert json.loads(json.dumps(trace))["seed"] == 118
    assert trace == build_game(118)


def test_review_draw_records_revealed_tile():
    trace = build_game(42)
    for frame in trace["frames"][1:]:
        if frame["action"]["kind"] == "draw" and frame["pending"]:
            assert frame["action"]["tile"] == frame["pending"]["tile"]


def test_seed118_drawer_chi_window_after_peng_pass():
    trace = build_game(118)
    draw = trace["frames"][6]
    assert draw["action"]["kind"] == "draw"
    assert draw["action"]["player"] == 1 and draw["action"]["tile"] == 15
    assert draw["legal"][0]["kind"] == "peng" and draw["legal"][0]["player"] == 2
    peng_pass = trace["frames"][7]
    assert peng_pass["action"]["kind"] == "pass" and peng_pass["action"]["player"] == 2
    assert peng_pass["legal"][0]["kind"] == "chi" and peng_pass["legal"][0]["player"] == 1
    assert any(a["kind"] == "pass" and a["player"] == 1 for a in peng_pass["legal"])


def test_seed322_cannot_peng_into_discard_dead_end():
    trace = build_game(322)
    assert trace["frames"][-1]["terminal"]
    assert all(not f["error"] for f in trace["frames"])


def test_arrangement_preserves_tiles_and_locked_kans():
    from collections import Counter

    from zimortal.engine import RuleEngine
    from zimortal.web.layout import arrange_hand

    engine = RuleEngine()
    for seed in range(100):
        for player in engine.new_game(seed).players:
            columns = arrange_hand(player.hand, player.kans)
            assert Counter(t for c in columns for t in c) == Counter(player.hand)
            assert all(1 <= len(c) <= 4 for c in columns)
            assert 6 <= len(columns) <= 8
            for kan in player.kans:
                assert [kan] * 3 in columns
            assert columns == arrange_hand(player.hand, player.kans)


def test_arrangement_groups_special_sequences_and_mixed_rank():
    from zimortal.web.layout import arrange_hand

    # Enough unrelated columns to leave room for the preferred combinations.
    hand = [0, 1, 2, 11, 16, 19, 4, 14, 3, 3, 5, 5, 7, 7, 8, 8, 12, 12, 15, 15]
    columns = arrange_hand(hand)
    assert any({0, 1, 2} <= set(c) for c in columns)
    assert any({11, 16, 19} <= set(c) for c in columns)
    assert any({4, 14} <= set(c) for c in columns)


def test_discard_records_follow_each_players_actual_history():
    trace = build_game(118)
    expected = [[], [], []]
    for frame in trace["frames"]:
        action = frame["action"]
        if action and action["kind"] == "discard":
            expected[action["player"]].append((frame["index"], action["tile"]))
        for seat, player in enumerate(frame["players"]):
            assert [(d["step"], d["tile"]) for d in player["discards"]] == expected[seat]
    assert not any(p["discards"] for p in trace["frames"][0]["players"])


def test_discard_claim_and_landing_are_distinguished():
    from zimortal.engine import Action, ActionType, SourceType
    from zimortal.web.server import discard_records

    history = [Action(ActionType.DISCARD, 0, 3)]
    assert discard_records(history)[0][0]["status"] == "pending"
    history.append(Action(ActionType.PENG, 1, 3, source_player=0, source_type=SourceType.DISCARD))
    record = discard_records(history)[0][0]
    assert record["status"] == "claimed" and record["claimed_by"] == 1
    history.append(Action(ActionType.DISCARD, 1, 8))
    history.append(
        Action(ActionType.PASS, 1, 8, source_player=1, source_type=SourceType.DISCARD, forced=True)
    )
    assert discard_records(history)[1][0]["status"] == "landed"


def test_last_actions_include_revealed_draw_tile():
    trace = build_game(118)
    frame = trace["frames"][6]
    assert frame["players"][1]["last_action"]["kind"] == "draw"
    assert frame["players"][1]["last_action"]["tile"] == 15
    assert trace["frames"][7]["players"][1]["last_action"] == frame["players"][1]["last_action"]


def test_analysis_replays_ui_annotated_draws():
    from zimortal.engine import RuleEngine
    from zimortal.web.server import replay_frame, snapshot

    engine = RuleEngine()
    trace = json.loads(json.dumps(build_game(42)))
    state = engine.new_game(42)
    for frame in trace["frames"][1:]:
        state = replay_frame(engine, state, frame)
        state.validate()
        actual = json.loads(json.dumps(snapshot(engine, state)))
        for key in ("remaining", "pending", "river", "legal", "phase", "terminal", "winner"):
            assert actual[key] == frame[key]
        assert [p["hand"] for p in actual["players"]] == [p["hand"] for p in frame["players"]]
