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
