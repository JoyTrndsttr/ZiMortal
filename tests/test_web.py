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
