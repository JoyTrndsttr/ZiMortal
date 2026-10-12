"""Independent visible contingent fitting, fixed-policy testing and persistence."""

import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from zimortal.engine import RuleEngine
from zimortal.model.network import PolicyValueNet
from zimortal.training import autonomous as auto
from zimortal.training.active import serialize_observation
from zimortal.training.branch_teacher import action_data, visible_key
from zimortal.training.runtime import load_model, save_model


def visible_root():
    engine = RuleEngine()
    state = engine.new_game(319000007)
    while len(engine.legal_actions(state)) == 1:
        state = engine.step(state, engine.legal_actions(state)[0])
    return engine.observation(state, engine.legal_actions(state)[0].player)


def raw_example(obs, offset=0):
    return [
        {
            "particle": p,
            "branches": [
                {
                    "information_set": visible_key(obs),
                    "observation": serialize_observation(obs),
                    "second_action": action_data(action),
                    "first_index": 0,
                    "cash": offset + (10 if i == p % 2 else -10),
                    "reference_selected": i == 0,
                }
                for i, action in enumerate(obs.legal_actions)
            ],
        }
        for p in range(2)
    ]


def test_visible_regression_keeps_candidates_not_hidden_max():
    obs = visible_root()
    rows = auto.search_rows(raw_example(obs))
    assert len(rows) == 1 and rows[0]["support"] == 2
    assert rows[0]["q"][:2].tolist() == [0.0, 0.0]
    assert set(rows[0]) == {"x", "actions", "q", "se", "soft", "support", "root_weight"}
    assert rows[0]["root_weight"] == 1
    single = auto.search_rows(raw_example(obs)[:1])[0]
    assert np.all(single["se"] > 0)


def test_contingent_optimizer_matches_checkpoint_and_frozen_encoder(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    parent = PolicyValueNet("resnet", width=8, feature_version="huxi").eval()
    source, output = tmp_path / "parent.pt", tmp_path / "first.pt"
    save_model(parent, source)
    rows = [auto.search_rows(raw_example(visible_root()))]
    result = auto.fit_contingent(source, rows, output, device="cpu", epochs=2)
    assert result["frozen_encoder_verified"]
    loaded = load_model(output)
    assert all(
        torch.equal(v, loaded.parent.state_dict()[k]) for k, v in parent.state_dict().items()
    )
    auto.fit_contingent(
        source, rows, tmp_path / "second.pt", device="cpu", epochs=2, initialize=result["optimizer"]
    )
    with output.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="optimizer no longer matches"):
        auto.fit_contingent(
            source,
            rows,
            tmp_path / "third.pt",
            device="cpu",
            epochs=2,
            initialize=result["optimizer"],
        )


def test_independent_choice_is_visible_and_never_per_world_max(monkeypatch):
    obs = visible_root()
    first = obs.legal_actions[0]
    monkeypatch.setattr(
        auto, "choose_batch", lambda model, visible: [o.legal_actions[0] for o in visible]
    )
    monkeypatch.setattr(auto, "predict", lambda model, visible: (torch.tensor([1.0, 0.0]), None))
    raw = raw_example(obs)
    results = [(obs, {"raw_particles": raw}), (obs, {"raw_particles": raw_example(obs, 20)})]
    result = auto.independent_comparison(results, object(), object())
    assert result["mean_cash_delta"] == 0
    assert not result["accepted"]
    assert all(r["changed_particles"] == 0 for r in result["records"])
    assert action_data(first) == raw[0]["branches"][0]["second_action"]


def test_atomic_compression_roundtrip_and_disk_guard(tmp_path, monkeypatch):
    data = {"牌": [1, 2], "samples": list(range(1000))}
    path = tmp_path / "evidence.json.gz"
    auto.atomic_gzip(path, data)
    with gzip.open(path, "rt") as file:
        assert json.load(file) == data
    assert not (tmp_path / "evidence.json.gz.tmp").exists()
    monkeypatch.setattr(auto.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="disk reserve"):
        auto.atomic_gzip(path, {})
    with gzip.open(path, "rt") as file:
        assert json.load(file) == data
