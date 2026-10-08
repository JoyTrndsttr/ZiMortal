"""Prevent the NPZ backing-array leak and verify epoch recovery equivalence."""

import argparse
import json
from collections import Counter

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from zimortal.engine import RuleEngine
from zimortal.model.network import PolicyValueNet
from zimortal.training import planning
from zimortal.training.runtime import load_model, save_model


def test_npz_members_loaded_once_and_shared(tmp_path, monkeypatch):
    engine = RuleEngine()
    state = engine.new_game(118)
    obs = engine.observation(state, state.turn)
    row = planning.pack(obs, state.deck, np.ones(len(obs.legal_actions)), 0.25, 1)
    path = tmp_path / "rows.npz"
    planning.save_rows(path, [row, row, row])
    with pytest.raises(ValueError, match="memory budget"):
        planning.read_rows(path, max_bytes=1)
    calls = Counter()
    original = np.lib.npyio.NpzFile.__getitem__

    def counted(self, name):
        calls[name] += 1
        return original(self, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counted)
    rows = planning.read_rows(path)
    assert len(rows) == 3
    assert len(calls) == 8 and set(calls.values()) == {1}
    assert rows[0][1].base is rows[1][1].base
    for loaded in rows:
        for before, after in zip(row, loaded):
            np.testing.assert_equal(before, after)


def test_epoch_resume_matches_uninterrupted_training(tmp_path):
    import hashlib

    engine = RuleEngine()
    state = engine.new_game(118)
    obs = engine.observation(state, state.turn)
    row = planning.pack(obs, state.deck, None, distance=1)
    for name in ("train.npz", "valid.npz"):
        planning.save_rows(tmp_path / name, [row])
    files = {
        n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
        for n in ("train.npz", "valid.npz")
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"files": files}))
    parent = tmp_path / "parent.pt"
    save_model(PolicyValueNet(feature_version="huxi"), parent)

    def args(output, epochs, resume=False):
        return argparse.Namespace(
            data=str(tmp_path),
            resume=str(parent),
            device="cpu",
            output=str(tmp_path / output),
            report=str(tmp_path / "report.json"),
            epochs=epochs,
            continue_training=resume,
        )

    planning.train(args("continuous.pt", 2))
    planning.train(args("resumed.pt", 1))
    planning.train(args("resumed.pt", 2, True))
    a = load_model(tmp_path / "continuous.pt").state_dict()
    b = load_model(tmp_path / "resumed.pt").state_dict()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    checkpoint = tmp_path / "resumed.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="best checkpoint changed"):
        planning.train(args("resumed.pt", 3, True))
    checkpoint.write_bytes(original)
    with (tmp_path / "train.npz").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="dataset changed"):
        planning.train(args("resumed.pt", 3, True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA access")
def test_cuda_validation_includes_cash_soft_targets():
    import copy

    engine = RuleEngine()
    state = engine.new_game(118)
    obs = engine.observation(state, state.turn)
    row = planning.pack(obs, state.deck, np.ones(len(obs.legal_actions)), 0.25, 1)
    model = PolicyValueNet(feature_version="huxi", auxiliary_version="planning").eval()
    cpu = planning.metrics(model, [row])
    gpu = planning.metrics(copy.deepcopy(model).cuda(), [row])
    assert gpu == pytest.approx(cpu, rel=1e-3, abs=1e-4)
