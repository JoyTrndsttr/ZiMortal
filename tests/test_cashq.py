"""Cash-Q masks, conservative fallback, batching, and checkpoint round trips."""

import random

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from zimortal.engine import RuleEngine
from zimortal.model.cashq import CashQNet
from zimortal.model.network import PolicyValueNet
from zimortal.training.planning import minibatch, pack
from zimortal.training.rollout import teacher
from zimortal.training.runtime import choose, choose_batch, load_model, save_model


def observation():
    engine = RuleEngine()
    state = engine.new_game(118)
    while len(engine.legal_actions(state)) == 1:
        state = engine.step(state, engine.legal_actions(state)[0])
    return engine.observation(state, engine.legal_actions(state)[0].player)


def test_cashq_parent_is_frozen_and_disabled_gate_exactly_preserves_policy(tmp_path):
    obs = observation()
    model = CashQNet().eval()
    x, a, m, _ = minibatch([pack(obs, None)])
    original = model.parent(x, a, m)
    torch.testing.assert_close(model(x, a, m)[0], original[0], rtol=0, atol=0)
    ensemble, _ = model.forward_q(x, a, m)
    assert ensemble.shape == (*m.shape, 3)
    assert all(not p.requires_grad for p in model.parent.parameters())
    path = tmp_path / "q.pt"
    save_model(model, path, value_target="net cash/100")
    loaded = load_model(path)
    assert isinstance(loaded, CashQNet)
    torch.testing.assert_close(loaded(x, a, m)[0], original[0], rtol=0, atol=0)
    # Real gradient reaches Q heads and cannot change the pretrained encoder.
    ensemble.square().mean().backward()
    assert all(p.grad is None for p in model.parent.parameters())


def test_gate_uses_advantage_disagreement_and_masks_illegal_actions(monkeypatch):
    model = CashQNet().eval()
    model.enabled.fill_(True)
    model.margin_cash.fill_(10)
    x = torch.zeros(1, 55, 20)
    a = torch.zeros(1, 3, model.parent.action_encoder[0].in_features)
    mask = torch.tensor([[True, True, False]])
    original = (torch.tensor([[1.0, 0.0, -1e9]]), torch.zeros(1), torch.zeros(1, 20))
    values = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [100.0, 100.0, 100.0]]])
    monkeypatch.setattr(model, "forward_q", lambda *_: (values, original))
    assert int(model(x, a, mask)[0].argmax()) == 1
    values[0, 1] = torch.tensor([0.2, -0.2, 0.6])
    assert int(model(x, a, mask)[0].argmax()) == 0


def test_batched_rollout_matches_serial_common_particles_and_pair_errors():
    torch.set_num_threads(2)
    model = PolicyValueNet(feature_version="huxi").eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    obs = observation()
    assert choose_batch(model, [obs, obs]) == [choose(obs, random.Random(1), model)] * 2
    serial = teacher(obs, model=model, rollouts=2, seed=144, reference_index=1)
    batched = teacher(obs, model=model, rollouts=2, seed=144, reference_index=1, batched=True)
    assert serial == batched
    assert batched.paired_standard_errors[1] == 0


def test_cashq_epoch_recovery_preserves_optimizer_and_parent(tmp_path):
    import argparse
    import hashlib
    import json

    from zimortal.model.encoding import encode_action, encode_observation
    from zimortal.training.cashq import train
    from zimortal.training.rollout import soft_target

    parent = PolicyValueNet(width=8, feature_version="huxi").eval()
    parent_path = tmp_path / "parent.pt"
    save_model(parent, parent_path)
    records = {"train": [], "valid": []}
    engine = RuleEngine()
    for i in range(4):
        state = engine.new_game(118 + i)
        while len(engine.legal_actions(state)) == 1:
            state = engine.step(state, engine.legal_actions(state)[0])
        obs = engine.observation(state, engine.legal_actions(state)[0].player)
        n = len(obs.legal_actions)
        path = tmp_path / f"row-{i}.npz"
        q = np.linspace(-0.3, 0.4, n, dtype=np.float32)
        reference = obs.legal_actions.index(choose(obs, random.Random(0), parent))
        paired = np.ones(n, dtype=np.float32) * 0.05
        paired[reference] = 0
        np.savez_compressed(
            path,
            x=encode_observation(obs, "huxi"),
            actions=np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
            q=q,
            se=np.full(n, 0.1, dtype=np.float32),
            paired_se=paired,
            soft=np.array(soft_target(q * 100, 10), dtype=np.float32),
            reference=reference,
        )
        records["train" if i < 3 else "valid"].append(
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"records": records}))

    def args(name, epochs, resume=False):
        return argparse.Namespace(
            data=str(tmp_path),
            parent=str(parent_path),
            device="cpu",
            output=str(tmp_path / name),
            report=str(tmp_path / "report.json"),
            epochs=epochs,
            continue_training=resume,
        )

    train(args("continuous.pt", 2))
    train(args("resumed.pt", 1))
    train(args("resumed.pt", 2, True))
    a, b = load_model(tmp_path / "continuous.pt"), load_model(tmp_path / "resumed.pt")
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, b.parent.state_dict()[name], rtol=0, atol=0)
    changed = args("resumed.pt", 3, True)
    changed.selection = "cash_rmse"
    with pytest.raises(ValueError, match="selection changed"):
        train(changed)


def test_web_cashq_candidates_match_visible_legal_actions():
    from zimortal.web.server import cashq_analysis

    engine = RuleEngine()
    state = engine.new_game(118)
    while len(engine.legal_actions(state)) == 1:
        state = engine.step(state, engine.legal_actions(state)[0])
    model = CashQNet(width=8).eval()
    result = cashq_analysis(engine, state, model)
    assert result["player"] == engine.legal_actions(state)[0].player
    assert len(result["candidates"]) == len(engine.legal_actions(state))
    assert sum(c["selected"] for c in result["candidates"]) == 1
    assert sum(c["parent"] for c in result["candidates"]) == 1
    assert not result["changed_from_parent"]
    assert all(np.isfinite(c["cash_q"]) for c in result["candidates"])
    assert all(c["model_disagreement"] == 0 for c in result["candidates"])


def test_evaluation_reports_when_gate_changes_no_decisions():
    from zimortal.training.runtime import tournament

    torch.set_num_threads(2)
    model = CashQNet(width=8).eval()
    result = tournament(model, [118], include_games=True)
    baseline = tournament(model.parent, [118], include_games=True)
    assert result["cashq_decisions"] > 0
    assert result["cashq_changes"] == 0
    assert result["game_results"] == baseline["game_results"]
