import pytest
import torch

from zimortal.engine import RuleEngine
from zimortal.model.encoding import ACTION_DIM
from zimortal.model.mortal_q import MortalQNet
from zimortal.training.mortal_offline import mc_target, q_loss
from zimortal.training.runtime import load_model, save_model


@pytest.mark.parametrize("counts", [(1, 3), (2, 5)])
def test_dueling_shape_legal_mean_and_padding_invariance(counts):
    torch.manual_seed(31)
    model = MortalQNet(width=8, latent=32, blocks=1).eval()
    x = torch.randn(2, 55, 20)
    a = torch.randn(2, max(counts), ACTION_DIM)
    mask = torch.arange(max(counts))[None] < torch.tensor(counts)[:, None]
    q, value, aux = model.forward_q(x, a, mask)
    assert q.shape == mask.shape and value.shape == (2,) and not aux
    assert torch.isneginf(q[~mask]).all()
    assert torch.allclose(q.masked_fill(~mask, 0).sum(-1) / mask.sum(-1), value, atol=1e-6)
    for i, n in enumerate(counts):
        single = model.forward_q(
            x[i : i + 1], a[i : i + 1, :n], torch.ones(1, n, dtype=torch.bool)
        )[0]
        assert torch.allclose(q[i, :n], single[0], atol=1e-6)
    with pytest.raises(ValueError):
        model.forward_q(x, a, torch.zeros_like(mask))


def test_observation_only_and_checkpoint_roundtrip(tmp_path):
    model = MortalQNet(width=8, latent=32, blocks=1).eval()
    engine = RuleEngine()
    state = engine.new_game(118, 1)
    obs = engine.observation(state, engine.legal_actions(state)[0].player)
    with pytest.raises(TypeError):
        model.observation_q(state)
    before = model.observation_q(obs)[0]
    # Different unseen wall orders cannot change an identical Observation input.
    state.deck.reverse()
    assert engine.observation(state, obs.player) == obs
    assert torch.equal(model.observation_q(obs)[0], before)
    save_model(model, tmp_path / "q.pt")
    loaded = load_model(tmp_path / "q.pt")
    assert loaded.latent == 32 and loaded.blocks == 1
    assert torch.equal(loaded.observation_q(obs)[0], before)


@pytest.mark.parametrize("gamma", [1.0, 0.995, 0.99])
def test_mc_gamma_and_chosen_action_loss(gamma):
    cash = torch.tensor([2.0, -1.0])
    steps = torch.tensor([0, 3])
    target = mc_target(cash, steps, gamma)
    assert target.tolist() == pytest.approx([2.0, -(gamma**3)])
    q = torch.tensor([[2.0, 5.0, 999.0], [-1.0, 0.0, 999.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    chosen = torch.tensor([0, 0])
    loss, parts = q_loss(q, chosen, target, mask, alpha=0.1)
    assert torch.isfinite(loss)
    expected = torch.logsumexp(q[:, :2], -1).mean() - q[:, 0].mean()
    assert torch.allclose(parts["cql_loss"], expected)
    other = q.clone()
    other[:, -1] = -999
    assert torch.equal(q_loss(other, chosen, target, mask, 0.1)[0], loss)
    with pytest.raises(ValueError):
        q_loss(q, torch.tensor([2, 0]), target, mask)
