import pytest
import torch

from zimortal.model.encoding import ACTION_DIM
from zimortal.model.mortal_faithful import ChannelAttention, MortalFaithfulQNet
from zimortal.training.mortal_offline import q_loss
from zimortal.training.runtime import load_model, save_model


def test_shared_channel_attention_formula():
    torch.manual_seed(9)
    attention = ChannelAttention(16)
    x = torch.randn(2, 16, 20)
    expected = x * (
        attention.shared_mlp(x.mean(-1)) + attention.shared_mlp(x.amax(-1))
    ).sigmoid().unsqueeze(-1)
    assert torch.allclose(attention(x), expected, atol=1e-6)


def test_faithful_legal_action_padding_and_checkpoint(tmp_path):
    torch.manual_seed(31)
    model = MortalFaithfulQNet(width=16, blocks=1).eval()
    x, a = torch.randn(2, 55, 20), torch.randn(2, 4, ACTION_DIM)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    q, value, aux = model(x, a, mask)
    assert not aux
    assert torch.isneginf(q[~mask]).all()
    assert torch.allclose(q.masked_fill(~mask, 0).sum(-1) / mask.sum(-1), value, atol=1e-6)
    single = model(x[:1], a[:1, :2], mask[:1, :2])[0]
    assert torch.allclose(q[:1, :2], single, atol=1e-6)
    save_model(model, tmp_path / "faithful.pt")
    loaded = load_model(tmp_path / "faithful.pt")
    assert loaded.architecture == "mortal_faithful_v1"
    assert torch.equal(loaded(x, a, mask)[0], q)
    model.train()
    before = model.brain[1].res_unit[0].running_mean.clone()
    model(x, a, mask)[0][mask].square().mean().backward()
    assert not torch.equal(before, model.brain[1].res_unit[0].running_mean)


def test_official_half_mse_and_invalid_loss():
    q = torch.tensor([[3.0, 0.0]])
    loss, parts = q_loss(
        q, torch.tensor([0]), torch.tensor([0.0]), torch.ones(1, 2, dtype=torch.bool), 0, "mse"
    )
    assert loss.item() == 4.5
    assert parts["mc_loss"].item() == 4.5
    with pytest.raises(ValueError):
        q_loss(
            q,
            torch.tensor([0]),
            torch.tensor([0.0]),
            torch.ones(1, 2, dtype=torch.bool),
            regression_loss="other",
        )
