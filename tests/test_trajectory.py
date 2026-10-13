import numpy as np
import torch

from zimortal.engine import Action, RuleEngine
from zimortal.engine import ActionType as A
from zimortal.engine import SourceType as S
from zimortal.engine.scoring import Settlement
from zimortal.engine.types import Meld, MeldType
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.model.network import PolicyValueNet
from zimortal.model.trajectory import FAN_NAMES, GROUP_PATTERNS, terminal_route_targets
from zimortal.training.runtime import load_model, save_model
from zimortal.training.trajectory import (
    batch,
    future_intakes,
    huxi_baselines,
    initialize_group_prior,
    initialize_huxi_baseline,
    migrate,
    objective,
    route_warmup,
    verify_parent_policy,
)


def fixture():
    engine = RuleEngine()
    state = engine.new_game(118)
    actions = engine.legal_actions(state)
    actor = actions[0].player
    obs = engine.observation(state, actor)
    return encode_observation(obs, "huxi"), np.stack([encode_action(a, actor) for a in actions])


def test_migration_preserves_policy_and_roundtrip(tmp_path):
    parent = PolicyValueNet(feature_version="huxi").eval()
    model = migrate(parent).eval()
    x, a = fixture()
    inputs = torch.tensor(x)[None], torch.tensor(a)[None], torch.ones((1, len(a)), dtype=torch.bool)
    for old, new in zip(parent(*inputs), model(*inputs), strict=True):
        assert torch.equal(old, new)
    save_model(model, tmp_path / "model.pt")
    loaded = load_model(tmp_path / "model.pt")
    assert loaded.auxiliary_version == "trajectory"
    assert torch.equal(model(*inputs)[0], loaded(*inputs)[0])


def test_auxiliary_supervision_reaches_candidate_fusion_and_failure_mask():
    torch.set_num_threads(2)
    parent = PolicyValueNet(feature_version="huxi").eval()
    model = migrate(parent)
    x, a = fixture()
    row = (
        x,
        a,
        -0.1,
        False,
        [0] * 20,
        [0] * len(GROUP_PATTERNS),
        [0] * len(FAN_NAMES),
        [0.0] * 20,
        0.0,
        1.0,
        123,
    )
    tensors = batch([row], "cpu")
    loss = objective(model, parent, tensors)
    tensors[5] = torch.ones_like(tensors[5]) * 4
    tensors[6] = torch.ones_like(tensors[6]) * 7
    tensors[7] = torch.ones_like(tensors[7])
    assert torch.equal(loss, objective(model, parent, tensors))
    loss.backward()
    assert model.group_candidates.fusion.weight.grad.abs().sum() > 0
    assert model.final_groups.weight.grad.abs().sum() == 0


def test_future_acquisitions_count_offered_card_once():
    history = [
        Action(A.DRAW, 1, forced=True),
        Action(A.PAO, 0, 10, source_player=1, source_type=S.DRAW),
        Action(A.HU, 0, 10, source_player=1, source_type=S.DRAW),
    ]
    assert future_intakes(history, 0, 0, {0: 10})[10] == 0.25
    assert future_intakes(history, 0, 2, {0: 10})[10] == 0
    assert future_intakes(history, 1, 0, {0: 10})[10] == 0.25


def test_empty_wall_terminal_draw_is_not_a_future_card():
    history = [Action(A.DRAW, 0, forced=True)]
    assert future_intakes(history, 0, 0, {0: None}) == [0.0] * 20


def test_route_warmup_preserves_policy_after_learning_and_save(tmp_path):
    parent = PolicyValueNet(feature_version="huxi").eval()
    previous = migrate(parent)
    with torch.no_grad():
        previous.encoder[0].weight.add_(0.1)
        previous.final_fans.weight.add_(0.2)
    model = route_warmup(parent, previous)
    assert torch.equal(model.final_fans.weight, previous.final_fans.weight)
    verify_parent_policy(model, parent)
    x, a = fixture()
    row = (
        x,
        a,
        -0.1,
        False,
        [0] * 20,
        [0] * len(GROUP_PATTERNS),
        [0] * len(FAN_NAMES),
        [0.0] * 20,
        0.0,
        1.0,
        123,
    )
    inputs = batch([row], "cpu")
    old = [value.detach().clone() for value in parent(*inputs[:3])]
    weight = model.group_candidates.fusion.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    for _ in range(3):
        optimizer.zero_grad()
        objective(model, parent, inputs).backward()
        optimizer.step()
    verify_parent_policy(model, parent)
    assert not torch.equal(weight, model.group_candidates.fusion.weight)
    for left, right in zip(old, model(*inputs[:3]), strict=True):
        assert torch.equal(left, right)
    save_model(model, tmp_path / "warmup.pt")
    verify_parent_policy(load_model(tmp_path / "warmup.pt"), parent)


def test_terminal_targets_keep_duplicate_groups_and_multilabel_fan():
    groups = (Meld(MeldType.CHI, (10, 11, 12)),) * 2
    settlement = Settlement(0, 15, {"十一红": 3, "十八大": 5}, 80, (160, -80, -80), groups)
    tiles, counts, fans = terminal_route_targets(settlement)
    assert tiles[10:13] == [2, 2, 2]
    assert counts[GROUP_PATTERNS.index((10, 11, 12))] == 2
    assert fans[FAN_NAMES.index("十红系列")] == 1
    assert fans[FAN_NAMES.index("十八大")] == 1


def test_terminal_huxi_bounds_initialization_and_roundtrip(tmp_path):
    parent = PolicyValueNet(feature_version="huxi").eval()
    model = route_warmup(parent, version="trajectory_v3").eval()
    initialize_huxi_baseline(model, 6)
    x, a = fixture()
    x[22] = 36 / 60
    inputs = torch.tensor(x)[None], torch.tensor(a)[None], torch.ones((1, len(a)), dtype=torch.bool)
    _, forecast = model.forward_trajectory(*inputs)
    assert torch.allclose(forecast["huxi"] * 60, torch.tensor([42.0]))
    with torch.no_grad():
        model.final_huxi.bias.fill_(-100)
    assert (model.forward_trajectory(*inputs)[1]["huxi"] * 60 >= 36).all()
    model.terminal_huxi_constant.fill_(21)
    assert torch.allclose(model.forward_trajectory(*inputs)[1]["huxi"] * 60, torch.tensor([36.0]))
    inputs[0][:, 22] = 0
    assert torch.allclose(model.forward_trajectory(*inputs)[1]["huxi"] * 60, torch.tensor([21.0]))
    save_model(model, tmp_path / "bounded.pt")
    loaded = load_model(tmp_path / "bounded.pt")
    verify_parent_policy(loaded, parent)
    assert torch.equal(
        loaded.forward_trajectory(*inputs)[1]["huxi"], model.forward_trajectory(*inputs)[1]["huxi"]
    )


def test_huxi_baseline_uses_only_training_targets():
    x, a = fixture()
    x[22] = 0

    def row(value, seed):
        return (
            x,
            a,
            0.2,
            True,
            [0] * 20,
            [0] * len(GROUP_PATTERNS),
            [0] * len(FAN_NAMES),
            [0.0] * 20,
            value / 60,
            1.0,
            seed,
        )

    train = [row(18, 1), row(24, 2)]
    first = huxi_baselines(train, [row(21, 3)])
    second = huxi_baselines(train, [row(30, 3)])
    assert first["training_mean_huxi"] == second["training_mean_huxi"] == 21
    assert first["training_mean_extra_huxi"] == second["training_mean_extra_huxi"] == 6
    assert first["constant_mean_dev_mae"] == 0
    assert second["constant_mean_dev_mae"] == 9


def test_early_v3_checkpoint_defaults_optional_constant_guard_off(tmp_path):
    model = PolicyValueNet(feature_version="huxi", auxiliary_version="trajectory_v3")
    path = tmp_path / "early-v3.pt"
    save_model(model, path)
    saved = torch.load(path, weights_only=True)
    saved["state_dict"].pop("terminal_huxi_constant")
    saved["state_dict"].pop("positive_group_counts")
    torch.save(saved, path)
    loaded = load_model(path)
    assert loaded.terminal_huxi_constant.item() == 0
    assert not loaded.positive_group_counts.item()


def test_positive_group_prior_and_v3_continuation():
    parent = PolicyValueNet(feature_version="huxi").eval()
    model = route_warmup(parent, version="trajectory_v3")
    x, a = fixture()
    counts = [0] * len(GROUP_PATTERNS)
    counts[0] = 1
    row = (x, a, 0.2, True, [0] * 20, counts, [0] * len(FAN_NAMES), [0.0] * 20, 0.3, 1.0, 1)
    initialize_group_prior(model, [row])
    inputs = batch([row], "cpu")
    _, forecast = model.forward_trajectory(*inputs[:3])
    assert (forecast["groups"] >= 0).all()
    assert torch.allclose(forecast["groups"][0, 0], torch.tensor(1.0))
    assert torch.isfinite(objective(model, parent, inputs))
    continued = route_warmup(parent, model, version="trajectory_v3")
    assert continued.positive_group_counts.item()
    assert torch.equal(model.final_groups.bias, continued.final_groups.bias)
    verify_parent_policy(continued, parent)
