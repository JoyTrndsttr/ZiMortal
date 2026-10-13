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
