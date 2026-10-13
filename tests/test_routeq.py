from collections import Counter

import numpy as np
import pytest
import torch

from zimortal.engine import RuleEngine
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.model.network import PolicyValueNet
from zimortal.model.routeq import RouteQNet, plan_arrays, propose_routes
from zimortal.training.routeq import compare_routes, comparison_loss, root_row, route_choices
from zimortal.training.runtime import load_model, save_model


def fixture():
    torch.manual_seed(2)
    parent = PolicyValueNet(feature_version="huxi", auxiliary_version="trajectory_v4").eval()
    engine = RuleEngine()
    state = engine.new_game(118, dealer=1)
    actions = engine.legal_actions(state)
    obs = engine.observation(state, actions[0].player)
    plans = propose_routes(obs, parent, beam_size=16, count=3)
    assert len(plans) >= 2
    return parent, state, obs, plans


def test_route_structure_and_context_are_visible_and_source_bounded():
    parent, _state, obs, plans = fixture()
    groups, context = plan_arrays(obs, plans)
    assert context.shape == (len(plans), 88)
    assert np.all(groups.sum(-1) == 7)
    for plan in plans:
        counts = Counter(t for p in plan["groups"] for t in p)
        assert max(counts.values()) <= 4
    assert np.all(context[:, -7] >= context[:, -8])  # high >= low huxi
    with pytest.raises(TypeError):
        plan_arrays(_state, plans)
    # Repeating proposal on the same public state cannot use another hidden deal.
    assert plans == propose_routes(obs, parent, beam_size=16, count=3)


def test_routeq_freezes_prior_trains_route_delta_and_roundtrips(tmp_path):
    parent, _state, obs, plans = fixture()
    model = RouteQNet.from_parent(parent)
    outcomes = np.zeros((3, len(obs.legal_actions), len(plans) + 1), np.float32)
    outcomes[:, :, 0] = 1
    outcomes[:, :, 1] = -1
    row = root_row(obs, plans, outcomes, parent, "cpu")
    loss = comparison_loss(model, row, "cpu")
    assert torch.isfinite(loss)
    loss.backward()
    assert model.q_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parent.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(6):
        optimizer.zero_grad()
        comparison_loss(model, row, "cpu").backward()
        optimizer.step()
    from zimortal.training.routeq import model_root

    q, _value, _ = model_root(model, row, "cpu")
    assert (q[0, :, 0] - q[0, :, 1]).mean() > 0
    assert model.menzi.group[0].weight.grad.abs().sum() > 0
    x, a = torch.tensor(row["features"])[None], torch.tensor(row["actions"])[None]
    mask = torch.ones(a.shape[:2], dtype=torch.bool)
    assert torch.equal(model(x, a, mask)[0], parent(x, a, mask)[0])
    save_model(model, tmp_path / "route.pt")
    loaded = load_model(tmp_path / "route.pt")
    assert torch.equal(loaded(x, a, mask)[0], parent(x, a, mask)[0])
    assert torch.equal(model.q_head[-1].weight, loaded.q_head[-1].weight)


def test_shared_particles_replay_all_routes_and_zero_bias_recovers_parent():
    parent, state, obs, plans = fixture()
    engine = RuleEngine()

    class Sampler:
        def __init__(self):
            self.calls = []

        def sample(self, seed, _max_attempts):
            self.calls.append(seed)
            return state.clone(), 1

    sampler = Sampler()
    before = state.serialize()
    outcomes, evidence = compare_routes(
        obs, plans, parent, particles=2, seed=61, sampler=sampler, strength=0
    )
    assert sampler.calls == [61, 61 + 104729]
    assert state.serialize() == before
    assert outcomes.shape == (2, len(obs.legal_actions), len(plans) + 1)
    assert np.array_equal(outcomes[:, :, 0], outcomes[:, :, -1])
    for particle in evidence["raw_particles"]:
        assert len(particle["branches"]) == len(obs.legal_actions) * (len(plans) + 1)
        for branch in particle["branches"]:
            settlement = branch["settlement"]
            if settlement:
                assert sum(settlement["payments"]) == 0
                assert branch["cash"] == settlement["payments"][obs.player] / 100
    choices, _ = route_choices(parent, [obs, obs], [plans[0], plans[0]])
    assert choices[0] == choices[1]
    assert choices[0] in engine.legal_actions(state)


def test_padded_routes_and_actions_are_masked():
    parent, _state, obs, plans = fixture()
    model = RouteQNet.from_parent(parent)
    p, c = plan_arrays(obs, plans)
    x = torch.tensor(encode_observation(obs, "huxi"))[None]
    a = torch.tensor(np.stack([encode_action(v, obs.player) for v in obs.legal_actions]))[None]
    mask = torch.ones(a.shape[:2], dtype=torch.bool)
    mask[:, -1] = False
    route_mask = torch.ones((1, len(plans)), dtype=torch.bool)
    route_mask[:, -1] = False
    q, value, _ = model.forward_routes(
        x, a, mask, torch.tensor(p)[None], torch.tensor(c)[None], route_mask
    )
    assert (q[:, -1] == -1e9).all()
    assert (q[:, :, -1] == -1e9).all()
    assert value[0, -1] == -1e9


def test_claim_conditioned_routes_include_legal_chi_and_required_bi():
    import random

    from zimortal.engine import ActionType as A

    parent = PolicyValueNet(feature_version="huxi", auxiliary_version="trajectory_v4")
    engine = RuleEngine()
    state = engine.new_game(118, dealer=1)
    rng = random.Random(118)
    for _ in range(1000):
        actions = engine.legal_actions(state)
        if any(a.kind == A.CHI for a in actions):
            obs = engine.observation(state, actions[0].player)
            plans = propose_routes(obs, parent, beam_size=32, count=4)
            required = [
                Counter(tuple(sorted(p)) for p in (a.chi, *a.bi))
                for a in actions
                if a.kind == A.CHI
            ]
            assert any(
                all(Counter(tuple(p) for p in plan["groups"])[p] >= n for p, n in claim.items())
                for plan in plans
                for claim in required
            )
            return
        state = engine.step(state, rng.choice(actions))
        if state.terminal:
            break
    raise AssertionError("fixture did not expose a legal chi")


def test_locked_triples_can_upgrade_without_splitting_or_double_huxi():
    from zimortal.model.routeq import upgrade_plan_branches
    from zimortal.model.trajectory import GROUP_PATTERNS

    _parent, _state, obs, _ = fixture()
    branches = upgrade_plan_branches(obs, [0.0] * len(GROUP_PATTERNS), beam_size=64)
    assert branches
    for plan in branches:
        tile = plan["upgrade_path"]["tile"]
        assert obs.hand.count(tile) == 3 or any(
            m.tiles == (tile,) * 3 for m in obs.players[obs.player].melds
        )
        assert (tile,) * 4 in [tuple(p) for p in plan["groups"]]
        assert sum(len(p) == 2 for p in plan["groups"]) == 1
        _, context = plan_arrays(obs, [plan])
        assert context[0, -7] >= context[0, -8]
        assert context[0, -8] * 60 >= (9 if tile >= 10 else 6)
        from zimortal.training.routeq import route_potential

        assert route_potential(obs, plan) is not None


def test_dead_fourth_tile_does_not_create_locked_upgrade_branch():
    from dataclasses import replace

    from zimortal.model.routeq import upgrade_plan_branches
    from zimortal.model.trajectory import GROUP_PATTERNS

    _, _, obs, _ = fixture()
    tiles = [t for t in set(obs.hand) if obs.hand.count(t) == 3]
    assert tiles
    dead = replace(obs, river=obs.river + tuple(tiles))
    branches = upgrade_plan_branches(dead, [0.0] * len(GROUP_PATTERNS), beam_size=64)
    assert all(p["upgrade_path"]["tile"] not in tiles for p in branches)


def test_claim_preview_matches_actual_mandatory_discard_route_progress():
    import random

    from zimortal.engine import ActionType as A
    from zimortal.training.routeq import route_potential

    parent, state, _, _ = fixture()
    engine = RuleEngine()
    rng = random.Random(18)
    for _ in range(1000):
        legal = engine.legal_actions(state)
        claims = [a for a in legal if a.kind in (A.CHI, A.PENG)]
        if claims:
            obs = engine.observation(state, legal[0].player)
            plans = propose_routes(obs, parent, count=4)
            checked = 0
            for claim in claims:
                child = engine.step(state, claim)
                discards = engine.legal_actions(child)
                if not discards or any(a.kind != A.DISCARD for a in discards):
                    continue
                after = engine.observation(child, claim.player)
                for plan in plans:
                    values = [route_potential(after, plan, a) for a in discards]
                    valid = [v for v in values if v is not None]
                    expected = max(valid) if valid else None
                    assert route_potential(obs, plan, claim) == expected
                    checked += 1
            assert checked
            break
        state = engine.step(state, rng.choice(legal))
    else:
        raise AssertionError("no claim decision found")
