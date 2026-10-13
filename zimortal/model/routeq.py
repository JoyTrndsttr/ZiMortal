"""Visible group/complete-plan encoders and action-conditioned terminal cash Q.

The inherited policy stays frozen. Route Q is experimental until independent
cash evaluation; predictions do not replace legality or certify reachability.
"""

from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from zimortal.engine import ActionType, Meld, MeldType, meld_huxi
from zimortal.engine.observation import Observation
from zimortal.engine.tiles import is_red

from .encoding import ACTION_DIM, encode_action, encode_observation
from .network import PolicyValueNet
from .trajectory import GROUP_PATTERNS, complete_group_plans, observation_group_plans

PLAN_FEATURES = 88
REQUIREMENTS = np.array([[Counter(p)[t] for t in range(20)] for p in GROUP_PATTERNS], np.float32)


def plan_arrays(obs, plans):
    """Visible per-plan inventory/context. Future labels are never features."""
    if not isinstance(obs, Observation):
        raise TypeError("plan inputs require Observation")
    own = Counter(obs.hand)
    locked = Counter(t for m in obs.players[obs.player].melds for t in m.tiles)
    available = np.array([own[t] + locked[t] for t in range(20)])
    external = Counter(obs.river)
    for seat, p in enumerate(obs.players):
        if seat != obs.player:
            external.update(t for m in p.melds for t in m.tiles)
    upper = np.array([4 - external[t] for t in range(20)])
    counts, contexts = [], []
    for plan in plans:
        groups = Counter(tuple(sorted(p)) for p in plan["groups"])
        count = np.array([groups[p] for p in GROUP_PATTERNS], np.float32)
        desired = count @ REQUIREMENTS
        missing = np.maximum(desired - available, 0)
        # Source-dependent triplet/quad huxi cannot be known in advance.
        # These are bounds conditional on the structural target, not EV.
        remaining = groups.copy()
        low = high = 0
        for meld in obs.players[obs.player].melds:
            pattern = tuple(sorted(meld.tiles))
            if not remaining[pattern]:
                quad = (pattern[0],) * 4
                if len(pattern) != 3 or len(set(pattern)) != 1 or not remaining[quad]:
                    raise ValueError("plan fails to retain an exposed group")
                remaining[quad] -= 1
                # A public peng promotes to pao; wei may promote to pao/ti.
                low += meld_huxi(Meld(MeldType.PAO, quad))
                high += meld_huxi(
                    Meld(MeldType.PAO if meld.kind == MeldType.PENG else MeldType.TI, quad)
                )
            else:
                remaining[pattern] -= 1
                low += meld_huxi(meld)
                high += meld_huxi(meld)
        for tile, n in own.items():
            if n == 3:
                pattern = (tile,) * 3
                if not remaining[pattern]:
                    quad = (tile,) * 4
                    if not remaining[quad]:
                        raise ValueError("plan splits a protected kan")
                    remaining[quad] -= 1
                    low += meld_huxi(Meld(MeldType.PAO, quad))
                    high += meld_huxi(Meld(MeldType.TI, quad))
                else:
                    remaining[pattern] -= 1
                    value = meld_huxi(Meld(MeldType.KAN, pattern))
                    low += value
                    high += value
        for pattern, n in remaining.items():
            if not n:
                continue
            if len(pattern) == 2:
                lo = hi = 0
            elif len(pattern) == 4:
                lo = meld_huxi(Meld(MeldType.PAO, pattern))
                hi = meld_huxi(Meld(MeldType.TI, pattern))
            elif len(set(pattern)) == 1:
                lo = meld_huxi(Meld(MeldType.PENG, pattern))
                hi = meld_huxi(Meld(MeldType.WEI, pattern))
            else:
                lo = hi = meld_huxi(Meld(MeldType.CHI, pattern))
            low += n * lo
            high += n * hi
        reds = sum(desired[t] for t in range(20) if is_red(t))
        big = desired[10:].sum()
        sizes = REQUIREMENTS.sum(-1)
        scalars = [
            low / 60,
            high / 60,
            reds / 24,
            big / 24,
            float(all(len(set(p)) == 1 for p in groups)),
            float(big >= 18),
            float(desired.sum() - big >= 18),
            float(high >= 15),
        ]
        if count.sum() != 7 or np.any(desired > upper) or np.any(desired > 4):
            raise ValueError("invalid complete plan inventory")
        if count[sizes == 2].sum() != int(count[sizes == 4].sum() > 0):
            raise ValueError("invalid pair/quad structure")
        counts.append(count)
        contexts.append(
            np.concatenate(
                (
                    desired / 4,
                    missing / 4,
                    np.minimum(desired, available) / 4,
                    np.maximum(upper - available, 0) / 4,
                    scalars,
                )
            )
        )
    return np.asarray(counts, np.float32), np.asarray(contexts, np.float32)


def upgrade_plan_branches(obs, scores, *, beam_size=32):
    """First-upgrade alternatives from visible locked triples; no hidden inputs.

    Existing quads are fixed. Each branch upgrades at most one further locked
    triple; equivalent inventory targets deduplicate in the proposal layer.
    Additional upgrades are budgeted only if the existing locked huxi is <15.
    Source/arrival alternatives remain bounded events, not acquisition proofs.
    """
    own = Counter(obs.hand)
    melds = obs.players[obs.player].melds
    fixed = [tuple(sorted(m.tiles)) for m in melds]
    fixed += [(t,) * 3 for t, n in own.items() if n == 3]
    external = Counter(obs.river)
    for seat, player in enumerate(obs.players):
        if seat != obs.player:
            external.update(t for m in player.melds for t in m.tiles)
    existing = sum(len(p) == 4 for p in fixed)
    locked_huxi = sum(meld_huxi(m) for m in melds) + sum(
        meld_huxi(Meld(MeldType.KAN, (t,) * 3)) for t, n in own.items() if n == 3
    )
    if existing and locked_huxi >= 15:
        return []  # forced upgrades still execute in the rule engine
    candidates = []
    for i, group in enumerate(fixed):
        if len(group) != 3 or len(set(group)) != 1 or external[group[0]]:
            continue
        promoted = fixed.copy()
        promoted[i] = (group[0],) * 4
        result = complete_group_plans(
            scores,
            [4 - external[t] for t in range(20)],
            promoted,
            beam_size=beam_size,
            top_k=8,
            max_quads=existing + 1,
        )
        for plan in result["plans"]:
            if sum(len(p) == 4 for p in plan["groups"]) != existing + 1:
                continue
            plan["upgrade_path"] = {
                "stage": "first_upgrade" if not existing else "extra_upgrade_huxi_exception",
                "tile": group[0],
                "existing_quads": existing,
                "arrival": "public fourth -> pao; own draw kan/wei -> ti; peng -> pao",
                "status": "conditional structural target; timing/reachability unknown",
            }
            candidates.append(plan)
            break
    return candidates


def propose_routes(obs, prior, *, device="cpu", count=4, beam_size=32):
    """Diversified visible structural branches, never future settlement targets."""
    x = encode_observation(obs, "huxi")
    actions = np.stack([encode_action(a, obs.player) for a in obs.legal_actions])
    with torch.no_grad():
        _, forecast = prior.forward_trajectory(
            torch.as_tensor(x[None], device=device),
            torch.as_tensor(actions[None], device=device),
            torch.ones((1, len(actions)), dtype=torch.bool, device=device),
        )
        presence = forecast["presence"][0].cpu().numpy()
    own = Counter(obs.hand)
    owned = np.array([own[t] for t in range(20)], np.float32)
    covered = np.minimum(REQUIREMENTS, owned).sum(-1)
    missing = (REQUIREMENTS - owned).clip(0).sum(-1)
    huxi = []
    for p in GROUP_PATTERNS:
        kind = (
            MeldType.PAIR
            if len(p) == 2
            else MeldType.TI
            if len(p) == 4
            else MeldType.KAN
            if len(set(p)) == 1
            else MeldType.CHI
        )
        huxi.append(meld_huxi(Meld(kind, p)))
    big = REQUIREMENTS[:, 10:].sum(-1)
    red = REQUIREMENTS[:, [t for t in range(20) if is_red(t)]].sum(-1)
    base = presence * 0.25 + covered * 2 - missing * 2
    scorers = [
        base,
        base + np.asarray(huxi) * 0.3,
        base + big * 0.5,
        base + red * 0.5,
        base - red * 0.5,
    ]
    claims = []
    for action in obs.legal_actions:
        if action.kind == ActionType.CHI:
            claims.append((action.chi, *action.bi))
        elif action.kind == ActionType.PENG:
            claims.append(((action.tile,) * 3,))
    proposals = [(scorers[1], claim) for claim in claims[:2]]
    proposals += [(scorers[0], ())]
    proposals += [(score, ()) for score in scorers[1:]]
    candidates, seen = [], set()
    # Reserve a first-upgrade slot before ordinary terminal targets fill budget.
    upgraded = upgrade_plan_branches(obs, scorers[1], beam_size=beam_size)
    for plan in upgraded:
        _, context = plan_arrays(obs, [plan])
        if context[0, -1] and len(candidates) < max(1, count // 2):
            key = tuple(sorted(tuple(p) for p in plan["groups"]))
            if key not in seen:
                candidates.append(plan)
                seen.add(key)
    existing_quads = sum(len(m.tiles) == 4 for m in obs.players[obs.player].melds)
    for scores, required_claim in proposals:
        # A near-hand beam may contain only low-huxi structures. Try a bounded
        # huxi-biased proposal rather than reporting those as winning routes.
        for proposal_score in (scores, base + np.asarray(huxi) * 1.2):
            result = observation_group_plans(
                obs,
                proposal_score.tolist(),
                extra_fixed=required_claim,
                max_quads=existing_quads + 1,
                beam_size=beam_size,
                top_k=8,
            )
            added = False
            for plan in result["plans"]:
                key = tuple(sorted(tuple(p) for p in plan["groups"]))
                if key in seen:
                    continue
                if sum(len(p) == 4 for p in plan["groups"]) > existing_quads + 1:
                    continue
                plan["upgrade_path"] = {
                    "stage": "existing_upgrade"
                    if existing_quads
                    else "no_upgrade"
                    if not any(len(p) == 4 for p in plan["groups"])
                    else "first_upgrade_unlocked",
                    "existing_quads": existing_quads,
                    "status": "structural target; timing/reachability unknown",
                }
                _, context = plan_arrays(obs, [plan])
                if context[0, -1] == 0:
                    continue  # structural huxi upper bound is insufficient
                seen.add(key)
                candidates.append(plan)
                added = True
                break
            if added:
                break
        if len(candidates) >= count:
            break
    return candidates


class RouteQNet(nn.Module):
    architecture = "routeq"
    feature_version = "huxi"
    auxiliary_version = "routeq_v1"

    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.parent = PolicyValueNet("resnet", width, "huxi", "trajectory_v4")
        self.parent.requires_grad_(False)
        # Reuse trained local group embeddings, now kept separately per group.
        self.menzi = deepcopy(self.parent.group_candidates)
        self.menzi.requires_grad_(True)
        self.menzi.attention.requires_grad_(False)
        self.menzi.fusion.requires_grad_(False)
        self.plan_encoder = nn.Sequential(
            nn.Linear(64 + PLAN_FEATURES, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU()
        )
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, 64), nn.SiLU(), nn.Linear(64, 64))
        self.plan_value = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
        self.q_head = nn.Sequential(nn.Linear(320, 128), nn.SiLU(), nn.Linear(128, 1))
        for head in (self.plan_value, self.q_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @classmethod
    def from_parent(cls, parent):
        if parent.architecture != "resnet" or parent.auxiliary_version != "trajectory_v4":
            raise ValueError("route Q requires the trained v4 ResNet prior")
        model = cls(parent.width)
        model.parent.load_state_dict(parent.state_dict())
        model.menzi.load_state_dict(parent.group_candidates.state_dict())
        return model

    def forward_routes(self, features, actions, mask, plans, contexts, plan_mask):
        with torch.no_grad():
            base = self.parent.encoder(features)
            original = self.parent(features, actions, mask)
        groups = self.menzi.encode_groups(features)
        pooled = torch.einsum("brg,bgd->brd", plans, groups) / plans.sum(-1).clamp_min(1)[..., None]
        route = self.plan_encoder(torch.cat((pooled, contexts), -1))
        combined = torch.cat((base[:, None].expand(-1, route.shape[1], -1), route), -1)
        value = self.plan_value(combined).squeeze(-1) + original[1][:, None]
        action = self.action_encoder(actions)
        context = torch.cat(
            (
                combined[:, None].expand(-1, actions.shape[1], -1, -1),
                action[:, :, None].expand(-1, -1, route.shape[1], -1),
            ),
            -1,
        )
        q = self.q_head(context).squeeze(-1) + original[1][:, None, None]
        valid = mask[:, :, None] & plan_mask[:, None]
        return q.masked_fill(~valid, -1e9), value.masked_fill(~plan_mask, -1e9), original

    def forward(self, features, actions, mask):
        # Route inference is explicit; existing web/selfplay keeps the strong
        # policy until an independent cash gate validates route Q.
        return self.parent(features, actions, mask)
