"""Model policies and seat-rotated reproducible tournaments."""

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import ActionType as A
from zimortal.engine import RuleEngine
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.model.network import PolicyValueNet

from .data import teacher_scores


def save_model(model, path, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": model.architecture,
            "width": model.width,
            "feature_version": model.feature_version,
            "auxiliary_version": model.auxiliary_version,
            "metadata": metadata,
            "model_config": {
                k: getattr(model, k) for k in ("latent", "blocks") if hasattr(model, k)
            },
        },
        temporary,
    )
    with temporary.open("rb") as saved:
        os.fsync(saved.fileno())
    os.replace(temporary, path)


def load_model(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    if data["architecture"] == "mortal_q":
        from zimortal.model.mortal_q import MortalQNet

        model = MortalQNet(data["width"], **data.get("model_config", {})).to(device)
    elif data["architecture"] == "routeq":
        from zimortal.model.routeq import RouteQNet

        model = RouteQNet(data["width"]).to(device)
    elif data["architecture"] == "cashq":
        from zimortal.model.cashq import CashQNet

        model = CashQNet(data["width"]).to(device)
    else:
        model = PolicyValueNet(
            data["architecture"],
            data["width"],
            data.get("feature_version", "legacy"),
            data.get("auxiliary_version", "legacy"),
        ).to(device)
    state = dict(data["state_dict"])
    if data.get("auxiliary_version") == "trajectory_v3":
        # The optional constant guard defaults off for early v3 pilot checkpoints.
        state.setdefault("terminal_huxi_constant", model.terminal_huxi_constant)
        state.setdefault("positive_group_counts", model.positive_group_counts)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, obs, device="cpu"):
    features = (
        torch.from_numpy(encode_observation(obs, model.feature_version)).unsqueeze(0).to(device)
    )
    actions = (
        torch.from_numpy(np.stack([encode_action(a, obs.player) for a in obs.legal_actions]))
        .unsqueeze(0)
        .to(device)
    )
    mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value, _ = model(features, actions, mask)
    return logits[0], value[0]


@lru_cache(maxsize=2048)
def _visible_features(obs, version):
    features = encode_observation(obs, version)
    features.flags.writeable = False
    return features


@lru_cache(maxsize=4096)
def _action_features(actions, player):
    features = np.stack([encode_action(a, player) for a in actions])
    features.flags.writeable = False
    return features


def choose_batch(model, observations, device="cpu"):
    """Batch independent visible decisions without exposing hidden worlds."""
    if not observations:
        return []
    x = torch.from_numpy(
        np.stack([_visible_features(obs, model.feature_version) for obs in observations])
    ).to(device)
    size = max(len(obs.legal_actions) for obs in observations)
    encoded = [_action_features(obs.legal_actions, obs.player) for obs in observations]
    a = torch.zeros(len(observations), size, encoded[0].shape[-1], device=device)
    mask = torch.zeros(len(observations), size, dtype=torch.bool, device=device)
    for i, row in enumerate(encoded):
        a[i, : len(row)] = torch.from_numpy(row.copy()).to(device)
        mask[i, : len(row)] = True
    with torch.inference_mode():
        scores, *_ = model(x, a, mask)
        choices = scores.argmax(-1).cpu().tolist()
    return [obs.legal_actions[k] for obs, k in zip(observations, choices, strict=True)]


def choose(obs, rng, model=None, policy="model", device="cpu"):
    if len(obs.legal_actions) == 1:
        return obs.legal_actions[0]
    if policy == "random":
        return rng.choice(obs.legal_actions)
    if policy == "teacher":
        return obs.legal_actions[int(np.argmax(teacher_scores(obs)))]
    logits, _ = predict(model, obs, device)
    return obs.legal_actions[int(logits.argmax())]


def tournament(
    model,
    seeds,
    opponent="random",
    device="cpu",
    model_seats=None,
    model_policy="model",
    opponent_model=None,
    include_games=False,
):
    engine = RuleEngine()
    wins = draws = illegal = hu_pass = hu_opportunities = 0
    payoff = 0
    winning_huxi = winning_fan = winning_amount = 0
    reviews = []
    game_results = []
    cashq_decisions = cashq_changes = 0
    for seed in seeds:
        for seat in range(3) if model_seats is None else model_seats:
            state = engine.new_game(seed, dealer=seed % 3)
            rng = random.Random(seed * 3 + seat)
            for step in range(1, 1001):
                if state.terminal:
                    break
                actions = engine.legal_actions(state)
                actor = actions[0].player
                obs = engine.observation(state, actor)
                selected = choose(
                    obs,
                    rng,
                    model if actor == seat else opponent_model,
                    policy=model_policy if actor == seat else opponent,
                    device=device,
                )
                if (
                    actor == seat
                    and model_policy == "model"
                    and getattr(model, "architecture", None) == "cashq"
                    and len(actions) > 1
                ):
                    reference = choose(obs, rng, model.parent, device=device)
                    cashq_decisions += 1
                    cashq_changes += selected != reference
                if actor == seat and any(a.kind == A.HU for a in actions):
                    hu_opportunities += 1
                    hu_pass += selected.kind != A.HU
                    if selected.kind != A.HU and len(reviews) < 6:
                        reviews.append(
                            {
                                "seed": seed,
                                "dealer": seed % 3,
                                "seat": seat + 1,
                                "step": step,
                                "issue": "model_passed_hu",
                            }
                        )
                if selected not in actions:
                    illegal += 1
                    raise RuntimeError("illegal model action")
                state = engine.step(state, selected)
                state.validate()
            else:
                raise RuntimeError("nonterminal model game")
            wins += state.winner == seat
            draws += state.winner is None
            if include_games:
                game_results.append(
                    {
                        "seed": seed,
                        "seat": seat,
                        "payoff": state.settlement.payments[seat] if state.settlement else 0,
                        "winner": state.winner,
                    }
                )
            if state.settlement:
                payoff += state.settlement.payments[seat]
                if state.winner == seat:
                    winning_huxi += state.settlement.huxi
                    winning_fan += sum(state.settlement.fan.values()) or 1
                    winning_amount += state.settlement.amount_each
    games = len(seeds) * (3 if model_seats is None else len(model_seats))
    result = {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "draws": draws,
        "mean_payoff": payoff / games,
        "mean_winning_huxi": winning_huxi / wins if wins else 0,
        "mean_winning_fan": winning_fan / wins if wins else 0,
        "mean_winning_amount_each": winning_amount / wins if wins else 0,
        "illegal_actions": illegal,
        "hu_opportunities": hu_opportunities,
        "passed_hu": hu_pass,
        "reviews": reviews,
    }
    if include_games:
        result["game_results"] = game_results
    if getattr(model, "architecture", None) == "cashq":
        result["cashq_decisions"] = cashq_decisions
        result["cashq_changes"] = cashq_changes
    return result


def predict_group_plans(model, obs, device="cpu", **budget):
    """Expose v4 structural forecasts without changing the frozen action policy."""
    from zimortal.model.trajectory import observation_group_plans

    if model.auxiliary_version != "trajectory_v4":
        raise ValueError("nonempty-group forecasts require trajectory_v4")
    features = torch.from_numpy(encode_observation(obs, model.feature_version))[None].to(device)
    actions = torch.from_numpy(np.stack([encode_action(a, obs.player) for a in obs.legal_actions]))[
        None
    ].to(device)
    mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        _, forecasts = model.forward_trajectory(features, actions, mask)
        scores = torch.nn.functional.logsigmoid(forecasts["presence"][0])
    return observation_group_plans(obs, scores.cpu().tolist(), **budget)


def predict_route_q(model, obs, device="cpu", *, count=4, beam_size=32):
    """Explicit route/action analysis; existing action policy remains unchanged."""
    from zimortal.model.routeq import plan_arrays, propose_routes

    if model.architecture != "routeq":
        raise ValueError("route comparison requires RouteQNet")
    plans = propose_routes(obs, model.parent, device=device, count=count, beam_size=beam_size)
    if len(plans) < 2:
        return {"status": "unknown", "plans": plans}
    counts, contexts = plan_arrays(obs, plans)
    x = torch.as_tensor(encode_observation(obs, "huxi")[None], device=device)
    a = torch.as_tensor(
        np.stack([encode_action(v, obs.player) for v in obs.legal_actions])[None], device=device
    )
    with torch.inference_mode():
        q, value, original = model.forward_routes(
            x,
            a,
            torch.ones(a.shape[:2], dtype=torch.bool, device=device),
            torch.as_tensor(counts[None], device=device),
            torch.as_tensor(contexts[None], device=device),
            torch.ones((1, len(plans)), dtype=torch.bool, device=device),
        )
    return {
        "status": "experimental_prediction",
        "plans": plans,
        "action_route_q_cash": (q[0] * 100).cpu().tolist(),
        "route_value_cash": (value[0] * 100).cpu().tolist(),
        "parent_action": int(original[0][0].argmax()),
        "meaning": "predicted terminal cash under frozen route continuation; not optimal-route EV",
    }
