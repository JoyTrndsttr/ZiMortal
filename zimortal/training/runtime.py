"""Model policies and seat-rotated reproducible tournaments."""

import os
import random
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
        },
        temporary,
    )
    with temporary.open("rb") as saved:
        os.fsync(saved.fileno())
    os.replace(temporary, path)


def load_model(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    if data["architecture"] == "cashq":
        from zimortal.model.cashq import CashQNet

        model = CashQNet(data["width"]).to(device)
    else:
        model = PolicyValueNet(
            data["architecture"],
            data["width"],
            data.get("feature_version", "legacy"),
            data.get("auxiliary_version", "legacy"),
        ).to(device)
    model.load_state_dict(data["state_dict"])
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


def choose_batch(model, observations, device="cpu"):
    """Batch independent visible decisions without exposing hidden worlds."""
    if not observations:
        return []
    x = torch.from_numpy(
        np.stack([encode_observation(obs, model.feature_version) for obs in observations])
    ).to(device)
    size = max(len(obs.legal_actions) for obs in observations)
    encoded = [
        np.stack([encode_action(a, obs.player) for a in obs.legal_actions]) for obs in observations
    ]
    a = torch.zeros(len(observations), size, encoded[0].shape[-1], device=device)
    mask = torch.zeros(len(observations), size, dtype=torch.bool, device=device)
    for i, row in enumerate(encoded):
        a[i, : len(row)] = torch.from_numpy(row).to(device)
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
