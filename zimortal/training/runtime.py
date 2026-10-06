"""Model policies and seat-rotated reproducible tournaments."""

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
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": model.architecture,
            "width": model.width,
            "metadata": metadata,
        },
        path,
    )


def load_model(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    model = PolicyValueNet(data["architecture"], data["width"]).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model


def predict(model, obs, device="cpu"):
    features = torch.from_numpy(encode_observation(obs)).unsqueeze(0).to(device)
    actions = (
        torch.from_numpy(np.stack([encode_action(a, obs.player) for a in obs.legal_actions]))
        .unsqueeze(0)
        .to(device)
    )
    mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value, _ = model(features, actions, mask)
    return logits[0], value[0]


def choose(obs, rng, model=None, policy="model", device="cpu"):
    if len(obs.legal_actions) == 1:
        return obs.legal_actions[0]
    if policy == "random":
        return rng.choice(obs.legal_actions)
    if policy == "teacher":
        return obs.legal_actions[int(np.argmax(teacher_scores(obs)))]
    logits, _ = predict(model, obs, device)
    return obs.legal_actions[int(logits.argmax())]


def tournament(model, seeds, opponent="random", device="cpu", model_seats=None):
    engine = RuleEngine()
    wins = draws = illegal = hu_pass = hu_opportunities = 0
    payoff = 0
    reviews = []
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
                    obs, rng, model, policy="model" if actor == seat else opponent, device=device
                )
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
            if state.settlement:
                payoff += state.settlement.payments[seat]
    games = len(seeds) * (3 if model_seats is None else len(model_seats))
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "draws": draws,
        "mean_payoff": payoff / games,
        "illegal_actions": illegal,
        "hu_opportunities": hu_opportunities,
        "passed_hu": hu_pass,
        "reviews": reviews,
    }
