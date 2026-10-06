"""Audit model games independently from training and retain review frames."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import ActionType, RuleEngine

from .data import teacher_scores
from .runtime import choose, load_model


def audit(checkpoint, seeds, output):
    torch.set_num_threads(2)
    model = load_model(checkpoint)
    engine = RuleEngine()
    reviews = []
    steps = wins = 0
    for seed in seeds:
        initial = engine.new_game(seed, dealer=seed % 3)
        state = initial
        rng = random.Random(seed)
        for step in range(1, 1001):
            if state.terminal:
                break
            actions = engine.legal_actions(state)
            obs = engine.observation(state, actions[0].player)
            selected = choose(obs, rng, model)
            if len(actions) > 1:
                scores = teacher_scores(obs)
                chosen = actions.index(selected)
                best = int(np.argmax(scores))
                if scores[best] - scores[chosen] > 0.5 and len(reviews) < 12:
                    reviews.append(
                        {
                            "seed": seed,
                            "dealer": seed % 3,
                            "step": step,
                            "player": obs.player + 1,
                            "model_candidate": asdict(selected),
                            "teacher_candidate": asdict(actions[best]),
                            "model_action": selected.kind.value,
                            "model_tile": selected.tile,
                            "teacher_action": actions[best].kind.value,
                            "teacher_tile": actions[best].tile,
                            "heuristic_gap": float(scores[best] - scores[chosen]),
                            "hand": list(obs.hand),
                            "reason": "heuristic disagreement, not a rule violation",
                        }
                    )
            assert selected in actions
            if selected.kind == ActionType.CHI:
                assert selected.tile not in obs.passed_chi
                assert not any(
                    a.player == obs.player
                    and a.kind == ActionType.DISCARD
                    and a.tile == selected.tile
                    for a in state.history
                )
            state = engine.step(state, selected)
            state.validate()
            steps += 1
        else:
            raise RuntimeError("review exceeded step limit")
        assert engine.replay(initial, state.history).serialize() == state.serialize()
        wins += state.winner is not None
    report = {
        "checkpoint": checkpoint,
        "seeds": list(seeds),
        "games": len(seeds),
        "steps": steps,
        "winning_games": wins,
        "replay_verified": True,
        "conservation_verified": True,
        "reviews": reviews,
    }
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--start", type=int, default=9000)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.checkpoint, range(a.start, a.start + a.games), a.output)))
