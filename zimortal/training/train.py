"""Reproducible supervised training and dataset aggregation CLI."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zimortal.engine import RuleEngine
from zimortal.model.network import PolicyValueNet

from .data import example, generate_dataset
from .runtime import choose, load_model, save_model, tournament


def batch(rows):
    n = max(len(r[1]) for r in rows)
    actions = np.zeros((len(rows), n, rows[0][1].shape[-1]), np.float32)
    mask = np.zeros((len(rows), n), bool)
    for i, r in enumerate(rows):
        actions[i, : len(r[1])] = r[1]
        mask[i, : len(r[1])] = True
    return (
        torch.tensor(np.stack([r[0] for r in rows])),
        torch.tensor(actions),
        torch.tensor(mask),
        torch.tensor([r[2] for r in rows]),
        torch.tensor([r[3] for r in rows], dtype=torch.float32),
        torch.tensor(np.stack([r[4] for r in rows])),
    )


def aggregate(model, games, seed):
    engine = RuleEngine()
    rng = random.Random(seed)
    data = []
    outcomes = []
    for i in range(games):
        initial = engine.new_game(seed + i)
        state = initial
        rows = []
        for _ in range(1000):
            if state.terminal:
                break
            actions = engine.legal_actions(state)
            actor = actions[0].player
            obs = engine.observation(state, actor)
            if len(actions) > 1:
                rows.append((obs, actor))
            state = engine.step(state, choose(obs, rng, model))
            state.validate()
        else:
            raise RuntimeError("aggregation exceeded terminal limit")
        if engine.replay(initial, state.history).serialize() != state.serialize():
            raise RuntimeError("aggregation replay mismatch")
        # Readiness labels remain consistent with round-one pretraining;
        # outcome value learning is a separate explicit RL stage.
        data.extend(example(obs) for obs, _ in rows)
        outcomes.append(state.winner)
    return data, {"games": games, "winning_games": sum(w is not None for w in outcomes)}


def evaluate(model, data):
    correct = total = 0
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), 128):
            x, a, m, y, _v, _w = batch(data[start : start + 128])
            logits, _, _ = model(x, a, m)
            correct += int((logits.argmax(-1) == y).sum())
            total += len(y)
            loss += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
    return {
        "policy_accuracy": correct / total,
        "policy_cross_entropy": loss / total,
        "samples": total,
    }


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train, strata = generate_dataset(args.puzzles, args.games, args.seed)
    valid, _ = generate_dataset(max(128, args.puzzles // 5), 5, args.seed + 100)
    print(json.dumps({"dataset": len(train), "valid": len(valid), "strata": strata}), flush=True)
    model = load_model(args.resume) if args.resume else PolicyValueNet(args.architecture)
    aggregated = None
    if args.aggregate:
        extra, aggregated = aggregate(model, args.aggregate, args.seed * 100000 + 50000)
        train += extra
        print(json.dumps({"aggregated": len(extra), "outcomes": aggregated}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    logs = []
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train)
        losses = []
        for start in range(0, len(train), 128):
            x, a, m, y, v, w = batch(train[start : start + 128])
            logits, value, wait = model(x, a, m)
            loss = (
                nn.functional.cross_entropy(logits, y)
                + 0.2 * nn.functional.binary_cross_entropy_with_logits(value, v)
                + 0.1 * nn.functional.binary_cross_entropy_with_logits(wait, w)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        log = {"epoch": epoch + 1, "loss": sum(losses) / len(losses), **evaluate(model, valid)}
        logs.append(log)
        print(json.dumps(log), flush=True)
    save_model(
        model,
        args.output,
        seed=args.seed,
        value_target="immediate hu or own-draw readiness",
        training_samples=len(train),
    )
    report = {
        "seed": args.seed,
        "architecture": model.architecture,
        "training_samples": len(train),
        "validation_namespace": args.seed + 100,
        "strata": strata,
        "aggregation": aggregated,
        "epochs": logs,
        "checkpoint": args.output,
        "sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "device": "cpu",
        "torch": torch.__version__,
        "random_tournament": tournament(model, list(range(7000, 7000 + args.eval_seeds))),
        "teacher_tournament": tournament(
            model, list(range(8000, 8000 + max(3, args.eval_seeds // 3))), opponent="teacher"
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["random_tournament"]), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", choices=["mlp", "resnet"], default="resnet")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--puzzles", type=int, default=1024)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--resume")
    p.add_argument("--aggregate", type=int, default=0)
    p.add_argument("--eval-seeds", type=int, default=20)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
