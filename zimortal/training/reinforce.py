"""Clipped actor-critic self-play with a frozen opponent and imitation anchor."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zimortal.engine import RuleEngine
from zimortal.model.encoding import encode_action, encode_observation

from .data import example, make_puzzle
from .runtime import choose, load_model, predict, save_model, tournament
from .train import batch


def terminal_reward(settlement, seat, mode):
    if settlement is None:
        return 0.0
    if mode == "settlement":
        return settlement.payments[seat] / 100
    if mode == "utility":
        return 1.0 if settlement.winner == seat else -0.5
    raise ValueError("unknown reward mode")


def collect(model, opponent, games, seed, reward_mode="utility", opponent_mode="frozen"):
    engine = RuleEngine()
    rng = random.Random(seed)
    rows = []
    wins = draws = 0
    payoff = 0
    model.eval()
    for i in range(games):
        initial = engine.new_game(seed + i, dealer=(i // 3) % 3)
        state = initial
        seat = i % 3
        trajectory = []
        generator = torch.Generator().manual_seed(seed + i)
        for step in range(1000):
            if state.terminal:
                break
            actions = engine.legal_actions(state)
            actor = actions[0].player
            obs = engine.observation(state, actor)
            if actor == seat and len(actions) > 1:
                logits, value = predict(model, obs)
                probs = logits.softmax(-1)
                index = int(torch.multinomial(probs, 1, generator=generator))
                # Only observation and candidate actions enter the model.
                row = (
                    encode_observation(obs, model.feature_version),
                    np.stack([encode_action(a, actor) for a in actions]),
                    index,
                    0.0,
                    np.zeros(20, dtype=np.float32),
                )
                if model.feature_version == "huxi":
                    from .huxi import auxiliary

                    structural, hu_target = auxiliary(obs)
                    row = row[:4] + (structural, hu_target)
                trajectory.append((row, float(probs[index].log()), float(value), step))
                action = actions[index]
            else:
                policy = (
                    "random"
                    if i % 4 == 0
                    else ("teacher" if opponent_mode == "mixed" and i % 4 == 1 else "model")
                )
                action = choose(obs, rng, opponent, policy=policy)
            state = engine.step(state, action)
            state.validate()
        else:
            raise RuntimeError("self-play exceeded terminal limit")
        replay = engine.replay(initial, state.history)
        if replay.serialize() != state.serialize():
            raise RuntimeError("self-play replay mismatch")
        wins += state.winner == seat
        draws += state.winner is None
        # Settlement uses true payments; utility mode preserves the old experiment.
        reward = terminal_reward(state.settlement, seat, reward_mode)
        if state.settlement:
            payoff += state.settlement.payments[seat]
        for j, (row, oldlog, oldvalue, step) in enumerate(trajectory):
            discount = 1.0 if reward_mode == "settlement" else 0.99
            target = reward * discount ** (len(trajectory) - 1 - j)
            rows.append((row, oldlog, oldvalue, target))
    return rows, {
        "games": games,
        "wins": wins,
        "draws": draws,
        "decisions": len(rows),
        "replay_verified": True,
        "payoff": payoff,
        "reward_mode": reward_mode,
    }


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = load_model(args.resume)
    opponent = copy.deepcopy(model).eval()
    # Readiness logits have a different meaning from terminal utility.
    nn.init.zeros_(model.value[-1].weight)
    nn.init.zeros_(model.value[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if model.feature_version == "huxi":
        from .huxi import huxi_loss, rich_example

        make_example = rich_example
    else:
        make_example = example
    anchors = [make_example(make_puzzle(args.seed * 100000 + i)[0]) for i in range(256)]
    logs = []
    for iteration in range(args.iterations):
        rows, metrics = collect(
            model,
            opponent,
            args.games,
            args.seed * 1000 + iteration * args.games,
            args.reward,
            args.opponents,
        )
        advantages = np.array([r[3] - r[2] for r in rows], np.float32)
        scale = max(float(advantages.std()), 0.1)
        mean = float(advantages.mean())
        losses = []
        model.train()
        for epoch in range(3):
            order = list(range(len(rows)))
            rng.shuffle(order)
            for start in range(0, len(order), 128):
                indices = order[start : start + 128]
                selected = [rows[i] for i in indices]
                x, a, m, y, _v, _w = batch([r[0] for r in selected])
                logits, value, waits, hu_prediction = model.forward_all(x, a, m)
                distribution = torch.distributions.Categorical(logits=logits)
                logprob = distribution.log_prob(y)
                old = torch.tensor([r[1] for r in selected])
                adv = torch.tensor((advantages[indices] - mean) / scale)
                ratio = (logprob - old).exp()
                actor = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
                critic = nn.functional.mse_loss(value, torch.tensor([r[3] for r in selected]))
                anchor_rows = rng.sample(anchors, 32)
                anchor = batch(anchor_rows)
                al, _, aw, ah = model.forward_all(*anchor[:3])
                imitation = nn.functional.cross_entropy(
                    al, anchor[3]
                ) + 0.1 * nn.functional.binary_cross_entropy_with_logits(aw, anchor[5])
                loss = actor + 0.5 * critic - 0.01 * distribution.entropy().mean() + 0.2 * imitation
                if hu_prediction is not None:
                    hu_targets = torch.tensor(np.stack([r[0][5] for r in selected]))
                    anchor_hu = torch.tensor(np.stack([r[5] for r in anchor_rows]))
                    loss += (
                        0.15 * huxi_loss(hu_prediction, hu_targets)
                        + 0.05 * nn.functional.binary_cross_entropy_with_logits(waits, _w)
                        + 0.1 * huxi_loss(ah, anchor_hu)
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite RL loss")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss.detach()))
        log = {
            "iteration": iteration + 1,
            **metrics,
            "loss": sum(losses) / len(losses),
            "nonzero_rewards": sum(r[3] != 0 for r in rows),
        }
        logs.append(log)
        print(json.dumps(log), flush=True)
    value_target = (
        "undiscounted terminal payments/100"
        if args.reward == "settlement"
        else "discounted terminal win/draw/loss"
    )
    save_model(model, args.output, value_target=value_target, seed=args.seed)
    report = {
        "algorithm": "clipped actor-critic with supervised hu-xi anchor",
        "opponent_mode": args.opponents,
        "resume": args.resume,
        "seed": args.seed,
        "value_target": value_target,
        "reward_mode": args.reward,
        "device": "cpu",
        "iterations": logs,
        "checkpoint": args.output,
        "sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "random_tournament": tournament(model, list(range(args.eval_start, args.eval_start + 20))),
        "teacher_tournament": tournament(
            model, list(range(args.eval_start + 500, args.eval_start + 506)), opponent="teacher"
        ),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["random_tournament"]), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--reward", choices=["utility", "settlement"], default="utility")
    p.add_argument("--eval-start", type=int, default=7000)
    p.add_argument("--opponents", choices=["frozen", "mixed"], default="frozen")
    run(p.parse_args())


if __name__ == "__main__":
    main()
