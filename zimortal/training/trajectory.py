"""Whole-game supervised route pilot. Successful routes are feasible, not optimal."""

import argparse
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from zimortal.engine import ActionType as A
from zimortal.engine import RuleClarificationRequired, RuleEngine
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.model.network import PolicyValueNet
from zimortal.model.trajectory import FAN_NAMES, GROUP_PATTERNS, terminal_route_targets

from .active import atomic_json, atomic_npz
from .online_rl import digest, evaluation, source_hashes
from .runtime import choose_batch, load_model, save_model


def migrate(parent):
    if parent.feature_version != "huxi" or parent.auxiliary_version != "legacy":
        raise ValueError("pilot requires a huxi legacy policy parent")
    model = PolicyValueNet(parent.architecture, parent.width, "huxi", "trajectory")
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    prefixes = ("group_candidates.", "final_", "future_intakes.", "trajectory_win.")
    if unexpected or any(not name.startswith(prefixes) for name in missing):
        raise ValueError("incompatible policy migration")
    return model


def future_intakes(history, actor, start, draw_tiles):
    counts = Counter()
    offer, consumed = None, set()
    for index, action in enumerate(history):
        if action.kind in (A.DRAW, A.DISCARD):
            offer = index
        if action.player != actor:
            continue
        if action.kind == A.DRAW or (
            action.kind in (A.CHI, A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI, A.HU)
            and action.source_player is not None
            and action.source_player != actor
        ):
            if offer in consumed:
                continue
            consumed.add(offer)
            tile = draw_tiles[index] if action.kind == A.DRAW else action.tile
            if tile is not None and index >= start:
                counts[tile] += 1
    return [counts[t] / 4 for t in range(20)]


def collect(parent, seeds, device="cpu", progress=None):
    engine, rows, records = RuleEngine(), [], []
    for seed in seeds:
        state = engine.new_game(seed, dealer=seed % 3)
        snapshots, draw_tiles = [], {}
        try:
            for _ in range(1000):
                if state.terminal:
                    break
                actions = engine.legal_actions(state)
                actor = actions[0].player
                obs = engine.observation(state, actor)
                if len(actions) > 1:
                    snapshots.append(
                        (
                            actor,
                            len(state.history),
                            encode_observation(obs, "huxi"),
                            np.stack([encode_action(a, actor) for a in actions]),
                        )
                    )
                action = actions[0] if len(actions) == 1 else choose_batch(parent, [obs], device)[0]
                index = len(state.history)
                state = engine.step(state, action)
                if action.kind == A.DRAW:
                    if state.pending is None and not state.terminal:
                        raise RuntimeError("nonterminal draw has no offered tile")
                    draw_tiles[index] = state.pending.tile if state.pending else None
            if not state.terminal:
                raise RuntimeError("route generation failed to terminate")
            state.validate()
            replay = engine.replay(engine.new_game(seed, dealer=seed % 3), state.history)
            if replay.serialize() != state.serialize():
                raise RuntimeError("route terminal replay mismatch")
        except RuleClarificationRequired as exc:
            records.append({"seed": seed, "excluded": str(exc)})
            continue
        settlement = state.settlement
        tiles, groups, fans = (
            terminal_route_targets(settlement)
            if settlement
            else ([0] * 20, [0] * len(GROUP_PATTERNS), [0] * len(FAN_NAMES))
        )
        payments = settlement.payments if settlement else (0, 0, 0)
        for actor, start, features, actions in snapshots:
            won = bool(settlement and settlement.winner == actor)
            rows.append(
                (
                    features,
                    actions,
                    payments[actor] / 100,
                    won,
                    tiles,
                    groups,
                    fans,
                    future_intakes(state.history, actor, start, draw_tiles),
                    settlement.huxi / 60 if won else 0.0,
                    1 / len(snapshots),
                    seed,
                )
            )
        records.append(
            {
                "seed": seed,
                "winner": settlement.winner if settlement else None,
                "settlement": asdict(settlement) if settlement else None,
                "history": [asdict(a) for a in state.history],
                "decisions": len(snapshots),
                "replay_verified": True,
            }
        )
        if progress:
            progress(len(records), len(rows))
    return rows, records


def batch(rows, device):
    size = max(len(r[1]) for r in rows)
    actions = np.zeros((len(rows), size, rows[0][1].shape[-1]), np.float32)
    mask = np.zeros((len(rows), size), bool)
    for i, r in enumerate(rows):
        actions[i, : len(r[1])] = r[1]
        mask[i, : len(r[1])] = True
    values = [np.stack([r[0] for r in rows]), actions, mask]
    values.extend(np.array([r[k] for r in rows]) for k in range(2, 10))
    return [torch.as_tensor(v, device=device) for v in values]


def objective(model, parent, tensors):
    x, actions, mask, cash, won, tiles, groups, fans, intakes, huxi, weights = tensors
    outputs, forecast = model.forward_trajectory(x.float(), actions.float(), mask)
    with torch.no_grad():
        reference = parent(x.float(), actions.float(), mask)[0].softmax(-1)
    preservation = (
        reference * (reference.clamp_min(1e-12).log() - outputs[0].log_softmax(-1))
    ).sum(-1)
    winners = won.float()
    route = F.cross_entropy(forecast["tiles"].transpose(1, 2), tiles.long(), reduction="none").mean(
        -1
    )
    route += F.smooth_l1_loss(forecast["groups"], groups.float(), reduction="none").mean(-1)
    route += F.binary_cross_entropy_with_logits(
        forecast["fans"], fans.float(), reduction="none"
    ).mean(-1)
    route += F.smooth_l1_loss(forecast["huxi"], huxi.float(), reduction="none")
    intake = F.smooth_l1_loss(forecast["intakes"], intakes.float(), reduction="none").mean(-1)
    win = F.binary_cross_entropy_with_logits(forecast["win"], winners, reduction="none")
    value = F.smooth_l1_loss(outputs[1], cash.float(), reduction="none")
    # Each complete game has equal total state weight. Failure paths teach cash/win,
    # not fabricated all-zero successful routes. No successful-action cloning.
    loss = ((route * winners + intake + win + 0.1 * value + preservation) * weights).sum()
    loss = loss / weights.sum().clamp_min(1e-12)
    return loss


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "source_hashes": source_hashes(),
        "parent_sha256": digest(args.parent),
        "opponent_sha256": {p: digest(p) for p in args.opponents},
    }
    if (root / "config.json").exists():
        raise ValueError("use a new route training directory")
    atomic_json(root / "config.json", config)
    parent = load_model(args.parent, args.device)
    model = migrate(parent).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def progress(games, rows):
        atomic_json(
            root / "progress.json",
            {"status": "collecting", "games": games, "decisions": rows, "device": args.device},
        )

    train, train_games = collect(
        parent, range(args.train_start, args.train_start + args.games), args.device, progress
    )
    dev, dev_games = collect(
        parent, range(args.dev_start, args.dev_start + args.dev_games), args.device
    )
    if source_hashes() != config["source_hashes"] or digest(args.parent) != config["parent_sha256"]:
        raise ValueError("frozen trajectory producer changed during collection")
    if not train or not dev or not any(r[3] for r in train):
        raise RuntimeError("insufficient successful trajectories")
    for name, rows, records in (("train", train, train_games), ("dev", dev, dev_games)):
        atomic_json(root / f"{name}-games.json", records)
        atomic_npz(
            root / f"{name}.npz",
            features=np.stack([r[0] for r in rows]),
            actions=np.concatenate([r[1] for r in rows]),
            offsets=np.cumsum([0] + [len(r[1]) for r in rows]),
            cash=np.array([r[2] for r in rows]),
            won=np.array([r[3] for r in rows]),
            final_tiles=np.array([r[4] for r in rows]),
            final_groups=np.array([r[5] for r in rows]),
            final_fans=np.array([r[6] for r in rows]),
            intakes=np.array([r[7] for r in rows]),
            huxi=np.array([r[8] for r in rows]),
            weights=np.array([r[9] for r in rows]),
            game_seeds=np.array([r[10] for r in rows]),
        )
    rng, best = random.Random(args.seed), float("inf")
    for epoch in range(args.epochs):
        if (
            source_hashes() != config["source_hashes"]
            or digest(args.parent) != config["parent_sha256"]
        ):
            raise ValueError("frozen trajectory producer changed")
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), args.batch_size):
            tensors = batch([train[i] for i in order[start : start + args.batch_size]], args.device)
            loss = objective(model, parent, tensors)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            dev_loss = sum(
                objective(model, parent, batch(dev[i : i + args.batch_size], args.device)).item()
                * sum(r[9] for r in dev[i : i + args.batch_size])
                for i in range(0, len(dev), args.batch_size)
            ) / sum(r[9] for r in dev)
        metadata = {
            "value_target": "undiscounted terminal payments/100",
            "epoch": epoch + 1,
            "route_target": "realized winner decomposition, not optimal plan",
            "dev_loss": dev_loss,
            "parent_sha256": config["parent_sha256"],
        }
        save_model(model, root / "last-model.pt", **metadata)
        if dev_loss < best:
            best = dev_loss
            save_model(model, root / "best-dev.pt", **metadata)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "configuration": config,
                "epoch": epoch + 1,
                "rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
            },
            root / "optimizer.tmp",
        )
        (root / "optimizer.tmp").replace(root / "optimizer.pt")
        atomic_json(
            root / "progress.json",
            {
                "status": "training",
                "epoch": epoch + 1,
                "epochs": args.epochs,
                "train_loss": sum(losses) / len(losses),
                "dev_loss": dev_loss,
                "device": args.device,
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            },
        )
    atomic_json(root / "progress.json", {"status": "holdout_evaluation", "device": args.device})
    if source_hashes() != config["source_hashes"] or any(
        digest(p) != h for p, h in config["opponent_sha256"].items()
    ):
        raise ValueError("frozen route evaluation producer changed")
    candidate = load_model(root / "best-dev.pt", args.device)
    opponents = [parent] + [load_model(p, args.device) for p in args.opponents]
    holdout = evaluation(
        candidate,
        parent,
        opponents,
        args.holdout_start,
        args.holdout_seeds,
        args.device,
        32,
        audit=True,
    )
    atomic_json(root / "holdout.json", holdout)
    atomic_json(
        root / "result.json",
        {
            "status": "complete",
            "train_games": len(train_games),
            "dev_games": len(dev_games),
            "train_decisions": len(train),
            "dev_decisions": len(dev),
            "successful_train_decisions": sum(r[3] for r in train),
            "best_dev_loss": best,
            "formal_champion_changed": False,
            "cash_strength_evaluated": True,
            "holdout_valid": holdout["valid"],
            "holdout_aggregate": holdout.get("aggregate"),
            "excluded_train_games": sum("excluded" in g for g in train_games),
            "excluded_dev_games": sum("excluded" in g for g in dev_games),
        },
    )
    atomic_json(
        root / "progress.json",
        {"status": "complete", "device": args.device, "result": str(root / "result.json")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="data/generated/autonomous-v4/ppo/best-dev.pt")
    parser.add_argument("--directory", default="data/generated/trajectory-v1")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--dev-games", type=int, default=64)
    parser.add_argument("--train-start", type=int, default=640000000)
    parser.add_argument("--dev-start", type=int, default=650000000)
    parser.add_argument("--holdout-start", type=int, default=660000000)
    parser.add_argument("--holdout-seeds", type=int, default=64)
    parser.add_argument(
        "--opponents",
        nargs="*",
        default=[
            "checkpoints/huxi-resnet.pt",
            "checkpoints/scale100k-resnet.pt",
            "data/generated/league-v2/stage-02/best-dev.pt",
        ],
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=1315)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (
        min(args.games, args.dev_games, args.epochs, args.batch_size, args.holdout_seeds) < 1
        or args.lr <= 0
    ):
        parser.error("counts and learning rate must be positive")
    if not args.train_start + args.games <= args.dev_start:
        parser.error("training and development game seeds must be disjoint")
    if not args.dev_start + args.dev_games <= args.holdout_start:
        parser.error("development and holdout seeds must be disjoint")
    run(args)


if __name__ == "__main__":
    main()
