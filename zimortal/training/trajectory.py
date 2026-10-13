"""Whole-game supervised route pilot. Successful routes are feasible, not optimal."""

import argparse
import math
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


def migrate(parent, version="trajectory"):
    if parent.feature_version != "huxi" or parent.auxiliary_version != "legacy":
        raise ValueError("pilot requires a huxi legacy policy parent")
    model = PolicyValueNet(parent.architecture, parent.width, "huxi", version)
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    prefixes = (
        "group_candidates.",
        "final_",
        "future_intakes.",
        "trajectory_win.",
        "policy_plan_scale",
        "terminal_huxi_constant",
        "positive_group_counts",
    )
    if unexpected or any(not name.startswith(prefixes) for name in missing):
        raise ValueError("incompatible policy migration")
    return model


ROUTE_PREFIXES = ("group_candidates.", "final_", "future_intakes.", "trajectory_win.")


def route_warmup(parent, initialize=None, version="trajectory_v2"):
    model = migrate(parent, version)
    if initialize is not None:
        if (initialize.width, initialize.architecture, initialize.feature_version) != (
            parent.width,
            parent.architecture,
            parent.feature_version,
        ) or initialize.auxiliary_version not in (
            "trajectory",
            "trajectory_v2",
            "trajectory_v3",
            "trajectory_v4",
        ):
            raise ValueError("incompatible route initialization")
        model.load_state_dict(
            {k: v for k, v in initialize.state_dict().items() if k.startswith(ROUTE_PREFIXES)},
            strict=False,
        )
        if initialize.auxiliary_version in ("trajectory_v3", "trajectory_v4"):
            if version not in ("trajectory_v3", "trajectory_v4"):
                raise ValueError("bounded route heads cannot be downgraded to raw v2")
            model.positive_group_counts.copy_(initialize.positive_group_counts)
            model.terminal_huxi_constant.copy_(initialize.terminal_huxi_constant)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(ROUTE_PREFIXES))
    return model


def verify_parent_policy(model, parent):
    if (
        model.auxiliary_version not in ("trajectory_v2", "trajectory_v3", "trajectory_v4")
        or model.policy_plan_scale.item() != 0
    ):
        raise ValueError("route warmup must not change the policy gate")
    state = model.state_dict()
    if any(not torch.equal(state[k], v) for k, v in parent.state_dict().items()):
        raise ValueError("route pretraining changed frozen parent weights")


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
    if (
        model.auxiliary_version in ("trajectory_v3", "trajectory_v4")
        and model.positive_group_counts.item()
    ):
        route += F.poisson_nll_loss(
            forecast["groups"].clamp_min(1e-8), groups.float(), log_input=False, reduction="none"
        ).mean(-1)
    else:
        route += F.smooth_l1_loss(forecast["groups"], groups.float(), reduction="none").mean(-1)
    route += F.binary_cross_entropy_with_logits(
        forecast["fans"], fans.float(), reduction="none"
    ).mean(-1)
    route += F.smooth_l1_loss(forecast["huxi"], huxi.float(), reduction="none")
    if model.auxiliary_version in ("trajectory_v2", "trajectory_v3", "trajectory_v4"):
        expected_tiles = (forecast["tiles"].softmax(-1) * torch.arange(5, device=x.device)).sum(-1)
        group_tiles = forecast["groups"].clamp_min(0) @ model.group_candidates.requirements
        route += 0.1 * F.smooth_l1_loss(group_tiles / 4, expected_tiles / 4, reduction="none").mean(
            -1
        )
    if model.auxiliary_version == "trajectory_v4":
        present = (groups > 0).float()
        presence = F.binary_cross_entropy_with_logits(
            forecast["presence"], present, reduction="none"
        )
        # Balance present/absent patterns within each winner, not 91 empty groups
        # overwhelming the seven actual groups. Count errors focus on present groups.
        route += (presence * present).sum(-1) / present.sum(-1).clamp_min(1)
        route += (presence * (1 - present)).sum(-1) / (1 - present).sum(-1).clamp_min(1)
        route += ((forecast["groups"] - groups).abs() * present).sum(-1) / present.sum(
            -1
        ).clamp_min(1)
        counts = forecast["groups"]
        inventory = counts @ model.group_candidates.requirements
        route += (inventory - 4).clamp_min(0).square().mean(-1)
        route += (counts.sum(-1) - 7).square() / 7
        sizes = model.group_candidates.requirements.sum(-1)
        pair = counts[:, sizes == 2].sum(-1)
        quad = counts[:, sizes == 4].sum(-1)
        has_quad = (groups[:, sizes == 4].sum(-1) > 0).float()
        route += (pair - has_quad).square() + (quad * (1 - has_quad)).square()
    intake = F.smooth_l1_loss(forecast["intakes"], intakes.float(), reduction="none").mean(-1)
    win = F.binary_cross_entropy_with_logits(forecast["win"], winners, reduction="none")
    value = F.smooth_l1_loss(outputs[1], cash.float(), reduction="none")
    # Each complete game has equal total state weight. Failure paths teach cash/win,
    # not fabricated all-zero successful routes. No successful-action cloning.
    loss = ((route * winners + intake + win + 0.1 * value + preservation) * weights).sum()
    loss = loss / weights.sum().clamp_min(1e-12)
    return loss


def forecast_metrics(model, rows, device, batch_size):
    totals, denominator = Counter(), Counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            x, a, mask, _cash, won, tiles, groups, fans, intakes, hu, weights = batch(
                rows[start : start + batch_size], device
            )
            _, predictions = model.forward_trajectory(x.float(), a.float(), mask)
            success = weights * won
            tile_prediction = predictions["tiles"].argmax(-1)
            nonzero = (tiles > 0).float()
            values = {
                "final_tile_accuracy": ((tile_prediction == tiles).float().mean(-1), success),
                "final_nonzero_tile_accuracy": (
                    ((tile_prediction == tiles).float() * nonzero).sum(-1)
                    / nonzero.sum(-1).clamp_min(1),
                    success,
                ),
                "group_mae": ((predictions["groups"] - groups).abs().mean(-1), success),
                "negative_group_fraction": ((predictions["groups"] < 0).float().mean(-1), success),
                "positive_group_mae": (
                    ((predictions["groups"] - groups).abs() * (groups > 0)).sum(-1)
                    / (groups > 0).sum(-1).clamp_min(1),
                    success,
                ),
                "fan_brier": ((predictions["fans"].sigmoid() - fans).square().mean(-1), success),
                "huxi_mae": ((predictions["huxi"] * 60 - hu * 60).abs(), success),
                "conditional_terminal_huxi_below15_rate": (
                    (predictions["huxi"] * 60 < 15).float(),
                    success,
                ),
                "intake_mae": ((predictions["intakes"] * 4 - intakes * 4).abs().mean(-1), weights),
                "win_brier": ((predictions["win"].sigmoid() - won.float()).square(), weights),
            }
            if "presence" in predictions:
                selected = predictions["presence"] >= 0
                actual = groups > 0
                hits = (selected & actual).sum(-1)
                values["group_presence_precision"] = (hits / selected.sum(-1).clamp_min(1), success)
                values["group_presence_recall"] = (hits / actual.sum(-1).clamp_min(1), success)
            for name, (value, weight) in values.items():
                totals[name] += (value * weight).sum().item()
                denominator[name] += weight.sum().item()
    result = {k: totals[k] / denominator[k] if denominator[k] else None for k in totals}
    result["conditional_terminal_huxi_mae"] = result["huxi_mae"]
    winners = [r for r in rows if r[3]]
    result["winning_states"] = len(winners)
    result["winning_games"] = len({r[10] for r in winners})
    result["fan_positive_games"] = {
        name: len({r[10] for r in winners if r[6][i]}) for i, name in enumerate(FAN_NAMES)
    }
    return result


def huxi_baselines(train, dev):
    winners = [r for r in train if r[3]]
    weights = np.array([r[9] for r in winners])
    targets = np.array([r[8] * 60 for r in winners])
    floors = np.array([max(15, round(float(r[0][22, 0]) * 60)) for r in winners])
    if np.any(targets < floors - 1e-3):
        raise ValueError("terminal huxi contradicts locked-huxi lower bound")
    mean = float(np.average(targets, weights=weights))
    residual = float(np.average(targets - floors, weights=weights))
    heldout = [r for r in dev if r[3]]
    if not heldout:
        raise ValueError("development has no successful route to evaluate")
    y = np.array([r[8] * 60 for r in heldout])
    floor = np.array([max(15, round(float(r[0][22, 0]) * 60)) for r in heldout])
    if np.any(y < floor - 1e-3):
        raise ValueError("development terminal huxi violates locked lower bound")
    w = np.array([r[9] for r in heldout])
    return {
        "training_mean_huxi": mean,
        "training_mean_extra_huxi": residual,
        "constant_mean_dev_mae": float(np.average(abs(y - mean), weights=w)),
        "clipped_training_mean_dev_mae": float(
            np.average(abs(y - np.maximum(floor, mean)), weights=w)
        ),
        "locked_plus_training_residual_dev_mae": float(
            np.average(abs(y - floor - residual), weights=w)
        ),
        "meaning": "conditional terminal winning huxi; not current legal huxi",
        "zero_group_dev_mae": float(np.average([np.mean(r[5]) for r in heldout], weights=w)),
        "all_zero_tile_dev_accuracy": float(
            np.average([np.mean(np.array(r[4]) == 0) for r in heldout], weights=w)
        ),
        "training_fan_prior_dev_brier": float(
            np.average(
                (
                    (
                        np.array([r[6] for r in heldout])
                        - np.average([r[6] for r in winners], axis=0, weights=weights)
                    )
                    ** 2
                ).mean(-1),
                weights=w,
            )
        ),
    }


def initialize_huxi_baseline(model, extra):
    if model.auxiliary_version not in ("trajectory_v3", "trajectory_v4"):
        raise ValueError("bounded terminal huxi head required")
    with torch.no_grad():
        model.terminal_huxi_constant.zero_()
        model.final_huxi.weight.zero_()
        model.final_huxi.bias.fill_(math.log(math.expm1(max(extra / 15, 1e-5))))


def initialize_group_prior(model, rows):
    winners = [r for r in rows if r[3]]
    prior = np.average([r[5] for r in winners], axis=0, weights=[r[9] for r in winners])
    with torch.no_grad():
        model.positive_group_counts.fill_(True)
        model.final_groups.weight.zero_()
        model.final_groups.bias.copy_(
            torch.tensor(
                np.log(np.expm1(np.maximum(prior, 1e-5))),
                device=model.final_groups.bias.device,
                dtype=model.final_groups.bias.dtype,
            )
        )


def write_chunk(prefix, rows, records):
    """Atomic bounded-size shards; keep original replay evidence losslessly."""
    import gzip
    import json
    import os

    if not rows:
        raise RuntimeError("chunk has no decisions")
    atomic_npz(
        prefix.with_suffix(".npz"),
        features=np.stack([r[0] for r in rows]),
        actions=np.concatenate([r[1] for r in rows]),
        offsets=np.cumsum([0] + [len(r[1]) for r in rows]),
        **{
            name: np.array([r[i] for r in rows])
            for i, name in enumerate(
                (
                    "cash",
                    "won",
                    "final_tiles",
                    "final_groups",
                    "final_fans",
                    "intakes",
                    "huxi",
                    "weights",
                    "game_seeds",
                ),
                2,
            )
        },
    )
    path = prefix.with_suffix(".games.json.gz")
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(records, stream, ensure_ascii=False)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    atomic_json(
        prefix.with_suffix(".sha256.json"),
        {
            path.name: digest(path),
            prefix.with_suffix(".npz").name: digest(prefix.with_suffix(".npz")),
        },
    )


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
        "initialize_sha256": digest(args.initialize) if args.initialize else None,
    }
    if (root / "config.json").exists():
        raise ValueError("use a new route training directory")
    atomic_json(root / "config.json", config)
    parent = load_model(args.parent, args.device)
    initialize = load_model(args.initialize, args.device) if args.initialize else None
    model = route_warmup(parent, initialize, args.auxiliary_version).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def progress(games, rows):
        atomic_json(
            root / "progress.json",
            {"status": "collecting", "games": games, "decisions": rows, "device": args.device},
        )

    def collect_chunks(name, first, count):
        all_rows, all_records = [], []
        for offset in range(0, count, args.chunk_games):
            rows, records = collect(
                parent,
                range(first + offset, first + min(count, offset + args.chunk_games)),
                args.device,
                lambda games, decisions, offset=offset: atomic_json(
                    root / "progress.json",
                    {
                        "status": "collecting",
                        "split": name,
                        "games": offset + games,
                        "target_games": count,
                        "decisions": len(all_rows) + decisions,
                        "device": args.device,
                    },
                ),
            )
            write_chunk(root / f"{name}-{offset:05d}", rows, records)
            all_rows.extend(rows)
            all_records.extend(records)
            atomic_json(
                root / f"{name}-manifest.json",
                {
                    "games": len(all_records),
                    "decisions": len(all_rows),
                    "complete": offset + args.chunk_games >= count,
                    "first_seed": first,
                    "chunks": [f"{name}-{i:05d}" for i in range(0, offset + 1, args.chunk_games)],
                },
            )
        return all_rows, all_records

    train, train_games = collect_chunks("train", args.train_start, args.games)
    dev, dev_games = collect_chunks("dev", args.dev_start, args.dev_games)
    if source_hashes() != config["source_hashes"] or digest(args.parent) != config["parent_sha256"]:
        raise ValueError("frozen trajectory producer changed during collection")
    if not train or not dev or not any(r[3] for r in train):
        raise RuntimeError("insufficient successful trajectories")
    baselines = huxi_baselines(train, dev)
    if (
        initialize is None
        or initialize.auxiliary_version not in ("trajectory_v3", "trajectory_v4")
        or model.terminal_huxi_constant.item() > 0
    ):
        initialize_huxi_baseline(model, baselines["training_mean_extra_huxi"])
    if not model.positive_group_counts.item():
        initialize_group_prior(model, train)
    atomic_json(root / "huxi-baselines.json", baselines)
    initial_metrics = forecast_metrics(model, dev, args.device, args.batch_size)
    atomic_json(root / "initial-dev-metrics.json", initial_metrics)
    rng, best = random.Random(args.seed), float("inf")
    for epoch in range(args.epochs):
        if (
            source_hashes() != config["source_hashes"]
            or digest(args.parent) != config["parent_sha256"]
            or (args.initialize and digest(args.initialize) != config["initialize_sha256"])
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
        verify_parent_policy(model, parent)
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
            "huxi_target": "terminal huxi/60 conditional on winning; not current eligibility",
            "dev_loss": dev_loss,
            "parent_sha256": config["parent_sha256"],
            "policy_frozen": True,
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
                "dev_forecasts": forecast_metrics(model, dev, args.device, args.batch_size),
            },
        )
    atomic_json(root / "progress.json", {"status": "holdout_evaluation", "device": args.device})
    if source_hashes() != config["source_hashes"] or any(
        digest(p) != h for p, h in config["opponent_sha256"].items()
    ):
        raise ValueError("frozen route evaluation producer changed")
    candidate = load_model(root / "best-dev.pt", args.device)
    verify_parent_policy(candidate, parent)
    before_guard = forecast_metrics(candidate, dev, args.device, args.batch_size)
    baseline_mode = (
        "clipped_mean"
        if baselines["clipped_training_mean_dev_mae"]
        <= baselines["locked_plus_training_residual_dev_mae"]
        else "locked_residual"
    )
    baseline_error = min(
        baselines["clipped_training_mean_dev_mae"],
        baselines["locked_plus_training_residual_dev_mae"],
    )
    restored = before_guard["huxi_mae"] > baseline_error + 1e-4
    if restored:
        if baseline_mode == "clipped_mean":
            candidate.terminal_huxi_constant.fill_(baselines["training_mean_huxi"])
        else:
            initialize_huxi_baseline(candidate, baselines["training_mean_extra_huxi"])
        save_model(
            candidate,
            root / "selected-model.pt",
            value_target="undiscounted terminal payments/100",
            policy_frozen=True,
            huxi_baseline_restored=True,
        )
    else:
        save_model(
            candidate,
            root / "selected-model.pt",
            value_target="undiscounted terminal payments/100",
            policy_frozen=True,
            huxi_baseline_restored=False,
        )
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
            "policy_frozen": True,
            "initial_dev_forecasts": initial_metrics,
            "final_dev_forecasts": forecast_metrics(candidate, dev, args.device, args.batch_size),
            "huxi_baselines": baselines,
            "before_huxi_guard_forecasts": before_guard,
            "huxi_baseline_restored": restored,
            "huxi_baseline_mode": baseline_mode if restored else "learned",
            "selected_checkpoint": str(root / "selected-model.pt"),
        },
    )
    atomic_json(
        root / "progress.json",
        {"status": "complete", "device": args.device, "result": str(root / "result.json")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="data/generated/autonomous-v4/ppo/best-dev.pt")
    parser.add_argument(
        "--initialize", help="reuse only route modules; keep the strong parent policy"
    )
    parser.add_argument("--directory", default="data/generated/trajectory-warmup-v3")
    parser.add_argument(
        "--auxiliary-version", choices=("trajectory_v3", "trajectory_v4"), default="trajectory_v4"
    )
    parser.add_argument("--chunk-games", type=int, default=128)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--dev-games", type=int, default=64)
    parser.add_argument("--train-start", type=int, default=670000000)
    parser.add_argument("--dev-start", type=int, default=680000000)
    parser.add_argument("--holdout-start", type=int, default=690000000)
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
        min(
            args.games,
            args.dev_games,
            args.epochs,
            args.batch_size,
            args.holdout_seeds,
            args.chunk_games,
        )
        < 1
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
