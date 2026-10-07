"""Balanced structural-wait fixtures, masked hu-xi and ordinary-payout learning.

Fixtures conserve all 80 tiles and use engine transitions; they are synthetic
midgame states, not claimed to be reachable histories from the standard deal.
"""

import argparse
import hashlib
import inspect
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zimortal.engine import ActionType as A
from zimortal.engine import GameState, Meld, PlayerState, RuleEngine
from zimortal.engine import MeldType as M
from zimortal.engine.chi import CHI_PATTERNS
from zimortal.model.network import PolicyValueNet

from .huxi import read_rich, rich_example, source_signature, write_rich
from .runtime import load_model, save_model, tournament
from .scale import deduplicate
from .train import batch
from .valuation import draw_values

BUCKETS = ("low", "14", "15", "16", "high")


def bucket(hu):
    return str(hu) if hu in (14, 15, 16) else "low" if hu < 14 else "high"


def fixture(seed):
    """Return an observation before drawing a known completion, without leakage."""
    rng = random.Random(seed)
    engine = RuleEngine()
    for _ in range(200):
        groups = []
        exposed = []
        counts = Counter()
        quad = rng.random() < 0.18
        if quad:
            tile = rng.randrange(20)
            groups.append(Meld(rng.choice((M.TI, M.PAO)), (tile,) * 4))
            exposed.append(True)
            counts.update(groups[0].tiles)
        for _ in range(5 if quad else 7):
            for _ in range(100):
                if rng.random() < 0.5:
                    tile = rng.randrange(20)
                    show = rng.random() < 0.75
                    kind = rng.choice((M.PENG, M.PENG, M.WEI)) if show else M.KAN
                    group = Meld(kind, (tile,) * 3)
                else:
                    group = Meld(M.CHI, rng.choice(CHI_PATTERNS))
                    show = rng.random() < 0.45
                next_counts = counts + Counter(group.tiles)
                if max(next_counts.values()) <= (4 if quad else 3):
                    groups.append(group)
                    exposed.append(show)
                    counts = next_counts
                    break
            else:
                break
        if len(groups) != (6 if quad else 7):
            continue
        if quad:
            possible = [t for t in range(20) if counts[t] <= 2]
            tile = rng.choice(possible)
            groups.append(Meld(M.PAIR, (tile,) * 2))
            exposed.append(False)
            counts.update((tile, tile))
        hand = [t for g, show in zip(groups, exposed, strict=True) if not show for t in g.tiles]
        if not hand or max(Counter(hand).values()) > 3:
            continue
        target = rng.choice(hand)
        hand.remove(target)
        melds = [g for g, show in zip(groups, exposed, strict=True) if show]
        kans = {t for t, n in Counter(hand).items() if n == 3}
        outcomes = draw_values(tuple(sorted(hand)), tuple(melds), tuple(sorted(kans)))
        if outcomes[target][0] < 0:
            continue
        used = Counter(hand) + Counter(t for m in melds for t in m.tiles)
        rest = [t for t in range(20) for _ in range(4 - used[t])]
        rest.remove(target)
        rng.shuffle(rest)
        players = [PlayerState(sorted(hand), melds=melds, kans=kans, quad_count=int(quad))]
        players += [PlayerState([rest.pop() for _ in range(20)]) for _ in range(2)]
        river = [rest.pop()]
        deck = rest + [target]
        state = GameState(players, deck, dealer=0, phase="draw", river=river, seed=seed)
        state.validate()
        assert len(deck) == 19 and len(hand) + sum(len(m.tiles) for m in melds) == 20
        for player in state.players[1:]:
            player.kans = {t for t, n in Counter(player.hand).items() if n == 3}
        while engine.legal_actions(state)[0].kind == A.TI:
            state = engine.step(state, engine.legal_actions(state)[0])
        # Real draw and own mandatory wei/ti must agree with the analysis label.
        after = engine.step(state, engine.legal_actions(state)[0])
        while (
            (actions := engine.legal_actions(after))[0].forced
            and actions[0].player == 0
            and actions[0].kind in (A.WEI, A.STINKY_WEI, A.TI)
        ):
            after = engine.step(after, actions[0])
        own_hu = next((a for a in actions if a.player == 0 and a.kind == A.HU), None)
        if (own_hu is not None) != (outcomes[target][0] >= 15):
            raise RuntimeError("synthetic draw disagrees with rule engine")
        if own_hu:
            end = engine.step(after, own_hu)
            if end.settlement.amount_each != outcomes[target][1]:
                raise RuntimeError("ordinary payout label disagrees with settlement")
        return engine.observation(state, 0), target, state
    raise RuntimeError("fixture generation exhausted")


def row(obs):
    result = rich_example(obs)
    kans = tuple(t for t, n in Counter(obs.hand).items() if n == 3)
    outcomes = draw_values(obs.hand, obs.players[obs.player].melds, kans)
    payout = np.asarray([[amount, fan] for _hu, amount, fan in outcomes], np.float32)
    return result + (np.log1p(payout).T / np.log(501),)


def read_boundary(path):
    rows = read_rich(path)
    with np.load(path, allow_pickle=False) as z:
        payouts = z["payout"]
    if payouts.shape != (len(rows), 2, 20):
        raise ValueError("invalid payout labels")
    return [r + (payouts[i],) for i, r in enumerate(rows)]


def job(spec):
    root, split, shard, count = spec
    rows, records = [], []
    quota = Counter()
    attempts = 0
    seen = set()
    namespace = (2400 if split == "train" else 2700) + shard
    while len(rows) < count:
        seed = namespace * 100000 + attempts
        attempts += 1
        if attempts >= 100000:
            raise RuntimeError("boundary seed namespace exhausted")
        obs, target, state = fixture(seed)
        r = row(obs)
        hu = round(float(r[5][target + 1]) * 60)
        stratum = bucket(hu)
        key = hashlib.sha256(r[0].tobytes() + r[1].tobytes()).hexdigest()
        if quota[stratum] >= count // 5 or key in seen:
            continue
        quota[stratum] += 1
        seen.add(key)
        rows.append(r)
        records.append(
            {
                "seed": seed,
                "target": target,
                "huxi": hu,
                "bucket": stratum,
                "state_sha256": hashlib.sha256(state.serialize().encode()).hexdigest(),
            }
        )
    path = Path(root) / f"{split}-{shard:03d}.npz"
    write_rich(path, rows)
    with np.load(path, allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files}
    fields["payout"] = np.stack([r[6] for r in rows])
    np.savez_compressed(path, **fields)
    meta = {
        "file": path.name,
        "split": split,
        "samples": len(rows),
        "attempts": attempts,
        "quota": dict(quota),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fixtures": records,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return {k: v for k, v in meta.items() if k != "fixtures"}


def signature():
    return {
        "boundary_generator": hashlib.sha256(
            (
                repr(BUCKETS) + "".join(inspect.getsource(f) for f in (bucket, fixture, row, job))
            ).encode()
        ).hexdigest(),
        **source_signature(),
    }


def build(args):
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        if json.loads((root / "manifest.json").read_text())["source"] != signature():
            raise ValueError("generator changed; use fresh directory")
        raise ValueError("dataset already built; preserve existing fixtures")
    specs = [(str(root), "train", i, args.per_shard) for i in range(args.shards)]
    specs += [(str(root), "valid", i, args.per_shard) for i in range(4)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(job, specs):
            records.append(record)
            print(json.dumps(record), flush=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {"source": signature(), "shards": records, "synthetic_midgame_not_deal_replay": True},
            indent=2,
        )
        + "\n"
    )


def datasets(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source"] != signature():
        raise ValueError("source signature mismatch")
    train, valid = [], []
    for record in manifest["shards"]:
        path = root / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("corrupt boundary shard")
        (train if record["split"] == "train" else valid).extend(read_boundary(path))
    return deduplicate(train, valid)


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def outcome_loss(outputs, rows, hu_weight=40.0, hu_margin=0.0):
    _logits, _value, wait, hu, eligible, amount, fan = outputs
    targets = torch.tensor(np.stack([r[5] for r in rows]))
    structure = targets[:, 1:] >= 0
    legal = targets[:, 1:] * 60 >= 15
    hu_loss = nn.functional.mse_loss(hu[:, 0], targets[:, 0])
    hu_loss += hu_weight * masked_mean((hu[:, 1:] - targets[:, 1:]).square(), structure)
    hu_loss += hu_margin * masked_mean(
        nn.functional.binary_cross_entropy_with_logits(
            (hu[:, 1:] * 60 - 15) / 2, legal.float(), reduction="none"
        ),
        structure,
    )
    structural_loss = masked_mean(
        nn.functional.binary_cross_entropy_with_logits(wait, structure.float(), reduction="none"),
        structure,
    )
    structural_loss += masked_mean(
        nn.functional.binary_cross_entropy_with_logits(wait, structure.float(), reduction="none"),
        ~structure,
    )
    if eligible is None:
        return hu_loss + 0.2 * structural_loss
    eligibility_loss = masked_mean(
        nn.functional.binary_cross_entropy_with_logits(eligible, legal.float(), reduction="none"),
        structure,
    )
    with_payout = [i for i, r in enumerate(rows) if len(r) > 6]
    payout_loss = torch.zeros(())
    if with_payout:
        target = torch.tensor(np.stack([rows[i][6] for i in with_payout]))
        ok = legal[with_payout]
        payout_loss = masked_mean((amount[with_payout] - target[:, 0]).square(), ok)
        payout_loss += masked_mean((fan[with_payout] - target[:, 1]).square(), ok)
    return hu_loss + 0.2 * structural_loss + eligibility_loss + 2 * payout_loss


def diagnostics(model, rows):
    model.eval()
    sums = Counter()
    bins = {k: Counter() for k in BUCKETS}
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            selected = rows[start : start + 128]
            x, a, m, y, _v, _w = batch(selected)
            logits, _value, wait, hu, eligible, amount, fan = model.forward_aux(x, a, m)
            target = torch.tensor(np.stack([r[5] for r in selected])) * 60
            structure = target[:, 1:] >= 0
            legal = target[:, 1:] >= 15
            predicted = hu[:, 1:] * 60
            threshold = eligible >= 0 if eligible is not None else predicted >= 15
            predicted_structure = wait >= 0
            predicted_legal = predicted_structure & threshold
            for name, prediction, truth in (
                ("structure", predicted_structure, structure),
                ("legal", predicted_legal, legal),
            ):
                sums[name + "_tp"] += int((prediction & truth).sum())
                sums[name + "_fp"] += int((prediction & ~truth).sum())
                sums[name + "_fn"] += int((~prediction & truth).sum())
            sums["samples"] += len(selected)
            sums["policy_correct"] += int((logits.argmax(-1) == y).sum())
            sums["formed_error"] += float((hu[:, 0] * 60 - target[:, 0]).abs().sum())
            sums["structures"] += int(structure.sum())
            sums["draw_error"] += float((predicted - target[:, 1:]).abs()[structure].sum())
            sums["threshold_correct"] += int((threshold == legal)[structure].sum())
            sums["huxi_threshold_correct"] += int(((predicted >= 15) == legal)[structure].sum())
            for label in BUCKETS:
                actual = target[:, 1:].round()
                mask = structure & (
                    (actual < 14)
                    if label == "low"
                    else (actual > 16)
                    if label == "high"
                    else torch.isclose(actual, torch.tensor(float(label)), atol=0.001)
                )
                bins[label]["n"] += int(mask.sum())
                bins[label]["error"] += float((predicted - actual).abs()[mask].sum())
                bins[label]["correct"] += int((threshold == legal)[mask].sum())
            if amount is not None and len(selected[0]) > 6:
                payout = torch.tensor(np.stack([r[6] for r in selected]))
                true_amount = torch.expm1(payout[:, 0] * np.log(501))
                true_fan = torch.expm1(payout[:, 1] * np.log(501))
                predicted_amount = torch.expm1((amount * np.log(501)).clamp(0, 9))
                predicted_fan = torch.expm1((fan * np.log(501)).clamp(0, 5))
                sums["legal"] += int(legal.sum())
                sums["amount_error"] += float((predicted_amount - true_amount).abs()[legal].sum())
                sums["fan_error"] += float((predicted_fan - true_fan).abs()[legal].sum())
    result = {
        "samples": sums["samples"],
        "structural_tile_samples": sums["structures"],
        "teacher_policy_accuracy": sums["policy_correct"] / max(1, sums["samples"]),
        "formed_huxi_mae": sums["formed_error"] / max(1, sums["samples"]),
        "draw_huxi_mae": sums["draw_error"] / max(1, sums["structures"]),
        "fifteen_threshold_accuracy": sums["threshold_correct"] / max(1, sums["structures"]),
        "huxi_threshold_accuracy": sums["huxi_threshold_correct"] / max(1, sums["structures"]),
        "strata": {
            k: {
                "n": v["n"],
                "huxi_mae": v["error"] / max(1, v["n"]),
                "threshold_accuracy": v["correct"] / max(1, v["n"]),
            }
            for k, v in bins.items()
        },
    }
    if sums["legal"]:
        result.update(
            legal_tile_samples=sums["legal"],
            ordinary_amount_each_mae=sums["amount_error"] / sums["legal"],
            additive_fan_mae=sums["fan_error"] / sums["legal"],
        )
    for name in ("structure", "legal"):
        result[name + "_precision"] = sums[name + "_tp"] / max(
            1, sums[name + "_tp"] + sums[name + "_fp"]
        )
        result[name + "_recall"] = sums[name + "_tp"] / max(
            1, sums[name + "_tp"] + sums[name + "_fn"]
        )
    return result


def migrate(path):
    old = load_model(path)
    new = PolicyValueNet(old.architecture, old.width, old.feature_version, "boundary")
    new.load_state_dict(old.state_dict(), strict=False)
    return new


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(77)
    rng = np.random.default_rng(77)
    targeted, valid = datasets(args.data)
    root = Path("data/generated/huxi100k")
    manifest = json.loads((root / "manifest.json").read_text())
    base_train, base_valid = [], []
    for record in manifest["shards"]:
        (base_train if record["config"]["kind"].startswith("train") else base_valid).extend(
            read_rich(root / record["file"])
        )
    base_train, base_valid = deduplicate(base_train, base_valid)
    # No network-input overlap across either validation set and either training set.
    key = lambda r: hashlib.sha256(r[0].tobytes() + r[1].tobytes()).digest()
    training_keys = {key(r) for r in base_train + targeted}
    base_valid = [r for r in base_valid if key(r) not in training_keys]
    valid = [r for r in valid if key(r) not in training_keys]
    parent = load_model(args.resume)
    model = migrate(args.resume)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    logs = []
    before = {"boundary": diagnostics(parent, valid), "ordinary": diagnostics(parent, base_valid)}
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        base_order = rng.permutation(len(base_train))
        loss_sum = steps = 0
        for start in range(0, len(base_order), 64):
            general = [base_train[i] for i in base_order[start : start + 64]]
            focused = [targeted[i] for i in rng.integers(len(targeted), size=64)]
            rows = general + focused
            x, a, m, y, _v, _w = batch(rows)
            output = model.forward_aux(x, a, m)
            with torch.no_grad():
                previous = parent.forward_all(x, a, m)
            # Preserve the settlement critic and prior policy while improving auxiliaries.
            loss = outcome_loss(output, rows, args.hu_weight, args.hu_margin)
            loss += 0.15 * nn.functional.cross_entropy(output[0][: len(general)], y[: len(general)])
            loss += 0.5 * nn.functional.kl_div(
                output[0].log_softmax(-1), previous[0].softmax(-1), reduction="batchmean"
            )
            loss += 0.5 * nn.functional.mse_loss(output[1], previous[1])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite boundary loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1
        metrics = diagnostics(model, valid)
        score = metrics["draw_huxi_mae"] / 10 + 2 * (1 - metrics["fifteen_threshold_accuracy"])
        log = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "boundary": metrics,
            "ordinary": diagnostics(model, base_valid),
        }
        logs.append(log)
        if score < best:
            best = score
            save_model(
                model,
                args.output,
                value_target="distilled terminal payments/100",
                auxiliary_target="structural draw huxi, independent fifteen gate, log ordinary amount and additive fan",
                training_samples=len(base_train) + len(targeted),
                training_policy_decisions=len(base_train),
                training_auxiliary_draw_states=len(targeted),
                selected_epoch=epoch + 1,
            )
        print(json.dumps(log), flush=True)
    selected = load_model(args.output)
    result = {
        "train_base": len(base_train),
        "loss_weights": {"huxi_mse": args.hu_weight, "huxi_threshold_margin": args.hu_margin},
        "train_boundary": len(targeted),
        "validation_base": len(base_valid),
        "validation_boundary": len(valid),
        "before": before,
        "after": {
            "boundary": diagnostics(selected, valid),
            "ordinary": diagnostics(selected, base_valid),
        },
        "epochs": logs,
        "checkpoint": args.output,
        "sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "random_tournament": tournament(selected, list(range(28000, 28050))),
        "teacher_tournament": tournament(selected, list(range(28500, 28520)), opponent="teacher"),
    }
    Path(args.report).write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("build", "train"))
    p.add_argument("--data", default="data/generated/boundary20k")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shards", type=int, default=20)
    p.add_argument("--per-shard", type=int, default=800)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--hu-weight", type=float, default=40.0)
    p.add_argument("--hu-margin", type=float, default=0.0)
    p.add_argument("--resume", default="checkpoints/huxi-resnet.pt")
    p.add_argument("--output", default="checkpoints/boundary-warmup.pt")
    p.add_argument("--report", default="docs/training/boundary-warmup.json")
    args = p.parse_args()
    if args.per_shard < 5 or args.per_shard % 5 or min(args.workers, args.shards, args.epochs) < 1:
        p.error("positive settings required, per-shard divisible by five")
    (build if args.command == "build" else train)(args)


if __name__ == "__main__":
    main()
