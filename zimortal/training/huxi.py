"""Hu-xi-aware curriculum and supervised warmup for settlement self-play."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import ActionType as A
from zimortal.engine import RuleEngine
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.model.network import PolicyValueNet

from .data import make_puzzle
from .runtime import load_model, save_model, tournament
from .scale import deduplicate, read_shard, write_shard
from .train import batch, evaluate
from .valuation import action_scores, draw_values, formed_huxi


def auxiliary(obs):
    ms = obs.players[obs.player].melds
    ks = tuple(t for t, n in Counter(obs.hand).items() if n == 3)
    values = draw_values(obs.hand, ms, ks)
    structural = np.array([hu >= 0 for hu, _a, _f in values], np.float32)
    huxi = (
        np.array([formed_huxi(obs.hand, ms, ks)] + [hu for hu, _a, _f in values], np.float32) / 60
    )
    if obs.hu_disabled:
        structural[:] = 0
        huxi[1:] = -1 / 60
    return structural, huxi


def rich_example(obs):
    structural, hu = auxiliary(obs)
    scores = action_scores(obs)
    ready = float(any(a.kind == A.HU for a in obs.legal_actions) or bool((hu[1:] * 60 >= 15).any()))
    return (
        encode_observation(obs, "huxi"),
        np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
        int(np.argmax(scores)),
        ready,
        structural,
        hu,
    )


def write_rich(path, rows):
    write_shard(path, rows)
    with np.load(path, allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files}
    fields["huxi"] = np.stack([r[5] for r in rows])
    np.savez_compressed(path, **fields)


def read_rich(path):
    rows = read_shard(path)
    with np.load(path, allow_pickle=False) as z:
        hu = z["huxi"]
    if hu.shape != (len(rows), 21):
        raise ValueError("invalid hu-xi target dimensions")
    return [row + (hu[i],) for i, row in enumerate(rows)]


def generate(puzzles, games, namespace):
    rows = []
    rng = random.Random(namespace)
    engine = RuleEngine()
    for i in range(puzzles):
        rows.append(rich_example(make_puzzle(namespace * 100000 + i)[0]))
    for i in range(games):
        state = engine.new_game(namespace * 100000 + i)
        for _ in range(1000):
            if state.terminal:
                break
            actions = engine.legal_actions(state)
            if len(actions) > 1:
                rows.append(rich_example(engine.observation(state, actions[0].player)))
            state = engine.step(state, rng.choice(actions))
        else:
            raise RuntimeError("curriculum game did not terminate")
    return rows


def source_signature():
    package = Path(__file__).resolve().parents[1]
    paths = sorted((package / "engine").glob("*.py")) + [
        package / "model" / "encoding.py",
        Path(__file__),
        Path(__file__).with_name("valuation.py"),
        Path(__file__).with_name("data.py"),
    ]
    return {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def job(spec):
    root, kind, index, namespace, puzzles, games = spec
    path = Path(root) / f"{kind}-{index:04d}.npz"
    meta = path.with_suffix(".json")
    config = {"kind": kind, "namespace": namespace, "puzzles": puzzles, "games": games}
    if meta.exists() and path.exists():
        saved = json.loads(meta.read_text())
        if (
            saved["config"] != config
            or saved["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("rich shard mismatch")
        return saved
    start = time.time()
    rows = generate(puzzles, games, namespace)
    write_rich(path, rows)
    record = {
        "file": path.name,
        "config": config,
        "samples": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seconds": time.time() - start,
    }
    meta.write_text(json.dumps(record, indent=2) + "\n")
    return record


def build(args):
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    signature = source_signature()
    marker = root / "generator.json"
    if marker.exists() and json.loads(marker.read_text()) != signature:
        raise ValueError("generator changed; use fresh directory")
    if not marker.exists() and any(root.glob("*.npz")):
        raise ValueError("unversioned shards")
    marker.write_text(json.dumps(signature, indent=2) + "\n")
    jobs = [(str(root), "train-puzzle", i, 900 + i, 1000, 0) for i in range(50)]
    jobs += [(str(root), "train-game", i, 1100 + i, 0, 32) for i in range(64)]
    jobs += [(str(root), "valid-puzzle", i, 1500 + i, 1000, 0) for i in range(2)]
    jobs += [(str(root), "valid-game", i, 1550 + i, 0, 32) for i in range(3)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(job, jobs):
            records.append(record)
            print(
                json.dumps(
                    {
                        "completed": len(records),
                        "shards": len(jobs),
                        "file": record["file"],
                        "samples": record["samples"],
                    }
                ),
                flush=True,
            )
    manifest = {
        "shards": records,
        "source": signature,
        "features": "huxi55",
        "huxi_target": "formed scoring groups + twenty own-draw structural hu-xi; absent=-1",
        "value_target": "readiness warmup; replaced by terminal settlement in RL",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def transfer(path):
    old = load_model(path)
    new = PolicyValueNet(old.architecture, old.width, "huxi")
    oldstate = old.state_dict()
    state = new.state_dict()
    for key, value in oldstate.items():
        if key in state and state[key].shape == value.shape:
            state[key] = value
    if old.architecture == "resnet":
        state["encoder.0.weight"].zero_()
        state["encoder.0.weight"][:, : old.input_channels] = oldstate["encoder.0.weight"]
    else:
        raise ValueError("hu-xi migration currently supports resnet")
    new.load_state_dict(state)
    return new


def huxi_loss(pred, target):
    regression = torch.nn.functional.smooth_l1_loss(pred, target)
    eligibility = torch.nn.functional.binary_cross_entropy_with_logits(
        (pred[:, 1:] * 60 - 15) / 3, (target[:, 1:] * 60 >= 15).float()
    )
    return regression + 0.05 * eligibility


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(55)
    root = Path(args.data)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source"] != source_signature():
        raise ValueError("generator signature mismatch")
    training = []
    valid = []
    for record in manifest["shards"]:
        path = root / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("corrupt rich shard")
        rows = read_rich(path)
        if len(rows) != record["samples"]:
            raise ValueError("sample mismatch")
        (training if record["config"]["kind"].startswith("train") else valid).extend(rows)
    training, valid = deduplicate(training, valid)
    if len(training) < 100000:
        raise ValueError("100000 distinct decisions required")
    model = transfer(args.resume)
    rng = np.random.default_rng(55)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = float("inf")
    logs = []
    print(json.dumps({"train": len(training), "valid": len(valid)}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(training))
        loss_sum = 0.0
        for start in range(0, len(order), 128):
            rows = [training[i] for i in order[start : start + 128]]
            x, a, m, y, v, w = batch(rows)
            hu = torch.tensor(np.stack([r[5] for r in rows]))
            logits, value, wait, h = model.forward_all(x, a, m)
            loss = (
                torch.nn.functional.cross_entropy(logits, y)
                + 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(value, v)
                + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(wait, w)
                + 0.5 * huxi_loss(h, hu)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite hu-xi loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(rows)
        metrics = {"epoch": epoch + 1, "loss": loss_sum / len(training), **evaluate(model, valid)}
        logs.append(metrics)
        if metrics["policy_cross_entropy"] < best:
            best = metrics["policy_cross_entropy"]
            save_model(
                model,
                args.output,
                value_target="readiness",
                huxi_target=manifest["huxi_target"],
                training_samples=len(training),
                epoch=epoch + 1,
            )
        print(json.dumps(metrics), flush=True)
    model = load_model(args.output)
    report = {
        "train": len(training),
        "validation": len(valid),
        "source": manifest["source"],
        "checkpoint": args.output,
        "sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "epochs": logs,
        "random_tournament": tournament(model, list(range(19000, 19050))),
        "teacher_tournament": tournament(model, list(range(19500, 19520)), opponent="teacher"),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "train"])
    p.add_argument("--data", default="data/generated/huxi100k")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--resume", default="checkpoints/scale100k-resnet.pt")
    p.add_argument("--output", default="checkpoints/huxi-warmup.pt")
    p.add_argument("--report", default="docs/training/huxi-warmup.json")
    args = p.parse_args()
    if args.workers < 1 or args.epochs < 1:
        p.error("workers and epochs must be positive")
    if args.command == "build":
        build(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
