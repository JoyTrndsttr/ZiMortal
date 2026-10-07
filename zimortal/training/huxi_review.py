"""Dataset regeneration audit and frozen human-review positions."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import RuleEngine
from zimortal.model.encoding import encode_action, encode_observation

from .huxi import auxiliary, read_rich, rich_example, source_signature
from .runtime import choose, load_model, predict, tournament
from .train import batch
from .valuation import action_scores


def data_audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source"] != source_signature():
        raise ValueError("source signature mismatch")
    namespaces = [r["config"]["namespace"] for r in manifest["shards"]]
    assert len(namespaces) == len(set(namespaces))
    records = [r for r in manifest["shards"] if r["config"]["kind"] == "train-game"]
    engine = RuleEngine()
    games = decisions = steps = puzzles = 0
    for record in records[::8]:
        namespace = record["config"]["namespace"]
        rows = read_rich(root / record["file"])
        rng = random.Random(namespace)
        cursor = 0
        for i in range(3):
            initial = engine.new_game(namespace * 100000 + i)
            state = initial
            while not state.terminal:
                actions = engine.legal_actions(state)
                if len(actions) > 1:
                    expected = rich_example(engine.observation(state, actions[0].player))
                    saved = rows[cursor]
                    assert expected[2:4] == saved[2:4]
                    for index in (0, 1, 4, 5):
                        np.testing.assert_array_equal(expected[index], saved[index])
                    cursor += 1
                    decisions += 1
                state = engine.step(state, rng.choice(actions))
                state.validate()
                steps += 1
            assert engine.replay(initial, state.history).serialize() == state.serialize()
            games += 1
    from .data import make_puzzle

    for record in [r for r in manifest["shards"] if r["config"]["kind"] == "train-puzzle"][::10]:
        rows = read_rich(root / record["file"])
        for i in range(10):
            expected = rich_example(make_puzzle(record["config"]["namespace"] * 100000 + i)[0])
            saved = rows[i]
            assert expected[2:4] == saved[2:4]
            for field in (0, 1, 4, 5):
                np.testing.assert_array_equal(expected[field], saved[field])
            puzzles += 1
    return {
        "games": games,
        "decisions": decisions,
        "steps": steps,
        "puzzles": puzzles,
        "source_verified": True,
        "seed_namespaces_disjoint": True,
        "input_huxi_and_policy_targets_verified": True,
        "conservation_and_replay_verified": True,
    }


def frozen_positions(checkpoints):
    torch.set_num_threads(2)
    engine = RuleEngine()
    parent = load_model("checkpoints/scale100k-resnet.pt")
    state = engine.new_game(18000, 0)
    models = {name: load_model(path) for name, path in checkpoints.items()}
    reports = []
    for step in range(1, 18):
        actions = engine.legal_actions(state)
        obs = engine.observation(state, actions[0].player)
        if step in (6, 17):
            record = {
                "seed": 18000,
                "step": step,
                "player": obs.player + 1,
                "position_from": "frozen original scale100k game",
                "teacher_scores": action_scores(obs).tolist(),
                "exact_huxi_targets": [float(x) * 60 for x in auxiliary(obs)[1]],
                "models": {},
            }
            for name, model in models.items():
                logits, value = predict(model, obs)
                selected = actions[int(logits.argmax())]
                huxi_prediction = None
                if model.feature_version == "huxi":
                    x = torch.tensor(encode_observation(obs, "huxi")[None])
                    a = torch.tensor(
                        np.stack([encode_action(action, obs.player) for action in actions])[None]
                    )
                    with torch.no_grad():
                        _, _, _, hu = model.forward_all(
                            x, a, torch.ones(a.shape[:2], dtype=torch.bool)
                        )
                    huxi_prediction = [float(v) * 60 for v in hu[0]]
                record["models"][name] = {
                    "action": selected.kind.value,
                    "tile": selected.tile,
                    "softmax": [float(x) for x in logits.softmax(-1)],
                    "value_raw": float(value),
                    "feature_version": model.feature_version,
                    "huxi_prediction": huxi_prediction,
                }
            reports.append(record)
        state = engine.step(state, choose(obs, random.Random(18000), parent))
    return reports


def benchmark():
    checkpoints = {
        "original": "checkpoints/scale100k-resnet.pt",
        "warmup": "checkpoints/huxi-warmup.pt",
        "settlement": "checkpoints/huxi-resnet.pt",
    }
    report = {
        "seeds_random": list(range(19000, 19050)),
        "seeds_teacher": list(range(19500, 19520)),
        "frozen_positions": frozen_positions(checkpoints),
        "models": {},
    }
    for name, path in checkpoints.items():
        model = load_model(path)
        report["models"][name] = {
            "checkpoint": path,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "random_tournament": tournament(model, report["seeds_random"]),
            "teacher_tournament": tournament(model, report["seeds_teacher"], opponent="teacher"),
        }
        print(name, report["models"][name], flush=True)
    return report


def diagnostics(directory):
    torch.set_num_threads(2)
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source"] != source_signature():
        raise ValueError("source mismatch")
    train_keys = set()
    valid = []
    valid_keys = set()
    for record in manifest["shards"]:
        for row in read_rich(root / record["file"]):
            key = hashlib.sha256(row[0].tobytes() + row[1].tobytes()).digest()
            if record["config"]["kind"].startswith("train"):
                train_keys.add(key)
            elif key not in train_keys and key not in valid_keys:
                valid.append(row)
                valid_keys.add(key)
    result = {"validation_samples": len(valid), "models": {}}
    for name, path in [
        ("original", "checkpoints/scale100k-resnet.pt"),
        ("warmup", "checkpoints/huxi-warmup.pt"),
        ("settlement", "checkpoints/huxi-resnet.pt"),
    ]:
        model = load_model(path)
        correct = 0
        formed_error = draw_error = threshold_correct = struct_count = 0.0
        with torch.no_grad():
            for start in range(0, len(valid), 128):
                rows = valid[start : start + 128]
                x, a, m, y, _v, _w = batch(rows)
                if model.feature_version == "legacy":
                    x = x[:, :45]
                logits, _value, _wait, h = model.forward_all(x, a, m)
                correct += int((logits.argmax(-1) == y).sum())
                if h is not None:
                    targets = torch.tensor(np.stack([r[5] for r in rows])) * 60
                    pred = h * 60
                    struct = targets[:, 1:] >= 0
                    formed_error += float((pred[:, 0] - targets[:, 0]).abs().sum())
                    draw_error += float((pred[:, 1:] - targets[:, 1:]).abs()[struct].sum())
                    threshold_correct += float(
                        ((pred[:, 1:] >= 15) == (targets[:, 1:] >= 15))[struct].sum()
                    )
                    struct_count += float(struct.sum())
        result["models"][name] = {"teacher_policy_accuracy": correct / len(valid)}
        if model.feature_version == "huxi":
            result["models"][name].update(
                formed_huxi_mae=formed_error / len(valid),
                draw_huxi_mae_on_structural_tiles=draw_error / max(1, struct_count),
                fifteen_threshold_accuracy_on_structural_tiles=threshold_correct
                / max(1, struct_count),
                structural_tile_samples=int(struct_count),
            )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["data", "benchmark", "diagnostics"])
    p.add_argument("--data", default="data/generated/huxi100k")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "data":
        report = data_audit(args.data)
    elif args.command == "diagnostics":
        report = diagnostics(args.data)
    else:
        report = benchmark()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report) if args.command == "data" else "benchmark saved", flush=True)


if __name__ == "__main__":
    main()
