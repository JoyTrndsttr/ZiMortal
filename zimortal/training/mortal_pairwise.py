"""Bounded pairwise auxiliary experiment; hidden roots are rollout labels only."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from zimortal.engine import GameState, RuleEngine
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.model.mortal_q import MortalQNet

from .active import atomic_json
from .mortal_counterfactual import (
    Frequencies,
    correlation,
    paired_rollouts,
    paired_statistics,
    prepare,
    sample_roots,
)
from .mortal_data import load_rows, write_bytes
from .mortal_offline import BASELINE, fit, regression
from .mortal_support_experiment import evaluate
from .mortal_v2 import game_split
from .online_rl import digest, pack
from .runtime import load_model


def confidence_weight(stats):
    lo, hi = stats["paired_bootstrap_95"]
    delta = stats["paired_delta"]
    if lo <= 0 <= hi or abs(delta) < 1:
        return 0.0
    return min(abs(delta) / 20, 1) * max(0, 1 - 1.96 * stats["paired_standard_error"] / abs(delta))


def auxiliary_loss(model, rows, device):
    x, a, mask = pack([r["features"] for r in rows], [r["encoded_actions"] for r in rows], device)
    q = model.forward_q(x, a, mask)[0]
    idx = torch.arange(len(rows), device=device)
    parent = torch.tensor([r["parent"] for r in rows], device=device)
    selected = torch.tensor([r["chosen"] for r in rows], device=device)
    sign = torch.tensor([np.sign(r["stats"]["paired_delta"]) for r in rows], device=device)
    weight = torch.tensor([r["weight"] for r in rows], device=device)
    return (weight * F.softplus(-sign * (q[idx, selected] - q[idx, parent]) / 0.1)).mean()


def specs(start, count):
    return [(s, s % 3, seat, 0) for s in range(start, start + count) for seat in range(3)]


def collect(root, start, count, cap, teacher, baseline, config, frequencies, confirm=False):
    source = root / "source"
    evaluate(
        source / "screening",
        {"baseline": (None, None), "cql-support": (teacher, 0.1)},
        baseline,
        config,
        specs(start, count),
    )
    population, states = prepare(root, source, frequencies)
    samples = sample_roots(population, cap, start)
    atomic_json(root / "sampling.json", samples)
    results = []
    engine = RuleEngine()
    for i, row in enumerate(samples):
        state = GameState.deserialize(states[row["id"]])
        obs = engine.observation(state, row["seat"])
        seed = int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:4], "little")
        out, evidence = paired_rollouts(
            state,
            obs.legal_actions[row["parent"]],
            obs.legal_actions[row["chosen"]],
            baseline,
            count=32,
            seed=seed,
            device="cuda",
        )
        stats = paired_statistics(out, seed)
        weight = confidence_weight(stats)
        if confirm and weight:
            second, ev2 = paired_rollouts(
                state,
                obs.legal_actions[row["parent"]],
                obs.legal_actions[row["chosen"]],
                baseline,
                count=32,
                seed=seed ^ 0xA173BC29,
                device="cuda",
            )
            stats2 = paired_statistics(second, seed ^ 0xA173BC29)
            weight = (
                confidence_weight(stats2)
                if stats2["paired_delta"] * stats["paired_delta"] > 0
                else 0
            )
            evidence += ev2
            stats = stats2
        write_bytes(
            root / f"root-{i:04d}.json.gz",
            gzip.compress(
                json.dumps(
                    {"root": state.serialize(), "row": row, "stats": stats, "pairs": evidence}
                ).encode(),
                mtime=0,
            ),
        )
        result = {
            **row,
            "stats": stats,
            "weight": weight,
            "features": encode_observation(obs, "huxi"),
            "encoded_actions": np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
        }
        results.append(result)
        atomic_json(
            root / "progress.json",
            {
                "done": i + 1,
                "total": len(samples),
                "high_confidence": sum(r["weight"] > 0 for r in results),
            },
        )
    atomic_json(
        root / "labels.json",
        [{k: v for k, v in r.items() if k not in ("features", "encoded_actions")} for r in results],
    )
    return results


def ranking(model, rows):
    result = {}
    for family in ["all", "discard", "chi/pass", "peng/pass"]:
        group = rows if family == "all" else [r for r in rows if r["family"] == family]
        gaps, deltas = [], []
        with torch.no_grad():
            for r in group:
                x, a, m = pack([r["features"]], [r["encoded_actions"]], "cuda")
                q = model.forward_q(x, a, m)[0][0].cpu().numpy()
                gaps.append(float(q[r["chosen"]] - q[r["parent"]]) * 100)
                deltas.append(r["stats"]["paired_delta"])
        nonzero = np.asarray(deltas) != 0
        correct = np.sign(gaps) == np.sign(deltas)
        confident = np.asarray([r["weight"] > 0 for r in group])
        result[family] = {
            "roots": len(group),
            "nonzero": int(nonzero.sum()),
            "accuracy": float(correct[nonzero].mean()) if nonzero.any() else None,
            "confident_accuracy": float(correct[confident].mean()) if confident.any() else None,
            "pearson": correlation(gaps, deltas),
            "predicted_gap": gaps,
            "rollout_delta": deltas,
            "correct": correct.tolist(),
            "seeds": [r["seed"] for r in group],
        }
    return result


def run():
    torch.set_num_threads(2)
    torch.manual_seed(1414)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root = Path("data/generated/mortal-pairwise-v1")
    root.mkdir(parents=True, exist_ok=False)
    baseline = load_model(BASELINE, "cuda").requires_grad_(False)
    initial = MortalQNet().cuda()
    teacher_path = "data/generated/mortal-support-v1/cql01/best-dev.pt"
    teacher = load_model(teacher_path, "cuda").requires_grad_(False)
    data = Path("data/generated/mortal-v2/data")
    manifest = json.loads((data / "manifest.json").read_text())
    config = json.loads((data.parent / "config.json").read_text())["family_config"]
    atomic_json(
        root / "config.json",
        {
            "baseline_sha256": digest(BASELINE),
            "teacher_sha256": digest(teacher_path),
            "manifest_sha256": digest(data / "manifest.json"),
            "lambda": 0.05,
            "temperature_cash": 10,
            "support": 0.1,
            "alpha": 0.1,
            "train_seeds": [1050000000, 600],
            "dev_seeds": [1050100000, 200],
            "test_seeds": [1050200000, 200],
            "cash_seeds": [1050300000, 500],
            "definition": "Q(action, frozen-baseline continuation | fixed hidden root), not Q*",
            "selection": "same dev terminal RMSE early stopping; fixed auxiliary coefficient",
        },
    )
    train, dev, test = game_split(load_rows(data, manifest))
    frequencies = Frequencies(train)
    training = collect(
        root / "train", 1050000000, 600, 300, teacher, baseline, config, frequencies, True
    )
    development = collect(
        root / "dev", 1050100000, 200, 100, teacher, baseline, config, frequencies
    )
    pairs = [r for r in training if r["weight"] > 0]
    if not pairs:
        raise RuntimeError("no reliable training pairs")
    fits = {}
    for name, rows in [("A", None), ("B", pairs)]:
        fits[name] = fit(
            initial,
            train,
            dev,
            test,
            root / name,
            device="cuda",
            alpha=0.1,
            n_step=3,
            epochs=20,
            patience=4,
            evaluate_test=False,
            pair_rows=rows,
        )
    atomic_json(root / "selection-lock.json", {n: digest(root / n / "best-dev.pt") for n in fits})
    heldout = collect(root / "test", 1050200000, 200, 100, teacher, baseline, config, frequencies)
    models = {n: load_model(root / n / "best-dev.pt", "cuda") for n in fits}
    diagnostics = {
        n: {
            "dev_pair": ranking(m, development),
            "test_pair": ranking(m, heldout),
            "terminal_test": regression(m, test, device="cuda"),
        }
        for n, m in models.items()
    }
    atomic_json(root / "ranking.json", diagnostics)
    screening, _ = evaluate(
        root / "screening",
        {"baseline": (None, None), **{n: (m, 0.1) for n, m in models.items()}},
        baseline,
        config,
        specs(1050300000, 500),
    )
    atomic_json(
        root / "result.json",
        {
            "status": "complete_stop",
            "high_confidence_train": len(pairs),
            "sampled_train": len(training),
            "fits": fits,
            "ranking": diagnostics,
            "screening": screening,
        },
    )


if __name__ == "__main__":
    run()
