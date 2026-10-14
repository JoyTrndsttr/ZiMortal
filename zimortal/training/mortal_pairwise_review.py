"""Post-lock diagnostics for the finite pairwise experiment (never training)."""

import gzip
import json
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import GameState, RuleEngine
from zimortal.model.encoding import encode_action, encode_observation

from .active import atomic_json
from .mortal_counterfactual import correlation, ranks
from .mortal_offline import BASELINE
from .mortal_support import SupportPolicy
from .online_rl import digest, pack, paired_summary
from .runtime import load_model


def accuracy_gain(a, b, seed=1414):
    """Paired root accuracy with seed-cluster bootstrap; ties are excluded."""
    delta = np.asarray(a["rollout_delta"])
    valid = delta != 0
    difference = np.asarray(b["correct"], float) - np.asarray(a["correct"], float)
    seeds = np.asarray(a["seeds"])
    if a["seeds"] != b["seeds"] or a["rollout_delta"] != b["rollout_delta"]:
        raise ValueError("unmatched counterfactual tests")
    totals = np.array(
        [
            [difference[(seeds == s) & valid].sum(), ((seeds == s) & valid).sum()]
            for s in np.unique(seeds)
        ]
    )
    if not valid.any():
        return {"gain": None, "ci95": None}
    rng = np.random.default_rng(seed)
    sample = totals[rng.integers(len(totals), size=(4096, len(totals)))].sum(1)
    boot = sample[:, 0] / np.maximum(sample[:, 1], 1)
    return {
        "gain": float(difference[valid].mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def run():
    torch.set_num_threads(2)
    root = Path("data/generated/mortal-pairwise-v1")
    result = json.loads((root / "result.json").read_text())
    config_lock = json.loads((root / "config.json").read_text())
    if digest(BASELINE) != config_lock["baseline_sha256"]:
        raise RuntimeError("baseline changed")
    provenance = root / "source-provenance.json"
    source_checks = json.loads(provenance.read_text()) if provenance.exists() else {}
    for path, sha in source_checks.items():
        if digest(path) != sha:
            raise RuntimeError("experiment source changed during execution")
    lock = json.loads((root / "selection-lock.json").read_text())
    for name, sha in lock.items():
        if digest(root / name / "best-dev.pt") != sha:
            raise RuntimeError("checkpoint changed after selection")
    ranking = result["ranking"]
    gains = {
        family: accuracy_gain(ranking["A"]["test_pair"][family], ranking["B"]["test_pair"][family])
        for family in ranking["A"]["test_pair"]
    }
    for variant in ranking.values():
        for stats in variant["test_pair"].values():
            stats["spearman"] = correlation(
                ranks(stats["predicted_gap"]), ranks(stats["rollout_delta"])
            )
            gaps = np.asarray(stats["predicted_gap"])
            truth = np.asarray(stats["rollout_delta"])
            stats["margin_buckets"] = {}
            for lo, hi in [(0, 5), (5, 15), (15, float("inf"))]:
                take = (abs(gaps) >= lo) & (abs(gaps) < hi)
                stats["margin_buckets"][str(lo)] = {
                    "roots": int(take.sum()),
                    "mean_predicted_winner_advantage": float(
                        (np.sign(gaps[take]) * truth[take]).mean()
                    )
                    if take.any()
                    else None,
                }
    outcomes = json.loads((root / "screening/screening-outcomes.json").read_text())
    direct = paired_summary(outcomes["B"], outcomes["A"], 1414)
    epochs_a, epochs_b = result["fits"]["A"]["epochs"], result["fits"]["B"]["epochs"]
    hashes_match = all(
        a["batch_order_sha256"] == b["batch_order_sha256"]
        for a, b in zip(epochs_a, epochs_b, strict=False)
    )
    if not hashes_match or (
        result["fits"]["A"]["initial_parameter_sha256"]
        != result["fits"]["B"]["initial_parameter_sha256"]
    ):
        raise RuntimeError("A/B fairness violated")
    ids, seeds = {}, {}
    for split in ["train", "dev", "test"]:
        labels = json.loads((root / split / "labels.json").read_text())
        ids[split] = {r["id"] for r in labels}
        seeds[split] = {r["seed"] for r in labels}
    old = Path("data/generated/mortal-counterfactual-v1")
    old_labels = json.loads((old / "roots.json").read_text())
    if any(ids[s] & {r["id"] for r in old_labels} for s in ids):
        raise RuntimeError("old diagnostic roots reused")
    if any(seeds[a] & seeds[b] for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]):
        raise RuntimeError("split leakage")
    config = json.loads(Path("data/generated/mortal-v2/config.json").read_text())["family_config"]
    baseline = load_model(BASELINE, "cuda").requires_grad_(False)
    models = {n: load_model(root / n / "best-dev.pt", "cuda") for n in lock}
    from .mortal_pairwise import ranking as evaluate_ranking

    train_labels = json.loads((root / "train/labels.json").read_text())
    retained = []
    for i, label in enumerate(train_labels):
        if label["weight"] <= 0:
            continue
        saved = json.loads(gzip.decompress((root / "train" / f"root-{i:04d}.json.gz").read_bytes()))
        st = GameState.deserialize(saved["root"])
        ob = RuleEngine().observation(st, label["seat"])
        retained.append(
            {
                **label,
                "features": encode_observation(ob, "huxi"),
                "encoded_actions": np.stack(
                    [encode_action(a, ob.player) for a in ob.legal_actions]
                ),
            }
        )
    training_accuracy = {n: evaluate_ranking(m, retained) for n, m in models.items()}
    cases = []
    wanted = ["1040000339-2-45", "1040000207-1-46", "1040000031-0-26", "1040000435-2-32"]
    for item in old_labels:
        if item["id"] not in wanted:
            continue
        evidence = json.loads(gzip.decompress((old / item["evidence"]).read_bytes()))
        state = GameState.deserialize(evidence["root"])
        obs = RuleEngine().observation(state, item["seat"])
        x = encode_observation(obs, "huxi")
        actions = np.stack([encode_action(a, obs.player) for a in obs.legal_actions])
        case = {"id": item["id"], "rollout_delta": item["stats"]["paired_delta"], "models": {}}
        for name, model in models.items():
            with torch.no_grad():
                px, pa, pm = pack([x], [actions], "cuda")
                q = model.forward_q(px, pa, pm)[0][0].cpu().numpy()
                policy = SupportPolicy(model, baseline, config, 0.1).eval()
                scores = policy(px, pa, pm)
                # SupportPolicy follows the regular PolicyValueNet forward tuple contract.
                logits = scores[0] if isinstance(scores, tuple) else scores
                choice = int(logits[0].argmax())
            case["models"][name] = {
                "pair_gap_cash": float(q[item["chosen"]] - q[item["parent"]]) * 100,
                "fixed_original_pair": bool(q[item["parent"]] > q[item["chosen"]]),
                "deployment_chosen": choice,
                "original_parent": item["parent"],
                "original_bad": item["chosen"],
            }
        cases.append(case)
    review = {
        "training_pair_accuracy": training_accuracy,
        "accuracy_gains": gains,
        "cash_B_vs_A": direct,
        "same_main_batch_order": hashes_match,
        "known_eval_only_cases": cases,
        "ranking": ranking,
        "no_split_or_old_diagnostic_leakage": True,
    }
    atomic_json(root / "review.json", review)


if __name__ == "__main__":
    run()
