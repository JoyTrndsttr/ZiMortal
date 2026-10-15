"""Finite dev-locked extraction diagnostics. No training, data generation, or online."""

import argparse
import gzip
import json
from pathlib import Path

import torch

from .active import atomic_json
from .mortal_credit import audit_screening
from .mortal_extraction import ExtractionPolicy, summarize_families
from .mortal_offline import BASELINE
from .mortal_sarsa import parameter_hash
from .online_rl import digest, paired_summary, play, source_hashes
from .runtime import load_model

CHECKPOINT = "data/generated/mortal-coverage-v1/B-coverage/fit/selected-common.pt"
FAMILY_RULES = {
    "C-discard": {"mode": "family", "discard": True, "chi": False},
    "C-chi": {"mode": "family", "discard": False, "chi": True},
    "C-both": {"mode": "family", "discard": True, "chi": True},
}


def candidates():
    return {
        "baseline": {"mode": "baseline"},
        "A-raw": {"mode": "raw"},
        "B-support": {"mode": "support"},
        "F-hu-parent-control": {"mode": "family", "discard": False, "chi": False},
        **FAMILY_RULES,
        **{f"D-lambda-{v}": {"mode": "anchor", "lambda": v} for v in (.01, .03, .1, .3)},
        **{f"E-top-{d}-{c}": {"mode": "top", "discard_k": d, "chi_k": c} for d in (2, 3) for c in (2, 3)},
    }


def choose_development(outcomes):
    def best(prefix):
        names = sorted(n for n in outcomes if n.startswith(prefix))
        if not names:
            raise ValueError("missing development candidates")
        return max(names, key=lambda n: (sum(r["cash"] for r in outcomes[n]) / len(outcomes[n]), -names.index(n)))
    return {"C-family": best("C-"), "D-anchor": best("D-"), "E-top": best("E-")}


def evaluation(root, specs, settings, q, parent, behavior, frozen):
    root.mkdir(parents=True, exist_ok=False)
    outcomes, diagnostics = {}, {}
    state_hashes = {"q": parameter_hash(q), "parent": parameter_hash(parent)}
    for name, rule in settings.items():
        policy = ExtractionPolicy(q, parent, behavior, rule).eval()
        records = []
        atomic_json(root / "progress.json", {"policy": name, "games": len(specs), "mode": "fixed-Q evaluation only"})
        with gzip.open(root / f"screen-{name}-replays.jsonl.gz", "wt") as replay, gzip.open(root / f"decisions-{name}.jsonl.gz", "wt") as decisions:
            def sink(e):
                replay.write(json.dumps(e) + "\n")
            def decision_sink(batch, records=records):
                for r in batch:
                    decisions.write(json.dumps(r) + "\n")
                    records.append({k: v for k, v in r.items() if k not in ("q", "scores", "probabilities", "actions")})
            _, games, errors = play(
                policy, [parent], specs, device="cuda", slots=32, training=False,
                replay_sink=sink, decision_recorder=policy.record, decision_sink=decision_sink,
            )
        if errors or len(games) != len(specs):
            atomic_json(root / "failures.json", errors)
            raise RuntimeError("incomplete independent evaluation; no excluded pairs")
        outcomes[name] = games
        diagnostics[name] = summarize_families(records)
        atomic_json(root / "screening-outcomes.json", outcomes)
        atomic_json(root / "diagnostics.json", diagnostics)
        if parameter_hash(q) != state_hashes["q"] or parameter_hash(parent) != state_hashes["parent"]:
            raise RuntimeError("frozen weights or buffers changed")
        for path, sha in frozen.items():
            if digest(path) != sha:
                raise RuntimeError("frozen code/checkpoint changed")
    result = {
        "network_state_sha256": state_hashes,
        "audit": audit_screening(root),
        "comparisons": {n: paired_summary(g, outcomes["baseline"], 1414) for n, g in outcomes.items()},
        "diagnostics": diagnostics,
    }
    atomic_json(root / "result.json", result)
    return result, outcomes


def run(directory):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    original = json.loads(Path("data/generated/mortal-v2/config.json").read_text())
    lock = json.loads(Path("data/generated/mortal-coverage-v1/selection-lock.json").read_text())
    if digest(CHECKPOINT) != lock["checkpoints"]["B-coverage"] or digest(BASELINE) != original["baseline_sha256"]:
        raise RuntimeError("checkpoint/baseline changed")
    previous = json.loads(Path("data/generated/mortal-coverage-v1/config.json").read_text())
    for p, sha in previous["source_hashes"].items():
        if p.startswith(("zimortal/engine/", "zimortal/model/")) and digest(p) != sha:
            raise RuntimeError("rule engine or model changed")
    all_settings = candidates()
    frozen = source_hashes() | {CHECKPOINT: digest(CHECKPOINT), BASELINE: digest(BASELINE)}
    config = {
        "checkpoint": CHECKPOINT, "frozen_sha256": frozen,
        "behavior": original["family_config"], "candidates": all_settings,
        "dev_start": 1130000000, "dev_seeds": 200,
        "test_start": 1140000000, "test_seeds": 500,
        "selection": "maximum dev mean cash per C/D/E class, lexical name tie-break; lock before test",
        "epsilon": 1e-8, "Q_unit": "cash/100; lambda same units",
        "test_ablation": ["C-discard", "C-chi", "C-both"],
        "training_updates": 0, "dataset_changes": False, "online_started": False,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(root / "config.json", config)
    parent = load_model(BASELINE, "cuda").requires_grad_(False).eval()
    q = load_model(CHECKPOINT, "cuda").requires_grad_(False).eval()
    dev_specs = [(s, s % 3, seat, 0) for s in range(1130000000, 1130000200) for seat in range(3)]
    test_specs = [(s, s % 3, seat, 0) for s in range(1140000000, 1140000500) for seat in range(3)]
    atomic_json(root / "seed-lock.json", {"dev": dev_specs, "test": test_specs})
    development, dev_outcomes = evaluation(root / "dev", dev_specs, all_settings, q, parent, config["behavior"], frozen)
    selected = choose_development(dev_outcomes)
    test_names = ["baseline", "A-raw", "B-support", "F-hu-parent-control", *FAMILY_RULES, selected["D-anchor"], selected["E-top"]]
    test_settings = {n: all_settings[n] for n in test_names}
    atomic_json(root / "selection-lock.json", {"selected": selected, "test_settings": test_settings, "checkpoint_sha256": digest(CHECKPOINT), "seed_lock_sha256": digest(root / "seed-lock.json")})
    test, test_outcomes = evaluation(root / "test", test_specs, test_settings, q, parent, config["behavior"], frozen)
    comparisons = {n: {
        "vs_raw": paired_summary(g, test_outcomes["A-raw"], 1414),
        "vs_support": paired_summary(g, test_outcomes["B-support"], 1414),
    } for n, g in test_outcomes.items()}
    ablation = {
        "hu_parent_only_vs_raw": paired_summary(test_outcomes["F-hu-parent-control"], test_outcomes["A-raw"], 1414),
        "discard_only_vs_control": paired_summary(test_outcomes["C-discard"], test_outcomes["F-hu-parent-control"], 1414),
        "chi_only_vs_control": paired_summary(test_outcomes["C-chi"], test_outcomes["F-hu-parent-control"], 1414),
        "both_vs_control": paired_summary(test_outcomes["C-both"], test_outcomes["F-hu-parent-control"], 1414),
        "discard_effect_given_chi": paired_summary(test_outcomes["C-both"], test_outcomes["C-chi"], 1414),
        "chi_effect_given_discard": paired_summary(test_outcomes["C-both"], test_outcomes["C-discard"], 1414),
    }
    atomic_json(root / "result.json", {
        "status": "complete_fixed_Q_stop", "selected": selected, "development": development,
        "test": test, "comparisons": comparisons, "family_ablation": ablation,
        "training_updates": 0, "online_started": False, "baseline_replaced": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-extraction-v1")
    run(parser.parse_args().directory)
