"""Finite Mortal-faithful-v1 2x2 offline factorial; no online launch."""

import json
from pathlib import Path

import torch

from zimortal.model.mortal_faithful import MortalFaithfulQNet
from zimortal.model.mortal_q import MortalQNet

from .active import atomic_json
from .mortal_data import audit_dataset, load_rows
from .mortal_offline import BASELINE, fit, regression
from .mortal_support_experiment import evaluate, offline_support
from .mortal_v2 import game_split
from .online_rl import digest, paired_summary, source_hashes
from .runtime import load_model

ANCHOR = "data/generated/mortal-support-v1/cql01/best-dev.pt"


def run():
    torch.set_num_threads(2)
    torch.manual_seed(1414)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    root = Path("data/generated/mortal-faithful-v1")
    root.mkdir(parents=True, exist_ok=False)
    data = Path("data/generated/mortal-v2/data")
    original = json.loads((data.parent / "config.json").read_text())
    if digest(BASELINE) != original["baseline_sha256"]:
        raise RuntimeError("frozen parent changed")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provenance = source_hashes()
    files = [BASELINE, ANCHOR, str(manifest_path)]
    checksums = {p: digest(p) for p in files}
    protocol = {
        "name": "Mortal-faithful-v1",
        "source_hashes": provenance,
        "inputs_sha256": checksums,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "small": {"width": 32, "blocks": 3, "latent": 256},
        "faithful": {"width": 128, "blocks": 12, "latent": 1024},
        "regression_loss": "0.5 MSE",
        "alpha_cql": 0.1,
        "gamma": 1,
        "default_target": "terminal cash MC",
        "comparison_target": "3-step actual-action SARSA",
        "epochs": 20,
        "patience": 4,
        "batch_size": 256,
        "seed": 1414,
        "optimizer": "AdamW lr3e-4 weight_decay=.01",
        "target_sync_updates": 250,
        "selection": "same dev terminal-cash RMSE; no test hyperparameter selection",
        "cash_start": 1060000000,
        "cash_seeds": 500,
        "seats": 3,
        "primary_evaluation": "raw legal Q argmax",
        "support_guard_threshold": 0.1,
        "online_design_gate": "faithful MC guarded cash minus current anchor lower paired CI95 >=0; raw result must also be reported; no automatic online",
        "no_pairwise_no_extra_losses": True,
    }
    atomic_json(root / "config.json", protocol)
    atomic_json(root / "data-audit.json", audit_dataset(data, manifest))
    train, dev, test = game_split(load_rows(data, manifest))
    baseline = load_model(BASELINE, "cuda").requires_grad_(False)
    small = MortalQNet().cuda()
    torch.manual_seed(1414)
    faithful = MortalFaithfulQNet().cuda()
    fits = {}
    for architecture, initial in [("small", small), ("faithful", faithful)]:
        for target, horizon in [("mc", 0), ("sarsa3", 3)]:
            name = architecture + "-" + target
            fits[name] = fit(
                initial,
                train,
                dev,
                test,
                root / name,
                device="cuda",
                alpha=0.1,
                n_step=horizon,
                epochs=20,
                patience=4,
                regression_loss="mse",
                evaluate_test=False,
            )
    models = {n: load_model(root / n / "best-dev.pt", "cuda") for n in fits}
    atomic_json(root / "selection-lock.json", {n: digest(root / n / "best-dev.pt") for n in models})
    order = [f["epochs"] for f in fits.values()]
    for epochs in order[1:]:
        if any(
            x["batch_order_sha256"] != y["batch_order_sha256"]
            for x, y in zip(order[0], epochs, strict=False)
        ):
            raise RuntimeError("factorial batch order differs")
    for architecture in ["small", "faithful"]:
        if (
            fits[architecture + "-mc"]["initial_parameter_sha256"]
            != fits[architecture + "-sarsa3"]["initial_parameter_sha256"]
        ):
            raise RuntimeError("within-architecture initialization differs")
    offline = {
        n: {
            "terminal_test": regression(m, test, device="cuda"),
            "raw": offline_support(m, test, None),
            "guarded": offline_support(m, test, 0.1),
        }
        for n, m in models.items()
    }
    atomic_json(root / "offline-test.json", offline)
    anchor = load_model(ANCHOR, "cuda").requires_grad_(False)
    policies = {"baseline": (None, None), "current-cql-support": (anchor, 0.1)}
    policies.update({n + "-raw": (m, None) for n, m in models.items()})
    policies.update({n + "-guarded": (m, 0.1) for n, m in models.items()})
    specs = [(s, s % 3, seat, 0) for s in range(1060000000, 1060000500) for seat in range(3)]
    screening, outcomes = evaluate(
        root / "screening", policies, baseline, original["family_config"], specs
    )
    anchor_comparisons = {
        n: paired_summary(games, outcomes["current-cql-support"], 1414)
        for n, games in outcomes.items()
    }
    gate = anchor_comparisons["faithful-mc-guarded"]["0"]["paired_seed_bootstrap_95"][0] >= 0
    for path, sha in {**checksums, **provenance}.items():
        if digest(path) != sha:
            raise RuntimeError("frozen input/source changed during experiment")
    atomic_json(
        root / "result.json",
        {
            "status": "complete_offline_stop",
            "protocol": protocol,
            "fits": fits,
            "offline": offline,
            "screening": screening,
            "versus_current_anchor": anchor_comparisons,
            "online_design_gate_passed": gate,
            "online_started": False,
        },
    )


if __name__ == "__main__":
    run()
