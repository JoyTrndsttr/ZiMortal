"""Fixed-update 8k/16k MC+CQL experiment, then independent cash screening and stop."""

import argparse
import copy
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from zimortal.model.mortal_faithful import MortalFaithfulQNet

from .active import atomic_json
from .mortal_data import audit_dataset, generate_dataset, load_rows
from .mortal_faithful_diagnostics import coverage
from .mortal_offline import BASELINE, q_loss, regression, tensors
from .mortal_sarsa import parameter_hash
from .mortal_support_experiment import evaluate, offline_support
from .mortal_v2 import game_split
from .online_rl import atomic_checkpoint, digest, paired_summary, source_hashes
from .runtime import load_model, save_model


class ShuffledBatches:
    """Full fixed-size batches, concatenating the tail with the next shuffled cycle."""

    def __init__(self, size, seed=1414):
        if size < 1:
            raise ValueError("nonempty training set required")
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.order = self.rng.permutation(size)
        self.position = 0
        self.cycles = 1

    def next(self, count):
        if count < 1:
            raise ValueError("positive batch size required")
        parts = []
        remaining = count
        while remaining:
            take = min(remaining, self.size - self.position)
            parts.append(self.order[self.position : self.position + take])
            self.position += take
            remaining -= take
            if self.position == self.size:
                self.order = self.rng.permutation(self.size)
                self.position = 0
                self.cycles += 1
        return np.concatenate(parts)

    def state(self):
        return {
            "rng": self.rng.bit_generator.state,
            "order": self.order.tolist(),
            "position": self.position,
            "cycles": self.cycles,
        }


def train_fixed(initial, rows, dev, directory, *, updates=5200, interval=260, device="cuda"):
    if updates < 1 or interval < 1 or updates % interval:
        raise ValueError("update budget must be divisible by dev interval")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sampler = ShuffledBatches(len(rows))
    initial_hash = parameter_hash(model)
    best = float("inf")
    best_update = 0
    logs = []
    losses = []
    cql_losses = []
    norms = []
    order_hash = hashlib.sha256()
    for update in range(1, updates + 1):
        model.train()
        indices = sampler.next(256)
        order_hash.update(indices.astype(np.int64).tobytes())
        x, a, m, chosen, target = tensors([rows[i] for i in indices], device, 1.0)
        q = model.forward_q(x, a, m)[0]
        loss, parts = q_loss(q, chosen, target, m, 0.1, "mse")
        if not torch.isfinite(loss) or not torch.isfinite(q[m]).all():
            raise RuntimeError("nonfinite fixed-budget loss/Q")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite gradient")
        optimizer.step()
        losses.append(float(parts["mc_loss"].detach()))
        cql_losses.append(float(parts["cql_loss"].detach()))
        norms.append(float(norm))
        if update % interval:
            continue
        train = regression(model, rows, device=device)
        development = regression(model, dev, device=device)
        log = {
            "update": update,
            "examples_seen": update * 256,
            "batch_order_sha256": order_hash.hexdigest(),
            "completed_cycles": sampler.cycles - 1,
            "half_mse": float(np.mean(losses)),
            "cql_loss": float(np.mean(cql_losses)),
            "gradient_norm_mean": float(np.mean(norms)),
            "training": train,
            "development": development,
            "train_dev_rmse_gap": development["rmse"] - train["rmse"],
        }
        logs.append(log)
        atomic_json(root / "updates.json", logs)
        checkpoint = root / f"update-{update:04d}.pt"
        save_model(
            model,
            checkpoint,
            target="terminal payments[seat]/100",
            gamma=1.0,
            alpha_cql=0.1,
            regression_loss="mse",
            updates=update,
            learning_rate=3e-4,
        )
        atomic_checkpoint(
            root / f"update-{update:04d}-optimizer.pt",
            {
                "optimizer": optimizer.state_dict(),
                "updates": update,
                "checkpoint_sha256": digest(checkpoint),
                "sampler": sampler.state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            },
        )
        if development["rmse"] < best:
            best = development["rmse"]
            best_update = update
            shutil.copyfile(checkpoint, root / "best-dev.pt")
        atomic_json(
            root.parent / "progress.json",
            {
                "status": "fixed_update_training",
                "variant": root.name,
                "updates": update,
                "target": updates,
                "best_update": best_update,
            },
        )
        losses = []
        cql_losses = []
        norms = []
    selected = load_model(root / "best-dev.pt", device)
    result = {
        "initial_parameter_sha256": initial_hash,
        "updates": updates,
        "examples_seen": updates * 256,
        "best_update": best_update,
        "dev_checks": len(logs),
        "training_rows": len(rows),
        "selected_train": regression(selected, rows, device=device),
        "selected_dev": regression(selected, dev, device=device),
        "logs": logs,
    }
    atomic_json(root / "result.json", result)
    return result


def select_common_update(fits):
    schedules = [[e["update"] for e in f["logs"]] for f in fits.values()]
    if not schedules or any(v != schedules[0] for v in schedules[1:]):
        raise ValueError("common dev evaluation schedule required")
    return min(
        schedules[0],
        key=lambda u: (
            np.mean(
                [
                    next(e["development"]["rmse"] for e in f["logs"] if e["update"] == u)
                    for f in fits.values()
                ]
            ),
            u,
        ),
    )


def run(directory, data_source=None):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    old = Path("data/generated/mortal-v2/data")
    original = json.loads((old.parent / "config.json").read_text())
    if digest(BASELINE) != original["baseline_sha256"]:
        raise RuntimeError("baseline changed")
    original_faithful = json.loads(
        Path("data/generated/mortal-faithful-v1/config.json").read_text()
    )
    for p, sha in original_faithful["source_hashes"].items():
        if p.startswith(("zimortal/engine/", "zimortal/model/")) and digest(p) != sha:
            raise RuntimeError("engine/model representation changed")
    files = {BASELINE: digest(BASELINE), str(old / "manifest.json"): digest(old / "manifest.json")}
    config = {
        "data_start": 1080000000,
        "new_games": 10000,
        "cash_start": 1090000000,
        "cash_seeds": 500,
        "family_config": original["family_config"],
        "updates": 5200,
        "dev_interval": 260,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "alpha": 0.1,
        "support": 0.1,
        "optimizer": "AdamW defaults: betas(.9,.999), eps1e-8, weight_decay .01",
        "clip": 1.0,
        "gamma": 1.0,
        "target": "terminal MC cash/100",
        "loss": "0.5 MSE+CQL",
        "selection": "one common update minimizes mean A/B new-dev RMSE; same selected updates, equal 5200 budget, 20 dev checks; no test selection",
        "source_hashes": source_hashes(),
        "inputs_sha256": files,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "test_once_after_checkpoint_lock": True,
        "no_new_losses": True,
    }
    atomic_json(root / "config.json", config)
    baseline = load_model(BASELINE, "cuda").requires_grad_(False)
    data = Path(data_source) if data_source else root / "data"
    if data_source:
        manifest = json.loads((data / "manifest.json").read_text())
        metadata = manifest["metadata"]
        if metadata["family_config"] != original["family_config"] or metadata["inputs_sha256"][
            BASELINE
        ] != digest(BASELINE):
            raise RuntimeError("reused data behavior differs")
        for path, sha in metadata["source_hashes"].items():
            generator_file = path.startswith(("zimortal/engine/", "zimortal/model/")) or path in (
                "zimortal/training/mortal_data.py",
                "zimortal/training/rollout_engine.py",
            )
            if generator_file and digest(path) != sha:
                raise RuntimeError("reused data generator/representation differs")
    else:
        manifest = generate_dataset(
            baseline,
            data,
            games=10000,
            start=1080000000,
            shard_games=250,
            device="cuda",
            slots=32,
            metadata=config,
            family_config=original["family_config"],
        )
    if manifest["failures"] or sum(shard["games"] for shard in manifest["shards"]) != 10000:
        raise RuntimeError("require 10000 complete games without replacements")
    files[str(data / "manifest.json")] = digest(data / "manifest.json")
    config["data_source"] = str(data)
    atomic_json(root / "config.json", config)
    atomic_json(root / "data-audit.json", audit_dataset(data, manifest))
    old_manifest = json.loads((old / "manifest.json").read_text())
    old_train = game_split(load_rows(old, old_manifest))[0]
    new_train, dev, test = game_split(load_rows(data, manifest))
    seeds = {int(r["seed"]) for r in old_train}
    if seeds & {int(r["seed"]) for group in [new_train, dev, test] for r in group}:
        raise RuntimeError("old/new seed overlap")
    for group, remainder in [(new_train, 2), (dev, 1), (test, 0)]:
        expected = {
            s
            for s in range(1080000000, 1080010000)
            if (s % 10 >= 2 if remainder == 2 else s % 10 == remainder)
        }
        # Some legal games have no multichoice rows; count split games from all replays.
        actual = {
            int(g["seed"])
            for shard in manifest["shards"]
            for g in json.loads(gzip.decompress((data / shard["replays"]).read_bytes()))
            if (int(g["seed"]) % 10 >= 2 if remainder == 2 else int(g["seed"]) % 10 == remainder)
        }
        if actual != expected:
            raise RuntimeError("new complete-game split counts differ")
    groups = {"A-8k": old_train, "B-16k": old_train + new_train}
    atomic_json(root / "training-coverage.json", {n: coverage(rows) for n, rows in groups.items()})
    torch.manual_seed(1414)
    initial = MortalFaithfulQNet().cuda()
    fits = {n: train_fixed(initial, rows, dev, root / n) for n, rows in groups.items()}
    if len({f["initial_parameter_sha256"] for f in fits.values()}) != 1:
        raise RuntimeError("initialization mismatch")
    common_update = select_common_update(fits)
    for n in groups:
        src = root / n / f"update-{common_update:04d}.pt"
        shutil.copyfile(src, root / n / "selected-common.pt")
        shutil.copyfile(
            root / n / f"update-{common_update:04d}-optimizer.pt",
            root / n / "selected-common-optimizer.pt",
        )
    lock = {
        "common_update": common_update,
        "checkpoints": {n: digest(root / n / "selected-common.pt") for n in groups},
    }
    atomic_json(root / "selection-lock.json", lock)
    models = {n: load_model(root / n / "selected-common.pt", "cuda") for n in groups}
    for n, m in models.items():
        fits[n]["common_selected_update"] = common_update
        fits[n]["common_selected_train"] = regression(m, groups[n], device="cuda")
        fits[n]["common_selected_dev"] = regression(m, dev, device="cuda")
    offline = {
        n: {
            "terminal_test": regression(m, test, device="cuda"),
            "raw": offline_support(m, test, None),
            "guarded": offline_support(m, test, 0.1),
        }
        for n, m in models.items()
    }
    atomic_json(root / "offline-test.json", offline)
    atomic_json(root / "heldout-coverage.json", {"dev": coverage(dev), "test": coverage(test)})
    specs = [(s, s % 3, seat, 0) for s in range(1090000000, 1090000500) for seat in range(3)]
    policies = {
        "baseline": (None, None),
        **{n + "-raw": (m, None) for n, m in models.items()},
        **{n + "-guarded": (m, 0.1) for n, m in models.items()},
    }
    screening, outcomes = evaluate(
        root / "screening", policies, baseline, original["family_config"], specs
    )
    direct = {
        mode: paired_summary(outcomes["B-16k-" + mode], outcomes["A-8k-" + mode], 1414)
        for mode in ["raw", "guarded"]
    }
    for p, sha in {**files, **config["source_hashes"]}.items():
        if digest(p) != sha:
            raise RuntimeError("frozen source/input changed")
    atomic_json(
        root / "result.json",
        {
            "status": "complete_20k_stop",
            "fits": fits,
            "offline": offline,
            "screening": screening,
            "B_minus_A": direct,
            "online_started": False,
            "data_expanded_to50k": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-volume20k-v2")
    parser.add_argument("--data-source")
    args = parser.parse_args()
    run(args.directory, args.data_source)
