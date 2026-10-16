"""Read-only audit of the predeclared LR comparison; no new games or updates."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from zimortal.training.active import atomic_json
from zimortal.training.mortal_boltzmann import sampling_probability
from zimortal.training.mortal_data import load_rows
from zimortal.training.mortal_extraction_experiment import CHECKPOINT
from zimortal.training.mortal_lr_horizon import (
    bn_change,
    buffer_snapshot,
    optimizer_dynamics,
    q_dynamics,
    schedule,
    seed_lock,
)
from zimortal.training.mortal_online_long import parameter_change, q_probe
from zimortal.training.mortal_sarsa import parameter_hash
from zimortal.training.online_rl import digest
from zimortal.training.runtime import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=Path, default=Path("data/generated/mortal-lr-horizon-v1"))
parser.add_argument("--q-device", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()
root = args.directory
torch.set_num_threads(2)
if json.loads((root / "progress.json").read_text())["phase"] != "complete_stop":
    raise RuntimeError("audit requires complete fixed experiment")
config = json.loads((root / "config.json").read_text())
assert json.loads((root / "seed-lock.json").read_text()) == json.loads(json.dumps(seed_lock()))
for p, sha in config["frozen_sha256"].items():
    assert digest(p) == sha, p
for p, sha in config["schedule"]["history_sha256"].items():
    assert digest(p) == sha, p
assert (
    config["schedule"]["expected_updates"] == 1630
    and config["schedule"]["B_cosine_updates"] == 1609
)
rngs = {a: np.random.default_rng(20261015) for a in ("A", "B")}
updates = dict.fromkeys(rngs, 0)
previous = dict.fromkeys(rngs, CHECKPOINT)
review = []
previous_torch_rng = {}
for generation in range(1, 11):
    gen = root / f"gen-{generation:02d}"
    arms = {}
    root_hashes = {}
    probe_lock = json.loads((gen / "probe-lock.json").read_text())
    assert probe_lock["manifest_sha256"] == digest(gen / "A/data/manifest.json")
    a_rows = load_rows(gen / "A/data", json.loads((gen / "A/data/manifest.json").read_text()))
    indices = np.linspace(0, len(a_rows) - 1, min(4096, len(a_rows)), dtype=int).tolist()
    assert indices == probe_lock["A_row_indices"]
    probe = [a_rows[i] for i in indices]
    del a_rows
    for arm, rng in rngs.items():
        directory = gen / arm
        result = json.loads((directory / "training-result.json").read_text())
        manifest = json.loads((directory / "data/manifest.json").read_text())
        rows = load_rows(directory / "data", manifest)
        logs = json.loads((directory / "updates.json").read_text())
        assert len(rows) == result["decisions"] == sum(l["batch"] for l in logs)
        assert len(logs) == math.ceil(len(rows) / 256)
        rng.permutation(len(rows))
        length = 300 if arm == "A" else 1609
        for l in logs:
            assert l["update"] == updates[arm] + 1
            assert l["lr"] == schedule(updates[arm], length)
            assert l["cql_loss"] == l["auxiliary_loss"] == 0
            updates[arm] += 1
        opt = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=False)
        assert opt["updates"] == updates[arm] == result["updates"]
        assert opt["numpy_rng"] == rng.bit_generator.state
        if arm in previous_torch_rng:
            cpu_rng, cuda_rng = previous_torch_rng[arm]
            # This fixed architecture has no stochastic Torch layers; only NumPy sampling/shuffle consumes RNG.
            assert torch.equal(cpu_rng, opt["torch_rng"])
            assert len(cuda_rng) == len(opt["cuda_rng"])
            assert all(torch.equal(a, b) for a, b in zip(cuda_rng, opt["cuda_rng"], strict=True))
        previous_torch_rng[arm] = (opt["torch_rng"], opt["cuda_rng"])
        assert opt["schedule"]["cosine_updates"] == length
        assert (
            opt["checkpoint_sha256"]
            == digest(directory / "checkpoint.pt")
            == result["checkpoint_sha256"]
        )
        assert digest(directory / "optimizer.pt") == result["optimizer_sha256"]
        assert all(int(s["step"]) == updates[arm] for s in opt["optimizer"]["state"].values())
        assert all(
            g["lr"] == schedule(updates[arm] - 1, length) for g in opt["optimizer"]["param_groups"]
        )
        assert opt["metadata"]["manifest_sha256"] == digest(directory / "data/manifest.json")
        assert opt["metadata"]["seed_lock_sha256"] == digest(root / "seed-lock.json")
        producer = load_model(previous[arm], "cpu")
        current = load_model(directory / "checkpoint.pt", "cpu")
        assert parameter_hash(producer) == result["producer_state_sha256"]
        assert parameter_hash(current) == result["updated_state_sha256"]
        parameters = {n: p.detach().clone() for n, p in producer.named_parameters()}
        change = parameter_change(parameters, current)
        for k, v in change.items():
            assert np.isclose(result["parameter_change"][k], v, rtol=1e-10)
        assert bn_change(buffer_snapshot(producer), current) == result["bn_change"]
        norms = np.asarray([l["gradient_norm"] for l in logs])
        assert result["optimizer_dynamics"]["clip_fraction"] == float(np.mean(norms > 1))

        # Recompute all update statistics using the loaded optimizer state.
        class State:
            state = opt["optimizer"]["state"]

        dynamics = optimizer_dynamics(logs, State())
        assert all(result["optimizer_dynamics"][k] == v for k, v in dynamics.items())
        actual = []
        generators = {}
        family = Counter()
        own_anchor = 0
        for file in sorted(directory.glob("evidence-*/decisions-*.jsonl.gz")):
            with gzip.open(file, "rt") as stream:
                for line in stream:
                    r = json.loads(line)
                    key = (r["seed"], r["seat"])
                    q = np.asarray(r["q"], np.float32)
                    ref = np.asarray(r["probabilities"], np.float32)
                    score = q + np.float32(0.3) * np.log(ref + np.float32(1e-8))
                    np.testing.assert_allclose(r["scores"], score, atol=1e-6)
                    p = sampling_probability(
                        np.array(r["scores"]),
                        np.ones(len(q), bool),
                        conservative_hu=r["family"] == "hu/pass",
                    )
                    np.testing.assert_allclose(r["actor_probabilities"], p, atol=1e-12)
                    assert np.isclose(r["actual_behavior_probability"], p[r["chosen"]])
                    if r["family"] == "hu/pass":
                        assert r["chosen"] == int(p.argmax())
                    else:
                        if key not in generators:
                            generators[key] = np.random.default_rng(
                                np.random.SeedSequence([*key, 20261016])
                            )
                        assert r["chosen"] == int(generators[key].choice(len(p), p=p))
                    if r["changed"]:
                        family[r["family"]] += 1
                    own_anchor += r["chosen"] != int(np.argmax(r["scores"]))
                    actual.append((r["seed"], r["seat"], r["step"] - 1, r["chosen"]))
        trained = [
            (int(r["seed"]), int(r["seat"]), int(r["history_step"]), int(r["chosen"])) for r in rows
        ]
        assert actual == trained
        # Same initial actor, deals, sampling RNG implies exact Gen1 real trajectories/rows.
        if generation == 1:
            # gzip/npz timestamps differ: compare semantic actual rows instead of archive bytes.
            root_hashes[arm] = actual
        previous[arm] = str(directory / "checkpoint.pt")
        arms[arm] = {
            "rows_once": len(rows),
            "updates": updates[arm],
            "post_warmup_floor_this_generation": dynamics["post_warmup_floor_updates"],
            "actual_draws_verified": True,
            "Adam_and_shuffle_RNG_continuous": True,
            "BN_and_parameter_changes_independently_verified": True,
            "deviations_by_family": dict(family),
            "vs_own_anchor_deviations": own_anchor,
            "vs_own_anchor_deviation_rate": own_anchor / len(rows),
        }
        producer.to(args.q_device)
        current.to(args.q_device)
        verified_q = q_dynamics(
            q_probe(producer, probe, args.q_device), q_probe(current, probe, args.q_device), probe
        )
        for key, value in verified_q.items():
            if isinstance(value, (int, float)):
                assert np.isclose(value, result["q_drift"][key], rtol=1e-5, atol=1e-8), (
                    generation,
                    arm,
                    key,
                )
        assert parameter_hash(current) == result["updated_state_sha256"]
        arms[arm]["common_probe_Q_drift_verified"] = True
        del rows, producer, current, parameters
    if generation == 1:
        assert root_hashes["A"] == root_hashes["B"]
    review.append({"generation": generation, "arms": arms})
lock = json.loads((root / "final-candidate-lock.json").read_text())
assert lock["generation"] == 10
assert all(lock[f"{a}_sha256"] == digest(root / f"gen-10/{a}/checkpoint.pt") for a in rngs)
assert lock["seed_lock_sha256"] == digest(root / "seed-lock.json")
atomic_json(
    root / "independent-training-review.json",
    {
        "generations": review,
        "source_hashes_verified": True,
        "fixed_Gen10_candidates_verified": True,
    },
)
print(json.dumps(review, indent=2))
