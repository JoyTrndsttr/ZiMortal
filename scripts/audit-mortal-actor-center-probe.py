"""Independent read-only verification of the fixed actor-center experiment."""

import argparse
import ast
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from zimortal.training.active import atomic_json
from zimortal.training.mortal_actor_center_probe import (
    DEFAULT,
    SEED,
    grouped_behavior,
    matched_selection,
    old_seed_audit,
    ownership,
    probabilities,
    q_statistics,
    safety_stop,
    specs,
    verify,
)
from zimortal.training.mortal_credit import audit_screening
from zimortal.training.mortal_data import audit_dataset, load_rows
from zimortal.training.mortal_deviation_stats import cluster_cohort
from zimortal.training.mortal_extraction_experiment import CHECKPOINT
from zimortal.training.mortal_lr_horizon import (
    bn_change,
    buffer_snapshot,
    optimizer_dynamics,
    q_dynamics,
)
from zimortal.training.mortal_online import learning_rate
from zimortal.training.mortal_online_long import parameter_change, q_probe
from zimortal.training.mortal_sarsa import parameter_hash
from zimortal.training.online_rl import digest
from zimortal.training.runtime import load_model


def read(path):
    return json.loads(path.read_text())


def records(directory):
    for p in sorted(directory.glob("**/decisions-*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            yield from map(json.loads, f)


def close(a, b):
    if isinstance(b, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for key in b:
            close(a[key], b[key])
    elif isinstance(b, list):
        assert len(a) == len(b)
        for aa, bb in zip(a, b, strict=True):
            close(aa, bb)
    elif isinstance(b, (float, int)) and not isinstance(b, bool):
        assert np.isclose(a, b, rtol=3e-5, atol=2e-7), (a, b)
    else:
        assert a == b, (a, b)


def audit_actor(rows, mask, arm, training):
    flags = {(r["seed"], r["seat"]): r["probe"] for r in mask}
    rngs = {}
    for r in rows:
        key = r["seed"], r["seat"]
        assert key in flags and not r["filtered"]
        q, p = np.asarray(r["q"], np.float32), np.asarray(r["probabilities"], np.float32)
        anchor = q + np.float32(0.3) * np.log(p + np.float32(1e-8))
        raw_center = arm != "A" and flags[key] and r["family"] != "hu/pass"
        score = q if raw_center else anchor
        np.testing.assert_allclose(r["scores"], score, atol=1e-6, rtol=1e-6)
        assert r["probe"] == flags[key] and r["raw_center"] == raw_center
        assert r["anchor"] == int(anchor.argmax())
        assert r["center"] == int(score.argmax())
        assert (
            r["epsilon"] == (0.005 if training else 0.0)
            and r["temperature"] == 0.5
            and r["top_p"] == 1
        )
        prob = probabilities(r["scores"], r["epsilon"], hu=r["family"] == "hu/pass")
        np.testing.assert_allclose(r["actor_probabilities"], prob, atol=1e-12, rtol=1e-12)
        rng = rngs.setdefault(key, np.random.default_rng(np.random.SeedSequence([SEED, *key])))
        branch = training and r["family"] != "hu/pass" and bool(rng.random() < 0.005)
        if branch:
            score = np.asarray(r["scores"], float)
            weights = np.exp((score - score.max()) / 0.5)
            chosen = int(rng.choice(len(score), p=weights / weights.sum()))
        else:
            chosen = r["center"]
        assert r["sampled_branch"] == branch and r["chosen"] == chosen
        close(r["actual_behavior_probability"], prob[chosen])
        close(r["entropy"], float(-np.sum(prob[prob > 0] * np.log(prob[prob > 0]))))
        close(
            r["centered_legal_advantage_rms"],
            float(np.sqrt(np.mean((q.astype(float) - q.mean(dtype=float)) ** 2))),
        )
    return rngs


def main(root, device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = verify(root)
    assert read(root / "progress.json")["phase"] == "complete_stop"
    lock, masks = read(root / "seed-lock.json"), read(root / "ownership-manifest.json")
    seen = set()
    for g in range(1, 4):
        for key, ev in (("train", False), ("evaluation", True)):
            ss = specs(g, evaluation=ev, offset=config["seed_offset"])
            assert lock["generations"][str(g)][key] == json.loads(json.dumps(ss))
            current = {s[0] for s in ss}
            assert not seen & current
            seen |= current
            assert masks[str(g)][key] == ownership(
                ss, f"{'eval' if ev else 'train'}-{g}", evaluation=ev
            )
    ss = specs(holdout=True, offset=config["seed_offset"])
    assert lock["holdout"] == json.loads(json.dumps(ss)) and not seen & {s[0] for s in ss}
    seen |= {s[0] for s in ss}
    assert masks["holdout"] == ownership(ss, "holdout", evaluation=True)
    prior = read(root / "seed-overlap-audit.json")
    for p, sha in prior["files"].items():
        assert digest(p) == sha
    assert not old_seed_audit(seen, root)["overlaps"]
    # This module only uses ordinary real-play collect/update, not any diagnostic branch generator.
    module = Path("zimortal/training/mortal_actor_center_probe.py")
    tree = ast.parse(module.read_text())
    forbidden = (
        "counterfactual",
        "sampler",
        "posterior",
        "belief",
        "pairwise",
        "rollout",
        "target_network",
    )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [n.name for n in node.names]
            )
            assert not any(word in name for name in names for word in forbidden)
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            assert not any(word in name for word in forbidden)
    results = read(root / "generations.json")
    selection_rngs = {a: np.random.default_rng(SEED) for a in ("A", "B")}
    batches = {a: np.random.default_rng(SEED + 1) for a in ("A", "B")}
    updates = dict.fromkeys(batches, 0)
    previous = dict.fromkeys(batches, CHECKPOINT)
    torch_rngs, review = {}, []
    for g, result in enumerate(results, 1):
        gen = root / f"gen-{g:02d}"
        selection = read(gen / "row-selection.json")
        expected = matched_selection(selection["N_A"], selection["N_B"], selection_rngs)
        assert expected == selection["indices"]
        assert selection["selection_rng_after"] == {
            a: r.bit_generator.state for a, r in selection_rngs.items()
        }
        arms, first_non_probe = {}, {}
        for arm in ("A", "B"):
            directory = gen / arm
            manifest = read(directory / "data/manifest.json")
            summary = read(directory / "training-result.json")
            assert manifest["producer_checkpoint"] == previous[arm]
            assert manifest["producer_sha256"] == digest(previous[arm])
            assert manifest["ownership_sha256"] == config["ownership_sha256"]
            for p, sha in manifest["raw_logs_sha256"].items():
                assert digest(p) == sha
            replay = audit_dataset(directory / "data", manifest)
            rows = load_rows(directory / "data", manifest)
            assert len(rows) == selection[f"N_{arm}"] == summary["decisions"]
            indices = selection["indices"][arm]
            used = set(indices)
            assert len(used) == len(indices) == min(selection["N_A"], selection["N_B"]) // 256 * 256
            assert summary["used_rows"] == len(used) and summary["dropped_rows"] == len(rows) - len(
                used
            )
            assert summary["dropped_family"] == dict(
                Counter(str(r["family"]) for i, r in enumerate(rows) if i not in used)
            )
            assert summary["used_family"] == dict(Counter(str(rows[i]["family"]) for i in indices))
            evidence = list(records(directory))
            assert len(evidence) == len(rows)
            generators = audit_actor(evidence, masks[str(g)]["train"], arm, True)
            actual = [(r["seed"], r["seat"], r["step"] - 1, r["chosen"]) for r in evidence]
            assert actual == [
                (int(r["seed"]), int(r["seat"]), int(r["history_step"]), int(r["chosen"]))
                for r in rows
            ]
            for ev, row in zip(evidence, rows, strict=True):
                np.testing.assert_allclose(
                    ev["actor_probabilities"], row["behavior_probabilities"], atol=4e-8
                )
                assert np.isclose(ev["terminal_cash"] / 100, row["terminal_cash"])
            final_actor_rngs = read(directory / "actor-rng.json")
            for key, rng in generators.items():
                assert final_actor_rngs[f"{key[0]}:{key[1]}"] == rng.bit_generator.state
            close(summary["behavior"], grouped_behavior(evidence))
            if g == 1:
                # Exact same policy and RNG outside intervention in the first generation.
                first_non_probe[arm] = [
                    (r["seed"], r["seat"], r["step"], r["chosen"], r["actions"])
                    for r in evidence
                    if not r["probe"]
                ]
            logs = read(directory / "updates.json")
            assert len(logs) == len(indices) // 256
            batches[arm].permutation(len(indices))
            for log in logs:
                assert log["update"] == updates[arm] + 1 and log["batch"] == 256
                assert log["lr"] == learning_rate(updates[arm])
                assert log["cql_loss"] == log["auxiliary_loss"] == 0
                updates[arm] += 1
            opt = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=False)
            assert opt["numpy_rng"] == batches[arm].bit_generator.state
            assert opt["updates"] == updates[arm] == summary["updates"]
            assert opt["schedule"] == "linear20+cosine300"
            assert all(int(v["step"]) == updates[arm] for v in opt["optimizer"]["state"].values())
            groups = opt["optimizer"]["param_groups"]
            assert {group["weight_decay"] for group in groups} == {0.0, 0.01}
            assert all(group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8 for group in groups)
            assert all(group["lr"] == learning_rate(updates[arm] - 1) for group in groups)
            metadata = opt["metadata"]
            assert metadata["manifest_sha256"] == digest(directory / "data/manifest.json")
            assert metadata["row_selection_sha256"] == digest(gen / "row-selection.json")
            assert metadata["ownership_sha256"] == config["ownership_sha256"]
            assert metadata["seed_lock_sha256"] == config["seed_lock_sha256"]
            assert (
                opt["checkpoint_sha256"]
                == digest(directory / "checkpoint.pt")
                == summary["checkpoint_sha256"]
            )
            assert digest(directory / "optimizer.pt") == summary["optimizer_sha256"]
            if arm in torch_rngs:
                old = torch_rngs[arm]
                assert torch.equal(old[0], opt["torch_rng"])
                assert all(torch.equal(a, b) for a, b in zip(old[1], opt["cuda_rng"], strict=True))
            torch_rngs[arm] = opt["torch_rng"], opt["cuda_rng"]
            producer = load_model(previous[arm], device)
            current = load_model(directory / "checkpoint.pt", device)
            assert (
                parameter_hash(producer)
                == summary["producer_state_sha256"]
                == manifest["producer_state_sha256"]
            )
            assert parameter_hash(current) == summary["updated_state_sha256"]
            close(
                summary["parameter_change"],
                parameter_change(
                    {n: p.detach().cpu().clone() for n, p in producer.named_parameters()}, current
                ),
            )
            close(summary["bn_change"], bn_change(buffer_snapshot(producer), current))

            class State:
                state = opt["optimizer"]["state"]

            close(summary["optimizer_dynamics"], optimizer_dynamics(logs, State()))
            before, after = q_probe(producer, rows, device), q_probe(current, rows, device)
            numeric_max, centered_max, numeric_flips = 0.0, 0.0, 0
            for q, ev in zip(before, evidence, strict=True):
                # Real-play batches vary in size; audit uses 256. CUDA float32
                # kernels can produce small common shifts across these shapes.
                # Units cash/100: bounds are .05/.005 yuan. Also require
                # identical raw ranking winners on ALL rows, not just closeness.
                observed = np.asarray(ev["q"], float)
                delta = q.astype(float) - observed
                np.testing.assert_allclose(q, observed, atol=5e-4, rtol=3e-5)
                np.testing.assert_allclose(
                    q - q.mean(dtype=float), observed - observed.mean(), atol=5e-5, rtol=3e-5
                )
                numeric_max = max(numeric_max, float(abs(delta).max()))
                centered_max = max(centered_max, float(abs(delta - delta.mean()).max()))
                numeric_flips += int(q.argmax() != observed.argmax())
            assert numeric_flips == 0
            drift = q_dynamics(before, after, rows)
            drift["scope"] = summary["q_drift"]["scope"]
            close(summary["q_drift"], drift)
            close(summary["regression_before"], q_statistics(before, rows))
            close(summary["regression_after"], q_statistics(after, rows))
            previous[arm] = str(directory / "checkpoint.pt")
            arms[arm] = {
                "games": replay["games"],
                "decisions": len(rows),
                "used": len(indices),
                "cumulative_updates": updates[arm],
                "actor_choices_probabilities_RNG_verified": True,
                "MC_rule_replay_verified": True,
                "Q_parameter_BN_drift_verified": True,
                "matched_updates_no_replacement": True,
                "cross_batch_Q_numeric_check": {
                    "max_absolute_delta_cash100": numeric_max,
                    "max_centered_delta_cash100": centered_max,
                    "raw_argmax_changes": numeric_flips,
                },
            }
            del rows, evidence, producer, current, before, after
        assert updates["A"] == updates["B"]
        if g == 1:
            # Probe trajectories can change slot completion order, even while
            # non-probe games execute exactly the same per-game actions.
            assert sorted(first_non_probe["A"]) == sorted(first_non_probe["B"])
        review.append({"generation": g, "arms": arms})
        print(json.dumps(review[-1]), flush=True)
    assert len(results) == 3 or safety_stop(results)
    stopped = (root / "safety-stop.json").exists()
    assert stopped == safety_stop(results)
    if stopped:
        assert not (root / "holdout").exists()
    else:
        final_lock = read(root / "final-candidate-lock.json")
        assert final_lock["generation"] == 3
        for arm in ("A", "B"):
            assert final_lock[f"{arm}_sha256"] == digest(root / f"gen-03/{arm}/checkpoint.pt")
    evaluations = [
        (
            root / f"gen-{g:02d}/evaluation",
            lock["generations"][str(g)]["evaluation"],
            masks[str(g)]["evaluation"],
        )
        for g in range(1, len(results) + 1)
    ]
    if not stopped:
        evaluations.append((root / "holdout", lock["holdout"], masks["holdout"]))
    for directory, game_specs, mask in evaluations:
        replays = audit_screening(directory)
        outcomes, result = (
            read(directory / "screening-outcomes.json"),
            read(directory / "result.json"),
        )
        assert set(outcomes) == {"baseline", "initial", "A-D", "B-D", "A-raw", "B-raw"}
        keys = {(s[0], s[2]) for s in game_specs}
        baseline = {(g["seed"], g["seat"]): g for g in outcomes["baseline"]}
        assert set(baseline) == keys
        for name, games in outcomes.items():
            assert len(games) == len(keys) and {(g["seed"], g["seat"]) for g in games} == keys
            close(result["cash"][name], cluster_cohort(games, baseline))
            with gzip.open(directory / f"decisions-{name}.jsonl.gz", "rt") as f:
                evidence = [json.loads(line) for line in f]
            if name in ("initial", "A-D", "B-D"):
                audit_actor(evidence, mask, "D", False)
            elif name.endswith("-raw"):
                assert all(r["chosen"] == r["raw"] and r["filtered"] == 0 for r in evidence)
            else:
                assert all(r["chosen"] == r["parent"] for r in evidence)
            for seat in range(3):
                close(
                    result["seat_cash"][str(seat)][name],
                    cluster_cohort([g for g in games if g["seat"] == seat], baseline),
                )
        indexed = {n: {(g["seed"], g["seat"]): g for g in games} for n, games in outcomes.items()}
        for ref, name in (("A", "A-D"), ("initial", "initial"), ("baseline", "baseline")):
            close(
                result["contrasts"][f"B-minus-{ref}"],
                cluster_cohort(outcomes["B-D"], indexed[name]),
            )
        assert replays == result["replay_audit"]
        print(
            json.dumps(
                {
                    "evaluation_verified": str(directory),
                    "games": sum(len(v) for v in outcomes.values()),
                }
            ),
            flush=True,
        )
    verify(root)
    report = {
        "passed": True,
        "training": review,
        "evaluation_blocks": len(evaluations),
        "prior_manifest_files": len(prior["files"]),
        "fresh_seeds": len(seen),
        "source_checkpoint_seeds_masks_matching_updates_RNG_verified": True,
        "ordinary_engine_replay_all_actual_games_verified": True,
        "fixed_Gen3_no_selection": not stopped,
        "baseline_replaced": False,
        "no_forbidden_direct_calls": True,
        "boundary": "source-bound ordinary real play; full replay certifies legal execution/MC, not an independent proof of strategy quality or optimal Q",
    }
    atomic_json(root / "audit-report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--q-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    main(args.directory, args.q_device)
