"""Fixed 3x5000/arm actor-center intervention; MC learner is reused unchanged."""

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .active import atomic_json
from .mortal_credit import audit_screening
from .mortal_data import audit_dataset, load_rows, save_shard
from .mortal_deviation_stats import cluster_cohort
from .mortal_diagnostics import distribution
from .mortal_extraction import ExtractionPolicy, family_indices
from .mortal_extraction_experiment import CHECKPOINT
from .mortal_lr_horizon import bn_change, buffer_snapshot, optimizer_dynamics, q_dynamics
from .mortal_offline import BASELINE
from .mortal_online import checkpoint, collect, make_optimizer, update
from .mortal_online_long import parameter_change, q_probe
from .mortal_sarsa import parameter_hash
from .online_rl import digest, source_hashes
from .runtime import load_model

SEED = 20261018
EPSILON, TEMPERATURE = 0.005, 0.5
INITIAL_SHA = "4e2bbee3204630d949753ad92dcaa94a0ef6a2b6e19dc8c862b6dd13d4737a8f"
BASELINE_SHA = "9332fd07a09099dcdd38e180577a38887876c6aa2ee5ac1b8c896193c6ac1784"
DEFAULT = Path("data/generated/mortal-actor-center-probe-v1")


def hash_ticket(*parts):
    return hashlib.sha256(json.dumps([SEED, *parts], separators=(",", ":")).encode()).hexdigest()


def specs(generation=None, *, evaluation=False, holdout=False, offset=0):
    if holdout:
        if generation is not None or evaluation:
            raise ValueError("holdout is a separate block")
        start, n = 1720000000 + offset, 2000
    else:
        if generation not in (1, 2, 3):
            raise ValueError("generation must be 1..3")
        start = (1710000000 if evaluation else 1700000000) + offset + generation * 100000
        n = 500 if evaluation else 5000
    if evaluation or holdout:
        return [(s, s % 3, seat, 0) for s in range(start, start + n) for seat in range(3)]
    return [(start + i, i % 3, (i // 3) % 3, 0) for i in range(n)]


def ownership(game_specs, label, *, evaluation=False):
    """Only public trial metadata, decided BEFORE any engine/deal/model call."""
    if evaluation:
        seeds = sorted({s[0] for s in game_specs})
        chosen = set(sorted(seeds, key=lambda s: hash_ticket("eval", label, s))[: len(seeds) // 10])
        flags = {(s[0], s[2]): s[0] in chosen for s in game_specs}
    else:
        groups = defaultdict(list)
        for s in game_specs:
            groups[s[1], s[2]].append(s)
        if set(groups) != {(d, p) for d in range(3) for p in range(3)}:
            raise ValueError("all nine dealer/seat strata required")
        quota, remainder = divmod(len(game_specs) // 10, 9)
        flags = dict.fromkeys(((s[0], s[2]) for s in game_specs), False)
        for i, key in enumerate(sorted(groups)):
            chosen = sorted(groups[key], key=lambda s: hash_ticket("train", label, *s[:3]))[
                : quota + (i < remainder)
            ]
            for s in chosen:
                flags[s[0], s[2]] = True
    return [
        {
            "game_id": f"{label}-{i:05d}",
            "seed": s[0],
            "dealer": s[1],
            "seat": s[2],
            "probe": flags[s[0], s[2]],
        }
        for i, s in enumerate(game_specs)
    ]


def probabilities(score, epsilon, *, hu=False):
    score = np.asarray(score, float)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all() or not 0 <= epsilon <= 1:
        raise ValueError("finite legal scores and valid epsilon required")
    result = np.zeros(len(score), float)
    center = int(score.argmax())
    if hu or epsilon == 0:
        result[center] = 1.0
    else:
        weights = np.exp((score - score.max()) / TEMPERATURE)
        result = epsilon * weights / weights.sum()
        result[center] += 1 - epsilon
    return result


class OwnershipPolicy(ExtractionPolicy):
    def __init__(self, q, parent, reference, mask, *, arm="D", training=False):
        if arm not in ("A", "B", "D") or training and arm == "D":
            raise ValueError("invalid actor role")
        super().__init__(q, parent, reference, {"mode": "anchor", "lambda": 0.3})
        self.mask = {(r["seed"], r["seat"]): bool(r["probe"]) for r in mask}
        self.arm, self.sampling = arm, training
        self.epsilon = EPSILON if training else 0.0
        self.context, self.rngs = None, {}

    def prepare(self, game_specs):
        self.context = [(int(s[0]), int(s[2])) for s in game_specs]
        if len(set(self.context)) != len(self.context):
            raise ValueError("duplicate active contexts")
        for key in self.context:
            if key not in self.mask:
                raise ValueError("missing PRELOCKED ownership")
            self.rngs.setdefault(key, np.random.default_rng(np.random.SeedSequence([SEED, *key])))

    def forward(self, features, actions, legal):
        if self.context is None or len(self.context) != len(features):
            raise RuntimeError("public game context required")
        scores, value, aux = super().forward(features, actions, legal)
        q, p, parent, keep, anchor = self.last
        self.anchor = anchor.copy()
        self.families = family_indices(actions, legal).cpu().tolist()
        self.raw_center = [
            self.arm != "A" and self.mask[key] and f != 3
            for key, f in zip(self.context, self.families, strict=True)
        ]
        use = torch.tensor(self.raw_center, device=legal.device)
        scores = torch.where(use[:, None], torch.as_tensor(q, device=legal.device), scores)
        center_scores = scores.detach().cpu().numpy()
        self.last = q, p, parent, keep, center_scores
        self.actor_probabilities, self.choices, self.branches = [], [], []
        for i, (key, f) in enumerate(zip(self.context, self.families, strict=True)):
            n = int(legal[i].sum())
            s = center_scores[i, :n]
            prob = probabilities(s, self.epsilon, hu=f == 3)
            branch = self.sampling and f != 3 and bool(self.rngs[key].random() < self.epsilon)
            if branch:
                weights = np.exp((s.astype(float) - float(s.max())) / TEMPERATURE)
                choice = int(self.rngs[key].choice(n, p=weights / weights.sum()))
            else:
                choice = int(s.argmax())
            self.actor_probabilities.append(prob)
            self.branches.append(branch)
            self.choices.append(choice)
        dispatch = torch.full_like(scores, -torch.inf)
        dispatch.scatter_(1, torch.tensor(self.choices, device=legal.device)[:, None], 0.0)
        return dispatch, value, aux

    def record(self, obs, index, chosen, spec, step):
        if self.context[index] != (spec[0], spec[2]) or chosen != self.choices[index]:
            raise RuntimeError("context or executed action mismatch")
        r = super().record(obs, index, chosen, spec, step)
        p = self.actor_probabilities[index]
        q = np.array(r["q"], float)
        r.update(
            probe=self.mask[self.context[index]],
            raw_center=self.raw_center[index],
            center=int(np.argmax(r["scores"])),
            anchor=int(np.argmax(self.anchor[index, : len(q)])),
            sampled_branch=self.branches[index],
            epsilon=self.epsilon,
            temperature=TEMPERATURE,
            top_p=1,
            actor_probabilities=p.tolist(),
            actual_behavior_probability=float(p[chosen]),
            entropy=float(-np.sum(p[p > 0] * np.log(p[p > 0]))),
            centered_legal_advantage_rms=float(np.sqrt(np.mean((q - q.mean()) ** 2))),
            arm=self.arm,
        )
        return r


class DiagnosticGreedy(ExtractionPolicy):
    """Add diagnostics without changing the existing raw/baseline execution."""

    def record(self, obs, index, chosen, spec, step):
        r = super().record(obs, index, chosen, spec, step)
        q = np.asarray(r["q"], np.float32)
        p = np.asarray(r["probabilities"], np.float32)
        anchor = q + np.float32(0.3) * np.log(p + np.float32(1e-8))
        r.update(
            center=chosen, anchor=int(anchor.argmax()), actual_behavior_probability=1.0, entropy=0.0
        )
        return r


def grouped_behavior(records):
    groups = defaultdict(list)
    for r in records:
        keys = [
            "all",
            f"seat-{r['seat']}",
            r["family"],
            f"seat-{r['seat']}/{r['family']}",
            "probe" if r.get("probe") else "non-probe",
        ]
        for k in keys:
            groups[k].append(r)
    out = {}
    for k, rows in groups.items():
        n = len(rows)
        eligible = [r for r in rows if r["family"] != "hu/pass"]
        out[k] = {
            "decisions": n,
            "non_hu_decisions": len(eligible),
            "probe_decisions": sum(bool(r.get("probe")) for r in rows),
            "raw_center_decisions": sum(bool(r.get("raw_center")) for r in rows),
            "sampled_branch_count": sum(bool(r.get("sampled_branch")) for r in rows),
            "sampled_branch_rate_eligible": sum(bool(r.get("sampled_branch")) for r in rows)
            / max(1, len(eligible)),
            "deviation_vs_parent": sum(r["chosen"] != r["parent"] for r in rows) / n,
            "deviation_vs_center": sum(r["chosen"] != r.get("center", r["chosen"]) for r in rows)
            / n,
            "deviation_vs_anchor": sum(r["chosen"] != r.get("anchor", r["chosen"]) for r in rows)
            / n,
            "deviation_vs_raw": sum(r["chosen"] != r["raw"] for r in rows) / n,
            "low_support_fraction": sum(r["behavior_probability"] <= 0.1 for r in rows) / n,
            "low_support_among_parent_deviations": (
                sum(r["chosen"] != r["parent"] and r["behavior_probability"] <= 0.1 for r in rows)
                / max(1, sum(r["chosen"] != r["parent"] for r in rows))
            ),
            "terminal_cash_descriptive": distribution([r["terminal_cash"] for r in rows]),
            "selected_q_minus_parent": distribution([r["q_selected_minus_parent"] for r in rows]),
            "centered_advantage_rms": distribution(
                [
                    r["centered_legal_advantage_rms"]
                    for r in rows
                    if "centered_legal_advantage_rms" in r
                ]
            ),
            "chosen_p_reference": distribution([r["behavior_probability"] for r in rows]),
            "actual_actor_probability": distribution(
                [r.get("actual_behavior_probability", 1.0) for r in rows]
            ),
            "entropy": distribution([r.get("entropy", 0.0) for r in rows]),
            "raw_q_margin": distribution(
                [r["q_margin"] for r in rows if r["q_margin"] is not None]
            ),
        }
    return out


def q_statistics(qs, rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        for key in (
            "all",
            f"seat-{r['seat']}",
            str(r["family"]),
            f"seat-{r['seat']}/{r['family']}",
        ):
            groups[key].append(i)
    out = {}
    for key, ids in groups.items():
        chosen = np.array([qs[i][int(rows[i]["chosen"])] for i in ids], float)
        target = np.array([rows[i]["terminal_cash"] for i in ids], float)
        centered = np.concatenate([qs[i] - qs[i].mean() for i in ids])
        out[key] = {
            "rows": len(ids),
            "unit": "cash/100",
            "mae": float(np.mean(abs(chosen - target))),
            "rmse": float(np.sqrt(np.mean((chosen - target) ** 2))),
            "correlation": float(np.corrcoef(chosen, target)[0, 1])
            if chosen.std() > 1e-10 and target.std() > 1e-10
            else None,
            "centered_legal_advantage_rms": float(np.sqrt(np.mean(centered**2))),
            "raw_top1_top2_margin": distribution(
                [float(np.sort(qs[i])[-1] - np.sort(qs[i])[-2]) for i in ids]
            ),
        }
    return out


def matched_selection(n_a, n_b, rngs):
    used = min(n_a, n_b) // 256 * 256
    if not used:
        raise ValueError("no full matched batch")
    return {a: rngs[a].permutation(n)[:used].tolist() for a, n in (("A", n_a), ("B", n_b))}


def safety_stop(results):
    return len(results) >= 2 and all(
        r["cash"]["B-D"]["seed_bootstrap_95"][1] < -5 for r in results[-2:]
    )


def old_seed_audit(candidate, exclude):
    """Follow archived directory links; inspect all locked/config/input manifests."""
    hashes = {}
    hits = []
    seen = set()
    for directory, dirs, files in os.walk("data/generated", followlinks=True):
        real = Path(directory).resolve()
        if real == exclude.resolve() or real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        for name in sorted(files):
            if not name.endswith(".json") or not any(
                t in name for t in ("manifest", "seed", "lock", "config", "input-sha", "outcomes")
            ):
                continue
            path = Path(directory) / name
            data = json.loads(path.read_text())

            def check(value, path=path):
                if isinstance(value, dict):
                    for v in value.values():
                        check(v)
                elif isinstance(value, list):
                    for v in value:
                        check(v)
                elif isinstance(value, int) and value in candidate:
                    hits.append((str(path), value))

            check(data)
            hashes[str(path)] = digest(path)
    return {"files": hashes, "overlaps": hits}


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    assert digest(CHECKPOINT) == INITIAL_SHA and digest(BASELINE) == BASELINE_SHA
    offset = 0
    while True:
        lock = {
            "generations": {
                str(g): {
                    "train": specs(g, offset=offset),
                    "evaluation": specs(g, evaluation=True, offset=offset),
                }
                for g in (1, 2, 3)
            },
            "holdout": specs(holdout=True, offset=offset),
        }
        candidate = {s[0] for v in lock["generations"].values() for ss in v.values() for s in ss}
        candidate.update(s[0] for s in lock["holdout"])
        audit = old_seed_audit(candidate, root)
        if not audit["overlaps"]:
            break
        offset += 10000000
        if offset > 100000000:
            raise RuntimeError("no fresh contiguous seed block")
    masks = {
        str(g): {
            "train": ownership(v["train"], f"train-{g}"),
            "evaluation": ownership(v["evaluation"], f"eval-{g}", evaluation=True),
        }
        for g, v in lock["generations"].items()
    }
    masks["holdout"] = ownership(lock["holdout"], "holdout", evaluation=True)
    atomic_json(root / "seed-lock.json", lock)
    atomic_json(root / "ownership-manifest.json", masks)
    atomic_json(root / "seed-overlap-audit.json", audit)
    old = json.loads(Path("data/generated/mortal-online-v1/config.json").read_text())
    frozen = source_hashes() | {
        str(p): digest(p)
        for p in (
            Path(CHECKPOINT),
            Path(BASELINE),
            Path(__file__),
            Path("scripts/run-mortal-actor-center-probe.py"),
            Path("scripts/audit-mortal-actor-center-probe.py"),
        )
    }
    config = {
        "experiment": "restricted-Q-ownership-actor-center-v1",
        "generations": 3,
        "games_per_arm_generation": 5000,
        "training_games_total": 30000,
        "evaluation_games_total": 63000,
        "final_generation": 3,
        "epsilon": EPSILON,
        "temperature": TEMPERATURE,
        "top_p": 1,
        "lambda": 0.3,
        "probe_fraction": 0.1,
        "loss": "terminal cash/100 MC half MSE only",
        "CQL": 0,
        "aux": 0,
        "gamma": 1,
        "BN": "unchanged train/eval",
        "optimizer": "existing make_optimizer; AdamW .01 weight grouping; batch256 clip1 float32",
        "schedule": "existing learning_rate: warm20 cosine300 peak3e-5 floor3e-6; continuous",
        "row_matching": "floor(min(N_A,N_B)/256) batches; uniform without replacement; independent selection RNG, common batch-index RNG",
        "mask": "pre-deal SHA256 public metadata; nine dealer/seat strata, remainder fixed lexicographic order; eval seed-cluster10%",
        "reference": old["behavior"],
        "frozen_sha256": frozen,
        "seed_offset": offset,
        "seed_lock_sha256": digest(root / "seed-lock.json"),
        "ownership_sha256": digest(root / "ownership-manifest.json"),
        "bootstrap": 10000,
        "primary": ["B-minus-A", "B-minus-initial", "B-minus-baseline"],
        "safety": "B deployment vs baseline CI upper<-5 in two consecutive generations; no holdout if stopped",
        "promotion": False,
        "torch": str(torch.__version__),
    }
    atomic_json(root / "config.json", config)
    atomic_json(
        root / "lock.json",
        {"config_sha256": digest(root / "config.json"), "locked_before_dealing": True},
    )


def verify(root):
    config = json.loads((root / "config.json").read_text())
    lock = json.loads((root / "lock.json").read_text())
    assert digest(root / "config.json") == lock["config_sha256"]
    for p, h in config["frozen_sha256"].items():
        assert digest(p) == h, p
    assert digest(root / "seed-lock.json") == config["seed_lock_sha256"]
    assert digest(root / "ownership-manifest.json") == config["ownership_sha256"]
    return config


def evaluate(root, game_specs, mask, models, initial, parent, config):
    root.mkdir()
    before = {
        a: parameter_hash(m) for a, m in (models | {"initial": initial, "parent": parent}).items()
    }
    policies = {
        "baseline": DiagnosticGreedy(
            None, parent, config["reference"], {"mode": "baseline", "lambda": 0.3}
        ),
        "initial": OwnershipPolicy(initial, parent, config["reference"], mask),
    }
    for a, m in models.items():
        policies[f"{a}-D"] = OwnershipPolicy(m, parent, config["reference"], mask)
        policies[f"{a}-raw"] = DiagnosticGreedy(
            m, parent, config["reference"], {"mode": "raw", "lambda": 0.3}
        )
    outcomes = {}
    behavior = {}
    for name, policy in policies.items():
        atomic_json(root / "phase.json", {"policy": name, "games": len(game_specs)})
        games, records, _ = collect(policy.eval(), parent, game_specs, root, name, device="cuda")
        outcomes[name] = games
        behavior[name] = grouped_behavior(records)
        atomic_json(root / "screening-outcomes.json", outcomes)
    replay_audit = audit_screening(root)
    indexed = {n: {(g["seed"], g["seat"]): g for g in games} for n, games in outcomes.items()}
    cash = {n: cluster_cohort(games, indexed["baseline"]) for n, games in outcomes.items()}
    contrasts = {
        f"B-minus-{ref}": cluster_cohort(outcomes["B-D"], indexed[name])
        for ref, name in (("A", "A-D"), ("initial", "initial"), ("baseline", "baseline"))
    }
    seats = {
        str(s): {
            n: cluster_cohort([g for g in games if g["seat"] == s], indexed["baseline"])
            for n, games in outcomes.items()
        }
        for s in range(3)
    }
    assert before == {
        a: parameter_hash(m) for a, m in (models | {"initial": initial, "parent": parent}).items()
    }
    result = {
        "cash": cash,
        "contrasts": contrasts,
        "behavior": behavior,
        "seat_cash": seats,
        "replay_audit": replay_audit,
    }
    atomic_json(root / "result.json", result)
    return result


def run(root):
    root = Path(root)
    config = verify(root)
    if (root / "generations.json").exists():
        raise FileExistsError("no implicit restart/replay")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for authorized experiment")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    parent = load_model(BASELINE, "cuda").requires_grad_(False)
    initial = load_model(CHECKPOINT, "cuda").requires_grad_(False)
    models = {a: load_model(CHECKPOINT, "cuda") for a in ("A", "B")}
    assert parameter_hash(models["A"]) == parameter_hash(models["B"]) == parameter_hash(initial)
    opts = {a: make_optimizer(m) for a, m in models.items()}
    selects = {a: np.random.default_rng(SEED) for a in models}
    batches = {a: np.random.default_rng(SEED + 1) for a in models}
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state_all()
    torch_states = {a: (cpu.clone(), [v.clone() for v in gpu]) for a in models}
    updates = dict.fromkeys(models, 0)
    results = []
    locks = json.loads((root / "seed-lock.json").read_text())
    masks = json.loads((root / "ownership-manifest.json").read_text())
    atomic_json(
        root / "runtime.json",
        {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "device": "cuda",
            "dtype": "float32",
            "cpu_threads": 2,
        },
    )
    for generation in (1, 2, 3):
        gen = root / f"gen-{generation:02d}"
        gen.mkdir()
        all_rows = {}
        summary = {}
        ss = locks["generations"][str(generation)]
        mask = masks[str(generation)]["train"]
        for arm, model in models.items():
            directory = gen / arm
            directory.mkdir()
            data_dir = directory / "data"
            data_dir.mkdir()
            previous = (
                CHECKPOINT
                if generation == 1
                else str(root / f"gen-{generation - 1:02d}" / arm / "checkpoint.pt")
            )
            producer = parameter_hash(model)
            actor = OwnershipPolicy(
                model, parent, config["reference"], mask, arm=arm, training=True
            )
            manifest = {
                "shards": [],
                "producer_checkpoint": previous,
                "producer_sha256": digest(previous),
                "producer_state_sha256": producer,
                "ownership_sha256": config["ownership_sha256"],
                "generation": generation,
                "arm": arm,
            }
            records = []
            outcomes = []
            for start in range(0, 5000, 300):
                atomic_json(
                    root / "progress.json",
                    {
                        "phase": "collection",
                        "generation": generation,
                        "arm": arm,
                        "completed": start,
                    },
                )
                games, recs, data = collect(
                    actor.eval(),
                    parent,
                    ss["train"][start : start + 300],
                    directory / f"evidence-{start // 300:02d}",
                    f"{arm}-generation-{generation}",
                    training_rows=True,
                    device="cuda",
                )
                manifest["shards"].append(save_shard(data_dir, start // 300, data))
                records.extend(recs)
                outcomes.extend(games)
                atomic_json(data_dir / "manifest.json", manifest)
            assert parameter_hash(model) == producer
            manifest["raw_logs_sha256"] = {
                str(p): digest(p) for p in sorted(directory.glob("evidence-*/*.gz"))
            }
            atomic_json(data_dir / "manifest.json", manifest)
            atomic_json(
                directory / "actor-rng.json",
                {f"{k[0]}:{k[1]}": v.bit_generator.state for k, v in actor.rngs.items()},
            )
            atomic_json(directory / "training-outcomes.json", outcomes)
            audit = audit_dataset(data_dir, manifest)
            all_rows[arm] = load_rows(data_dir, manifest)
            summary[arm] = {
                "producer_state_sha256": producer,
                "data_audit": audit,
                "decisions": len(all_rows[arm]),
                "behavior": grouped_behavior(records),
                "sampling_execution_cash": distribution([g["cash"] for g in outcomes]),
                "probe_games": sum(r["probe"] for r in mask),
                "non_probe_games": sum(not r["probe"] for r in mask),
            }
            del actor, records, recs, data, outcomes
        selection = matched_selection(len(all_rows["A"]), len(all_rows["B"]), selects)
        atomic_json(
            gen / "row-selection.json",
            {
                "indices": selection,
                "N_A": len(all_rows["A"]),
                "N_B": len(all_rows["B"]),
                "selection_rng_after": {a: r.bit_generator.state for a, r in selects.items()},
            },
        )
        for arm, model in models.items():
            torch.set_rng_state(torch_states[arm][0])
            torch.cuda.set_rng_state_all(torch_states[arm][1])
            directory = gen / arm
            rows = all_rows[arm]
            indices = selection[arm]
            used = [rows[i] for i in indices]
            chosen = set(indices)
            params = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            buffers = buffer_snapshot(model)
            before = q_probe(model, rows)
            atomic_json(
                root / "progress.json",
                {
                    "phase": "MC_update",
                    "generation": generation,
                    "arm": arm,
                    "used_rows": len(used),
                },
            )
            updates[arm], logs = update(
                model, opts[arm], used, batches[arm], updates[arm], device="cuda"
            )
            after = q_probe(model, rows)
            checkpoint(
                model,
                opts[arm],
                directory / "checkpoint.pt",
                updates[arm],
                batches[arm],
                generation=generation,
                arm=arm,
                manifest_sha256=digest(directory / "data/manifest.json"),
                seed_lock_sha256=config["seed_lock_sha256"],
                ownership_sha256=config["ownership_sha256"],
                row_selection_sha256=digest(gen / "row-selection.json"),
                producer_state_sha256=summary[arm]["producer_state_sha256"],
                learning_rate_schedule="linear20+cosine300",
            )
            torch_states[arm] = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
            atomic_json(directory / "updates.json", logs)
            summary[arm].update(
                used_rows=len(used),
                dropped_rows=len(rows) - len(used),
                used_family=dict(Counter(str(r["family"]) for r in used)),
                dropped_family=dict(
                    Counter(str(r["family"]) for i, r in enumerate(rows) if i not in chosen)
                ),
                updates=updates[arm],
                optimizer_dynamics=optimizer_dynamics(logs, opts[arm]),
                parameter_change=parameter_change(params, model),
                bn_change=bn_change(buffers, model),
                q_drift=q_dynamics(before, after, rows),
                regression_before=q_statistics(before, rows),
                regression_after=q_statistics(after, rows),
                checkpoint_sha256=digest(directory / "checkpoint.pt"),
                optimizer_sha256=digest(directory / "optimizer.pt"),
                updated_state_sha256=parameter_hash(model),
            )
            summary[arm]["q_drift"]["scope"] = (
                "same own-generation visible training inputs before/after; diagnostic only, no extra loss/replay"
            )
            atomic_json(directory / "training-result.json", summary[arm])
            del params, buffers, before, after, used, chosen
        assert updates["A"] == updates["B"]
        del all_rows, rows, selection
        atomic_json(root / "progress.json", {"phase": "evaluation", "generation": generation})
        ev = evaluate(
            gen / "evaluation",
            ss["evaluation"],
            masks[str(generation)]["evaluation"],
            models,
            initial,
            parent,
            config,
        )
        result = {"generation": generation, "arms": summary, **ev}
        results.append(result)
        atomic_json(root / "generations.json", results)
        atomic_json(gen / "result.json", result)
        verify(root)
        print(
            json.dumps(
                {"generation": generation, "contrasts": ev["contrasts"], "cash": ev["cash"]}
            ),
            flush=True,
        )
        if safety_stop(results):
            atomic_json(
                root / "safety-stop.json",
                {
                    "reason": "B-D vs baseline upper<-5 in two consecutive generations",
                    "generation": generation,
                    "fallback": CHECKPOINT,
                    "baseline_replaced": False,
                },
            )
            break
    final = None
    if len(results) == 3 and not (root / "safety-stop.json").exists():
        atomic_json(
            root / "final-candidate-lock.json",
            {
                "generation": 3,
                "A_sha256": digest(root / "gen-03/A/checkpoint.pt"),
                "B_sha256": digest(root / "gen-03/B/checkpoint.pt"),
                "seed_lock_sha256": config["seed_lock_sha256"],
            },
        )
        atomic_json(root / "progress.json", {"phase": "final_holdout", "generation": 3})
        final = evaluate(
            root / "holdout", locks["holdout"], masks["holdout"], models, initial, parent, config
        )
    verify(root)
    passed = bool(final and all(v["seed_bootstrap_95"][0] > 0 for v in final["contrasts"].values()))
    atomic_json(
        root / "result.json",
        {
            "generations": results,
            "holdout": final,
            "reliable_policy_improvement": passed,
            "baseline_replaced": False,
            "stop": "complete fixed experiment" if final else "predeclared safety stop",
        },
    )
    atomic_json(
        root / "progress.json",
        {"phase": "complete_stop", "generations": len(results), "holdout_used": final is not None},
    )
