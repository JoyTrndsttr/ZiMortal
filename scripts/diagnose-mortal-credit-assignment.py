"""READ ONLY: replay recorded actions, aggregate labels, autograd without updates."""

import argparse
import gzip
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import GameState, RuleEngine
from zimortal.model.encoding import encode_action
from zimortal.training.active import atomic_json
from zimortal.training.mortal_credit_diagnostics import (
    FAMILIES,
    blob,
    coarse_keys,
    coverage,
    deck_bucket,
    encoded_key,
    exact_observation,
    gradient_probe,
    group_variance,
    hashed,
    relative_action,
    scale_summary,
    scales,
    summary,
)
from zimortal.training.mortal_data import load_rows
from zimortal.training.mortal_extraction_experiment import CHECKPOINT
from zimortal.training.mortal_offline import BASELINE, tensors
from zimortal.training.mortal_online_long import q_probe
from zimortal.training.mortal_sarsa import parameter_hash
from zimortal.training.online_rl import digest
from zimortal.training.runtime import load_model

SEED = 20261017
STREAMS = {
    "long": ("mortal-online-long-v1", ""),
    "LR-A": ("mortal-lr-horizon-v1", "A"),
    "LR-B": ("mortal-lr-horizon-v1", "B"),
}
DTYPE = np.dtype(
    [(k, "V32") for k in ("exact", "encoded", "fine", "broad", "compatible", "legal", "action")]
    + [
        (k, "i8")
        for k in (
            "game",
            "generation",
            "seat",
            "dealer",
            "huxi",
            "hand_length",
            "deck",
            "meld_cards",
            "family",
            "aliases",
        )
    ]
    + [(k, "f8") for k in ("cash", "prediction", "residual", "margin", "anchor_gain", "adv_std")]
)


class Sample:
    """Uniform min-hash DISTINCT states, uniform occurrence representative; no outcomes in priorities."""

    def __init__(self, salt, capacity=5000):
        self.salt, self.capacity, self.heap, self.rows = salt, capacity, [], {}

    def offer(self, key, locator, row):
        priority = int.from_bytes(hashlib.sha256(blob([SEED, self.salt]) + key).digest(), "big")
        occurrence = hashlib.sha256(blob([SEED, locator])).digest()
        if key in self.rows:
            if occurrence < self.rows[key][0]:
                self.rows[key] = (occurrence, row)
            return
        if len(self.rows) < self.capacity or priority < -self.heap[0][0]:
            if len(self.rows) == self.capacity:
                _, removed = heapq.heappop(self.heap)
                del self.rows[removed]
            heapq.heappush(self.heap, (-priority, key))
            self.rows[key] = (occurrence, row)

    def result(self):
        return [r for _, (_, r) in sorted(self.rows.items())]


def stream_path(name, generation):
    dataset, arm = STREAMS[name]
    base = Path("data/generated") / dataset / f"gen-{generation:02d}"
    return base / arm if arm else base


def grouped_key(*columns):
    a = np.ascontiguousarray(np.column_stack(columns), dtype="<i8")
    return a.view(f"V{a.shape[1] * 8}").ravel()


def analyze_array(data):
    result = {"rows": len(data), "families": {}, "variance_factors": {}}
    game = data["game"]
    cash = data["cash"]
    for f in ("all", *FAMILIES):
        ix = np.ones(len(data), bool) if f == "all" else data["family"] == FAMILIES.index(f)
        d = data[ix]
        if not len(d):
            result["families"][f] = {"rows": 0}
            continue
        result["families"][f] = {
            "rows": len(d),
            "target_yuan": summary(d["cash"]),
            "residual_yuan": summary(d["residual"]),
            "margin_yuan": summary(d["margin"]),
            "encoder_alias_row_fraction": float(np.mean(d["aliases"] > 0)),
            "coverage": {
                k: coverage(d[k], d["action"], d["cash"], d["game"])
                for k in ("exact", "encoded", "fine", "broad", "compatible")
            },
        }
    result["compatible_same_generation"] = coverage(
        grouped_key(data["generation"], np.unique(data["compatible"], return_inverse=True)[1]),
        data["action"],
        cash,
        game,
    )
    for field in ("seat", "dealer", "huxi", "hand_length", "deck", "family", "generation"):
        values = data[field]
        if field == "huxi":
            values = values // 3
        if field == "deck":
            values = np.array([deck_bucket(int(v)) for v in values])
        result["variance_factors"][field] = group_variance(values, cash, game)
    joint = grouped_key(
        data["generation"],
        data["family"],
        (data["dealer"] - data["seat"]) % 3,
        data["huxi"] // 3,
        data["hand_length"] // 3,
        [deck_bucket(int(v)) for v in data["deck"]],
        data["meld_cards"] // 3,
    )
    result["variance_factors"]["joint_public_plus_generation"] = group_variance(joint, cash, game)
    result["per_generation"] = {
        str(g): {
            "rows": int(np.sum(data["generation"] == g)),
            "target_yuan": summary(cash[data["generation"] == g]),
            "residual_yuan": summary(data["residual"][data["generation"] == g]),
            "margin_yuan": summary(data["margin"][data["generation"] == g]),
        }
        for g in range(1, 11)
    }
    _, first = np.unique(game, return_index=True)
    result["between_game_target_yuan_decision_games_only"] = summary(cash[first])
    return result


def measure_stream(name, root, inputs):
    manifests = {
        g: json.loads((stream_path(name, g) / "data/manifest.json").read_text())
        for g in range(1, 11)
    }
    total = sum(s["decisions"] for m in manifests.values() for s in m["shards"])
    table = np.empty(total, dtype=DTYPE)
    sample = Sample(name)
    cursor = 0
    per_game = []
    current_sources = {}
    for g in range(1, 11):
        base = stream_path(name, g)
        manifest = manifests[g]
        current_sources[str(base / "data/manifest.json")] = digest(base / "data/manifest.json")
        current_sources[str(base / "training-result.json")] = digest(base / "training-result.json")
        oldconfig = Path("data/generated") / STREAMS[name][0] / "config.json"
        old = json.loads(oldconfig.read_text())
        for p, sha in old["frozen_sha256"].items():
            if p.startswith(("zimortal/engine/", "zimortal/model/")) and digest(p) != sha:
                raise RuntimeError("historical rules/model/encoder changed")
        for si, shard in enumerate(manifest["shards"]):
            atomic_json(
                root / "progress.json",
                {
                    "phase": "recorded_history_reconstruction",
                    "stream": name,
                    "generation": g,
                    "shard": si,
                    "rows": cursor,
                    "total": total,
                },
            )
            rows = load_rows(base / "data", {"shards": [shard]})
            evidence = list((base / f"evidence-{si:02d}").glob("decisions-*.jsonl.gz"))
            if len(evidence) != 1:
                raise RuntimeError("ambiguous recorded decisions")
            with gzip.open(evidence[0], "rt") as f:
                records = [json.loads(line) for line in f]
            current_sources[str(evidence[0])] = digest(evidence[0])
            if len(records) != len(rows):
                raise RuntimeError("record/row mismatch")
            indexed = {}
            for ri, (r, row) in enumerate(zip(records, rows, strict=True)):
                key = (int(row["seed"]), int(row["seat"]), int(row["history_step"]))
                if (
                    key != (r["seed"], r["seat"], r["step"] - 1)
                    or int(row["chosen"]) != r["chosen"]
                ):
                    raise RuntimeError("execution row ordering mismatch")
                indexed[key] = (ri, r, row)
            replay = base / "data" / shard["replays"]
            if digest(replay) != shard["replay_sha256"]:
                raise RuntimeError("replay SHA changed")
            current_sources[str(replay)] = shard["replay_sha256"]
            current_sources[str(base / "data" / shard["data"])] = shard["sha256"]
            with gzip.open(replay, "rt") as f:
                games = json.load(f)
            for game in games:
                terminal = GameState.deserialize(game["state"])
                # Reconstruct ONLY the stored history. No policy, play(), rollout or generated action.
                engine = RuleEngine()
                state = engine.new_game(game["seed"], game["dealer"])
                prefix = hashlib.sha256()
                targets, states, by_family = [], set(), defaultdict(set)
                previous = None
                adjacent = []
                for step, action in enumerate(terminal.history):
                    item = indexed.get((game["seed"], game["seat"], step))
                    if item is not None:
                        ri, r, row = item
                        obs = engine.observation(state, game["seat"])
                        if obs.legal_actions[int(row["chosen"])] != action:
                            raise RuntimeError("stored choice is not executed action")
                        encoded = np.stack(
                            [encode_action(a, obs.player) for a in obs.legal_actions]
                        )
                        if not np.array_equal(encoded, row["actions"]):
                            raise RuntimeError("candidate encoding changed")
                        hand = np.bincount(obs.hand, minlength=20) / 4
                        if not np.array_equal(hand, row["features"][0]):
                            raise RuntimeError("visible hand differs")
                        key, _absolute_legal = exact_observation(obs, prefix.digest())
                        legal = hashed(
                            sorted(
                                blob(relative_action(a, obs.player)).hex()
                                for a in obs.legal_actions
                            )
                        )
                        fine, broad, compatible = coarse_keys(
                            row["features"], r["family"], len(encoded), legal
                        )
                        cash = float(
                            terminal.settlement.payments[game["seat"]] if terminal.settlement else 0
                        )
                        if r["terminal_cash"] != cash or not np.isclose(
                            float(row["terminal_cash"]), cash / 100
                        ):
                            raise RuntimeError("recorded target differs from terminal cash")
                        values = scales(
                            r["q"],
                            r["probabilities"],
                            r["parent"],
                            r["chosen"],
                            cash,
                            scores=r["scores"],
                        )
                        locator = [name, g, si, ri, game["seed"], game["seat"], step]
                        values.update(family=r["family"], locator=locator)
                        sample.offer(key, locator, values)
                        x = row["features"]
                        hu = round(float(x[22, 0]) * 60)
                        deck = round(float(x[27, 0]) * 20)
                        if deck != obs.remaining_tiles:
                            raise RuntimeError("public deck differs")
                        table[cursor] = (
                            np.void(key),
                            np.void(encoded_key(x, encoded)),
                            np.void(fine),
                            np.void(broad),
                            np.void(compatible),
                            np.void(legal),
                            np.void(hashed(relative_action(action, obs.player))),
                            game["seed"] * 3 + game["seat"],
                            g,
                            game["seat"],
                            game["dealer"],
                            hu,
                            len(obs.hand),
                            deck,
                            round(float(x[4:22].sum()) * 4),
                            FAMILIES.index(r["family"]),
                            len(encoded) - len(np.unique(encoded, axis=0)),
                            cash,
                            values["chosen_q_yuan"],
                            values["residual_yuan"],
                            values["raw_margin_yuan"],
                            values["anchor_parent_delta_yuan"],
                            values["centered_advantage_std_yuan"],
                        )
                        cursor += 1
                        states.add(key)
                        by_family[r["family"]].add(key)
                        targets.append(cash)
                        if previous is not None:
                            adjacent.append(cash - previous)
                        previous = cash
                    state = engine.step(state, action)
                    raw = blob(action)
                    prefix.update(len(raw).to_bytes(4, "little") + raw)
                if state.serialize() != terminal.serialize():
                    raise RuntimeError("recorded replay terminal differs")
                row = {
                    "decisions": len(targets),
                    "states": len(states),
                    "unique_targets": len(set(targets)),
                    "target_variance": float(np.var(targets)) if targets else None,
                    "max_adjacent_target_difference": max(map(abs, adjacent), default=0),
                    "cash": float(
                        terminal.settlement.payments[game["seat"]] if terminal.settlement else 0
                    ),
                    **{f: len(by_family[f]) for f in FAMILIES},
                }
                per_game.append(row)
            del rows, records, indexed
    if cursor != total or len(per_game) != 50000:
        raise RuntimeError("incomplete historical cohort")
    atomic_json(
        root / "progress.json", {"phase": "variance_and_coverage", "stream": name, "rows": cursor}
    )
    measured = analyze_array(table)
    measured["granularity"] = {
        "games": len(per_game),
        "games_without_decisions": sum(r["decisions"] == 0 for r in per_game),
        "decisions_per_game": summary([r["decisions"] for r in per_game]),
        "distinct_observations_per_game": summary([r["states"] for r in per_game]),
        "unique_MC_targets_per_game": dict(Counter(str(r["unique_targets"]) for r in per_game)),
        "within_game_target_variance_yuan2": summary(
            [r["target_variance"] for r in per_game if r["target_variance"] is not None]
        ),
        "max_adjacent_target_difference_yuan": summary(
            [r["max_adjacent_target_difference"] for r in per_game]
        ),
        "game_weighted_terminal_yuan": summary([r["cash"] for r in per_game]),
        "families": {
            f: {
                "states_per_all_games": summary([r[f] for r in per_game]),
                "states_per_present_game": summary([r[f] for r in per_game if r[f]]),
            }
            for f in FAMILIES
        },
    }
    sampled = sample.result()
    if len(sampled) != 5000:
        raise RuntimeError("not enough distinct sampled states")
    atomic_json(root / f"{name}-samples.json", sampled)
    measured["sample_scales"] = {
        f: scale_summary([r for r in sampled if f == "all" or r["family"] == f])
        for f in ("all", *FAMILIES)
        if any(f == "all" or r["family"] == f for r in sampled)
    }
    measured["sampling"] = (
        "5000 distinct full Observation hashes uniform min-hash; representative occurrence uniform min-hash; seed20261017"
    )
    np.savez_compressed(root / f"{name}-statistics.npz", rows=table)
    atomic_json(root / f"{name}-result.json", measured)
    inputs.update(current_sources)
    return measured


def gradients(root, inputs):
    base = stream_path("LR-B", 10)
    manifest = json.loads((base / "data/manifest.json").read_text())
    rows = load_rows(base / "data", manifest)
    indices = np.random.default_rng(SEED).choice(len(rows), 4096, replace=False)
    selected = [
        {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rows[int(i)].items()}
        for i in indices
    ]
    del rows
    atomic_json(
        root / "gradient-sample-lock.json",
        {
            "stream": "LR-B",
            "generation": 10,
            "row_indices": indices.tolist(),
            "seed": SEED,
            "manifest_sha256": digest(base / "data/manifest.json"),
        },
    )
    models = {
        "initial": Path(CHECKPOINT),
        "long10": stream_path("long", 10) / "checkpoint.pt",
        "A10": stream_path("LR-A", 10) / "checkpoint.pt",
        "B9-producer": stream_path("LR-B", 9) / "checkpoint.pt",
        "B10": base / "checkpoint.pt",
    }
    result = {}
    for name, path in models.items():
        atomic_json(root / "progress.json", {"phase": "read_only_autograd", "checkpoint": name})
        inputs[str(path)] = digest(path)
        model = load_model(path, "cuda")
        if name == "B9-producer":
            wanted = {(int(r["seed"]), int(r["seat"]), int(r["history_step"])): r for r in selected}
            for file in sorted(base.glob("evidence-*/decisions-*.jsonl.gz")):
                with gzip.open(file, "rt") as f:
                    for line in f:
                        r = json.loads(line)
                        key = (r["seed"], r["seat"], r["step"] - 1)
                        if key in wanted:
                            wanted[key]["recorded_q"] = np.array(r["q"])
            qs = q_probe(model, selected, "cuda")
            for r, q in zip(selected, qs, strict=True):
                np.testing.assert_allclose(q, r["recorded_q"], atol=5e-4, rtol=1e-5)
        state = parameter_hash(model)
        result[name] = {
            "checkpoint_sha256": inputs[str(path)],
            "model_state_sha256": state,
            "eval": gradient_probe(model, selected, tensors, device="cuda"),
            "train_isolated_BN": gradient_probe(
                model, selected, tensors, device="cuda", train_mode=True
            ),
        }
        if parameter_hash(model) != state or digest(path) != inputs[str(path)]:
            raise RuntimeError("gradient probe changed checkpoint")
        atomic_json(root / "gradients.json", result)
        del model
    return result


def run(directory):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for bounded read-only gradient probes")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    inputs = {p: digest(p) for p in (BASELINE, CHECKPOINT)}
    source = {str(p): digest(p) for p in sorted(Path("zimortal").glob("**/*.py"))}
    source["scripts/diagnose-mortal-credit-assignment.py"] = digest(
        "scripts/diagnose-mortal-credit-assignment.py"
    )
    config = {
        "date": "2026-10-17",
        "streams": STREAMS,
        "seed": SEED,
        "new_games": 0,
        "optimizer_steps": 0,
        "exact": "entire Observation including ordered history prefix digest, river, pending, phase, pass flags, public hands/melds; canonical full legal action set",
        "encoded": "55x20 tensor plus sorted encoded legal candidate multiset; not exact full Observation",
        "fine": "exact hand20 counts + exact locked huxi + family + K + deck buckets 0/1-5/6-10/11-15/16-19 + 18 relative-seat meld-kind tile totals + full legal set hash",
        "broad": "small/big hand multiplicity histograms0..4 + red hand count + huxi//3 + family + K + deck bucket + meld totals18 + pending tile",
        "compatible": "broad + identical full legal candidate set",
        "variance": "descriptive row-weighted ANOVA, cross-game collapsed within variance and leave-one-game-out means; singleton noise unidentifiable",
        "primary_coverage": "each independent stream separately; LR A/B common-seed copies never pooled as independent visits",
        "sample": "5000 distinct exact states per stream; minhash seed20261017, outcome-blind occurrence representative",
        "gradient": "same4096 uniformly random LR-B Gen10 rows for initial/long10/A10/B9/B10; eval and training-BN-on-functional-clones; autograd.grad only",
        "units": "targets/Q/residual yuan; gradients on cash/100 half MSE; anchor score x100 is yuan-equivalent PRIOR score, not cash EV",
        "decision_rule": "assess label sharing, residual/margin>10, low exact/fine different-action coverage(<1% rows), mean-coupling derivative; descriptive support cannot prove principal causal bottleneck",
        "source_sha256": source,
    }
    atomic_json(root / "config.json", config)
    results = {}
    for name in STREAMS:
        results[name] = measure_stream(name, root, inputs)
    grad = gradients(root, inputs)
    for p, sha in (inputs | source).items():
        if digest(p) != sha:
            raise RuntimeError(f"read-only input changed: {p}")
    atomic_json(root / "input-sha256.json", inputs)
    atomic_json(
        root / "result.json",
        {
            "status": "complete_stop",
            "streams": results,
            "gradients": grad,
            "new_games": 0,
            "optimizer_steps": 0,
            "checkpoint_and_BN_unchanged": True,
            "input_sha_verified": True,
            "automatic_training": False,
        },
    )
    atomic_json(root / "progress.json", {"phase": "complete_stop"})
    print(
        "MC credit diagnosis complete; no games generated, no optimizer steps, all original states unchanged",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-credit-diagnostics-v1")
    run(parser.parse_args().directory)
