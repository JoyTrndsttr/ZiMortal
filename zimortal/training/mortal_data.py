"""Complete-game, observation-only bootstrap data with audited terminal cash."""

import gzip
import hashlib
import io
import json
import math
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import RuleClarificationRequired, RuleEngine
from zimortal.engine.observation import Observation
from zimortal.model.encoding import encode_action, encode_observation

from .active import atomic_json
from .online_rl import pack
from .rollout_engine import RolloutEngine

RESERVE = 2 * 1024**3


def write_bytes(path, value):
    import shutil

    path = Path(path)
    if shutil.disk_usage(path.parent).free < RESERVE + len(value):
        raise OSError("insufficient 2GiB reserve for mortal evidence")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as file:
        file.write(value)
        file.flush()
        os.fsync(file.fileno())
    tmp.replace(path)
    return hashlib.sha256(value).hexdigest()


def encode_visible(obs):
    if not isinstance(obs, Observation):
        raise TypeError("dataset inputs require Observation")
    return encode_observation(obs, "huxi"), np.stack(
        [encode_action(a, obs.player) for a in obs.legal_actions]
    )


def exploration(logits, *, temperature=1.5, epsilon=0.0, top_p=1.0):
    """Temperature/nucleus distribution plus optional UNIFORM epsilon mixture.

    This epsilon differs from Mortal's greedy/Boltzmann mixture; disabled v1.
    Input contains only legal candidates, so padding never receives probability.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all() or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("finite logits and positive temperature required")
    if not 0 <= epsilon <= 1 or not 0 < top_p <= 1:
        raise ValueError("invalid exploration settings")
    p = np.exp((logits - logits.max()) / temperature)
    p /= p.sum()
    order = np.argsort(-p, kind="stable")
    keep = np.cumsum(p[order]) - p[order] < top_p
    p[order[~keep]] = 0
    p /= p.sum()
    return (1 - epsilon) * p + epsilon / len(p)


def specs(start, count):
    # Three seats independently crossed with three dealer positions.
    return [(start + i, i % 3, (i // 3) % 3) for i in range(count)]


def terminal_rows(rows, state, *, seed, seat, game_id, version):
    if not state.terminal:
        raise ValueError("terminal labels require a finished game")
    state.validate()
    if state.settlement is None:
        if state.winner is not None:
            raise ValueError("winner without settlement")
        cash = 0.0  # genuine terminal draw only
    else:
        if sum(state.settlement.payments) != 0 or state.winner != state.settlement.winner:
            raise ValueError("inconsistent settlement")
        cash = state.settlement.payments[seat] / 100.0
    return [
        {
            **r,
            "terminal_cash": cash,
            "steps_to_terminal": len(rows) - j - 1,
            "seed": seed,
            "seat": seat,
            "game_id": game_id,
            "policy_version": version,
        }
        for j, r in enumerate(rows)
    ]


@contextmanager
def solver_profile(counters, engine):
    from zimortal.engine import game

    legal = engine.legal_actions

    def measured_legal(*a, **kw):
        started = time.perf_counter()
        try:
            return legal(*a, **kw)
        finally:
            counters["legal_actions_seconds"] += time.perf_counter() - started

    engine.legal_actions = measured_legal
    originals = {}
    for name, label in [
        ("evaluate_hand", "hu_solver"),
        ("enumerate_chi_with_required_bi", "chi_enumeration"),
    ]:
        originals[name] = getattr(game, name)

        def wrapped(*a, _fn=originals[name], _label=label, **kw):
            started = time.perf_counter()
            try:
                return _fn(*a, **kw)
            finally:
                counters[_label + "_seconds"] += time.perf_counter() - started

        setattr(game, name, wrapped)
    try:
        yield
    finally:
        engine.legal_actions = legal
        for name, fn in originals.items():
            setattr(game, name, fn)


def generate_games(
    baseline,
    game_specs,
    *,
    device="cpu",
    slots=32,
    temperature=1.5,
    epsilon=0.0,
    top_p=1.0,
    engine=None,
    version="ZiMortal-v0",
):
    """Frozen v0 with one exploratory learner vs two greedy v0; all rows atomic per game."""
    engine = engine or RolloutEngine()
    baseline.eval()
    states, cursor, completed, failures = {}, 0, [], []
    counters = Counter()
    started = time.perf_counter()
    cpu_started = sum(os.times()[:2])

    def timed(name, fn, *a):
        begin = time.perf_counter()
        try:
            return fn(*a)
        finally:
            counters[name + "_seconds"] += time.perf_counter() - begin

    def fail(index, exc):
        st, spec, _, _ = states.pop(index)
        failures.append(
            {
                "seed": spec[0],
                "dealer": spec[1],
                "seat": spec[2],
                "error": str(exc),
                "state": st.serialize(),
                "status": "whole game excluded",
            }
        )

    with solver_profile(counters, engine):
        while cursor < len(game_specs) or states:
            for index in range(slots):
                if index not in states and cursor < len(game_specs):
                    spec = game_specs[cursor]
                    states[index] = [
                        engine.new_game(spec[0], spec[1]),
                        spec,
                        [],
                        np.random.default_rng(spec[0]),
                    ]
                    cursor += 1
            waiting = []
            for index, (state, spec, rows, rng) in list(states.items()):
                try:
                    if state.terminal:
                        initial = RuleEngine().new_game(spec[0], spec[1])
                        checked = timed(
                            "replay_validation", RuleEngine().replay, initial, state.history
                        )
                        if checked.serialize() != state.serialize():
                            raise RuntimeError("complete game replay mismatch")
                        labeled = terminal_rows(
                            rows,
                            state,
                            seed=spec[0],
                            seat=spec[2],
                            game_id=spec[0],
                            version=version,
                        )
                        completed.append(
                            {
                                "seed": spec[0],
                                "dealer": spec[1],
                                "seat": spec[2],
                                "rows": labeled,
                                "state": state.serialize(),
                            }
                        )
                        del states[index]
                        continue
                    if len(state.history) >= 1000:
                        raise RuntimeError("nonterminal game budget exceeded")
                    actions = engine.legal_actions(state)
                    if len(actions) == 1:
                        states[index][0] = timed("step", engine.step, state, actions[0])
                        states[index][0].validate()
                    else:
                        obs = timed("observation", engine.observation, state, actions[0].player)
                        begin = time.perf_counter()
                        features, candidates = encode_visible(obs)
                        counters["encoding_seconds"] += time.perf_counter() - begin
                        waiting.append((index, obs, features, candidates))
                except RuleClarificationRequired as exc:
                    fail(index, exc)
            if not waiting:
                continue
            begin = time.perf_counter()
            with torch.inference_mode():
                scores = (
                    baseline(*pack([r[2] for r in waiting], [r[3] for r in waiting], device))[0]
                    .cpu()
                    .numpy()
                )
            counters["network_inference_seconds"] += time.perf_counter() - begin
            for j, (index, obs, features, candidates) in enumerate(waiting):
                state, spec, rows, rng = states[index]
                logits = scores[j, : len(candidates)]
                choice = int(logits.argmax())
                if obs.player == spec[2]:
                    p = exploration(logits, temperature=temperature, epsilon=epsilon, top_p=top_p)
                    chosen = int(rng.choice(len(p), p=p))
                    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
                    family = (
                        "hu/pass"
                        if any(a.kind.value == "hu" for a in obs.legal_actions)
                        else "chi/pass"
                        if any(a.kind.value == "chi" for a in obs.legal_actions)
                        else "peng/pass"
                        if any(a.kind.value == "peng" for a in obs.legal_actions)
                        else "discard"
                    )
                    counters["entropy_sum"] += entropy
                    counters["changed_from_greedy"] += chosen != choice
                    counters["decisions"] += 1
                    counters[family + "_decisions"] += 1
                    counters[family + "_changed"] += chosen != choice
                    rows.append(
                        {
                            "features": features,
                            "actions": candidates,
                            "chosen": chosen,
                            "behavior_logprob": float(np.log(p[chosen])),
                            "behavior_probabilities": p.astype(np.float32),
                            "history_step": len(state.history),
                            "family": family,
                            "parent_greedy": choice,
                        }
                    )
                    choice = chosen
                action = obs.legal_actions[choice]
                if action not in engine.legal_actions(state):
                    raise RuntimeError("illegal behavior action")
                try:
                    states[index][0] = timed("step", engine.step, state, action)
                    states[index][0].validate()
                except RuleClarificationRequired as exc:
                    fail(index, exc)
    elapsed = time.perf_counter() - started
    counters["wall_seconds"] = elapsed
    counters["games_per_second"] = len(completed) / elapsed
    counters["decisions_per_second"] = counters["decisions"] / elapsed
    counters["process_cpu_percent"] = 100 * (sum(os.times()[:2]) - cpu_started) / elapsed
    counters["timing_note"] = (
        "inclusive timers overlap: step/replay include hu/chi; CPU%=process seconds/wall, 100%=one core"
    )
    return completed, failures, dict(counters)


def save_shard(directory, index, games):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rows = [row for g in games for row in g["rows"]]
    offsets = np.cumsum([0] + [len(r["actions"]) for r in rows])
    arrays = {
        "features": np.stack([r["features"] for r in rows]),
        "actions": np.concatenate([r["actions"] for r in rows]),
        "action_offsets": offsets,
        "behavior_probabilities": np.concatenate([r["behavior_probabilities"] for r in rows]),
        **{
            k: np.array([r[k] for r in rows])
            for k in (
                "chosen",
                "terminal_cash",
                "steps_to_terminal",
                "seed",
                "seat",
                "game_id",
                "behavior_logprob",
                "history_step",
                "parent_greedy",
                "policy_version",
            )
        },
        "family": np.array([r["family"] for r in rows]),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    data_name = f"shard-{index:04d}.npz"
    data_hash = write_bytes(root / data_name, buffer.getvalue())
    replay_name = f"shard-{index:04d}-replays.json.gz"
    evidence = [{k: v for k, v in g.items() if k != "rows"} for g in games]
    replay_hash = write_bytes(
        root / replay_name, gzip.compress(json.dumps(evidence).encode(), mtime=0)
    )
    return {
        "data": data_name,
        "sha256": data_hash,
        "replays": replay_name,
        "replay_sha256": replay_hash,
        "games": len(games),
        "decisions": len(rows),
        "seeds": [g["seed"] for g in games],
    }


def load_rows(directory, manifest):
    """Read each NPZ member once; rows share backing slices, never repeated decompression."""
    rows = []
    for shard in manifest["shards"]:
        path = Path(directory) / shard["data"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError("shard hash mismatch")
        with np.load(path, allow_pickle=False) as file:
            arrays = {k: file[k] for k in file.files}
        for i in range(len(arrays["chosen"])):
            lo, hi = arrays["action_offsets"][i : i + 2]
            rows.append(
                {
                    k: v[i]
                    for k, v in arrays.items()
                    if k not in ("actions", "action_offsets", "behavior_probabilities")
                }
                | {
                    "actions": arrays["actions"][lo:hi],
                    "behavior_probabilities": arrays["behavior_probabilities"][lo:hi],
                }
            )
    return rows


def generate_dataset(
    baseline,
    directory,
    *,
    games=1000,
    start=1000000000,
    shard_games=256,
    device="cpu",
    slots=32,
    temperature=1.5,
    epsilon=0.0,
    top_p=1.0,
    metadata=None,
):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": "mortal-v1",
        "metadata": metadata or {},
        "shards": [],
        "failures": [],
        "profiles": [],
    }
    cursor = 0
    for index in range(math.ceil(games / shard_games)):
        wanted = min(shard_games, games - index * shard_games)
        accepted = []
        for _ in range(8):
            need = wanted - len(accepted)
            if need == 0:
                break
            batch = specs(start + cursor, need)
            # Rotate independently of shard boundaries/replacements.
            batch = [
                (s, (cursor + i) % 3, ((cursor + i) // 3) % 3) for i, (s, _, _) in enumerate(batch)
            ]
            result, errors, profile = generate_games(
                baseline,
                batch,
                device=device,
                slots=slots,
                temperature=temperature,
                epsilon=epsilon,
                top_p=top_p,
            )
            accepted.extend(result)
            cursor += need
            manifest["failures"].extend(errors)
            manifest["profiles"].append(profile)
        if len(accepted) != wanted:
            raise RuntimeError("insufficient complete games; no fabricated labels")
        manifest["shards"].append(save_shard(root, index, accepted))
        atomic_json(root / "manifest.json", manifest)
        atomic_json(
            root.parent / "progress.json",
            {
                "status": "data_generation",
                "games": sum(s["games"] for s in manifest["shards"]),
                "target": games,
            },
        )
    return manifest


def audit_dataset(directory, manifest):
    """Verify saved rows against complete histories, independently of behavior net."""
    from zimortal.engine.types import GameState

    rows = load_rows(directory, manifest)
    by_game = {}
    for row in rows:
        by_game.setdefault(int(row["seed"]), []).append(row)
    verified = 0
    for shard in manifest["shards"]:
        path = Path(directory) / shard["replays"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != shard["replay_sha256"]:
            raise ValueError("replay hash mismatch")
        for game in json.loads(gzip.decompress(raw)):
            engine = RuleEngine()
            terminal = GameState.deserialize(game["state"])
            terminal.validate()
            state = engine.new_game(game["seed"], game["dealer"])
            game_rows = by_game.get(game["seed"], [])
            indexed = {int(r["history_step"]): r for r in game_rows}
            if len(indexed) != len(game_rows):
                raise ValueError("duplicate decision step")
            seen = 0
            for step, action in enumerate(terminal.history):
                legal = engine.legal_actions(state)
                if action not in legal:
                    raise ValueError("saved history contains illegal action")
                if len(legal) > 1 and action.player == game["seat"]:
                    r = indexed[step]
                    obs = engine.observation(state, action.player)
                    features, actions = encode_visible(obs)
                    if not np.array_equal(features, r["features"]) or not np.array_equal(
                        actions, r["actions"]
                    ):
                        raise ValueError("saved visible inputs differ from replay")
                    if legal[int(r["chosen"])] != action:
                        raise ValueError("chosen candidate differs from replay")
                    probabilities = r["behavior_probabilities"]
                    if not np.isclose(probabilities.sum(), 1) or (probabilities < 0).any():
                        raise ValueError("invalid behavior probabilities")
                    if not np.isclose(
                        math.log(float(probabilities[int(r["chosen"])])),
                        float(r["behavior_logprob"]),
                        atol=1e-6,
                    ):
                        raise ValueError("chosen behavior logprob differs")
                    cash = (
                        terminal.settlement.payments[game["seat"]] / 100
                        if terminal.settlement
                        else 0
                    )
                    if (
                        float(r["terminal_cash"]) != np.float32(cash)
                        or int(r["steps_to_terminal"]) != len(game_rows) - seen - 1
                    ):
                        raise ValueError("terminal cash/discount step label mismatch")
                    seen += 1
                state = engine.step(state, action)
                state.validate()
            if seen != len(game_rows) or state.serialize() != terminal.serialize():
                raise ValueError("incomplete saved game replay")
            verified += 1
    return {
        "games": verified,
        "decisions": len(rows),
        "replay_verified": True,
        "visible_rows_verified": True,
    }
