"""Finite MC-only online pilot. Actors are anchored; raw Q is diagnostic only."""

import gzip
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zimortal.engine import GameState

from .active import atomic_json
from .mortal_data import encode_visible, terminal_rows
from .mortal_offline import tensors
from .online_rl import atomic_checkpoint, play
from .runtime import save_model


def mc_loss(q, chosen, cash, legal):
    if not legal.gather(1, chosen[:, None]).all() or not torch.isfinite(q[legal]).all():
        raise ValueError("nonfinite or illegal online Q")
    return .5 * nn.functional.mse_loss(q.gather(1, chosen[:, None]).squeeze(1), cash)


def make_optimizer(model):
    decay = {id(p) for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv1d))
             for n, p in m.named_parameters(recurse=False) if n == "weight"}
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in decay], "weight_decay": .01},
        {"params": [p for p in model.parameters() if id(p) not in decay], "weight_decay": 0.},
    ], lr=3e-5, betas=(.9, .999), eps=1e-8)


def learning_rate(step):
    # Same linear warmup / cosine shape as Mortal; fixed pilot horizon, not tuned.
    if step < 20:
        return 3e-6 + (3e-5 - 3e-6) * step / 20
    phase = min(1., (step - 20) / 300)
    return 3e-6 + .5 * (3e-5 - 3e-6) * (1 + math.cos(math.pi * phase))


def update(model, optimizer, rows, rng, updates, *, device="cpu", batch_size=256,
           lr_schedule=None):
    if not rows:
        raise ValueError("no current-policy online rows")
    model.requires_grad_(True).train()
    logs = []
    for start in range(0, len(rows), batch_size):
        if start == 0:
            order = rng.permutation(len(rows))
        batch = [rows[i] for i in order[start:start + batch_size]]
        x, a, legal, chosen, target = tensors(batch, device, 1.)
        for group in optimizer.param_groups:
            group["lr"] = (learning_rate if lr_schedule is None else lr_schedule)(updates)
        q = model.forward_q(x, a, legal)[0]
        loss = mc_loss(q, chosen, target, legal)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite online MC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite online gradient")
        optimizer.step()
        updates += 1
        logs.append({"update": updates, "loss": float(loss.detach()),
                     "gradient_norm": float(norm), "lr": optimizer.param_groups[0]["lr"],
                     "batch": len(batch), "target_mean": float(target.mean()),
                     "cql_loss": 0., "auxiliary_loss": 0.})
    model.eval()
    return updates, logs


def collect(policy, parent, specs, directory, name, *, training_rows=False, device="cpu"):
    """Greedy deployed actor visits real states. No PPO, counterfactuals or raw training."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    pending, data_games, records = {}, [], []
    completed = 0
    with gzip.open(root / f"screen-{name}-replays.jsonl.gz", "wt") as replays, gzip.open(
        root / f"decisions-{name}.jsonl.gz", "wt"
    ) as decisions:
        def record(obs, index, chosen, spec, step):
            r = policy.record(obs, index, chosen, spec, step)
            if training_rows:
                r["_features"], r["_actions"] = encode_visible(obs)
            return r

        def sink(e):
            replays.write(json.dumps(e) + "\n")
            pending.update(e=e, state=GameState.deserialize(e["state"]))

        def decision_sink(batch):
            nonlocal completed
            e, state = pending["e"], pending["state"]
            rows = []
            for r in batch:
                if (r["seed"], r["seat"]) != (e["seed"], e["seat"]):
                    raise RuntimeError("cross-game online decisions")
                actual = json.loads(json.dumps(asdict(state.history[r["step"] - 1])))
                if actual != json.loads(json.dumps(r["actions"][r["chosen"]])):
                    raise RuntimeError("online decision differs from executed history")
                if r["terminal_cash"] != (state.settlement.payments[e["seat"]] if state.settlement else 0):
                    raise RuntimeError("online terminal cash differs from settlement")
                evidence = {k: v for k, v in r.items() if not k.startswith("_")}
                decisions.write(json.dumps(evidence) + "\n")
                records.append({k: v for k, v in evidence.items() if k not in ("actions", "q", "scores", "probabilities")})
                if training_rows:
                    # Actual execution distribution is separate from frozen reference p.
                    if "actor_probabilities" in r:
                        probability = np.asarray(r["actor_probabilities"], dtype=np.float32)
                    else:
                        probability = np.zeros(len(r["actions"]), np.float32)
                        probability[r["chosen"]] = 1.
                    if (probability < 0).any() or not np.isclose(probability.sum(), 1) or probability[r["chosen"]] <= 0:
                        raise RuntimeError("invalid executed actor probability")
                    rows.append({
                        "features": r["_features"], "actions": r["_actions"],
                        "chosen": r["chosen"], "parent_greedy": r["parent"],
                        "behavior_probabilities": probability, "behavior_logprob": float(np.log(float(probability[r["chosen"]]))),
                        "reference_probabilities": np.array(r["probabilities"], np.float32),
                        "history_step": r["step"] - 1, "family": r["family"],
                        "chosen_kind": r["actions"][r["chosen"]]["kind"],
                        "behavior_rank": int(np.flatnonzero(np.argsort(-np.array(r["probabilities"]), kind="stable") == r["chosen"])[0]) + 1,
                    })
            if training_rows:
                data_games.append({**e, "rows": terminal_rows(
                    rows, state, seed=e["seed"], seat=e["seat"], game_id=e["seed"], version=name
                )})
            pending.clear()
            completed += 1
            if completed % 250 == 0:
                atomic_json(root / "progress.json", {"name": name, "completed": completed, "total": len(specs)})
        unused, games, errors = play(policy, [parent], specs, device=device,
                                    decision_recorder=record, decision_sink=decision_sink,
                                    replay_sink=sink, training=False,
                                    decision_context=getattr(policy, "prepare", None))
    if unused or errors or len(games) != len(specs) or completed != len(specs):
        atomic_json(root / "errors.json", errors)
        raise RuntimeError("incomplete online generation/evaluation; no replacements")
    keys = {(g["seed"], g["seat"]) for g in games}
    if len(keys) != len(specs) or keys != {(s[0], s[2]) for s in specs}:
        raise RuntimeError("duplicate/unmatched online games")
    return games, records, data_games


def checkpoint(model, optimizer, path, updates, rng, **metadata):
    save_model(model, path, online=True, target="terminal_cash/100", cql=0., updates=updates, **metadata)
    atomic_checkpoint(Path(path).with_name("optimizer.pt"), {
        "optimizer": optimizer.state_dict(), "updates": updates,
        "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
        "checkpoint_sha256": __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest(),
        "schedule": metadata.get("learning_rate_schedule", "linear20+cosine300"),
        "metadata": metadata,
    })
