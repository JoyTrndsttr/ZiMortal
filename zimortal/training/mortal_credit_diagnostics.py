"""Descriptive MC credit diagnostics only. Never choose actions or update models."""

import dataclasses
import hashlib
import json
from enum import Enum

import numpy as np
import torch
from torch.func import functional_call

FAMILIES = ("discard", "chi/pass", "peng/pass", "hu/pass")


def canonical(value):
    if dataclasses.is_dataclass(value):
        return {f.name: canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def blob(value):
    return json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()


def hashed(value):
    return hashlib.sha256(blob(value)).digest()


def relative_action(action, player):
    result = canonical(action)
    result["player"] = (result["player"] - player) % 3
    if result["source_player"] is not None:
        result["source_player"] = (result["source_player"] - player) % 3
    return result


def exact_observation(obs, history_digest):
    # Canonical set of full structured actions, not just candidate indices/encodings.
    actions = sorted(blob(a) for a in obs.legal_actions)
    legal = hashlib.sha256(b"".join(len(a).to_bytes(4, "little") + a for a in actions)).digest()
    visible = {
        f.name: getattr(obs, f.name)
        for f in dataclasses.fields(obs)
        if f.name not in ("history", "legal_actions")
    }
    return hashlib.sha256(blob(visible) + history_digest + legal).digest(), legal


def encoded_key(features, actions):
    x = np.asarray(features, dtype="<f4").tobytes()
    a = b"".join(sorted(np.asarray(r, dtype="<f4").tobytes() for r in actions))
    return hashlib.sha256(x + a).digest()


def deck_bucket(n):
    return 0 if n == 0 else (n - 1) // 5 + 1


def coarse_keys(features, family, nlegal, legal):
    hand = np.rint(features[0] * 4).astype(int)
    hu = round(float(features[22, 0]) * 60)
    deck = round(float(features[27, 0]) * 20)
    melds = np.rint(features[4:22].sum(1) * 4).astype(int).tolist()
    fine = [hand.tolist(), hu, family, nlegal, deck_bucket(deck), melds, legal.hex()]
    multiplicities = [[int(np.sum(half == k)) for k in range(5)] for half in (hand[:10], hand[10:])]
    red = int(hand[[1, 6, 9, 11, 16, 19]].sum())
    pending = np.flatnonzero(features[38] > 0.5).tolist()
    broad = [multiplicities, red, hu // 3, family, nlegal, deck_bucket(deck), melds, pending]
    return hashed(fine), hashed(broad), hashed([broad, legal.hex()])


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0}
    return {
        "count": len(a),
        "mean": float(a.mean()),
        "std": float(a.std()),
        **{f"p{k}": float(np.percentile(a, k)) for k in (5, 25, 50, 75, 95)},
    }


def key_pairs(a, b):
    result = np.empty(len(a), dtype=[("a", a.dtype), ("b", b.dtype)])
    result["a"], result["b"] = a, b
    return result


def group_variance(key, cash, game):
    """ANOVA is descriptive. Leave-one-GAME-out means avoid same-game label leakage."""
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=cash)
    means = sums / counts
    residual = cash - means[inv]
    total = float(np.var(cash))
    within = float(np.mean(residual**2))
    pair = key_pairs(inv, game)
    _, first, game_inv, game_counts = np.unique(
        pair, return_index=True, return_inverse=True, return_counts=True
    )
    unique_groups = inv[first]
    ngames = np.bincount(unique_groups, minlength=len(counts))
    # One cash per (signature, game), irrespective of repeated decisions inside that game.
    unique_cash = cash[first]
    usum = np.bincount(unique_groups, weights=unique_cash, minlength=len(counts))
    umean = usum / np.maximum(1, ngames)
    usse = float(np.sum((unique_cash - umean[unique_groups]) ** 2))
    df = len(first) - len(counts)
    other = counts[inv] - game_counts[game_inv]
    eligible = other > 0
    predictions = np.zeros(len(cash))
    predictions[eligible] = (
        sums[inv[eligible]] - game_counts[game_inv[eligible]] * cash[eligible]
    ) / other[eligible]
    return {
        "groups": len(counts),
        "visits": summary(counts),
        "independent_games": summary(ngames),
        "total_variance_yuan2": total,
        "within_group_variance_yuan2": within,
        "between_group_variance_yuan2": max(0.0, total - within),
        "within_fraction": within / total if total else None,
        "cross_game_repeated_groups": int(np.sum(ngames >= 2)),
        "cross_game_within_std_yuan": (usse / df) ** 0.5 if df else None,
        "cross_game_within_df": df,
        "leave_one_game_out_coverage": float(eligible.mean()),
        "leave_one_game_out_residual_yuan": summary((cash - predictions)[eligible]),
        "scope": "coarsened state heterogeneity + continuation/policy variation + outcome randomness; not identified irreducible noise",
    }


def coverage(key, action, cash, game):
    variance = group_variance(key, cash, game)
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    pairs = np.unique(key_pairs(inv, action))
    distinct = np.bincount(pairs["a"], minlength=len(counts))
    multi = distinct >= 2
    repeated = counts >= 2
    return {
        **variance,
        "states": len(counts),
        "rows": len(key),
        "repeated_states": int(repeated.sum()),
        "repeated_row_fraction": float(np.mean(repeated[inv])),
        "multi_action_states": int(multi.sum()),
        "multi_action_state_fraction": float(multi.mean()),
        "multi_action_row_fraction": float(np.mean(multi[inv])),
        "multi_action_among_repeated_states": float(multi[repeated].mean())
        if repeated.any()
        else None,
        "distinct_executed_actions": summary(distinct),
        "state_action_variance": group_variance(key_pairs(key, action), cash, game),
    }


def scales(q, reference, parent, chosen, cash, *, scores=None):
    q = np.asarray(q, float) * 100
    score = q + 30 * np.log(np.asarray(reference, dtype=np.float32) + np.float32(1e-8))
    if scores is not None:
        np.testing.assert_allclose(np.asarray(scores) * 100, score, atol=1e-4, rtol=1e-5)
        score = np.asarray(scores, float) * 100
    ranked, scores = np.sort(q), np.sort(score)
    anchor = int(np.argmax(score))
    residual = float(cash - q[chosen])
    return {
        "legal_actions": len(q),
        "cash_yuan": float(cash),
        "chosen_q_yuan": float(q[chosen]),
        "raw_margin_yuan": float(ranked[-1] - ranked[-2]),
        "raw_parent_delta_yuan": float(q.max() - q[parent]),
        "anchor_parent_delta_yuan": float(q[anchor] - q[parent]),
        "anchor_score_margin_yuan_equivalent": float(scores[-1] - scores[-2]),
        "centered_advantage_std_yuan": float(q.std()),
        "residual_yuan": residual,
        "abs_residual_yuan": abs(residual),
    }


def scale_summary(rows):
    fields = rows[0].keys()
    result = {k: summary([r[k] for r in rows]) for k in fields if k not in ("family", "locator")}
    residual = np.array([r["residual_yuan"] for r in rows])
    target = np.array([r["cash_yuan"] for r in rows])
    pred = np.array([r["chosen_q_yuan"] for r in rows])
    result["chosen_fit"] = {
        "mae_yuan": float(abs(residual).mean()),
        "rmse_yuan": float(np.sqrt(np.mean(residual**2))),
        "error_std_yuan": float(residual.std()),
        "target_std_yuan": float(target.std()),
        "r2": 1 - float(np.mean(residual**2)) / float(target.var()) if target.var() else None,
        "correlation": float(np.corrcoef(pred, target)[0, 1])
        if pred.std() and target.std()
        else None,
    }
    for field in (
        "raw_margin_yuan",
        "anchor_score_margin_yuan_equivalent",
        "centered_advantage_std_yuan",
        "anchor_parent_delta_yuan",
    ):
        denominator = np.abs([r[field] for r in rows])
        valid = denominator > 0
        result[f"abs_residual_over_{field}"] = {
            "positive_denominator_rows": int(valid.sum()),
            "zero_denominator_rows": int((~valid).sum()),
            "ratios": summary(abs(residual[valid]) / denominator[valid]),
        }
    return result


def gradient_probe(model, rows, tensors, *, device="cuda", train_mode=False, batch=256):
    """Autograd only; functional cloned BN buffers, no optimizer/.backward()/state update."""
    from .mortal_sarsa import parameter_hash

    before = parameter_hash(model)
    flags = [(m, m.training) for m in model.modules()]
    params = dict(model.named_parameters())
    names = tuple(params)
    groups = {
        g: [n for n in names if n.startswith(g + ".")]
        for g in ("brain", "value", "advantage", "action_encoder")
    }
    counts = {g: sum(params[n].numel() for n in ns) for g, ns in groups.items()}
    families = ["all", *FAMILIES]
    sizes = {f: int(sum(f == "all" or r["family"] == f for r in rows)) for f in families}
    accum = {
        f: {g: torch.zeros(counts[g], dtype=torch.float64) for g in groups}
        for f in FAMILIES
        if sizes[f]
    }
    batch_norms = {g: [] for g in groups}
    capture = []
    handle = model.advantage.register_forward_hook(lambda module, args, out: capture.append(out))
    max_direct_error = 0.0
    try:
        model.train(train_mode)
        for start in range(0, len(rows), batch):
            selected = rows[start : start + batch]
            x, a, mask, chosen, target = tensors(selected, device, 1.0)
            capture.clear()
            # Clone EVERY buffer. Original BN running mean/var/count remain bit-identical.
            buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
            q, _, _ = functional_call(model, (params, buffers), (x, a, mask), strict=True)
            err = q.gather(1, chosen[:, None]).squeeze(1) - target
            losses = 0.5 * err.square()
            raw = capture[-1]
            direct = torch.autograd.grad(losses.mean(), raw, retain_graph=True)[0].squeeze(-1)
            theory = -mask.to(q.dtype) / mask.sum(1)[:, None]
            theory.scatter_add_(1, chosen[:, None], torch.ones((len(selected), 1), device=device))
            theory *= err.detach()[:, None] / len(selected)
            max_direct_error = max(max_direct_error, float((direct - theory).abs().max()))
            present = [f for f in FAMILIES if any(r["family"] == f for r in selected)]
            weighted = {g: torch.zeros(counts[g], dtype=torch.float64) for g in groups}
            for i, f in enumerate(present):
                indices = [j for j, r in enumerate(selected) if r["family"] == f]
                grads = torch.autograd.grad(
                    losses[indices].mean(),
                    tuple(params.values()),
                    retain_graph=i < len(present) - 1,
                    allow_unused=True,
                )
                by_name = dict(zip(names, grads, strict=True))
                for g, ns in groups.items():
                    vec = torch.cat(
                        [
                            (
                                by_name[n].detach().cpu().double()
                                if by_name[n] is not None
                                else torch.zeros_like(params[n], device="cpu", dtype=torch.float64)
                            ).flatten()
                            for n in ns
                        ]
                    )
                    accum[f][g] += vec * (len(indices) / sizes[f])
                    weighted[g] += vec * (len(indices) / len(selected))
            for g in groups:
                batch_norms[g].append(float(weighted[g].norm()))
        result = {}
        for f in families:
            if not sizes[f]:
                result[f] = {"rows": 0}
                continue
            vectors = (
                accum[f]
                if f != "all"
                else {
                    g: sum(v[g] * (sizes[k] / sizes["all"]) for k, v in accum.items())
                    for g in groups
                }
            )
            result[f] = {
                "rows": sizes[f],
                "groups": {
                    g: {
                        "parameters": counts[g],
                        "mean_gradient_l2": float(v.norm()),
                        "gradient_rms_per_parameter": float(v.norm()) / counts[g] ** 0.5,
                        "parameter_l2": sum(
                            float(params[n].detach().cpu().double().square().sum())
                            for n in groups[g]
                        )
                        ** 0.5,
                    }
                    for g, v in vectors.items()
                },
            }
            result[f]["last_advantage_bias_gradient"] = float(vectors["advantage"][-1])
            result[f]["last_value_bias_gradient"] = float(vectors["value"][-1])
            result[f]["advantage_plus_encoder_l2"] = (
                sum(
                    result[f]["groups"][g]["mean_gradient_l2"] ** 2
                    for g in ("advantage", "action_encoder")
                )
                ** 0.5
            )
        result["all"]["batch_gradient_norms"] = {g: summary(v) for g, v in batch_norms.items()}
        result["direct_derivative_max_abs_error"] = max_direct_error
        result["mode"] = (
            "train BN on isolated functional buffers" if train_mode else "eval frozen BN"
        )
        result["loss_units"] = (
            "half squared cash/100 error, gradients pre-clipping; no parameter update"
        )
    finally:
        handle.remove()
        for m, mode in flags:
            m.training = mode
    if parameter_hash(model) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError(
            "read-only gradient diagnostic changed parameters/BN or accumulated .grad"
        )
    result["state_hash_unchanged"] = True
    return result
