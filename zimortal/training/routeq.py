"""Paired route/action cash comparison under frozen visible continuation policies.

A structural route is a goal, not a winning guarantee. Its label is the cash
of a specified bounded route-following policy, never oracle best future play.
"""

import argparse
import hashlib
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from zimortal.belief.conditional import HistorySampler
from zimortal.belief.sampling import SamplingFailure
from zimortal.engine import ActionType as A
from zimortal.engine import RuleClarificationRequired, RuleEngine
from zimortal.engine.observation import Observation
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.model.routeq import RouteQNet, plan_arrays, propose_routes

from .active import atomic_json, serialize_observation, tags
from .autonomous import atomic_gzip
from .branch_teacher import action_data, producer_hashes, visible_key
from .online_rl import digest
from .planning import atomic_save
from .rollout_engine import RolloutEngine
from .runtime import choose_batch, load_model, predict, save_model


def route_potential(obs, plan, action=None):
    """Bounded route progress from own visible cards only; never engine.step.

    A forced upgrade or disappearing supply may invalidate this fixed branch.
    Then the continuation returns to the frozen parent rather than declaring
    the player cannot win by another route.
    """
    if not isinstance(obs, Observation):
        raise TypeError("route policy requires Observation")
    if obs.hu_disabled:
        return None
    hand = Counter(obs.hand)
    protected = {t for t, n in hand.items() if n == 3}
    melds = [tuple(sorted(m.tiles)) for m in obs.players[obs.player].melds]
    if action is not None:
        if action not in obs.legal_actions:
            raise ValueError("route preview requires a legal action")
        if action.kind == A.DISCARD:
            hand[action.tile] -= 1
        elif action.kind == A.CHI:
            consumed = Counter(action.chi)
            consumed[action.tile] -= 1
            for bi in action.bi:
                consumed.update(bi)
            hand.subtract(consumed)
            melds.extend(tuple(sorted(p)) for p in (action.chi, *action.bi))
        elif action.kind == A.PENG:
            hand[action.tile] -= 2
            melds.append((action.tile,) * 3)
        elif action.kind not in (A.PASS, A.HU):
            return None  # all other transitions are forced in free decisions
    if any(n < 0 for n in hand.values()):
        raise ValueError("negative visible hand in route preview")
    desired_groups = Counter(tuple(sorted(p)) for p in plan["groups"])
    locked_groups = Counter(melds)
    locked_groups.update((t,) * 3 for t, n in hand.items() if n == 3)
    slots = desired_groups.copy()
    for pattern, n in locked_groups.items():
        if slots[pattern] >= n:
            slots[pattern] -= n
        elif len(pattern) == 3 and len(set(pattern)) == 1 and slots[(pattern[0],) * 4] >= n:
            # A newly acquired peng/wei can precede the intended fourth tile.
            # This does not permit splitting an opening protected kan.
            slots[(pattern[0],) * 4] -= n
        else:
            return None
    desired = Counter(t for p in plan["groups"] for t in p)
    available = hand + Counter(t for p in melds for t in p)
    external = Counter(obs.river)
    for seat, player in enumerate(obs.players):
        if seat != obs.player:
            external.update(t for m in player.melds for t in m.tiles)
    # A contemplated discard is public and no longer available to this player.
    if action is not None and action.kind == A.DISCARD:
        external[action.tile] += 1
    if any(desired[t] > 4 - external[t] for t in range(20)):
        return None

    def after_discard(tile=None):
        inventory = available.copy()
        public = external.copy()
        if tile is not None:
            inventory[tile] -= 1
            public[tile] += 1
        if any(desired[t] > 4 - public[t] for t in range(20)):
            return None
        missing = sum(max(desired[t] - inventory[t], 0) for t in range(20))
        # Route template occupation models protected pairs/alternatives without
        # awarding heuristic cash. Cash labels still come from terminal play.
        return -missing / 3

    if action is not None and action.kind in (A.CHI, A.PENG):
        # Opening double-ti exemption is visible in own action history.
        own_history = [a for a in obs.history if a.player == obs.player]
        opening_tis = sum(a.kind == A.TI for a in own_history)
        intervening = any(a.kind in (A.CHI, A.PENG, A.WEI, A.PAO, A.DISCARD) for a in own_history)
        if opening_tis >= 2 and not intervening:
            return after_discard()
        choices = [after_discard(t) for t, n in hand.items() if n > 0 and t not in protected]
        valid = [v for v in choices if v is not None]
        return max(valid) if valid else None
    return after_discard()


def route_choices(parent, observations, plans, *, device="cpu", strength=2.0, diagnostics=None):
    """Same observation and root plan always yield the same continuation action."""
    x = torch.as_tensor(
        np.stack([encode_observation(o, "huxi") for o in observations]), device=device
    )
    size = max(len(o.legal_actions) for o in observations)
    a = torch.zeros(
        len(observations),
        size,
        encode_action(observations[0].legal_actions[0], observations[0].player).shape[0],
        device=device,
    )
    mask = torch.zeros(len(observations), size, dtype=torch.bool, device=device)
    for i, obs in enumerate(observations):
        a[i, : len(obs.legal_actions)] = torch.as_tensor(
            np.stack([encode_action(v, obs.player) for v in obs.legal_actions]), device=device
        )
        mask[i, : len(obs.legal_actions)] = True
    with torch.inference_mode():
        logits = parent(x, a, mask)[0].cpu().numpy()
    choices, abandoned = [], 0
    for i, (obs, plan) in enumerate(zip(observations, plans, strict=True)):
        scores = logits[i, : len(obs.legal_actions)].copy()
        before = route_potential(obs, plan) if plan else None
        if before is None:
            abandoned += int(plan is not None)
        elif not any(v.kind == A.HU for v in obs.legal_actions):
            # Do not override the parent HU/pass decision with progress shaping.
            for j, action in enumerate(obs.legal_actions):
                after = route_potential(obs, plan, action)
                delta = -2 if after is None else max(-2, min(2, after - before))
                scores[j] += strength * delta
        choice = int(scores.argmax())
        if diagnostics is not None and plan is not None:
            diagnostics["goal_decisions"] += 1
            diagnostics["goal_changed_actions"] += int(
                choice != int(logits[i, : len(obs.legal_actions)].argmax())
            )
        choices.append(obs.legal_actions[choice])
    return choices, abandoned


def compare_routes(
    obs,
    plans,
    parent,
    *,
    particles=8,
    seed=0,
    device="cpu",
    max_attempts=256,
    sampler=None,
    strength=2.0,
):
    """Every (first legal action, route) uses identical replay-consistent worlds.

    A separate parent-only continuation supplies a paired reference. Different
    PASS information sets choose from their own visible observations; terminal
    outcomes never select a future action. Any error excludes the entire root.
    """
    if not isinstance(obs, Observation) or particles < 2 or len(plans) < 2:
        raise ValueError("comparison requires Observation, two routes and two particles")
    plan_arrays(obs, plans)  # reject invalid visible goals before sampling
    engine = RolloutEngine()
    sampler = sampler or HistorySampler(obs, engine=RolloutEngine())
    outcomes, raw, attempts, abandoned = [], [], 0, 0
    diagnostics = Counter()
    for particle in range(particles):
        world, used = sampler.sample(seed + particle * 104729, max_attempts)
        attempts += used
        world.validate()
        if engine.observation(world, obs.player) != obs:
            raise ValueError("particle differs from the public information set")
        before = world.serialize()
        variants = [*plans, None]
        states, descriptors = [], []
        for ai, action in enumerate(obs.legal_actions):
            for ri, plan in enumerate(variants):
                states.append(engine.step(world, action))
                descriptors.append((ai, ri, plan))
        if world.serialize() != before:
            raise RuntimeError("shared particle mutated")
        for _ in range(1000):
            indices, visible, goals = [], [], []
            for index, state in enumerate(states):
                if state.terminal:
                    continue
                actions = engine.legal_actions(state)
                if len(actions) == 1:
                    states[index] = engine.step(state, actions[0])
                else:
                    indices.append(index)
                    observation = engine.observation(state, actions[0].player)
                    visible.append(observation)
                    goals.append(descriptors[index][2] if actions[0].player == obs.player else None)
            if all(s.terminal for s in states):
                break
            if visible:
                chosen, fallback = route_choices(
                    parent,
                    visible,
                    goals,
                    device=device,
                    strength=strength,
                    diagnostics=diagnostics,
                )
                abandoned += fallback
                for index, action in zip(indices, chosen, strict=True):
                    states[index] = engine.step(states[index], action)
        else:
            raise RuntimeError("route continuation exceeded transition budget")
        cash = np.zeros((len(obs.legal_actions), len(variants)))
        branches = []
        for state, (ai, ri, _plan) in zip(states, descriptors, strict=True):
            state.validate()
            suffix = state.history[len(world.history) :]
            checked = RuleEngine().replay(world, suffix)
            if checked.serialize() != state.serialize():
                raise RuntimeError("route terminal standard-engine replay mismatch")
            settlement = state.settlement
            cash[ai, ri] = settlement.payments[obs.player] / 100 if settlement else 0
            branches.append(
                {
                    "action": ai,
                    "route": ri,
                    "cash": cash[ai, ri],
                    "settlement": asdict(settlement) if settlement else None,
                    "history_suffix": [action_data(a) for a in suffix],
                    "own_upgrade_events": [
                        action_data(a)
                        for a in suffix
                        if a.player == obs.player and a.kind in (A.PAO, A.TI)
                    ],
                    "claim_followup_discards": [
                        action_data(a)
                        for a in suffix
                        if a.player == obs.player and a.kind == A.DISCARD
                    ],
                    "terminal_sha256": hashlib.sha256(state.serialize().encode()).hexdigest(),
                }
            )
        outcomes.append(cash)
        raw.append(
            {
                "particle": particle,
                "world": before,
                "world_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "branches": branches,
            }
        )
    return np.asarray(outcomes, np.float32), {
        **dict(diagnostics),
        "particles": particles,
        "sampling_attempts": attempts,
        "route_fallback_decisions": abandoned,
        "raw_particles": raw,
        "policy": "v2 frozen parent + visible route progress with mandatory claim discard preview; opponents frozen; HU/pass unmodified",
        "cash_units": "actual terminal payments/100",
        "strength": strength,
        "meaning": "Q(action, route-following continuation), not optimal route EV",
    }


def mine_observation(seed, prior, device, preferred):
    """Select by visible state only; finish and verify the source game first."""
    engine = RuleEngine()
    initial = engine.new_game(seed, seed % 3)
    state, candidates = initial, []
    for _ in range(1000):
        if state.terminal:
            break
        actions = engine.legal_actions(state)
        obs = engine.observation(state, actions[0].player)
        if len(actions) > 1 and not obs.hu_disabled:
            kinds = {a.kind for a in actions}
            family = (
                "hu"
                if A.HU in kinds
                else "chi"
                if A.CHI in kinds
                else "peng"
                if A.PENG in kinds
                else "discard"
            )
            # Prioritize early roots to keep exact public-history sampling cheap.
            score = 10 * (family == preferred) + 2 * (family != "discard") - len(obs.history) / 100
            candidates.append((score, obs, family))
        chosen = actions[0] if len(actions) == 1 else choose_batch(prior, [obs], device)[0]
        state = engine.step(state, chosen)
    if not state.terminal:
        raise RuntimeError("source game did not terminate")
    state.validate()
    if engine.replay(initial, state.history).serialize() != state.serialize():
        raise RuntimeError("source game replay mismatch")
    for _, obs, family in sorted(candidates, key=lambda v: v[0], reverse=True)[:8]:
        plans = propose_routes(obs, prior, device=device)
        if len(plans) >= 2:
            return obs, plans, family
    return None


def root_row(obs, plans, outcomes, prior, device):
    groups, contexts = plan_arrays(obs, plans)
    logits, _ = predict(prior, obs, device)
    return {
        "features": encode_observation(obs, "huxi"),
        "actions": np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
        "plans": groups,
        "contexts": contexts,
        "outcomes": outcomes,
        "probabilities": logits.softmax(-1).cpu().numpy(),
        "baseline_action": int(logits.argmax()),
    }


def root_tensors(row, device):
    return [
        torch.as_tensor(row[k][None], device=device)
        for k in ("features", "actions", "plans", "contexts")
    ]


def model_root(model, row, device):
    x, a, p, c = root_tensors(row, device)
    return model.forward_routes(
        x,
        a,
        torch.ones(a.shape[:2], dtype=torch.bool, device=device),
        p,
        c,
        torch.ones(p.shape[:2], dtype=torch.bool, device=device),
    )


def comparison_loss(model, row, device):
    q, value, _ = model_root(model, row, device)
    cash = torch.as_tensor(row["outcomes"][:, :, :-1], device=device)
    target = cash.mean(0)
    se = cash.std(0, unbiased=True) / math.sqrt(len(cash))
    # Root-local weights, with a floor to avoid declaring rare zero-variance
    # observations exact. All roots have equal total optimization weight.
    weight = (1 / (0.25 + se)).clamp_max(4)
    regression = (F.smooth_l1_loss(q[0], target, reduction="none") * weight).sum() / weight.sum()
    delta = cash[:, :, :, None] - cash[:, :, None, :]
    delta_target = delta.mean(0)
    delta_se = delta.std(0, unbiased=True) / math.sqrt(len(cash))
    predicted = q[0, :, :, None] - q[0, :, None, :]
    pairs = ~torch.eye(target.shape[1], dtype=torch.bool, device=device)[None].expand_as(
        delta_target
    )
    pair_weight = (1 / (0.25 + delta_se)).clamp_max(4) * pairs
    paired = (
        F.smooth_l1_loss(predicted, delta_target, reduction="none") * pair_weight
    ).sum() / pair_weight.sum().clamp_min(1)
    probabilities = torch.as_tensor(row["probabilities"], device=device)
    # Route V uses the frozen first-action distribution; no oracle max.
    value_target = (target * probabilities[:, None]).sum(0)
    value_loss = F.smooth_l1_loss(value[0], value_target)
    return regression + paired + value_loss


def metrics(model, rows, device):
    errors, deltas, agreements, cash_differences = [], [], [], []
    distinct_routes = 0
    model.eval()
    with torch.no_grad():
        for row in rows:
            q, _value, _ = model_root(model, row, device)
            prediction = q[0].cpu().numpy()
            samples = row["outcomes"][:, :, :-1]
            target = samples.mean(0)
            distinct_routes += int(np.any(np.ptp(samples, axis=2) > 0))
            errors.append(float(np.abs(prediction - target).mean() * 100))
            true_delta = target[:, :, None] - target[:, None, :]
            pred_delta = prediction[:, :, None] - prediction[:, None, :]
            diff = samples[:, :, :, None] - samples[:, :, None, :]
            se = diff.std(0, ddof=1) / math.sqrt(len(samples))
            comparable = np.abs(true_delta) > 2 * se + 0.01
            if comparable.any():
                agreements.extend(
                    (np.sign(true_delta[comparable]) == np.sign(pred_delta[comparable])).tolist()
                )
            deltas.append(float(np.abs(true_delta - pred_delta).mean() * 100))
            ai, ri = np.unravel_index(prediction.argmax(), prediction.shape)
            cash_differences.append(
                float(
                    (
                        row["outcomes"][:, ai, ri] - row["outcomes"][:, row["baseline_action"], -1]
                    ).mean()
                    * 100
                )
            )
    return {
        "roots": len(rows),
        "roots_with_route_cash_variation": distinct_routes,
        "q_mae_cash": float(np.mean(errors)),
        "within_state_route_delta_mae_cash": float(np.mean(deltas)),
        "diagnostic_pair_sign_accuracy": float(np.mean(agreements)) if agreements else None,
        "diagnostic_pairs": len(agreements),
        "selected_root_switch_mean_cash_delta": float(np.mean(cash_differences)),
        "meaning": "root-local evaluation; noisy fixed-route continuation MC; not certified ranking or full-agent cash strength",
    }


def evaluate_full_games(model, *, start, games=16, device="cpu"):
    """Fresh full games, rotating actor/dealer; parent opponents, seed clusters.

    The Q actor replans each free decision. This is an experimental agent,
    separate from the frozen route-following policy used to label Q.
    """
    engine = RuleEngine()
    records, excluded = [], []
    for seed in range(start, start + games):
        paired = []
        try:
            for seat in range(3):
                payments = []
                traces = []
                for experiment in (False, True):
                    initial = engine.new_game(seed, seed % 3)
                    state = initial
                    changed = 0
                    for _ in range(1000):
                        if state.terminal:
                            break
                        legal = engine.legal_actions(state)
                        action = legal[0]
                        if len(legal) > 1:
                            obs = engine.observation(state, legal[0].player)
                            baseline = choose_batch(model.parent, [obs], device)[0]
                            action = baseline
                            if experiment and obs.player == seat and not obs.hu_disabled:
                                plans = propose_routes(obs, model.parent, device=device)
                                if plans:
                                    groups, contexts = plan_arrays(obs, plans)
                                    row = {
                                        "features": encode_observation(obs, "huxi"),
                                        "actions": np.stack(
                                            [encode_action(a, obs.player) for a in legal]
                                        ),
                                        "plans": groups,
                                        "contexts": contexts,
                                    }
                                    with torch.inference_mode():
                                        q = model_root(model, row, device)[0][0].cpu().numpy()
                                    ai, _ri = np.unravel_index(q.argmax(), q.shape)
                                    action = legal[ai]
                                    changed += int(action != baseline)
                        state = engine.step(state, action)
                    if not state.terminal:
                        raise RuntimeError("full cash game exceeded transition budget")
                    state.validate()
                    if engine.replay(initial, state.history).serialize() != state.serialize():
                        raise RuntimeError("full cash replay mismatch")
                    payments.append(state.settlement.payments[seat] if state.settlement else 0)
                    traces.append(
                        {
                            "experiment": experiment,
                            "changed": changed,
                            "history": [action_data(a) for a in state.history],
                            "settlement": asdict(state.settlement) if state.settlement else None,
                        }
                    )
                paired.append(
                    {
                        "seat": seat,
                        "cash_delta": payments[1] - payments[0],
                        "payments": payments,
                        "traces": traces,
                    }
                )
            records.append(
                {
                    "seed": seed,
                    "mean_delta": float(np.mean([r["cash_delta"] for r in paired])),
                    "seats": paired,
                }
            )
        except (RuleClarificationRequired, ValueError) as exc:
            excluded.append({"seed": seed, "error": str(exc), "status": "whole seed excluded"})
    if not records:
        raise RuntimeError("no valid independent full-game cash seeds")
    means = [r["mean_delta"] for r in records]
    bootstrap = (
        np.random.default_rng(start).choice(means, (2000, len(means)), replace=True).mean(-1)
    )
    return {
        "independent_seeds": len(records),
        "mean_cash_delta": float(np.mean(means)),
        "approximate_seed_bootstrap_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "opponents": "frozen parent only",
        "candidate_policy": "greedy route-Q replanning every free own decision",
        "meaning": "small independent full-agent experiment; not multi-opponent champion certification",
        "records": records,
        "excluded": excluded,
    }


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "source_hashes": producer_hashes(), "prior_sha256": digest(args.prior)}
    atomic_json(root / "config.json", config)
    prior = load_model(args.prior, args.device)
    model = RouteQNet.from_parent(prior).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def frozen():
        if (
            producer_hashes() != config["source_hashes"]
            or digest(args.prior) != config["prior_sha256"]
        ):
            raise RuntimeError("frozen route producer changed")

    def generate(split, count, first, particle_start, particles=None):
        particles = particles or args.particles
        rows, failures = [], []
        for attempt in range(count * args.attempt_multiplier):
            if len(rows) == count:
                break
            frozen()
            seed = first + attempt
            obs, plans = None, []
            try:
                mined = mine_observation(
                    seed, prior, args.device, ("chi", "peng", "discard", "hu")[len(rows) % 4]
                )
                if mined is None:
                    raise ValueError("budgeted candidate proposals returned fewer than two routes")
                obs, plans, family = mined
                outcomes, evidence = compare_routes(
                    obs,
                    plans,
                    prior,
                    particles=particles,
                    seed=particle_start + attempt * particles * 104729,
                    device=args.device,
                    max_attempts=args.max_attempts,
                    strength=args.strength,
                )
                rows.append(root_row(obs, plans, outcomes, prior, args.device))
                atomic_gzip(
                    root / f"{split}-{len(rows) - 1:05d}.json.gz",
                    {
                        "seed": seed,
                        "family": family,
                        "tags": tags(obs),
                        "observation": serialize_observation(obs),
                        "visible_sha256": visible_key(obs),
                        "plans": plans,
                        "outcomes": outcomes.tolist(),
                        **evidence,
                    },
                )
            except (SamplingFailure, RuleClarificationRequired, ValueError) as exc:
                failures.append({"seed": seed, "error": str(exc), "error_type": type(exc).__name__})
                atomic_gzip(
                    root / f"{split}-excluded-{attempt:05d}.json.gz",
                    {
                        "seed": seed,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "observation": serialize_observation(obs) if obs is not None else None,
                        "plans": plans,
                        "status": "whole root excluded; no label",
                    },
                )
                atomic_json(root / f"{split}-failures.json", failures)
            atomic_json(
                root / "progress.json",
                {
                    "status": "collecting",
                    "split": split,
                    "roots": len(rows),
                    "target_roots": count,
                    "attempted_games": attempt + 1,
                    "excluded_roots": len(failures),
                    "particles_per_root": particles,
                    "device": args.device,
                },
            )
        if len(rows) != count:
            raise RuntimeError(
                f"{split}: insufficient complete comparison roots; see failure evidence"
            )
        return rows, failures

    train, failures = generate("train", args.roots, args.train_start, args.particle_start)
    dev, dev_failures = generate(
        "dev",
        args.dev_roots,
        args.dev_start,
        args.particle_start + args.roots * args.attempt_multiplier * args.particles * 104729,
    )
    initial = metrics(model, dev, args.device)
    atomic_json(root / "initial-dev.json", initial)
    best, rng = float("inf"), random.Random(args.seed)
    for epoch in range(args.epochs):
        frozen()
        order = list(train)
        rng.shuffle(order)
        model.train()
        loss_total = 0
        for row in order:
            loss = comparison_loss(model, row, args.device)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite route-comparison loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item()
        for key, tensor in prior.state_dict().items():
            if not torch.equal(model.parent.state_dict()[key], tensor):
                raise RuntimeError("route Q changed frozen prior")
        with torch.no_grad():
            dev_loss = sum(comparison_loss(model, row, args.device).item() for row in dev) / len(
                dev
            )
        scores = metrics(model, dev, args.device)
        if dev_loss < best:
            best = dev_loss
            save_model(
                model,
                root / "best-dev.pt",
                epoch=epoch + 1,
                policy_frozen=True,
                route_target="cash under explicit frozen route-following continuation",
                prior_sha256=config["prior_sha256"],
            )
            atomic_save(
                {
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "checkpoint_sha256": digest(root / "best-dev.pt"),
                    "configuration": config,
                    "random_state": rng.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                },
                root / "best-optimizer.pt",
            )
        save_model(model, root / "last-model.pt", epoch=epoch + 1, policy_frozen=True)
        atomic_json(
            root / "progress.json",
            {
                "status": "training",
                "epoch": epoch + 1,
                "epochs": args.epochs,
                "train_loss": loss_total / len(train),
                "dev_loss": dev_loss,
                "dev_metrics": scores,
                "device": args.device,
            },
        )
    selected = load_model(root / "best-dev.pt", args.device)
    frozen()
    atomic_json(root / "progress.json", {"status": "independent_test", "device": args.device})
    test, test_failures = generate(
        "test",
        args.test_roots,
        args.test_start,
        args.particle_start
        + (args.roots + args.dev_roots) * args.attempt_multiplier * args.particles * 104729,
        args.test_particles,
    )
    test_metrics = metrics(selected, test, args.device)
    root_deltas = []
    with torch.no_grad():
        for row in test:
            prediction = model_root(selected, row, args.device)[0][0].cpu().numpy()
            ai, ri = np.unravel_index(prediction.argmax(), prediction.shape)
            root_deltas.append(
                float(
                    (
                        row["outcomes"][:, ai, ri] - row["outcomes"][:, row["baseline_action"], -1]
                    ).mean()
                    * 100
                )
            )
    bootstrap = (
        np.random.default_rng(args.seed + 1)
        .choice(root_deltas, (2000, len(root_deltas)), replace=True)
        .mean(-1)
    )
    holdout = {
        "metrics": test_metrics,
        "root_cash_deltas": root_deltas,
        "approximate_root_bootstrap_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "selected_checkpoint_sha256": digest(root / "best-dev.pt"),
        "meaning": "fresh roots/particles after checkpoint selection; one root switch then frozen route continuation; not full-agent cash gate",
    }
    atomic_json(root / "independent-test.json", holdout)
    full_cash = (
        evaluate_full_games(
            selected, start=args.cash_start, games=args.cash_games, device=args.device
        )
        if args.cash_games
        else None
    )
    if full_cash is not None:
        atomic_gzip(root / "independent-full-cash.json.gz", full_cash)
    result = {
        "status": "complete",
        "train_roots": len(train),
        "dev_roots": len(dev),
        "test_roots": len(test),
        "independent_test": holdout,
        "excluded_test_roots": len(test_failures),
        "excluded_train_roots": len(failures),
        "excluded_dev_roots": len(dev_failures),
        "initial_dev": initial,
        "final_dev": metrics(selected, dev, args.device),
        "best_dev_loss": best,
        "policy_frozen": True,
        "formal_champion_changed": False,
        "selected_checkpoint": str(root / "best-dev.pt"),
        "cash_strength_evaluated": full_cash is not None,
        "full_cash_summary": {
            k: v for k, v in full_cash.items() if k not in ("records", "excluded")
        }
        if full_cash
        else None,
        "next_gate": "root-switch test review, uncertainty calibration and full-agent cash evaluation before policy integration",
    }
    atomic_json(root / "result.json", result)
    atomic_json(
        root / "progress.json",
        {"status": "complete", "device": args.device, "result": str(root / "result.json")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", default="data/generated/trajectory-v4-5000/selected-model.pt")
    parser.add_argument("--directory", default="data/generated/routeq-v1")
    parser.add_argument("--roots", type=int, default=128)
    parser.add_argument("--dev-roots", type=int, default=32)
    parser.add_argument("--test-roots", type=int, default=16)
    parser.add_argument("--test-particles", type=int, default=16)
    parser.add_argument("--test-start", type=int, default=810000000)
    parser.add_argument("--cash-start", type=int, default=850000000)
    parser.add_argument("--cash-games", type=int, default=0)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=256)
    parser.add_argument("--attempt-multiplier", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--train-start", type=int, default=750000000)
    parser.add_argument("--dev-start", type=int, default=760000000)
    parser.add_argument("--particle-start", type=int, default=770000000)
    parser.add_argument("--seed", type=int, default=1351)
    parser.add_argument("--strength", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (
        min(
            args.roots,
            args.dev_roots,
            args.test_roots,
            args.max_attempts,
            args.attempt_multiplier,
            args.epochs,
        )
        < 1
        or min(args.particles, args.test_particles) < 2
    ):
        parser.error("positive budgets and at least two particles required")
    if (
        not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.strength)
        or args.strength < 0
    ):
        parser.error("invalid learning rate or route strength")
    if args.train_start + args.roots * args.attempt_multiplier > args.dev_start:
        parser.error("train/development game seed ranges overlap")
    if args.dev_start + args.dev_roots * args.attempt_multiplier > args.test_start:
        parser.error("development/test game seed ranges overlap")
    new_directory = not Path(args.directory).exists()
    try:
        run(args)
    except Exception as exc:
        if new_directory and Path(args.directory).is_dir():
            atomic_json(
                Path(args.directory) / "progress.json",
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "device": args.device,
                    "labels_complete": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
