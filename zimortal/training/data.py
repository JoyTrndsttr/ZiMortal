"""Visible-information teachers and seeded near-hu/discard curriculum."""

import random
from collections import Counter
from functools import lru_cache

import numpy as np

from zimortal.engine import ActionType as A
from zimortal.engine import GameState, Meld, PlayerState, RuleEngine, evaluate_hand
from zimortal.engine import MeldType as M
from zimortal.engine.chi import CHI_PATTERNS
from zimortal.model.encoding import encode_action, encode_observation


@lru_cache(maxsize=32768)
def winning_tiles(hand, melds, kans):
    """Exact own-draw wins, applying forced wei/ti before evaluating hu."""
    own = Counter(hand)
    used = own + Counter(t for m in melds for t in m.tiles)
    result = []
    for tile in range(20):
        if used[tile] >= 4:
            continue
        h = list(hand)
        ms = list(melds)
        ks = set(kans)
        upgrade = next(
            (
                i
                for i, m in enumerate(ms)
                if m.tiles[0] == tile and m.kind in (M.WEI, M.STINKY_WEI, M.PENG)
            ),
            None,
        )
        if upgrade is not None:
            kind = M.PAO if ms[upgrade].kind == M.PENG else M.TI
            ms[upgrade] = Meld(kind, (tile,) * 4)
        elif own[tile] in (2, 3):
            for _ in range(own[tile]):
                h.remove(tile)
            ms.append(Meld(M.WEI if own[tile] == 2 else M.TI, (tile,) * (own[tile] + 1)))
            ks.discard(tile)
        else:
            h.append(tile)
        if evaluate_hand(h, ms, quad_requires_pair=True, protected=ks):
            result.append(tile)
    return tuple(result)


def hand_quality(hand, melds, kans, visible=None):
    """Not an exact shanten number: exact waits + local cohesion tie-breaker."""
    counts = Counter(hand)
    waits = winning_tiles(tuple(sorted(hand)), tuple(melds), tuple(sorted(kans)))
    visible = visible or Counter()
    live = sum(max(0, 4 - counts[t] - visible[t]) for t in waits)
    cohesion = sum(n * (n - 1) for n in counts.values())
    for pattern in CHI_PATTERNS:
        needed = Counter(pattern)
        filled = sum(min(counts[t], n) for t, n in needed.items())
        cohesion += 0.15 * filled * filled
    return live + len(waits) * 0.5 + cohesion * 0.015


def teacher_scores(obs):
    own_melds = obs.players[obs.player].melds
    kans = {t for t, n in Counter(obs.hand).items() if n == 3}
    visible = Counter(obs.river)
    for p in obs.players:
        visible.update(t for m in p.melds for t in m.tiles)
    if obs.pending:
        visible[obs.pending.tile] += 1
    scores = []
    for action in obs.legal_actions:
        if action.kind == A.HU:
            scores.append(1000.0)
            continue
        if action.forced:
            scores.append(0.0)
            continue
        if action.kind == A.PASS:
            scores.append(hand_quality(obs.hand, own_melds, kans, visible) - 0.1)
            continue
        hand = list(obs.hand)
        melds = list(own_melds)
        ks = set(kans)
        if action.kind == A.DISCARD:
            hand.remove(action.tile)
        elif action.kind == A.CHI:
            consumed = list(action.chi)
            consumed.remove(action.tile)
            for t in consumed:
                hand.remove(t)
            melds.append(Meld(M.CHI, action.chi))
            for group in action.bi:
                for t in group:
                    hand.remove(t)
                melds.append(Meld(M.CHI, group))
        elif action.kind == A.PENG:
            hand.remove(action.tile)
            hand.remove(action.tile)
            melds.append(Meld(M.PENG, (action.tile,) * 3))
        post_visible = visible + (Counter(obs.hand) - Counter(hand))
        # Optional claims usually require a discard: inspect every discard,
        # except opening double-ti's first intake (inferred from public history).
        initial_ti = sum(
            a.player == obs.player and a.kind == A.TI and a.source_type.value == "initial"
            for a in obs.history
        )
        previous_intake = any(
            a.player == obs.player
            and a.kind in (A.CHI, A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI)
            and a.source_type.value != "initial"
            for a in obs.history
        )
        waived = initial_ti >= 2 and not previous_intake
        if action.kind in (A.CHI, A.PENG) and not waived:
            candidates = [t for t in set(hand) if t not in ks]
            quality = max(
                (
                    hand_quality(
                        tuple(x for i, x in enumerate(hand) if i != hand.index(t)),
                        melds,
                        ks,
                        post_visible + Counter({t: 1}),
                    )
                    for t in candidates
                ),
                default=-10.0,
            )
        else:
            quality = hand_quality(hand, melds, ks, post_visible)
        scores.append(quality)
    return np.asarray(scores, dtype=np.float32)


def make_puzzle(seed):
    rng = random.Random(seed)
    engine = RuleEngine()
    for _ in range(100):
        groups = []
        counts = Counter()
        for _ in range(7):
            choices = [Meld(M.KAN, (t,) * 3) for t in range(20)] + [
                Meld(M.CHI, p) for p in CHI_PATTERNS
            ]
            rng.shuffle(choices)
            for group in choices:
                next_counts = counts + Counter(group.tiles)
                if max(next_counts.values()) <= 3:
                    counts = next_counts
                    groups.append(group)
                    break
        if sum(counts.values()) != 21:
            continue
        hand = list(counts.elements())
        # 0..3 corruptions create a curriculum around known complete structures;
        # no claim that corruption count equals exact shanten.
        corruptions = seed % 4
        for _ in range(corruptions):
            old = rng.choice(hand)
            hand.remove(old)
            choices = [t for t in range(20) if hand.count(t) < 3 and t != old]
            hand.append(rng.choice(choices))
        deck = [t for t in range(20) for _ in range(4 - hand.count(t))]
        rng.shuffle(deck)
        opponents = [PlayerState([deck.pop() for _ in range(20)]) for _ in range(2)]
        own = PlayerState(sorted(hand), kans={t for t, n in Counter(hand).items() if n == 3})
        s = GameState([own] + opponents, deck, phase="opening", seed=seed)
        s.validate()
        legal = engine.legal_actions(s)
        while legal[0].kind == A.TI and legal[0].forced:
            s = engine.step(s, legal[0])
            legal = engine.legal_actions(s)
        if legal[0].kind != A.HU:
            s = engine.step(s, legal[0])
        if engine.legal_actions(s):
            return engine.observation(s, 0), corruptions
    raise RuntimeError("puzzle generation failed")


def example(obs, value=None):
    scores = teacher_scores(obs)
    teacher = int(np.argmax(scores))
    waits = winning_tiles(
        tuple(sorted(obs.hand)),
        tuple(obs.players[obs.player].melds),
        tuple(t for t, n in Counter(obs.hand).items() if n == 3),
    )
    if obs.hu_disabled:
        waits = ()
    target = np.zeros(20, dtype=np.float32)
    target[list(waits)] = 1
    if value is None:
        value = float(any(a.kind == A.HU for a in obs.legal_actions) or bool(waits))
    return (
        encode_observation(obs),
        np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
        teacher,
        float(value),
        target,
    )


def generate_dataset(puzzles, games, seed):
    data = []
    strata = Counter()
    for i in range(puzzles):
        obs, level = make_puzzle(seed * 100000 + i)
        data.append(example(obs))
        strata[level] += 1
    engine = RuleEngine()
    rng = random.Random(seed)
    for i in range(games):
        state = engine.new_game(seed * 100000 + i)
        for _ in range(1000):
            if state.terminal:
                break
            actions = engine.legal_actions(state)
            if len(actions) > 1:
                obs = engine.observation(state, actions[0].player)
                data.append(example(obs))
            state = engine.step(state, rng.choice(actions))
        else:
            raise RuntimeError("nonterminal simulation")
    return data, dict(strata)
