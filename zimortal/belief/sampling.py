"""History-consistent hidden worlds built solely from player observations."""

import random
from collections import Counter

from zimortal.engine import ActionType as A
from zimortal.engine import GameState, PlayerState, RuleEngine
from zimortal.engine import MeldType as M
from zimortal.engine import SourceType as S


class SamplingFailure(RuntimeError):
    pass


def reverse_hands(obs, hands, draws):
    draw_cursor = len(draws) - 1
    previous = {}
    seen = [{} for _ in range(3)]
    for i, a in enumerate(obs.history):
        if a.kind in (A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI):
            old = seen[a.player].get(a.tile)
            previous[i] = old if a.kind in (A.PAO, A.TI) and a.source_type != S.INITIAL else None
            seen[a.player][a.tile] = M(a.kind.value)
    for i in range(len(obs.history) - 1, -1, -1):
        a = obs.history[i]
        h = hands[a.player]
        if a.kind == A.DRAW:
            tile = draws[draw_cursor]
            draw_cursor -= 1
            following = obs.history[i + 1] if i + 1 < len(obs.history) else obs.legal_actions[0]
            mandatory = following.player == a.player and following.kind in (
                A.WEI,
                A.STINKY_WEI,
                A.TI,
            )
            count = h.count(tile)
            permitted = (
                (2,) if mandatory and following.kind != A.TI else (0, 3) if mandatory else (0, 1)
            )
            if count not in permitted:
                raise ValueError("particle contradicts forced draw evidence")
        elif a.kind == A.DISCARD:
            h.append(a.tile)
            if h.count(a.tile) >= 3:
                raise ValueError("particle discarded a locked kan")
        elif a.kind == A.CHI:
            used = list(a.chi)
            used.remove(a.tile)
            h.extend(used)
            h.extend(t for g in a.bi for t in g)
        elif a.kind in (A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI) and previous[i] is None:
            count = 4 if a.source_type == S.INITIAL else 3 if a.kind in (A.PAO, A.TI) else 2
            h.extend([a.tile] * count)
    return hands


def observed_draws(obs):
    draws = []
    for i, a in enumerate(obs.history):
        if a.kind != A.DRAW:
            continue
        tile = None
        for later in obs.history[i + 1 :]:
            if later.kind in (A.DRAW, A.DISCARD):
                break
            if later.source_type == S.DRAW and later.tile is not None:
                tile = later.tile
                break
        if tile is None:
            if (
                obs.pending is None
                or obs.pending.source != S.DRAW
                or any(x.kind in (A.DRAW, A.DISCARD) for x in obs.history[i + 1 :])
            ):
                raise SamplingFailure("draw tile not recoverable from public history")
            tile = obs.pending.tile
        draws.append(tile)
    return draws


def sample_world(obs, seed, max_attempts=2000):
    """Reconstruct an initial deal, replay ALL public actions, then compare obs.

    Rejection checks forced wei/ti, no-wei evidence, locked kans, chi/bi,
    priority, passes and the exact pending decision. No true hidden state is
    accepted as an input, and no inconsistent fallback is allowed.
    """
    rng = random.Random(seed)
    known = Counter(obs.hand) + Counter(obs.river)
    for p in obs.players:
        known.update(t for m in p.melds for t in m.tiles)
    if obs.pending:
        known[obs.pending.tile] += 1
    if any(n > 4 for n in known.values()):
        raise ValueError("invalid known physical counts")
    pool = [t for t in range(20) for _ in range(4 - known[t])]
    opponents = [p for p in range(3) if p != obs.player]
    if len(pool) != obs.remaining_tiles + sum(obs.players[p].hand_count for p in opponents):
        raise ValueError("hidden slot counts disagree")
    first_discard = next((a for a in obs.history if a.kind == A.DISCARD), None)
    dealer = first_discard.player if first_discard else obs.turn
    engine = RuleEngine()
    draws = observed_draws(obs)
    for attempt in range(1, max_attempts + 1):
        rng.shuffle(pool)
        hands = [list(obs.hand) if p == obs.player else [] for p in range(3)]
        cursor = 0
        for p in opponents:
            n = obs.players[p].hand_count
            hands[p] = pool[cursor : cursor + n]
            cursor += n
        deck = pool[cursor:]
        try:
            initial_hands = reverse_hands(obs, [h.copy() for h in hands], draws)
        except ValueError:
            continue
        if any(
            len(h) != (21 if p == dealer else 20) or max(Counter(h).values(), default=0) > 4
            for p, h in enumerate(initial_hands)
        ):
            continue
        initial_deck = deck + list(reversed(draws))
        players = [
            PlayerState(h, kans={t for t, n in Counter(h).items() if n == 3}) for h in initial_hands
        ]
        initial = GameState(players, initial_deck, dealer=dealer, turn=dealer, phase="opening")
        try:
            initial.validate()
            state = engine.replay(initial, obs.history)
            state.validate()
            if engine.observation(state, obs.player) == obs:
                return state, attempt
        except ValueError:
            continue
    raise SamplingFailure(f"no history-consistent particle in {max_attempts} attempts")
