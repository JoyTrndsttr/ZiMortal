"""Public-only, current-slot prior conditioned on authoritative prefix replay.

Not a policy-likelihood posterior. No full state, truth, outcome, suffix, model,
or checkpoint is accepted by the generator. Ordinary RuleEngine is the final
arbiter; RuleClarificationRequired is fatal rather than evidence of impossibility.
"""

import hashlib
import json
import random
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import factorial

from zimortal.belief.conditional import count_constraints
from zimortal.belief.sampling import SamplingFailure, observed_draws
from zimortal.engine import Action, GameState, Observation, PendingTile, PlayerState, RuleEngine
from zimortal.engine import ActionType as A
from zimortal.engine import SourceType as S
from zimortal.engine.observation import PublicPlayer


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=lambda x: sorted(x))


def observation_bytes(obs):
    return canonical(asdict(obs)).encode()


@dataclass(frozen=True)
class PublicInformationState:
    """The existing observation contract, including the visible action prefix.

    Opponents' pass sets/kans, true wall, original deal, and future are absent.
    Dealer is recovered from the first discard (or turn during opening).
    """

    observation: Observation

    def __post_init__(self):
        if type(self.observation) is not Observation:
            raise TypeError("sampler requires Observation, never GameState/audit truth")
        obs = self.observation
        if (
            type(obs.players) is not tuple
            or len(obs.players) != 3
            or any(type(p) is not PublicPlayer for p in obs.players)
            or obs.pending is not None
            and type(obs.pending) is not PendingTile
            or any(type(a) is not Action for a in (*obs.history, *obs.legal_actions))
        ):
            raise TypeError("only public observation records accepted")
        if self.observation.player not in range(3):
            raise ValueError("invalid perspective")

    @property
    def dealer(self):
        first = next((a for a in self.observation.history if a.kind == A.DISCARD), None)
        return first.player if first else self.observation.turn

    @property
    def sha256(self):
        return hashlib.sha256(observation_bytes(self.observation)).hexdigest()


@dataclass(frozen=True)
class HiddenWorld:
    """Canonical opponent hand multisets and ordered wall (end is next draw)."""

    hands: tuple[tuple[int, ...], ...]
    wall: tuple[int, ...]

    @classmethod
    def from_state(cls, state):
        return cls(tuple(tuple(sorted(p.hand)) for p in state.players), tuple(state.deck))

    @property
    def key(self):
        return canonical(asdict(self))


def physical_pool(public):
    obs = public.observation
    known = Counter(obs.hand) + Counter(obs.river)
    for player in obs.players:
        known.update(t for meld in player.melds for t in meld.tiles)
    if obs.pending:
        known[obs.pending.tile] += 1
    if set(known) - set(range(20)) or any(n > 4 for n in known.values()):
        raise ValueError("invalid public physical tile count")
    pool = tuple(4 - known[t] for t in range(20))
    opponents = tuple(p for p in range(3) if p != obs.player)
    sizes = tuple(obs.players[p].hand_count for p in opponents)
    if sum(pool) != sum(sizes) + obs.remaining_tiles:
        raise ValueError("public hand/wall slots disagree with 80-card conservation")
    return pool, opponents, sizes


class SlotPrior:
    """Exact integer DP for exchangeable physical unknown slots.

    Per-type count-vector mass m!/(a! b! d!); ordered wall uniformly shuffled
    conditional on its multiset. Restrictions must be necessary constraints.
    Also used with NO history restrictions by the exhaustive toy oracle.
    """

    def __init__(self, pool, sizes, allowed=None):
        self.pool, self.sizes = tuple(pool), tuple(sizes)
        self.options = []
        for tile, total in enumerate(pool):
            choices = []
            for a in range(total + 1):
                for b in range(total - a + 1):
                    if allowed is not None and (
                        a not in allowed[0][tile] or b not in allowed[1][tile]
                    ):
                        continue
                    d = total - a - b
                    w = factorial(total) // (factorial(a) * factorial(b) * factorial(d))
                    choices.append((a, b, d, w))
            self.options.append(tuple(choices))

        @lru_cache(None)
        def ways(tile, a_left, b_left):
            if min(a_left, b_left) < 0:
                return 0
            if tile == len(pool):
                return int(a_left == b_left == 0)
            return sum(
                w * ways(tile + 1, a_left - a, b_left - b) for a, b, _, w in self.options[tile]
            )

        @lru_cache(None)
        def transitions(tile, a_left, b_left):
            cumulative, options, total = [], [], 0
            for a, b, d, w in self.options[tile]:
                mass = w * ways(tile + 1, a_left - a, b_left - b)
                if mass:
                    total += mass
                    cumulative.append(total)
                    options.append((a, b, d))
            return tuple(cumulative), tuple(options)

        self.ways, self.transitions = ways, transitions
        self.mass = ways(0, *self.sizes)
        if not self.mass:
            raise SamplingFailure("necessary count constraints have empty support")

    def draw(self, rng):
        first, second, wall = [], [], []
        left = self.sizes
        for tile in range(len(self.pool)):
            cumulative, options = self.transitions(tile, *left)
            a, b, d = options[bisect_right(cumulative, rng.randrange(cumulative[-1]))]
            first.extend([tile] * a)
            second.extend([tile] * b)
            wall.extend([tile] * d)
            left = left[0] - a, left[1] - b
        rng.shuffle(wall)
        return tuple(first), tuple(second), tuple(wall)

    def contains(self, first, second, wall):
        counts = [Counter(x) for x in (first, second, wall)]
        return (len(first), len(second)) == self.sizes and all(
            any((a, b, d) == tuple(c[t] for c in counts) for a, b, d, _ in options)
            for t, options in enumerate(self.options)
        )


def reconstruct_initial(public, world):
    """Reverse only public consumption; legality is NOT inferred here.

    Unlike reverse_hands, no forced-draw or strategy rejection is performed.
    Reverse operations are additive: drawn cards were offered, never silently
    inserted in a hand. Prefix replay verifies all mandatory transitions.
    """
    obs = public.observation
    hands = [list(h) for h in world.hands]
    seen, previous = [set() for _ in range(3)], {}
    meld_kinds = (A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI)
    for i, action in enumerate(obs.history):
        if action.kind in meld_kinds:
            previous[i] = action.tile in seen[action.player] and action.source_type != S.INITIAL
            seen[action.player].add(action.tile)
    for i in range(len(obs.history) - 1, -1, -1):
        action = obs.history[i]
        hand = hands[action.player]
        if action.kind == A.DISCARD:
            hand.append(action.tile)
        elif action.kind == A.CHI:
            used = list(action.chi)
            used.remove(action.tile)
            hand.extend(used)
            hand.extend(t for group in action.bi for t in group)
        elif action.kind in meld_kinds and not previous[i]:
            n = 4 if action.source_type == S.INITIAL else 3 if action.kind in (A.PAO, A.TI) else 2
            hand.extend([action.tile] * n)
    dealer = public.dealer
    if any(len(h) != (21 if p == dealer else 20) for p, h in enumerate(hands)):
        raise ValueError("reverse initial deal sizes are not 21/20/20")
    return GameState(
        [PlayerState(sorted(h), kans={t for t, n in Counter(h).items() if n == 3}) for h in hands],
        list(world.wall) + list(reversed(observed_draws(obs))),
        dealer=dealer,
        turn=dealer,
        phase="opening",
    )


def certify_world(public, world, engine=None):
    """Unpruned, ordinary-engine replay oracle; no policy/outcome conditioning."""
    if type(public) is not PublicInformationState or type(world) is not HiddenWorld:
        raise TypeError("explicit public and hidden-world contracts required")
    obs = public.observation
    if len(world.hands) != 3 or world.hands[obs.player] != obs.hand:
        raise ValueError("own hand must be fixed")
    if len(world.wall) != obs.remaining_tiles or any(
        len(hand) != obs.players[p].hand_count for p, hand in enumerate(world.hands)
    ):
        raise ValueError("root hidden slot sizes differ")
    initial = reconstruct_initial(public, world)
    initial.validate()
    engine = engine or RuleEngine()
    state = engine.replay(initial, obs.history)
    state.validate()
    if observation_bytes(engine.observation(state, obs.player)) != observation_bytes(obs):
        raise ValueError("replayed Observation bytes differ")
    if HiddenWorld.from_state(state) != world:
        raise ValueError("replayed hidden assignment differs")
    return state


class InformationSetSampler:
    """Constraint-conditioned current-slot distribution; behavioral inference OFF.

    count_constraints is a necessary per-tile reverse-history optimization.
    Historical elective PASS availability is NOT used as an extra prefilter;
    ordinary replay alone decides whether the observed action was legal.
    """

    def __init__(self, public):
        if type(public) is not PublicInformationState:
            raise TypeError("pass PublicInformationState, never a simulator full state")
        self.public = public
        pool, self.opponents, sizes = physical_pool(public)
        allowed = count_constraints(public.observation)
        self.prior = SlotPrior(pool, sizes, tuple(allowed[p] for p in self.opponents))
        self.engine = RuleEngine()

    def proposal(self, rng):
        a, b, wall = self.prior.draw(rng)
        hands = [self.public.observation.hand] * 3
        hands[self.opponents[0]], hands[self.opponents[1]] = a, b
        return HiddenWorld(tuple(hands), wall)

    def contains(self, world):
        """Audit support membership, not likelihood of the true assignment."""
        a, b = (world.hands[p] for p in self.opponents)
        if not self.prior.contains(a, b, world.wall):
            return False
        try:
            certify_world(self.public, world, self.engine)
        except ValueError:
            return False
        return True

    def sample(self, seed, max_attempts=200000):
        rng, rejects = random.Random(seed), Counter()
        for attempt in range(1, max_attempts + 1):
            world = self.proposal(rng)
            try:
                certify_world(self.public, world, self.engine)
                return world, attempt, dict(rejects)
            except ValueError as exc:
                # Unknown rule boundaries must propagate, never lose support.
                rejects[str(exc).split(":")[0]] += 1
        raise SamplingFailure(f"budget exhausted after {max_attempts} proposals: {dict(rejects)}")


def public_from_json(data):
    """Deserialize ONLY the observation schema; unknown/private keys error."""
    from dataclasses import fields

    from zimortal.engine import Action, ActionType, Meld, MeldType, PendingTile, SourceType
    from zimortal.engine.observation import PublicPlayer

    if set(data) != {f.name for f in fields(Observation)}:
        raise TypeError("exact public Observation JSON schema required")
    data = dict(data)

    def action(a):
        return Action(
            ActionType(a["kind"]),
            a["player"],
            a["tile"],
            tuple(a["chi"]),
            tuple(tuple(g) for g in a["bi"]),
            a["source_player"],
            SourceType(a["source_type"]) if a["source_type"] else None,
            a["forced"],
        )

    data["players"] = tuple(
        PublicPlayer(
            p["hand_count"], tuple(Meld(MeldType(m["kind"]), tuple(m["tiles"])) for m in p["melds"])
        )
        for p in data["players"]
    )
    for key in ("hand", "river"):
        data[key] = tuple(data[key])
    for key in ("passed_peng", "passed_chi"):
        data[key] = frozenset(data[key])
    for key in ("history", "legal_actions"):
        data[key] = tuple(action(a) for a in data[key])
    if data["pending"]:
        p = data["pending"]
        data["pending"] = PendingTile(p["tile"], p["player"], SourceType(p["source"]))
    return PublicInformationState(Observation(**data))


def enumerate_physical_worlds(public, limit=100000):
    """Small-space oracle: NO count_constraints or sampler prefilters.

    Exhaust all hand count vectors and distinct ordered walls satisfying A.
    Caller uses certify_world for B. Limit errors rather than truncation.
    """
    from itertools import permutations

    pool, opponents, sizes = physical_pool(public)
    prior = SlotPrior(pool, sizes)
    emitted = 0

    def walk(tile, left, first, second, wall):
        nonlocal emitted
        if tile == len(pool):
            if left != (0, 0):
                return
            # This oracle is intentionally restricted to tiny hidden spaces.
            if len(wall) > 8:
                raise ValueError("toy wall too large to enumerate")
            for ordered in sorted(set(permutations(wall))):
                emitted += 1
                if emitted > limit:
                    raise ValueError("exhaustive oracle limit exceeded; not truncated")
                hands = [public.observation.hand] * 3
                hands[opponents[0]], hands[opponents[1]] = first, second
                yield HiddenWorld(tuple(hands), ordered)
            return
        for a, b, d, _ in prior.options[tile]:
            if a <= left[0] and b <= left[1]:
                yield from walk(
                    tile + 1,
                    (left[0] - a, left[1] - b),
                    first + (tile,) * a,
                    second + (tile,) * b,
                    wall + (tile,) * d,
                )

    yield from walk(0, sizes, (), (), ())
