"""Data contracts shared by legality, transitions, and evaluation."""

from dataclasses import dataclass, field
from enum import Enum

from .tiles import validate


class ActionType(str, Enum):
    DISCARD = "discard"
    DRAW = "draw"
    CHI = "chi"
    PENG = "peng"
    WEI = "wei"
    STINKY_WEI = "stinky_wei"
    PAO = "pao"
    TI = "ti"
    HU = "hu"
    PASS = "pass"


class MeldType(str, Enum):
    CHI = "chi"
    PENG = "peng"
    WEI = "wei"
    STINKY_WEI = "stinky_wei"
    KAN = "kan"  # Evaluation only; never published as an opening meld.
    PAO = "pao"
    TI = "ti"
    PAIR = "pair"


class SourceType(str, Enum):
    DRAW = "draw"
    DISCARD = "discard"
    INITIAL = "initial"


@dataclass(frozen=True)
class Meld:
    kind: MeldType
    tiles: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "tiles", tuple(self.tiles))
        for tile in self.tiles:
            validate(tile)
        size = (
            4
            if self.kind in (MeldType.PAO, MeldType.TI)
            else 2
            if self.kind == MeldType.PAIR
            else 3
        )
        if len(self.tiles) != size:
            raise ValueError("invalid meld size")
        if self.kind != MeldType.CHI and len(set(self.tiles)) != 1:
            raise ValueError("identical tiles required")
        if self.kind == MeldType.CHI:
            from .chi import CHI_PATTERNS

            if tuple(sorted(self.tiles)) not in CHI_PATTERNS:
                raise ValueError("invalid chi pattern")


@dataclass(frozen=True)
class Action:
    kind: ActionType
    player: int
    tile: int | None = None
    chi: tuple[int, ...] = ()
    bi: tuple[tuple[int, ...], ...] = ()
    source_player: int | None = None
    source_type: SourceType | None = None
    forced: bool = False

    def __post_init__(self):
        object.__setattr__(self, "chi", tuple(self.chi))
        object.__setattr__(self, "bi", tuple(tuple(group) for group in self.bi))


@dataclass
class PlayerState:
    hand: list[int] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    passed_peng: set[int] = field(default_factory=set)
    passed_chi: set[int] = field(default_factory=set)
    kans: set[int] = field(default_factory=set)
    hu_disabled: bool = False
    quad_count: int = 0
    opening_double_ti_pending: bool = False

    def needs_discard_after(self, kind: ActionType) -> bool:
        if kind in (ActionType.PAO, ActionType.TI):
            self.quad_count += 1
        if self.opening_double_ti_pending:
            self.opening_double_ti_pending = False
            return False
        if kind in (ActionType.PAO, ActionType.TI):
            return self.quad_count < 2
        return True


@dataclass(frozen=True)
class PendingTile:
    tile: int
    player: int
    source: SourceType


@dataclass
class GameState:
    players: list[PlayerState]
    deck: list[int]
    turn: int = 0
    dealer: int = 0
    phase: str = "draw"
    pending: PendingTile | None = None
    passed: set[int] = field(default_factory=set)
    hu_passed: set[int] = field(default_factory=set)
    resume_phase: str = "discard"
    resume_turn: int = 0
    history: list[Action] = field(default_factory=list)
    river: list[int] = field(default_factory=list)
    winner: int | None = None
    settlement: object | None = None
    seed: int | None = None

    @property
    def terminal(self) -> bool:
        return self.phase == "terminal"

    def validate(self):
        """Check full-deck conservation without exposing it to observations."""
        from collections import Counter

        if len(self.players) != 3 or self.turn not in range(3) or self.dealer not in range(3):
            raise ValueError("invalid seats")
        tiles = list(self.deck) + list(self.river)
        for player in self.players:
            tiles.extend(player.hand)
            for kan in player.kans:
                if player.hand.count(kan) != 3:
                    raise ValueError("kan must remain as three tiles in hand")
            for meld in player.melds:
                if meld.kind in (MeldType.KAN, MeldType.PAIR):
                    raise ValueError("kan/pair cannot be an exposed meld")
                tiles.extend(meld.tiles)
        if self.pending:
            if self.pending.player not in range(3):
                raise ValueError("invalid pending source")
            tiles.append(self.pending.tile)
        for tile in tiles:
            validate(tile)
        if Counter(tiles) != Counter({t: 4 for t in range(20)}):
            raise ValueError("each tile type must have exactly four physical copies")

    def clone(self):
        import copy

        # Actions, melds and pending tiles are immutable. Copy their containers
        # while keeping every mutable hand, pass set and settlement isolated.
        out = copy.copy(self)
        out.players = [
            PlayerState(
                list(p.hand),
                list(p.melds),
                set(p.passed_peng),
                set(p.passed_chi),
                set(p.kans),
                p.hu_disabled,
                p.quad_count,
                p.opening_double_ti_pending,
            )
            for p in self.players
        ]
        out.deck = list(self.deck)
        out.passed = set(self.passed)
        out.hu_passed = set(self.hu_passed)
        out.history = list(self.history)
        out.river = list(self.river)
        out.settlement = copy.deepcopy(self.settlement)
        return out

    def serialize(self) -> str:
        import json
        from dataclasses import asdict

        return json.dumps(asdict(self), default=lambda x: sorted(x), sort_keys=True)

    @classmethod
    def deserialize(cls, payload: str):
        import json

        data = json.loads(payload)
        data["players"] = [
            PlayerState(
                hand=p["hand"],
                melds=[Meld(MeldType(m["kind"]), tuple(m["tiles"])) for m in p["melds"]],
                passed_peng=set(p["passed_peng"]),
                passed_chi=set(p["passed_chi"]),
                kans=set(p["kans"]),
                hu_disabled=p.get("hu_disabled", False),
                quad_count=p["quad_count"],
                opening_double_ti_pending=p["opening_double_ti_pending"],
            )
            for p in data["players"]
        ]
        data["pending"] = (
            PendingTile(
                data["pending"]["tile"],
                data["pending"]["player"],
                SourceType(data["pending"]["source"]),
            )
            if data["pending"]
            else None
        )
        data["passed"] = set(data["passed"])
        data["hu_passed"] = set(data["hu_passed"])
        data["history"] = [
            Action(
                ActionType(a["kind"]),
                a["player"],
                a["tile"],
                tuple(a["chi"]),
                tuple(tuple(b) for b in a["bi"]),
                a["source_player"],
                SourceType(a["source_type"]) if a["source_type"] else None,
                a["forced"],
            )
            for a in data["history"]
        ]
        if data["settlement"] is not None:
            from .scoring import Settlement

            data["settlement"]["payments"] = tuple(data["settlement"]["payments"])
            data["settlement"]["groups"] = tuple(
                Meld(MeldType(m["kind"]), tuple(m["tiles"]))
                for m in data["settlement"].get("groups", [])
            )
            data["settlement"] = Settlement(**data["settlement"])
        return cls(**data)
