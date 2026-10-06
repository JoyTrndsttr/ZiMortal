"""Ningxiang Paohuzi rules engine; no policy or neural-network dependencies."""

from .chi import enumerate_chi_with_required_bi
from .evaluator import evaluate_hand
from .game import RuleClarificationRequired, RuleConfig, RuleEngine
from .observation import Observation, observe
from .scoring import base_amount, detect_fan, meld_huxi, settle
from .tiles import Tile, is_big, is_red, make_deck, rank, tile_name
from .types import (
    Action,
    ActionType,
    GameState,
    Meld,
    MeldType,
    PendingTile,
    PlayerState,
    SourceType,
)

FullGameState = GameState

__all__ = [
    "Action",
    "ActionType",
    "FullGameState",
    "GameState",
    "Meld",
    "MeldType",
    "Observation",
    "PendingTile",
    "PlayerState",
    "RuleClarificationRequired",
    "RuleConfig",
    "RuleEngine",
    "SourceType",
    "Tile",
    "base_amount",
    "detect_fan",
    "enumerate_chi_with_required_bi",
    "evaluate_hand",
    "is_big",
    "is_red",
    "make_deck",
    "meld_huxi",
    "observe",
    "rank",
    "settle",
    "tile_name",
]
