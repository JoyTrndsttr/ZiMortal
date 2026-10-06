# ZiMortal

**ZiMortal — A strong self-play AI for Ningxiang Paohuzi (Zi Pai), inspired by Mortal.**

ZiMortal explores imperfect-information reasoning for Ningxiang Paohuzi: estimating the unseen deck, learning high-EV decisions, and turning self-play behavior into human-readable strategy.

## Goals

- Reproduce Ningxiang Paohuzi rules exactly.
- Build a fast simulation engine for large-scale self-play.
- Learn belief over hidden hands and the remaining deck.
- Train policy/value models for action selection.
- Support stronger search-based analysis for review and study.
- Distill learned behavior into practical human strategy.

## Project Structure

```text
ZiMortal/
├── docs/
│   ├── rules/
│   │   └── ningxiang-paohuzi.md
│   └── strategy/
│       └── belief-and-live-tiles.md
├── zimortal/
│   ├── engine/
│   ├── belief/
│   ├── model/
│   ├── selfplay/
│   └── analysis/
├── tests/
├── pyproject.toml
└── README.md
```

## Roadmap

1. Rule engine and legality checks
2. Deterministic hand evaluator
3. Information-set state representation
4. Belief estimation for hidden tiles
5. Monte Carlo rollout baseline
6. Self-play policy/value learning
7. Search-enhanced "big model" analysis
8. Strategy mining and human-readable review

## Why ZiMortal?

The name combines **Zi Pai** (字牌) with a tribute to **Mortal**, the open-source Japanese mahjong AI project that inspired the overall direction.

## Status

Early-stage research and engineering project. The first priority is correctness of the game engine before any large-scale learning.
