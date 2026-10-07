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

## 神经网络训练与复盘

已完成三轮实验：MLP／1D ResNet 监督预训练、模型状态聚合、自博弈 actor-critic。
输入仅使用玩家可见 Observation，网络有 policy／value／听牌辅助头。
安装训练依赖：`uv pip install --python .venv/bin/python -e '.[training,dev]'`。
训练命令、完整结果和限制见 [训练记录](docs/training/README.md)。
本机权重保存在 `checkpoints/`，不纳入 Git；各轮报告记录校验值。

复盘网页新增各轮模型策略，默认仍为随机合法动作。
例如 `http://127.0.0.1:8765/?seed=9002&dealer=2&policy=round3`，第12步可查看
模型与教师对吃牌／下比方案的分歧。所有座位使用所选策略，网页为全信息审查，
网络输入仍经过 Observation 隔离。模型目前尚未超过启发式教师。

第五轮新增 [胡息与结算收益训练](docs/training/huxi.md)。复盘策略 `huxi` 显示锁定胡息、组合潜力及不足15胡的结构进张；当前收益模型对教师仍为负收益。

第六轮加入 [十五胡边界与名堂金额校准](docs/training/boundary.md)，网页策略 `boundary` 可复盘新的收益训练模型，`boundary_warmup` 可对照校准模型。
