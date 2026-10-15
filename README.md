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

## 当前研究主线（2026-10-15）

当前建立[Mortal-faithful-v1主线](docs/training/mortal-style.md)：55×20可见编码，
128通道×12块的1D ResNet＋Channel Attention、1024 latent和结构化合法动作Dueling Q。
默认terminal现金MC＋offline CQL，3-step SARSA仅作对照；support只作离线安全护栏。
固定10k的小模型/新模型×MC/3-step四组CUDA对照和新500种子×3座位筛选已完成。
新模型MC护栏现金−0.46元/局；相对旧CQL+support为+2.07，95%区间[−0.65,5.07]，
未证明达到预先设定门槛；raw仍为−17.87元/局。停止，不进入online或替换baseline。
停止扩展pairwise/counterfactual训练；RouteQ、trajectory、PPO与历史证据全部保留。
最新离线诊断：LR/patience对照支持快速过拟合，未修复原epoch1最佳；胡／过负相关
主要受稀少的高收益PASS影响。低支持argmax仍是raw风险，暂不直接扩50k，详见同一报告。
20k过渡实验完成：同更新预算下16k改善新test收益预测，但raw现金相对8k差−0.99元/局，
95%区间[−4.11,1.98]；护栏差+1.60，区间[−0.76,3.94]，低支持推翻比例未下降。
停止同分布扩量，不进入50k/online；下一步建议研究behavior coverage，baseline保持冻结。
小规模behavior coverage对照也已完成：弃牌/吃过替代动作执行显著增加，但raw现金
B−A仅+0.63元/局[−2.18,3.53]，回归略差，护栏依赖仍大；停止追加探索，
下一步建议诊断固定权重的policy extraction，尚未启动online或替换baseline。
以下训练记录按历史阶段保留。

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

第七轮试验加入[精确进张距离、现金rollout与belief](docs/training/planning.md)，以及[WSL训练隔离与断点恢复](docs/training/wsl.md)。收益基线仍为第五轮，试验模型须经独立评估。

第八轮采用[无人工牌谱的模拟现金Q训练](docs/training/cashq.md)。网页 `policy=cashq` 展示实验模型每个合法动作的净收益预测与成员分歧；它们是模拟策略下的估计，不能视为精确EV或校准置信区间。

下一阶段启动[主动状态挖掘与自适应rollout](docs/training/active.md)，优先高分歧与关键规则状态，
依据共同粒子的收益差区间停止；至少5000个合格训练根后才训练下一代实验模型。
