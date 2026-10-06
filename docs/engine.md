# 第一版规则引擎

## 使用

```python
import random
from zimortal.engine import RuleEngine, observe

engine = RuleEngine()
initial = engine.new_game(seed=42, dealer=0)
state = initial.clone()
policy_rng = random.Random(42)
while not state.terminal:
    observation = observe(engine, state, engine.decision_player(state))
    actions = engine.legal_actions(state)
    state = engine.step(state, policy_rng.choice(actions))

assert engine.replay(initial, state.history).serialize() == state.serialize()
```

随机选动作仅在调用端；引擎不包含策略。`step` 返回独立新状态，非法动作抛出
`ValueError`，不修改输入。洗牌和策略随机源相互独立。牌堆末端为下一张摸牌。
庄家的决策之前先处理所有起手提，再检查天胡，可 PASS 后出第一张。

## 模块

- `tiles.py`：20种稳定编码、80张牌堆、本地随机种子。
- `types.py`：完整状态、动作、牌组、过碰/过张、暗坎、夹比、历史、JSON与克隆。
- `chi.py`：全部吃牌模式与递归完整下比枚举，保护起手坎。
- `evaluator.py`：有界缓存的精确覆盖分解，固定牌组与坎、将、15胡门槛。
- `scoring.py`：胡息替换、名堂累加、基础金额、两家完整付款。
- `game.py`：发牌、开局、强制动作、响应优先级、胡牌和终局、重放。
- `observation.py`：不可变可见快照。包含自己的手牌及过牌记录，不包含对手手牌、
  坎的牌型、对手过牌记录或牌堆顺序。公开的同牌牌组可由亮出的牌推知全部牌型。

`MeldType.KAN` 只用于分解；开局坎保留在 `PlayerState.hand/kans` 中。
`DRAW`、无响应的强制 `PASS` 也为合法动作，因此重放不依赖隐式随机操作。
强制偎/提后设置独立胡牌决策窗口，PASS 后恢复夹比决定的出牌/摸牌阶段。
胡牌 PASS 与碰/吃 PASS 分开，避免不胡时同时放弃较低优先级的动作。
胡牌有多个结构时，计分采用最终金额最高的结构。

## 检查

```sh
python -m pytest -q
ruff check zimortal tests
```

当前版本以规则正确性为目标，未实现神经网络、策略、搜索或性能优化。
后续 belief 模块应只接收 `Observation`，通过显式接口采样完整状态。
规则边界与已确认口径见 `docs/rules/ningxiang-paohuzi.md`。

## 吃后与偎后无牌可打

维护者于2026-10-07确认：

- 吃及全部下比后，如本次需要出牌但只剩坎或空手，该吃牌方案非法。
  起手双提的首次进张原本免出牌，仍允许执行。
- 强制偎／臭偎后无牌可打且不能胡时，免出牌，由下家摸牌；该玩家本局
  不能胡、不再吃碰。后续强制跑／提仍执行，且不要求其从坎中出牌。
- 若偎后已满足胡牌条件，仍先提供胡牌机会；选择不胡且无牌可打时进入上述状态。

`PlayerState.hu_disabled` 为该状态的持久标记，随克隆、重放及序列化保存。
自己能在 Observation 中看到该标记。`RuleClarificationRequired` 保留用于报告
其他尚未定义的无牌可出边界，不自动拆坎。

## 本地逐步复盘网页

```sh
python -m zimortal.web.server --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。无需安装额外运行依赖。
选择种子和庄家后生成整局；使用上一步、下一步、播放、进度条或动作时间线
检查每步。左右方向键也可切换步骤。默认种子118用于回归原先的非法吃边界。

页面展示每步三家手牌、坎、公开牌组及胡息、过碰/过张、跑提次数、不能胡状态、
墩牌来源、落桌牌、下一步合法动作、规则说明与结算。摸牌事件明确记录当次牌型。
这里的全信息快照仅用于人工审查，与 AI 使用的 `Observation` 接口分离。
网页中的验证是合法动作成员检查、四张守恒与整局重放一致性，并非独立规则裁判。

服务器仅监听本机，`--host` 和 `--port` 可配置。随机策略仅用于生成核对样本，
不是 AI。相同种子与庄家产生相同轨迹。
