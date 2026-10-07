# 从零训练实验

训练依赖：`uv pip install --python .venv/bin/python -e '.[training,dev]'`。
本机实验使用 PyTorch 2.14.1+cpu、NumPy 2.5.2，CPU 2 线程；虽检测到 RTX 5080，本轮未使用 GPU。

## 第一轮：监督预训练

```bash
.venv/bin/python -m zimortal.training.train --architecture mlp --puzzles 2048 --games 40 --epochs 10 --output checkpoints/round1-mlp.pt --report docs/training/round1-mlp.json
.venv/bin/python -m zimortal.training.train --architecture resnet --puzzles 2048 --games 40 --epochs 10 --output checkpoints/round1-resnet.pt --report docs/training/round1-resnet.json
.venv/bin/python -m zimortal.training.review --checkpoint checkpoints/round1-resnet.pt --output docs/training/round1-review.json
```

输入是玩家 Observation 的 45×20 张量。模型对每个引擎提供的完整合法动作打分；吃与下比保留各组组合特征，填充动作被 mask。共享 encoder 输出 policy、value、20 种牌的听牌辅助头。MLP 与两层 1D ResNet 均从随机参数开始。

2048 个临胡扰动题加 40 场随机合法对局，共 3147 个训练决策；543 个验证决策使用独立种子命名空间。0～3 次扰动不是精确向听数；教师是“精确自摸听牌＋可见余张上界＋局部搭子”的启发式，不使用暗手。余张上界不是对手手中牌的概率。value 此阶段表示即时可胡／自摸听牌 readiness，不是终局胜率。

固定随机对手评估种子 7000～7019，每种子轮换模型三个座位、庄家 seed%3。MLP 10/60 胜、ResNet 11/60 胜；ResNet 对教师种子 8000～8005 为 1/18 胜。这只是小样本基线，不构成棋力结论。验证教师动作准确率分别 42.54%、44.01%；它也不是胡牌正确率。

审查保留种子 9000～9049，逐步验证牌张守恒、合法选择、禁吃自己打过的牌与完整重放。review JSON 中的差异是启发式策略分歧，不是已确认规则 bug；步骤号按引擎动作计，包含过牌。

修正：样本发牌可能给对手起手四张，必须先完成强制提，否则庄家 Observation 的动作集合为空；已加回归测试。强制动作不需要网络训练；训练与网页全信息审查严格分离。

模型权重位于本机 `checkpoints/`（忽略 Git），报告保留 SHA256。所有训练命令可复现 CPU 实验，跨硬件／版本不保证逐位一致。

下一步：聚合模型状态继续监督训练，再用真实终局胜负进行带监督锚点的自博弈 actor-critic。仍缺精确向听、多步搜索、隐藏信息信念、可信的大样本棋力评估。

## 第二轮：模型状态聚合

```bash
.venv/bin/python -m zimortal.training.train --seed 22 --resume checkpoints/round1-resnet.pt --puzzles 4096 --games 100 --aggregate 200 --epochs 14 --lr 0.0005 --output checkpoints/round2-resnet.pt --report docs/training/round2-resnet.json
.venv/bin/python -m zimortal.training.review --checkpoint checkpoints/round2-resnet.pt --output docs/training/round2-review.json
```

4096 扰动题＋100 随机对局得到 6841 个决策，再聚合第一轮模型的 200 场对局（78 场有人胡牌）中的 5758 个决策，总计 12599。聚合过程检查守恒和完整重放。监督标签仍来自可见信息教师；这是一次 DAgger 风格的数据聚合，尚不是强化学习。独立验证 933 个样本、教师动作准确率 57.23%；验证集与第一轮不同，不能直接比较准确率。

同一固定评估：随机对手 13/60 胜，教师 2/18 胜，均无非法动作、无漏胡。还不足以证明稳定提升，更没有超过教师。修正教师模拟动作后新增公开牌的计数，避免高估余张；禁止胡牌状态的辅助听牌标签归零。聚合加入完整重放检查。

网页新增随机／三轮模型选择和种子 URL：`http://127.0.0.1:8765/?seed=9000&dealer=0&policy=round2`。这里三个座位均使用所选模型，和“模型对随机对手”评估不同。审查 JSON 的种子、庄家、步骤可在对应轮次网页中定位；模型始终仅收到 Observation。模型不存在时提示错误，不自动换成随机。浏览器已验证模型选择、URL 初始化、首尾导航与切回随机。

剩余问题：教师仍只看一步，模型仍会选择明显弱于教师的出牌；没有精确 n 向听标签。下一轮尝试真实终局反馈，并保留监督锚点限制遗忘。

第二轮重放审查定位到 seed 9049、终局第44步：protected 坎遍历顺序不稳定，使相同结算的牌组顺序不同。evaluator 现在按牌种排序固定坎，补充不同 protected 顺序的回归测试；胡息和结算数值不变。修正后重新完成同组审查。

## 第三轮：真实终局反馈

```bash
.venv/bin/python -m zimortal.training.reinforce --resume checkpoints/round2-resnet.pt --iterations 10 --games 40 --output checkpoints/round3-resnet.pt --report docs/training/round3-resnet.json
.venv/bin/python -m zimortal.training.review --checkpoint checkpoints/round3-resnet.pt --output docs/training/round3-review.json
.venv/bin/python -m zimortal.training.benchmark
```

从第二轮继续，400 场训练对局、3819 个决策、10 次采样迭代，每批做三遍裁剪 actor-critic 更新。75% 对局对手是冻结的第二轮模型，25% 对手随机；当前玩家按概率采样，座位与庄家分别轮换。裁剪比率 0.8～1.2、entropy 0.01、value MSE 权重 0.5、监督锚点 0.2、AdamW 学习率 1e-4、梯度范数上限1。256 个独立种子临胡题用作监督锚点。冻结对手池尚未滚动扩充，也没有多代 league。

每场终局和完整重放都验证。训练玩家赢59场、流局217场，其余124场输；1627 个决策获得非零终局反馈。终局收益采用赢家+1、两个输家各-0.5、流局0，按自己的剩余决策次数以0.99折扣；尚未直接优化结算金额。value 从 readiness 转为此收益估计，切换时将最后一层归零，不能解释为校准后的胡牌概率。模型 encoder／policy 参数相对第二轮 L2 差异0.680678，确实发生参数更新。

固定旧评估：随机对手15/60胜、教师2/18胜，无非法动作和漏胡。另进行独立评估，种子10000～10049，每种子轮换三个座位，150场：

| 策略 | 对随机胜场 | 胜率 | 平均结算收益 |
| --- | ---: | ---: | ---: |
| 随机合法动作 | 4 | 2.67% | 0.50 |
| 启发式教师 | 75 | 50.00% | 25.60 |
| 第一轮 ResNet | 21 | 14.00% | 4.67 |
| 第二轮 ResNet | 34 | 22.67% | 15.80 |
| 第三轮 ResNet | 42 | 28.00% | 12.73 |

新种子12000～12019对教师，各60场，第二／三轮都仅8胜，平均结算分别-7.00／-13.25。种子11000～11019的交叉评估：第三轮对两个第二轮8/60胜，第二轮对两个第三轮10/60胜。三人对局不是两人配对零和比较；每个种子的多个座位也相关，不能当450个独立伯努利样本。所有指标是探索性小样本结果，尚未检验显著性。

结论：监督聚合和终局训练使模型对随机对手更常胡牌，但第三轮没有证明综合强于第二轮，结算收益甚至下降；不自动覆盖第二轮作为唯一“最佳模型”。仍明显弱于教师。保留两轮供复盘，下一步优先采用实际结算收益训练、增加样本和滚动对手池，再加入基于可见信息的多步搜索与隐藏信息信念。

三轮审查各50场，分别3467／3300／3333步；第三轮25场有人胡。规则检查不能证明所有规则边界正确。第三轮审查保存完整动作（包含 chi／bi），避免同为吃同一张时遗漏方案差异。

### 留给人工审查的策略分歧

未发现需要新增规则确认的未定义边界；以下请在人有空时评价策略，而非判定规则 bug：

- 第三轮，seed9002，庄家玩家三，第12步，玩家一吃三：模型选择“三三叁”，教师选择“一二三＋下比一二三”。两种方案均合法，教师认为后者保留更多自摸听牌机会。链接：`http://127.0.0.1:8765/?seed=9002&dealer=2&policy=round3`。
- 第三轮，seed9000，庄家玩家一，第2步：模型打壹，教师打三。是否应优先保留大小搭／对子，留待人工判断。链接：`http://127.0.0.1:8765/?seed=9000&dealer=0&policy=round3`。

步骤号包含强制动作与过牌。启发式差值只是教师评分，不是胡率或真实 EV。检查使用的全信息帧不进入模型，教师也不读对手暗手或真实牌堆顺序。

最终验证：165 个测试通过，Ruff 与 `git diff --check` 通过。浏览器验证第三轮 seed9002 第12步与记录的“三三叁、无下比”一致、首尾导航和终局可查看，并在390px宽度下无 JavaScript 错误。坎排序回归测试移入不依赖 PyTorch 的引擎测试文件，基础安装也会运行该检查。

## 第四次迭代：十万级扩容

已完成107031条不同决策输入、4674条独立验证输入的10epoch训练。新种子对随机41/150胜，对教师8/60胜，100场完整模型审查通过。数据分片、断点续跑、去重、代码及文件哈希验证、命令和限制见 [十万级训练记录](scale100k.md)。网页新增 `policy=scale100k`，旧权重保留。
