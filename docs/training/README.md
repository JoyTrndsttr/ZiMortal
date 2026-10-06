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
