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
