# 第五轮：十五胡门槛与真实结算收益

本轮针对原 scale100k 策略在 seed 18000 的两个问题：玩家二第6步拆掉壹贰叁，第17步在所有结构进张都不足15胡时放弃碰肆。规则没有变化；不足15胡的结构分析不能作为合法胡牌。

## 表示与训练目标

保留原45通道模型兼容性，新模型使用20牌种×55通道。新增十五胡门槛、锁定胡息缺口、手牌完成组合的胡息潜力、各牌出掉造成的组合损失、坎标识以及大小123／2710完成比例。完成组合潜力通过不重叠分组求值，不代表已经满足完整胡牌分解。

1D ResNet 共享编码器保留 policy／value／结构进张辅助头，增加21维胡息头：手牌组合潜力以及20种自摸后的结构胡息。低于15胡的结构有标签，但真实规则引擎仍拒绝胡牌。进张分析模拟强制偎、提、跑，普通结算使用正式胡息、名堂和金额公式。未来进张不假设天胡／地胡，也不读取对手暗手或牌堆顺序。

监督教师同时考虑胡息缺口、成组胡息和可见剩余牌的普通结算金额上界。这是启发式评分，不是进张概率或精确EV。收益强化学习使用真正终局 payments/100，保留赢家收两家的金额、输家实际付款和流局零收益，不截断大名堂收益，不折扣较晚的结算，也不额外奖励虚构的胡息。价值头切换目标时重新初始化输出层，不能解释为胜率。

## 数据、训练与复现

重新生成107,390条不同训练决策和4,633条独立验证决策，包含临胡扰动题及随机完整对局，剔除重复输入。数据保存在本机 `data/generated/huxi100k/`，旧数据和权重保留。12个epoch监督训练后，进行20轮×100场混合对手结算收益训练，共18,111条学习者决策。混合对手包括随机、旧教师和冻结监督模型；座位轮换。

```bash
.venv/bin/python -m zimortal.training.huxi build --workers 4
.venv/bin/python -m zimortal.training.huxi train --epochs 12
.venv/bin/python -m zimortal.training.reinforce --resume checkpoints/huxi-warmup.pt --output checkpoints/huxi-resnet.pt --report docs/training/huxi-settlement.json --seed 66 --iterations 20 --games 100 --reward settlement --opponents mixed --eval-start 21000
.venv/bin/python -m zimortal.training.huxi_review data --output docs/training/huxi-data-audit.json
.venv/bin/python -m zimortal.training.huxi_review diagnostics --output docs/training/huxi-diagnostics.json
.venv/bin/python -m zimortal.training.huxi_review benchmark --output docs/training/huxi-comparison.json
.venv/bin/python -m zimortal.training.review --checkpoint checkpoints/huxi-resnet.pt --start 22000 --games 100 --teacher huxi --output docs/training/huxi-game-review.json
```

完整参数默认值以模块 `--help` 为准。报告包含数据源与权重校验值，训练与评估种子隔离。数据抽查重新生成24局、663条决策及50题，核对编码、策略和胡息标签；100局模型复盘共5,244步，82局有人胡，全部检查守恒和确定性重放。策略与教师分歧是候选审查点，不直接判为规则错误。

## 相同种子、相同对手的对照

每个种子轮换三个学习者座位，对手两家为随机或旧启发式教师。评估网页则三家均使用所选模型，不要混淆两种设置。

| 模型 | 随机对手胡牌/150 | 平均结算收益 | 教师对手胡牌/60 | 平均结算收益 |
| --- | ---: | ---: | ---: | ---: |
| 原 scale100k | 45 | 23.80 | 9 | -12.33 |
| 胡息监督 | 53 | 16.70 | 9 | -10.67 |
| 胡息结算收益 | 65 | 26.93 | 13 | -7.25 |

样本量仍小，随机对手弱，不能据此声称高强度或统计显著提升。监督模型更常胡却收益更低，说明不能只优化胡牌次数。结算模型随机对手的获胜平均胡息19.26、平均番数1.77，原模型19.29、2.07：本轮尚未证明更擅长做大胡息或大名堂，收益改善主要伴随更常胡牌。

冻结原 seed 18000 的状态重新给三个模型评分：第6步新模型打玖，保留壹贰叁；第17步新模型碰肆（候选softmax约99.54%），旧模型选择过。此时玩家二所有自摸结构最高13胡，不足15；碰肆增加3胡息并继续调整手牌。冻结状态来自旧模型轨迹，新模型重新打一局后步数与状态会改变，不能把这些结果冒称为新局第17步。

## 已知差距与下一步

胡息辅助预测仍然薄弱：验证中只有90个有完整结构的进张标签，结算模型这些牌的胡息MAE约11.99胡，十五胡阈值准确率58.89%。普通对局大量“不成结构”标签淹没了临胡关键样本，因此下一步优先补充真实合法的14／15胡边界题，均衡有结构与无结构进张，分别评估自摸和他人来牌，强化名堂金额预测。网络预测不能替代精确引擎裁决，网页使用精确结构分析。

随后扩大独立收益评估，加入隐藏牌信念和搜索，减少启发式教师依赖。当前对教师仍负收益，没有证明高强度棋力，也没有证明价值头已校准成动作EV。

网页策略 `huxi` 为结算收益模型，`huxi_warmup` 为监督模型。页面标明各座位策略、锁定胡息、组合潜力和仍可见的结构进张是否达到15胡；公开耗尽的牌不列作进张。需先出牌或执行起手提牌时暂不展示未来进张。
