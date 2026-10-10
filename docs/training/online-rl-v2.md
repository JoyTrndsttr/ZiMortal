# 扩容自博弈与决策审查

首批12代PPO未证明收益提升。本批从保留的huxi冠军重新开始，用更多真实对局
和对手多样性降低小批次更新的风险；旧批的第二代和第十二代也作为冻结对手。

每代2048局，共24代49152局，50%三人自博弈。冻结对手包含huxi、scale100k、
boundary-calibrated、planning-cuda、round3、online-rl-v1第二代与第十二代。
再加入最多三个近期训练快照，限制池大小。所有对手只接收自身Observation。
比赛类型先独立打乱，再轮换庄家和训练座位，修正旧调度中自博弈只分配庄家0的偏差。
冻结七模型用于开发与留出，近期快照仅用于训练；同一批每代评估对手保持不变。

```bash
scripts/train-online-rl-wsl.sh \
  --directory data/generated/online-rl-v2 --reports docs/training/online-rl-v2 \
  --iterations 24 --games 2048 --slots 64 --epochs 2 --batch-size 512 --lr 0.00005 \
  --seed 209 --train-start 7000000 --dev-start 8000000 --dev-seeds 30 \
  --holdout-start 9000000 --holdout-seeds 200 \
  --opponents checkpoints/boundary-calibrated.pt checkpoints/planning-cuda-resnet.pt \
  checkpoints/round3-resnet.pt data/generated/online-rl-v1/generation-002.pt \
  data/generated/online-rl-v1/generation-012.pt \
  --snapshot-pool 3 --selfplay-fraction 0.5 --audit-decisions --commit-reports
```

仍使用最终净现金/100，不裁剪奖励。学习率减半、两个epoch、较大mini-batch；
这是待检验的训练设置，不能保证更强。正式留出种子此前未用于训练／开发／诊断。
每代保存报告与断点并实际commit，内存8G、交换1G、CPU配额300%、Nice10。
GPU承担网络推理与梯度，CPU执行规则和每局独立完整重放。

## 决策审查口径

各代decision_audit记录训练中的吃／比、碰／过碰、胡／不胡决策；development
和最终holdout记录双方贪心评估决策。仅统计完成且合法重放的整局。
报告包含动作次数、父策略到新动作的变化、实际终局净金额与获胜比例。
例子保留种子、庄家、玩家、动作步号、自己的手牌、公开胡息、剩余牌数、全部
合法方案与模型概率。训练例子采用随机采样，不能仅靠seed重放，须保留原批配置
和检查点恢复；贪心评估例子可以固定种子与对手复现。

动作分组来自不同状态，平均实战金额**不是因果动作EV**。训练中的动作与父策略
不同还包含探索效果；判断策略实际变化应看贪心评估。吃的不同下比方案完整保留，
同为chi但方案不同也计为决策变化。

胡牌机会额外记录当场HU的精确净结算（两家付款之和），与该局后续实际净收益比较。
该差值是本次未来轨迹的实现值，不是未来收益期望。只用规则引擎计算反事实当场结算，
不让暗手和牌堆进入策略。6500000起30个诊断种子用于上批专项复盘，不能再称未见测试。

实际新批进展见data/generated/online-rl-v2/progress.json与逐代报告。
目前距离强AI仍缺在多种对手和新的独立种子上稳定提高现金收益的证据。
下一步根据贪心评估中吃碰和胡过的具体变化调整训练，不把探索局的表面平均收益当棋力。
