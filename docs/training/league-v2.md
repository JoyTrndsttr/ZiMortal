# 从新最佳模型继续训练与拆组审核

起点是league-v1/stage-03/best-dev.pt及stage-03/last.pt中匹配的best_optimizer。
该模型上一批最终1000独立种子、八对手的收益差为+7.62元/局，近似95%区间
[6.29,8.93]；新批仍需证明相对这个更强起点的增益，不能沿用旧成绩。

```bash
ZIMORTAL_TRAINING_MODULE=zimortal.training.league scripts/train-online-rl-wsl.sh \
  --directory data/generated/league-v2 --reports docs/training/league-v2 \
  --initial data/generated/league-v1/stage-03/best-dev.pt \
  --initial-optimizer data/generated/league-v1/stage-03/last.pt \
  --games 8192 --later-games 8192 --dev-seeds 60 --screen-seeds 600 --final-seeds 1500 \
  --train-start 100000000 --dev-start 110000000 --screen-start 120000000 \
  --final-start 130000000 --seed 511 --commit-reports
```

四阶段各12代，共393216局。种子段与上一批及人工审核诊断90000000段分离。
沿用现金PPO、自博弈50%、冻结历史对手、近期快照、收益门槛和优化器连续恢复。
CUDA使用RTX5080，CPU负责规则与每局独立重放，WSL内存8G/交换1G/CPU配额300%。
所有源码在启动前冻结，每代及阶段报告自动commit；开发、晋级检查、最终评估口径
同[联赛约定](league.md)，概率接近和静态胡息分析不进入PPO奖励或策略规则。

## 人工审核

540局完整真实诊断，初批筛出85个拆组或大小搭相关的低margin状态，挑选8个案例。
对手不同但历史和状态相同的题去重，兼顾吃比与碰过。见[审核题](league-v2/human-review.md)。
玩家自己的手牌、公开牌组、过张、合法动作概率和完整历史前缀保留；每题从原seed
重放前缀并校验手牌及合法选项一致。网页全信息只供人类审核，不传入策略网络。
审核链接使用review ID，不靠重新生成同seed全员同模型的对局。

静态指标只覆盖不重叠123／2710完整手牌组合，排除三张锁坎。吃及全部下比、碰的
新胡息补回，不能仅看到拆了123就断言总胡息损失。指标不是完整成牌、向听或EV。
“板”按来小字吃手中一对大字定义；其他大小搭单独标识。
概率top2差≤0.2作为筛题条件，它不是经过校准的EV不确定性或棋力证明。
待审选择前展示状态，未来实际输赢不作为人工题目提示，以减少结果偏见。
八个ID已发布后保持稳定；初批标识中的板方向已核对并修正，不改变轨迹或概率。

下一步将用户意见独立记录，分析抢速与胡息／名堂取舍，不直接转为精确EV标签，
也不把经验变成非法动作限制。主要差距是收益虽提升，复杂拆组决策的可靠性仍待审查。
验证：264项测试通过、1项跳过；静态组合重叠、坎、全部下比、碰新增胡息和混合
轨迹重放均有回归检查。网页HTTP接口及JavaScript语法检查通过，未做真实浏览器视觉检查。
