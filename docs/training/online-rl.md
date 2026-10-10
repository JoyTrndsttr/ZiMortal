# 在线现金PPO

2026-10-10按用户要求暂停active-v8精算，转向真实对局的在线强化学习。
保留旧认证证据；当前训练不消费这些未达标的搜索标签。

沿用huxi 1D ResNet状态编码器与合法候选动作policy、现金value头。
网络只接收Observation。终局payments[seat]/100同时作为蒙特卡洛价值目标和
PPO优势的回报来源，不裁剪金额。三个epoch、0.2 ratio clipping、0.03 KL早停、
父策略KL正则及熵正则限制更新；原现金value单位一致，不重置该头。
训练轨迹来自随机采样当前策略，混合三人自博弈及冻结huxi/scale100k对手。
每局通过公开RuleEngine完整重放和四张守恒检查，未知规则排除整局并保留问题。

```bash
scripts/train-online-rl-wsl.sh --iterations 12 --games 256 --slots 32 \
  --dev-seeds 30 --holdout-seeds 100 --commit-reports
```

每代256局，共12代。默认训练种子4000000起，开发5000000起、留出6000000起。
开发集对两个冻结对手各轮换三座位，并配对比较候选和父模型。
报告按种子聚合现金差、bootstrap描述性95%区间及改变轨迹的局数。
反复开发集选择会有选择偏差，其区间不能作为正式棋力认证。
训练模型持续更新，best-dev.pt单独保存开发最佳候选；最后锁定候选后一次留出评估。
留出评估不参与选择、训练和参数调整，正式冠军不会自动替换。

CUDA环境为.venv-cuda，RTX5080负责网络批量推理和梯度更新；规则与重放仍用CPU。
WSL服务MemoryMax8G、SwapMax1G、CPUQuota300%、Nice10，线程数2。
权重与原子恢复文件在data/generated/online-rl-v1，报告在docs/training/online-rl-v1。
progress.json记录真实状态；日志在logs/zimortal-online-rl-*.log。
相同命令、目录与源码可恢复last.pt中的模型、优化器和随机状态；配置或源码变化须新版本。
运行期间冻结生产源码。逐代只commit本轮报告，遇已有暂存改动跳过提交以避免混入。

三轮小规模GPU原型已验证48局训练、更新和保存；原型不是正式能力提升证据。
当前缺少对强对手的可靠收益提升证据，后续优先分析独立留出收益、吃碰/胡过的决策变化，
再扩大真实自博弈与历史对手池，避免把训练局数直接当作棋力。
