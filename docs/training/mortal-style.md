# Mortal风格终局现金Q：独立主线

2026-10-14按用户本轮明确要求，暂停扩展trajectory／RouteQ／显式路线。
routeq-v2任务已停止，保存50个训练根及1个整根排除，未完成训练，全部旧代码、
权重和结果保留。此前PPO、现金Q和路线试验均不删除、不混入新数据。
本轮只执行A/B/C：独立模型、1000局审计、两种小训练，完成后停下汇报。
不自动扩大10k／50k，不进入online，也不替换baseline。

## 官方实现调研与对应

参考2026-10-14读取的官方源码：
[model.py](https://github.com/Equim-chan/Mortal/blob/main/mortal/model.py)、
[train.py](https://github.com/Equim-chan/Mortal/blob/main/mortal/train.py)、
[dataloader.py](https://github.com/Equim-chan/Mortal/blob/main/mortal/dataloader.py)、
[player.py](https://github.com/Equim-chan/Mortal/blob/main/mortal/player.py)、
[engine.py](https://github.com/Equim-chan/Mortal/blob/main/mortal/engine.py)。
本项目独立实现训练思想，未复制或导入官方实现代码。

| 官方Mortal | ZiMortal第一版 |
| --- | --- |
| Brain：1D ResNet，通道注意力、BatchNorm、Mish，34牌位置、1024 latent | 既有ResidualBlock思路：GroupNorm／SiLU，20位置、55通道，small宽32、3块、256 latent；保留表示以隔离算法改变 |
| 固定动作DQN：V＋A−合法A均值 | encode_action结构化候选（包括完整chi／bi），状态＋动作MLP产生A，mask内均值；padding Q为负无穷 |
| gamma^steps_to_done × kyoku_reward；训练用0.5 MSE | gamma^剩余本家非强制决策数 × payments[seat]/100；用用户指定Huber，不裁剪回报 |
| 离线CQL：合法Q的logsumexp−chosen Q；正系数相加，在线关闭此项 | MC-only alpha=0与MC+CQL alpha=0.1对照；合法候选内计算，不加入PPO |
| 日麻牌谱loader、按玩家筛选、GRP计算跨局排名相关奖励 | 无高质量真人牌谱，以冻结强v0产生bootstrap完整局；单局真实现金，不复用日麻奖励或GRP |
| 冻结baseline，训练／测试OneVsThree | 一探索learner＋两冻结v0，三个座位与庄家独立交叉轮换 |
| Boltzmann epsilon在greedy／温度Q采样间切换，支持top-p | v0 policy logits温度采样T=1.5；top-p=1，无epsilon。可选epsilon明确为uniform混合，与官方语义不同 |
| 测试avg_rank／avg_pt优于旧best时保存；并无本项目所要求的bootstrap认证 | 后续晋级用独立现金、种子聚合区间与历史对手退步检查；不能宣称严格复制官方晋级公式 |

来源奖励、损失、探索和晋级均有明确适配差异，不能把Dueling结构相同说成
完全复刻Mortal训练效果。不能直接复用固定动作空间、日麻规则、四人竞技、
GRP、排名收益、oracle输入或日麻和牌保护规则。

## 模型与数据

`model.mortal_q.MortalQNet`不覆盖PolicyValueNet，独立随机初始化，
small默认width32／latent256／3块；可配置width64／latent512做后续base对照。
只接受55×20 Observation编码及结构化合法候选。没有路线／未来赢家组／
人工偏好输入，没有wait、huxi等辅助head；既有55通道中的公开胡息表示保持不变。
模型可经runtime.save_model/load_model原子保存／恢复结构与权重。

behavior固定为`data/generated/league-v2/stage-02/best-dev.pt`，称ZiMortal-v0。
只保存探索learner非强制决策；其他两人greedy，非法动作概率始终为0。
每个有效完整局保留seed、dealer、learner seat、完整历史与终局审计状态。
决策字段包括features、actions、chosen、terminal_cash、steps_to_terminal、
seed／seat／game_id、behavior_logprob、完整行为概率、parent_greedy、history_step、
动作族；policy version和checkpoint／源码哈希存manifest元数据。
折扣步数是之后本家自由决策数，不是摸牌／过牌等引擎总步数。

256完整局一shard（尾块232），features保持float32；动作通过offsets存ragged，
mask在batch时重建，只有实际候选为True。NPZ每成员一次读取后共享切片，避免
逐行重复解压。manifest带数据／重放SHA256，完整状态仅用于审计和标签。
训练／开发／检验按完整游戏seed%10切分为80%／10%／10%，没有同局状态跨集。
生成时逐step验证守恒，终局标准引擎完整重放；落盘后再次逐决策比较可见features、
legal actions、chosen、行为概率与现金标签。未知规则整局排除，保存错误和前缀，
独立新seed补足有效局，不伪造0回报；真实流局才为0。

## 有限实验与性能记录

```bash
ZIMORTAL_TRAINING_MODULE=zimortal.training.mortal_offline \
  bash scripts/train-online-rl-wsl.sh \
  --directory data/generated/mortal-v1 --games 1000 --device cuda \
  --temperature 1.5 --alpha-cql 0.1 --gamma 1 --epochs 8
```

两变体使用相同初始权重、数据、batch shuffle种子，AdamW lr3e-4、Huber、
梯度范数1；开发RMSE选型，锁定后才看留出回归／reranking。
恢复文件保存匹配optimizer、权重哈希与随机状态；本版入口只允许新目录，
不自动覆盖或续写已有数据。逐epoch结果只留本地。

采样32slots批量网络推理；CPU执行Python规则和独立重放，CUDA做网络与更新。
默认WSL上限8GiB、Swap1GiB、300%CPU、2线程；保存前保留2GiB磁盘空间。
profile分别计编码、网络、legal_actions、step、胡求解、吃枚举、重放时间；
计时为inclusive，有重叠，不能直接相加当流水线占比。
CPU为进程累计CPU秒／wall，100%对应一核；nvidia-smi每0.5秒采GPU／显存利用率，
若WSL未提供数值如实报告不可测。无CUDA不悄悄回退CPU。

本轮不进行实战晋级，回归或reranking变化不证明牌力提升。
结果完成后在本节追加实测参数量、有效决策、速度、探索、Q数值和loss。

## 后续门槛

先汇报1000局A/B/C后停止。确认无明显问题再做D：10k局回归与reranking；
E：50k局MC／CQL对照；F：开发集选择后独立3000种子×三座位实战，
按seed聚合现金差与近似区间，检查改变轨迹比例和各动作族。
有可靠现金改善才实施online：同代C/B/B，baseline固定，300开发、1000中期、
5000或更多正式晋级种子并检查历史对手；未通过不换baseline。
没有证据证明此主线一定强于PPO；也不能因本轮转向就宣称旧路线实验错误。

## A/B/C实际结果：1000局，已停止

实际服务`zimortal-online-rl-20261014-010052.service`退出成功、MainPID=0。
PyTorch2.7.1+cu128，RTX5080；249,922参数，small32／256／3块。
1000有效完整局，8381个探索learner决策，规则排除0。
训练800局6722决策、开发100局871决策、独立检验100局788决策。
四个shard为256／256／256／232局；哈希、守恒、完整重放及所有可见输入逐行核对通过。
权重与配对optimizer保存在本地`mortal-v1/mc`及`mortal-v1/mc-cql`，未更换baseline。
本批共享behavior版本由manifest中的固定v0 checkpoint路径与SHA256标识；
落盘字段后续已增加显式policy_version，未修改本批训练输入或现金标签。

T=1.5、epsilon0、top-p1：平均动作entropy0.406 nat，1209/8381（14.43%）
不同于v0 greedy。探索覆盖：

| 决策族 | 数量 | 非greedy数量／比例 |
| --- | ---: | --- |
| 弃牌 | 4047 | 741／18.31% |
| 吃／过 | 2087 | 409／19.60% |
| 碰／过 | 1973 | 59／2.99% |
| 胡／过 | 274 | 0／0% |

碰、胡决策仍很集中，温度1.5没有产生胡／过探索；这是后续需要诊断的覆盖限制，
不能把“有温度采样”当成所有关键决策已充分探索。

两组各8轮，MC训练2.36秒，MC+CQL训练1.96秒；训练MC loss分别从0.1796
降至0.0586／0.0623，没有NaN、非法Q或非有限梯度。两组开发RMSE均在第1轮最优，
后续训练loss继续下降但开发误差上升，提示小数据过拟合。以下是开发选定模型，
不是第8轮最后权重；Q、回报单位均为现金/100：

| 检验指标 | MC | MC+CQL（0.1） |
| --- | ---: | ---: |
| chosen-Q MAE | 0.4207 | 0.4208 |
| chosen-Q RMSE | 0.8579 | 0.8581 |
| correlation | 0.1144 | 0.1128 |
| 全合法Q范围 | −0.2397～0.2600 | −0.2505～0.2606 |
| 全合法Q均值／std | −0.0257／0.0865 | −0.0281／0.0870 |
| chosen target均值／std | −0.1525／0.8527 | −0.1525／0.8527 |
| 相对父greedy改变动作率 | 58.76% | 59.14% |

MC的检验reranking：弃牌326/398、吃过134/198、碰过3/163、胡过0/29；
CQL分别328/398、135/198、3/163、0/29。变化大不表示策略强；没做现金实战。
这些是同局多个相关决策的描述性回归指标，不将788决策视为788独立局。
辅助wait/huxi loss均0，辅助head未启用。CQL loss从约1.4203降至1.3643，
但留出回归没有证明CQL优于MC；训练稀疏现金的variance仍很大。

## 性能与局限

生成和写shard总21.35秒，46.83局/秒；纯模拟20.80秒，402.99探索决策/秒。
进程CPU平均约100.0%（一核），未持续耗满300%限制。计时：

| 项目 | 秒 |
| --- | ---: |
| Observation编码 | 7.47 |
| 网络batch推理及传输 | 3.32 |
| legal_actions（含内部调用） | 4.28 |
| step | 0.72 |
| 胡求解（与上层计时重叠） | 2.36 |
| 吃枚举（与上层计时重叠） | 0.97 |
| 生成期间完整终局重放 | 2.21 |
| 落盘后二次逐行审计，另计 | 9.95 |

本轮最大的采样单项是编码；独立审计也占明显时间。不能将inclusive计时相加。
GPU百分比在原实验中不可用：systemd PATH找不到nvidia-smi，错误原样保留。
随后修正为检测路径并回退`/usr/lib/wsl/lib/nvidia-smi`；仅使用已保存Q推理验证
采集可用（3%、3%、30%三次样本），没有生成新局或继续训练。
这三个数不代表原始1000局或训练阶段GPU利用率，不据此夸大硬件瓶颈判断。

完整测试333通过、1跳过，Ruff与diff检查通过。标准顺序曾触发磁盘保留线；
清理本次测试临时目录、无损重压旧归档并调整测试顺序后，全套通过。
2GiB保护线没有降低，旧实验没有删除；磁盘空间仍紧，扩大50k前需增加空间。

结论：独立Dueling Q流水线、真实终局标签和小CUDA训练可正常运行，
暂未证明强于PPO或baseline。主要差距是小数据过拟合、胡过／碰过探索不足、
chosen-Q与未尝试动作之间缺少可靠收益排序证据。按用户要求此处停止。
建议下一阶段先审查探索覆盖、按完整局衡量回报方差与编码profile，再经确认
做10k对照；50k与独立3000种子现金实战在后，online不提前实施。
