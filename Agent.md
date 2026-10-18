# ZiMortal 开发约定

## 工作范围与规则来源

- 在 `/home/wangke/project/ZiMortal` 中工作。
- 开始开发前阅读 `README.md`、`docs/rules/ningxiang-paohuzi.md` 和
  `docs/strategy/belief-and-live-tiles.md`；修改现有模块时同时阅读相关代码与测试。
- `docs/rules/ningxiang-paohuzi.md` 是规则 specification，优先于现有代码。
  用户在会话中的最新明确规则优先，并应同步到规则文档。
- 不自行猜测影响合法性、状态转移或结算的规则边界。提出具体问题，先完成
  不依赖答案的部分；未知边界应明确报错，不能静默拆坎或跳过动作。

## 架构与实现

- 正确性优先，保持模块化，避免不必要的大规模重写。
- 规则引擎与策略解耦：引擎负责合法动作、动作优先级、状态转移和结算；
  调用方从合法动作中选择，不能把策略决策写入引擎。
- 保留固定种子、独立随机源、状态克隆、确定性 step、动作历史重放与序列化。
- 严格区分完整状态与玩家 Observation。Observation 不能包含对手暗手、
  对手坎的牌型或牌堆顺序。
- 起手坎保留在手牌中，不作为公开 meld，不能拆。
- 全信息复盘网页仅供规则审查，不能用它的数据替代 AI 的 Observation。
- 当前优先完善规则引擎与复盘工具；未经用户要求不开展神经网络训练、
  自博弈训练或搜索算法开发。

## 已确认的重要边界

- 2026-10-11：全部成胡牌只有两张同一字号同牌红、其余全黑为红胡2番。
  十红也称红胡，超过十张每多一红加一番，显示十一红、十二红等实际名称。
  计分规则变化必须隔离训练批次，不混用旧终局收益标签。

- 吃牌并完成全部下比后，如果本次需要出牌却只剩不可拆的坎或空手，
  该吃牌方案非法；满足夹比免出牌条件时不受此限制。
- 强制偎／臭偎后不能胡且无牌可打时，免出牌，由下家摸牌。
  该玩家本局不能胡，不再吃碰；后续强制跑／提仍须执行。
- 自己主动打过的牌不能再吃；胡别人摸出的第三张，同牌三张组合按碰计胡息。
- 下比只强制处理来牌的同一字号牌型，大小搭中另一字号的同点牌可以留手。
- 其他发牌、天胡／地胡、下比、名堂和多跑／提结构，以规则文档为准。

## 验证与交付

- 改动规则或状态转移时补充有意义的回归测试；完整运行
  `.venv/bin/python -m pytest -q`。
- 运行 `.venv/bin/ruff check zimortal tests` 和 `git diff --check`。
- 改动对局流程时验证固定种子重放、四张守恒与合法执行到终局。
- 改动网页交互时验证受影响的交互与页面显示；无法完成检查时明确说明。
- 本地复盘网页启动命令为
  `.venv/bin/python -m zimortal.web.server --port 8765`。
  改动已运行服务的 Python 代码后，应重启本次开发启动的服务使改动生效。
- 用中文简明说明改动、验证结果和实际剩余问题。

## 目标差距与下一步（用户要求）

- 每次交付除了说明做了什么，还必须说明距离项目目标还有哪些具体差距，
  并给出下一步优先做什么；纯状态说明也应包含这两项。
- 第一阶段目标是规则正确、状态清晰、可测试、可高速模拟的宁乡跑胡子引擎。
  能运行对局、偶尔胡牌或测试通过，不代表已证明所有规则正确。
- 长期目标是面向宁乡跑胡子的高强度 AI：高速模拟、自博弈、隐藏信息推理、
  动作 EV 评估及复盘分析。明确区分规则引擎进度与 AI 策略能力。
- 用已验证的里程碑、尚缺的模块和已知规则边界描述差距；没有依据时不编造
  完成百分比、预计棋力或工期。
- 说明当前采用的策略时，以当前代码为准。随机从合法动作中选择仅用于
  模拟与规则审查，不是高强度 AI；胡与 PASS 也可能被随机选中。
- 示例交付结构：本次改动与验证；目标差距；下一步；适用时附 CSS 提交消息。

## 提交消息格式（用户要求）

- 每次完成代码修改，在最终回复中提供可直接复制的 Git 提交消息。
- 用户所说的“CSS 格式”在此指使用语言标记为 `css` 的 Markdown 代码块
  展示提交消息；消息本身是 Git 提交文本，不是 CSS 样式规则。
- 标题采用 `type(scope): 中文概述`，例如 `feat(engine): 实现宁乡跑胡子规则引擎`。
  按改动选择 `feat`、`fix`、`docs`、`test` 或 `chore`。
- 必要时用空行分隔标题与正文，正文简要描述改动和验证结果。
- 提供提交消息不代表执行 `git commit` 或推送；仅在用户要求时执行这些操作。

示例：

```css
fix(engine): 禁止吃后无法出牌的方案

过滤无法完成出牌的吃牌及下比组合，保留夹比免出牌例外。
补充回归测试并验证固定种子对局。
```

## 已授权的训练迭代

- 用户已授权从零训练、至少三轮训练／对局审查／修正，每轮保存本地记录；代码修改和重要总结实际 commit，不逐代提交机器指标。
- 固定数据及评估种子，记录模型、真实指标和限制；疑问记录到训练报告。
- 扰动次数不是精确向听数；监督启发式教师不代表最优策略。
- 训练仅使用 Observation，价值预训练标签与终局收益标签必须明确区分。
- 胜率与平均结算收益分开报告；第三轮更常胡牌不代表综合强于第二轮。
- 价值头切换训练目标时记录语义与初始化方式，不能把终局收益输出称为胜率。
- 训练记录与复现命令见 `docs/training/README.md`；新评估种子不要用于训练或标签生成。
- 十万级数据按不同的实际决策输入计数，不能把epoch重复、强制动作或对局步数冒称为新样本。
- 扩容数据在 `data/generated/scale100k/`，训练入口 `python -m zimortal.training.scale`；数据和权重保留本机，报告纳入Git。

## 胡息与收益建模优先级

- 结构成牌与达到15胡必须分开建模；不能把低胡息的结构听牌标作可胡。
- 显式考虑大小123／2710、碰偎跑提的胡息与拆组损失，潜力组合不可重复计分。
- 策略最终优化实际结算收益，涵盖胡息和名堂，不仅优化胡牌次数。
- 普通未来进张金额与天胡／地胡触发条件分开；启发式收益上界不能称为精确EV。
- 胡息预测头应单独评估有结构进张及15胡边界，模型预测不能替代规则引擎。
- 第五轮训练报告见 `docs/training/huxi.md`；保留旧模型复现原局，新模型轨迹不沿用旧步号。

## 专项辅助任务与泛化检查

- 合成中局题必须守恒并核对引擎实际来牌、强制动作、胡牌资格和结算，明确没有完整开局可达历史。
- 待摸牌只有强制动作的题用于辅助任务，不计作新增策略决策；报告不同状态数与重复采样分开。
- 不成结构的牌不能淹没有结构的胡息回归；胡息数值、15胡门槛、名堂金额分别检查。
- 除条件准确率之外，报告全20种进张的结构／合法进张precision与recall，检查误报和漏报。
- 真实收益价值头延续训练前核对金额单位，辅助训练须检查对策略收益的影响。
- 同种子各座位收益相关，比较收益差异时按种子聚合；小样本及弱对手不能证明高强度。
- 第六轮入口 `zimortal.training.boundary`，报告见 `docs/training/boundary.md`。

- 第六轮胡息预测改善但收益未证明提升，当前收益基线继续使用第五轮 `huxi`；不得把 `boundary` 描述为更强策略。

## 第七轮规划与策略审查

- 用户要求按精确completion distance、最终净收益EV、rollout teacher、belief的顺序推进。
- “板”（一对大字吃小字）通常低优先级；足胡且明确接近听牌或抢速时可能合理，这是策略倾向，不是规则禁令。
- solver必须明确来源和乐观口径，复用合法进张／出牌规则；搜索未完成不能产生精确标签。
- rollout只接受Observation，采样世界必须满足公开历史；不许读取真实隐藏状态给搜索老师。
- value金额单位和冻结后续策略明确，MC估计不等于最优EV；记录逐动作Q、不确定性和soft target。
- belief的真实隐藏牌只作为训练标签，检查物理不可能类别，对照未知牌均匀分配基线的NLL与accuracy。
- 详见 `docs/training/planning.md` 和 `docs/strategy/huxi-and-speed.md`。

## WSL训练资源保护

- 禁止逐行访问NPZ整数组；成员只读取一次，切片共享backing array。
- 训练优先独立systemd用户服务，默认MemoryMax=8G、MemorySwapMax=1G；不通过增大WSL内存掩盖泄漏。
- CPU环境`.venv`保留；CUDA使用独立`.venv-cuda`，训练启动须记录实际device、PyTorch及GPU。
- 每轮原子保存模型、优化器、随机状态及数据校验值；恢复前验证数据与父模型未变化。
- WSL故障导致进程消失时不得声称仍在训练，先检查服务、日志与checkpoint。

## 无人工牌谱的现金Q迭代

- 当前没有人工高质量牌谱；使用合法模拟轨迹、公开历史一致隐藏世界与终局现金收益生成标签。
- 区分独立搜索根、候选动作数和rollout次数；十万级旧数据不等于十万级搜索标签。
- 搜索标签注明冻结后续策略；同一批隐藏粒子比较候选，报告相对基线的配对误差。
- bootstrap网络成员分歧不等于校准置信区间；保守换动作阈值只用开发种子选择。
- 正式晋级必须经过独立种子、轮换座位、多种对手的现金收益比较；未通过保留原冠军和实验权重。
- 记录数据审计、CUDA训练、终局重放、实际收益及未解决的能力差距；见 `docs/training/cashq.md`。
- 现金数值拟合误差与动作排序分开检查；门控策略须报告真实换动作次数，零变化不算能力提升。
- 第八轮两份现金Q实验未通过晋级，收益冠军仍为第五轮；网络分歧状态优先增加配对推演精度。

## 主动状态挖掘与自适应精算（历史阶段）

- 停止继续扩大低精度随机数据。优先模型／父策略／搜索分歧，覆盖吃比、碰过碰、胡不胡、名堂临界与晚局。
- 共同粒子独立采样并通过全部公开历史重放；筛选粒子与认证粒子分离，保留原始配对收益证据。
- 自适应停止必须使用覆盖多次查看、多动作对和多根择优的误差预算。区间包含零不能判tie；未收敛根不凑数量。
- 至少5000个不同的认证训练根及独立牌局验证集，审计通过后才能训练下一代policy/Q；尚未达标如实报告。
- 规则未定义的未来分支记录并排除整个根，不把它当流局、零收益或重采样掉。
- 后台运行期间冻结源代码、采样模型与统计配置；修改后新建数据版本，不混合不同语义证据。
- 运行与认证口径见 `docs/training/active.md`。默认top_set认证最优集合，full_order另要求所有动作对排序；两者报告中明确区分。

- 2026-10-08确认：单钓仅有跑／提时允许6组＋将，无跑／提仍需7组；支持替代分解的多面听。
- seed44501的人类策略审查保留为独立案例，不能直接作为EV真值或加入训练根。
  门子、保留不重叠123／2710、优先小字及名堂组牌计划是策略倾向，不能改为合法性限制。

- 2026-10-08调度调整：使用active_schedule与active-v3，轻量候选持续补充，最多一个深度worker。
  兼容迁移仅在原标签生产源码、模型和统计配置均未变化时复制已有证据；保留误差预算编号及来源链。
  调度预算单独记录，不把延期根当合格根，不用已观测最大金额代替安全收益上界。

- 2026-10-09：active_schedule_v2与active-v4使用最多三个深度worker、两个新状态worker。
  用实际收益差与区间宽度估算续算成本；此估算仅决定调度，不代替认证。原始证据与逐根alpha继承保留。

- 2026-10-09性能迭代：active_schedule_v3/active-v5使用rollout私有状态访问缓存，
  胡牌求解与吃比方案纯函数各限8192项；公共可变状态引擎不做身份缓存。
  必须保留逐步历史合法性检查、采样序列和逐粒子收益一致性；三根诊断不代表整体产出。
  缓存模块额外源码哈希纳入scheduler及manifest，原alpha和认证公式不变。

- 2026-10-10：active_schedule_v4/active-v7对未认证根采用公开物理库存与大小字
  容量收益界，原alpha、tie和两档规则不变。已认证根按固定legacy名单保留原证据
  及旧界审计；更窄区间可能触发tie边界变化，不能假定分类／稳定性单调。
  新证据重算必须守恒原始outcomes及预算编号，不能复活规则或采样失败根。
  完整训练审计和worker必须使用相同按根分派规则；部分审计不代表5k门槛达标。

- 2026-10-10数值修复优先于前条：发现HU int16支持端点使Bernstein范围乘法溢出。
  adaptive必须在算术前提升为Python float；active-v8默认旧认证豁免名单为空，
  所有根安全重审，旧qualified失败须撤销，不保留未经修复验证的合格数量。
  新config显式升级adaptive源码与numeric_evidence_version，重签派生配置哈希，
  原收益粒子、alpha和预算编号不变。active-v6/v7不用于训练，旧计数仅属历史记录。

## 在线现金强化学习（2026-10-10最新要求）

- 用户要求暂停精算产出，直接迭代强化学习。5000认证根门槛只约束上述搜索监督训练，
  不阻塞当前on-policy PPO；不得把真实自博弈收益冒称为认证搜索Q标签。
- 使用当前现金收益基线起步，混合自博弈、冻结父策略和历史模型；持续更新训练模型，
  单独保留开发集选出的候选，不因一代开发收益波动重置累计训练。
- 奖励为终局payments[seat]/100，不裁剪、不改成胜负分；仅Observation进入网络。
- CUDA批量推理和PPO更新，CPU执行规则。完整对局重放验证，未知规则排除整局并记录。
- 每代保存权重、优化器、随机状态、源码与配置哈希和报告，报告逐代保留本地，不自动commit。
- 训练、开发、独立留出种子分离；开发选定候选后一次独立评估，不用留出集挑轮次。
- 旧冠军保留，开发集更好不等于正式晋级。入口和复现见docs/training/online-rl.md。
- 扩容批使用独立于座位／庄家的比赛类型分配、多个冻结历史模型与有限近期快照池；
  开发和留出对手固定，不能用不断变化的对手池比较代际强弱。
- 吃／比、碰／过碰、胡／不胡报告包含合法方案、概率、父策略选择及实际终局收益。
  动作选择后的分组收益为描述性结果，不是因果Q；放弃胡牌比较当场精确净结算。

## 连续强化学习联赛（最新授权）

- 用户要求扩大局数、迭代换更强模型持续打；入口zimortal.training.league。
- 从上一批最佳实验权重和对应优化器继续，不把旧最佳实验模型宣称为正式冠军。
- 阶段内累计训练；阶段后用新种子检查候选。通过实验收益门槛才更换训练基准，
  否则下一阶段延续当前训练权重与优化器，保留原基准；已通过模型加入后续对手池。
- 所有阶段训练、开发、晋级检查种子互不重叠；最终独立评估不参与换模。
- 按种子同时聚合三个座位和多种对手，不能把相关对局当独立证据。
- bootstrap门槛为近似实验检查，不冒称严格全局认证；记录各对手退步与局限。
- 每代及每阶段保留本地记录，保存来源与换模链；续训优化器必须与所载权重精确对应。
- 持续训练期间冻结源码，后台状态同时查看league-progress.json与当前stage的progress.json。

## 吃碰拆组人工审核

- 用户授权从league-v1第三阶段最佳继续训练，并将模型拿不准的吃碰决策交由人工判断。
- 静态123／2710潜力必须避免重叠计分并排除不可拆坎，吃与所有下比及碰的新增胡息计入。
- “板”须包含来小字、手中同点一对大字；来大字形成相同大小搭不能误标成板。
- policy概率差小只作为筛题条件，不声称为动作EV置信区间；静态拆组不等于错误动作。
- 人工审核保存原动作前缀并检查真实重放，保留case ID；混合模型轨迹不能用全员同策略重生成。
- 人类意见与收益标签分开保存，不改合法性或当作精确EV。审核题来自独立诊断种子，
  不进入正式留出测试。审核不阻塞已授权的强化学习训练。
- 首批反馈见 `docs/strategy/reviews/human-review-v1.md` 与同名JSON：
  接近合法听牌时允许合理拆组抢速；保留灵活门子，结合红胡／十八大计划审查早期弃牌。
  早打红色大字的风险是策略倾向，不是一律禁打；“区别不大”不得强制转换为确定动作标签。
  审查吃碰须连同后续弃牌和前序组牌计划，不只统计静态123／2710损失。

## 训练记录与 Git（2026-10-11最新要求）

- 不再逐代或逐JSON自动提交训练记录；不要启用 `--commit-reports`。
- 逐代JSON、详细日志、检查点留本地；代码／规则修改、人工审核意见及重要实验总结再提交。
- 本要求优先于旧报告和旧启动命令中的逐代commit约定；启动脚本过滤旧的自动提交参数。
- 已推送历史若需压缩，先保留本地bundle备份、核对只含训练记录的提交；改写远端前获得明确确认。

- 平台期探索可调整采样温度，但收集旧logprob与PPO新logprob必须使用同一温度，
  冻结对手及greedy评估口径保持固定；采样概率不是动作收益或置信区间。
  每轮同时记录actor、critic、entropy及anchor KL，不能仅凭标量损失推断梯度主导。

## 两步 Branch Teacher（2026-10-12最新要求）

- 本阶段只做两步分支生成、人工案例诊断及性能验证，不启动league-v5、大网络或蒸馏。
- teacher只接收Observation，复用HistorySampler完整公开历史重放；所有分支共享粒子与冻结后续策略。
- 第一动作后自动执行单一合法动作，等待原玩家下一次多选决策；终局结算使用实际payments现金。
- PASS后的不同公开信息集分别统计，不能按隐藏世界选最优动作后混成max-Q标签。
- 提案预算不足、规则未定义使整个root失败，失败证据保留；不填零、不用真实隐藏世界替代。
- 报告配对差、SE、CI与未收敛状态；近似正态诊断不冒充严格序贯认证，不因CI包含零判tie。
- 原始证据只留data/generated/branch-teacher-v1，Git仅提交实现、重要总结及人工原始prefix。
- 第一阶段完成后先报告结果与瓶颈，再决定是否生成1k～5k高价值根；不得自动进入第二阶段训练。

## Branch Teacher 性能与 PASS 信息集（2026-10-12）

- 最新授权先优化采样、可见编码与PASS处理；本轮不启动1k～5k根生产、蒸馏或新PPO。
- 采样转移CDF缓存和历史弃牌PASS必要条件预过滤不得消耗额外随机数；保持提案序列、
  接受粒子及尝试次数。预过滤忽略坎保护／下比／优先级等更强限制，完整历史重放仍为唯一接受依据。
- 编码缓存仅保存可见输入的纯特征，有界且只读；按feature version隔离，不缓存随权重变化的logits。
- PASS新增固定可见后续策略的配对现金估计，明确区分Q(a1, frozen a2 policy)与max-a2；
  不用当前粒子的现金挑a2。同一公开信息集必须选择同一动作，低支持信息集不变成最优动作标签。
- max-Q与固定策略诊断分享误差预算；近似正态区间仍非严格认证，不据参考策略排序生成训练标签。
- v2证据将Observation按信息集索引保存，保留每粒子世界哈希、全部终局现金及可重建可见状态；
  原始文件／逐次指标留data/generated/branch-teacher-v2，Git仅提交代码与重要总结。
- producer哈希包含belief采样源码；源码／模型冻结，不混用v1、缓存消融和最终优化目录。

## 自动迭代授权（2026-10-12最新用户要求）

- 用户明确要求每轮完成后自行决定并执行下一步，自动优化和训练；此要求覆盖前两节
  “本轮只诊断、不得进入下一阶段”的阶段性限制。规则不明或无法合理判断的方向才交用户审核。
- 入口zimortal.training.autonomous：先独立搜索／评估条件现金Q试验，再继续三阶段真实现金PPO。
- 条件Q只用新的完整重放牌局及Observation；搜索与检验牌局／粒子、PPO训练／开发／筛选／最终留出种子分离。
- 低支持a2状态可用于明确标记的噪声MC辅助回归，不能冒称合格根、max-Q标签或直接蒸馏policy；
  singleton不赋零误差，同一根总权重相同。人工审查题和原始人类倾向不进入该训练集。
- 条件Q通过新牌局配对现金试验才用于下一轮挖掘行为；未通过仍保留父行为，继续扩展关键状态覆盖。
  此试验不更换正式冠军、不作为PPO权重初始化；PPO从保留的最佳实验policy与匹配优化器继续。
- 每轮Q权重／优化器匹配校验，父编码器冻结；证据gzip原子写入、校验哈希、保持2GiB磁盘保留线。
  规则／采样失败整根排除并留失败证据；日志和每代指标只留本地，不逐JSON commit。
- 后台源码冻结，CPUQuota300%、MemoryMax8G、SwapMax1G；Q更新与PPO使用CUDA，CPU处理规则和粒子。
- 自动联赛用独立实验门槛决定换模型和对手池，正式冠军仍需最终审查。进度同时查看autonomous progress与league/stage progress。

## 显式组牌规划（2026-10-13最新要求）

- 用户要求重新规划门子、名堂路线、进张方式与活张变化后的策略调整；执行路线见
  `docs/strategy/plan-aware-ai.md`。下一阶段先完成有预算的可见规划诊断，再做辅助训练和现金RL。
- 当前PPO主模型没有在线推演／计划集合，不因收益提高而声称已具备可验证的规划能力。
- 候选门子可以重叠，每个完整计划不得重复占牌；已公开组、不可拆坎与吃比弃牌限制固定。
- 碰碰胡包括坎／偎／跑／提，不要求实际碰六对；某保留方案失败不等于全部替代方案不可达。
- 活张硬界、belief估计、乐观completion和真实时限胡牌概率分开；预算耗尽必须为unknown。
- 新模型与上一代及固定基线分别评估；训练主模型可更新，正式冠军认证不阻塞实验迭代。

- 2026-10-13用户授权改结构并用成功胡牌轨迹监督规划；实现入口`training.trajectory`，
  `auxiliary_version=trajectory`，见`docs/training/trajectory.md`。成功路线不是最优动作标签。
  赢家组牌／名堂监督与全体现金／是否胡牌监督分开；未来信息仅作标签，不进入输入。
  本版是局部门子集合与结果预测，尚无互斥完整计划求解或在线推演；不能夸大能力。

- trajectory-v1现金评估退步，后续使用trajectory_v2隔离预训练：原主干／policy／value
  冻结，策略接入系数保持零，辅助模块继续学习并逐项验证父权重。现金零差是保护验证，
  不能冒称牌力改善。各预测头单独评估，之后保留辅助回放再做现金RL接入。

- trajectory_v3明确条件终局胡息语义，监督输出不低于15与已锁定胡息；不替代引擎
  当前胡牌判定。加入训练常量基线和开发选型回退，最终使用selected-model.pt；
  组牌计数非负、报告非空组误差与名堂独立牌局覆盖。审查见trajectory-review-v2.md。

- trajectory_v4加入非空组平衡监督与完整结构解码；解码只用Observation，固定公开组和坎，
  约束七组、四张上限、跑提与将。固定坎升级分支未展开，空结果为unknown；结构方案
  不冒充>=15胡息、合法吃比路径或时限可达证明。父策略冻结不冒称牌力提高。
- 最新授权先修完整组牌约束与非空组预测，再分128局块扩大5,000训练／1,000开发局；
  原始重放gzip保留、块哈希与manifest校验，指标留本地，不逐块commit。

## 同状态完整路线比较（2026-10-13）

- 用户授权保留ResNet，加入门子／完整方案编码及动作条件Q；实现 `model.routeq`、
  `training.routeq`，细节见 `docs/training/route-comparison.md`。
- 复用5,000局trajectory_v4门子权重，父ResNet／policy／value冻结；新头学习真实现金
  和共同粒子下同动作不同路线的收益差，未采用的门子不能直接当错误标签。
- 路线提案加入合法吃及全部下比、碰的具体组；完整方案互斥用牌。
  当前标签为明确、冻结的可见路线跟随策略收益，不冒称最优路线EV。
- 原玩家路线失效回退父策略，后续胡／过不受进展偏好覆盖；对手父策略冻结。
  PASS后只看各自Observation，终局现金不可挑选后续动作。
- 训练／开发／选型后独立检验根及粒子分离；错误整根排除，完整物理重放和失败证据保留。
  普通模型forward保持父策略，路线预测可靠性与完整代理现金门槛通过后才讨论策略接入。

- 路线Q人工审核入口 `training.route_review`：用独立种子生成父策略完整重放，
  网页同时标出父动作、试验Q动作及互斥完整路线。父策略概率不能标成Q策略概率。
  人工判断用于路线／动作偏好诊断，不冒充实际现金标签；审核根不混入既有评测集。

## 路线人工反馈（2026-10-14）

- 三题倾向：split-102过吃九，split-104优先打三并保留对子／坎，split-105过吃三。
  详情与实施顺序见 `docs/strategy/reviews/routeq-human-review-20261014.md`。
- 下一轮先补固定三张组升级路径，通常优先无跑／一次跑及已有跑的继续完成，
  再比较吃碰全部下比与随后弃牌、灵活进张损失和15胡／名堂路径，之后新根现金训练。
- 多跑提是合法规则，不硬禁第二次升级；按来源计算增量，不能把任意跑统一加6胡。
  “任意跑一个”按备选升级集合表达，不能重复计胡息或忽略活张。
- 第二题保存手牌有六陆；此前“无小六”是整理时漏读编号5，已纠正。
  三约45%、六约40%为近似偏好，不擅自补足为完整soft标签。
  三题人工意见为诊断证据，不伪造现金标签或污染独立检验集。

- 2026-10-14用户授权：以后完成提交后自动push到已有远端；失败如实报告，不强推。
- 跑路线首要区分是否需跑才能胡；具体跑牌作为备选升级事件，近似等价来源去重，
  仅在胡息门槛、活张、进张合法性或现金差异重要时展开区别。

- routeq-v2开始按用户顺序补升级与吃碰后弃牌比较：固定三张组可升级，
  保留无升级／首次／已有升级和胡息不足例外；默认最多新增一个四张组。
  升级仍是条件结构路径，尚无跑胡到达概率或精确替代进张价值。
  新标签目录与种子独立，正式部署保持父策略；完整对局实验按种子聚合并保留重放。

## Mortal风格主线（2026-10-14最新要求）

- 本轮暂停trajectory／RouteQ／显式路线扩展，保留所有旧代码和实验；主线见
  `docs/training/mortal-style.md`。不使用人工路线、未来组牌或搜索route作为输入／标签。
- 独立MortalQNet：55×20可见编码、ResNet Brain、结构化合法动作Dueling Q，
  均值与CQL仅合法动作，终局payments[seat]/100，默认gamma1，Huber＋正系数CQL。
- 冻结league-v2/stage-02/best-dev.pt为ZiMortal-v0；一探索learner对两greedy baseline，
  座位／庄家轮换，完整局shard，未知规则整局排除，全部重放及可见数据核对。
- 用户明确要求A/B/C：模型和测试、1000局生成、MC与CQL小训练完成后停下来汇报。
  此停止要求覆盖此前自动无限迭代授权；不要自动做10k／50k或online／替换baseline。
- Git仅提交代码与重要总结，数据／逐epoch日志／权重保留本机；按既有授权自动push。

## Mortal 10k离线扩容（2026-10-14最新授权）

- 本轮仅按动作族探索pilot与独立10,000局MC／CQL对照，保持55×20、32宽3块256 latent；
  开发RMSE早停、开发选择门控阈值，保留全量审计。完成后停止，不自动50k、online或晋级。
- 数据、原始profile和逐epoch日志仅留mortal-v2本地；重要代码与总结提交并自动push。

## Mortal credit assignment对照（2026-10-14最新授权）

- 复用mortal-v2原10k，同结构／初始权重／optimizer／batch序列／开发早停，仅比较MC、3-step、5-step行为SARSA。
- bootstrap只用同局同玩家`t+n`真实动作；独立冻结target network，按optimizer update可配置hard sync。
- 开发及检验均对真实终局现金评估，不用TD标签评价收益预测；不得max-Q bootstrap或跨游戏／split链接。
- 离线明显改善才做独立500 seed×3座位现金screening，原始完整重放留本地；不正式晋级。
- 完成后停止，不扩50k、online、RouteQ／trajectory或替换baseline。

## 高支持排序反事实诊断（2026-10-14最新授权）

- 本轮只复用现有10k、3-step SARSA+CQL0.1+support0.1权重做诊断，不重新训练、调参、改架构、扩50k或进入online；完成后停止。
- 复制真实完整根、固定三家暗手和历史，只对尚未摸出的牌堆做共同随机置换，允许用于本轮条件诊断；所有网络始终只接收Observation。
- 该条件baseline-continuation现金不是Q*、不是对未知暗手积分后的Observation EV，不能推广为旧搜索老师读取真实暗手的授权。
- 抽样按动作族/Q差分层，不按原终局收益筛选；训练频率只取train，动作全局频率和精确输入覆盖分开。
- 配对SE/CI、零均值与未区分根分开；一次独立粒子复核不冒称严格排序认证。本轮标签全部not_for_training。
- 下一步优先研究选择性pairwise ranking，处理继续策略及隐藏世界口径；未经新指令不启动后续训练、探索或替换baseline。

## Mortal-faithful-v1（2026-10-14最新主线）

- 停止扩展pairwise/counterfactual训练；保留历史代码与证据，不混入新主线。
- 默认terminal MC现金/100 + offline CQL，以½MSE回归；3-step实际行为SARSA仅作对照。
- 保留55×20可见Observation、structured legal action、合法均值Dueling与真实现金结算。
- Mortal-like主干为更宽更深1D ResNet、Channel Attention、约1024 latent；具体宽深和官方差异见mortal-style.md。
- support仅为offline评估/部署护栏，不进入损失、采样或主干合法均值；同时报告raw结果。
- 新主线不加入pairwise、RouteQ、trajectory等辅助loss。固定10k与2×2对照，先锁定权重再看test。
- 与当前旧CQL+support在独立同种子现金筛选比较；达到证据门槛才设计受控online，不能自动启动。
- 本轮训练和评估完成后停止，冻结baseline，详细日志/权重只留本地，重要代码/总结commit并按既有授权push。

## Mortal-faithful离线诊断（2026-10-15）

- 本轮只做LR3e-4/1e-4/3e-5与patience4/8诊断，保持网络、MC、CQL=.1与support=.1；不新增loss或对局。
- 逐epoch train/dev/test指标允许用于诊断，checkpoint仍只按dev选择；已重复观察的test不能当下一轮盲测。
- 区分HU与PASS条件统计、全局动作模板覆盖与给定状态行为支持；相关性/实现收益不能冒充反事实EV。
- 原500种子raw/support只读取已有日志做同状态比较，不新增counterfactual训练。
- 六组完成后停止；20k过渡及50k仅为后续建议，未经新指令不采集、训练、online或晋级。

## 20k过渡实验完成（2026-10-15）

- 固定faithful/MC/CQL=.1/support=.1/探索，新增独立10k；A旧8k与B合并16k各5200更新、每260检查，按新dev共同选择同一update520，锁SHA后看新test。
- 新test预测与train-dev gap改善，但raw现金B−A=−0.99，95%[−4.11,1.98]；护栏+1.60，95%[−0.76,3.94]；低支持推翻比例77.12%→77.37%，未改善。
- 本轮停止，不继续同分布50k、online、改baseline或增loss。下一步建议先设计behavior coverage/受控探索验证，未经新指令不执行。
- 完整协议、结果与复核见docs/training/mortal-style.md；日志、JSON、权重本地保留，重要代码/总结commit并自动push。

## Behavior coverage对照完成（2026-10-15）

- 仅改变discard/chi行为探索，各4k独立训练局、共享新1k dev/1k test；同模型/MC/CQL=.1/support=.1，均5200更新，共同选中260；冻结baseline。
- p_B=.5p_A+.5×原p≥.03的前三非greedy候选均匀分布，无合格替代不变；peng、HU/PASS不变。全部概率及终局现金本地可审计。
- 真实替代执行增加，但新test原概率低支持推翻76.37%→75.86%不稳定，chi31.65%→44.49%；raw B−A+0.63元，95%[−2.18,3.53]，护栏−1.11元[−3.91,1.62]，回归点估计略差。
- 本轮结束停止加探索，不自动50k、online、改baseline或增loss。下一步仅建议固定权重的policy extraction诊断，未经新指令不执行。
- 主支持口径/现金护栏对两组均为原探索p>.1，避免用改变概率定义机械宣称覆盖改善。完整细节见docs/training/mortal-style.md。

## Fixed-Q policy extraction诊断（2026-10-15最新授权）

- 本轮禁止训练：固定B-coverage update260的selected-common.pt、冻结baseline、原behavior概率与规则引擎；不改变网络/Q值/训练数据/loss，不进入online。
- 全新dev1130000000起200seed×3座位选择C family规则、D λ∈{.01,.03,.1,.3}、E弃牌/吃牌k∈{2,3}；锁定后全新test1140000000起500seed×3座位，一次评估不调参。
- family约束仅弃牌/吃牌，peng raw、HU/PASS父策略；另有HU父策略控制组，保证仅弃牌/仅吃牌/两者消融可解释。当前support全族>.1、D平滑anchor保留全合法动作。
- 记录分族改动作、低支持比例、Q margin、父候选保留及实际选择率，现金按种子簇配对区间；不能用退回父动作的收益恢复证明Q已强于baseline。
- 完成报告、pytest/ruff/diff后停止；下一步仅根据结果建议受控actor设计或重审Q ranking，不自动启动。

- 本轮已完成：dev锁C-both、λ=.3、top(3,2)；test raw−19.29、support/family+0.12、anchor+0.45、top−12.95元/局。anchor相对raw+19.74[16.56,22.91]，对baseline[−1.13,1.94]，未证明超越。
- 弃牌约束单独+14.76元；已有弃牌约束再约束吃牌+4.64元。anchor98.30%仍选父动作，不能把恢复父行为当Q已更强。
- 实验优化更新0，权重及BN buffers、规则与模型SHA不变；22500评估局完整审计。停止；下一步仅建议受控行为锚定actor设计，不自动online或重训。

## Anchored actor偏离价值验证（2026-10-15最新授权）

- 本轮不训练：固定上一轮同一checkpoint/baseline/原behavior及Q+.3log(p+1e-8)，不改网络/Q/规则/support/loss。
- 预声明1150000000起3000seed×3座位；baseline、all、first(1次)、max2(2次)各9000局。不搜索budget、λ或分桶，不用test选规则，结束停止。
- 只统计实际非强制决策对同局面父greedy的偏离，按(seed,seat)计数，禁用并行slot作局标识。额度耗尽回父策略，不改变引擎。
- 分桶与bootstrap协议见mortal-style.md；单次/多次偏离子集为描述统计，不冒称动作反事实EV。全部95%CI跨零不宣称改进，不自动进入online。

- 本轮已完成36000局/1945930步审计：all相对baseline+.113元95%[−.45,.71]，first+.103[−.46,.70]，max2+.105[−.46,.70]；全部区间跨零，不宣称稳定policy-improvement、不进入online。
- all偏离1174次（discard711/chi207/peng255/hu1），1106局受影响；一次1042局、多次64局。预算对比也未确认“越偏离越差”，不得从test挑新预算。
- 下一步仅建议审查本地全部偏离CSV及配对轨迹，关注拆组/对子/123/15胡；未启动此下一步、不改loss或训练。重要总结/代码commit并按既有授权push，原始记录仅本地。

## Anchor-all 30k独立统计检验（2026-10-15最新授权）

- 本轮只扩大独立评估：1160000000起30,000seed×3座位，baseline与anchor-all共180,000局；不训练、不人工挑case、不改λ/网络/Q/规则/Observation/action，不再运行first/max2或其他网格。
- 固定同一checkpoint、冻结baseline、原behavior及Q+.3log(p+1e-8)，主指标按完整seed簇配对bootstrap；分族现金仅描述，不按结果改过滤器。
- 完整样本现金差+.0225元/局，95%[−.153444,+.196411]，未证明稳定正收益；偏离12,127次/752,303决策（1.611983%），11,382局受影响。
- 区间已接近零，不继续为同一offline actor扩检验；下一步仅建议正式online闭环设计，未经新指令不启动online或替换baseline。本轮审计、文档与pytest/ruff/diff完成后停止，原始记录仅本地，重要实现/总结commit并自动push。

## 首轮受控Mortal-style online完成（2026-10-15最新授权）

- 用户授权的小闭环已执行3代×3000真实新局；每代当代anchored trainee对两家固定baseline，生成→MC-only更新→新actor，评估独立500seed×3座位。官方commit及差异详见mortal-style.md。
- 按官方online逻辑关闭CQL，唯一½MSE终局现金/100 loss；未新增pairwise/RouteQ/trajectory/辅助头，不用raw收集训练数据。AdamW跨代延续，累计296更新，9,000训练局/75,297决策。
- 三代anchor−baseline为−1.08[−2.51,.18]、−.62[−1.63,.30]、−.73[−1.86,.28]元/局；都未通过95%或预声明多代校正门槛。raw仍显著负，偏离率1.47%→1.18%→1.24%，不得宣称online已提升棋力。
- 连续3代无可靠改善已停止；本地active-checkpoint回指原B-coverage offline文件，baseline从未替换。27,000局完整审计，392测试通过/1跳过、ruff/diff通过。
- 本轮停止覆盖此前自动连续训练授权；仅建议下一步小预算研究anchored score的官方式Boltzmann混合访问，不自动执行、不扩量或加loss。原始日志/数据/权重本地保留，重要实现与总结commit并自动push。

## 第二轮anchored Boltzmann A/B完成（2026-10-15最新授权）

- 固定offline共同起点、baseline、λ=.3、T=.5，各组3代×3000新训练局；A greedy、B在anchored score上采样，HU/PASS均保持原anchor greedy。MC-only/CQL0/optimizer/网络/规则不变。
- 每代500新seed×3座位严格配对；主评估A/B训练后的greedy anchor，另报B采样执行与A/B raw，不能择优偷换评估口径。无人工筛case、新loss或offline扩量。
- B真实训练偏离约25%，A约1%～2%，配对访问率差95%均正；但三代B-greedy−A为+.1033[0,.2333]、−.2533[−.8267,.1033]、+.0067[−.3167,.34]，未证明探索优于greedy。
- 第1代B对baseline+2.4533[.7899,4.6567]，多代校正下界也正，按预设口径保留方向候选；A同期也正，不能只归功Boltzmann。第2代未延续，第3代A/B均负，直接采样三代均显著负。
- 45,000局/2,501,884步、149,641训练决策全部核验；A/B各294更新，初始化/优化器连续性/源文件与冻结模型SHA复核通过。396测试通过、1跳过，ruff/diff通过。
- 本轮已停止，baseline未替换；不扩generation或温度搜索。仅建议下一步用现有真实轨迹诊断MC噪声、Q尺度与优势头变化，不自动执行后续实验。代码/重要总结commit并自动push，原始数据/日志/权重仅本地。

## 长期尺度Mortal-style online完成（2026-10-16）

- 单线Boltzmann T=.5/λ=.3，从原offline起点完成10×5000真实新训练局，MC-only/CQL0，一次消费，无replay；optimizer/RNG跨代连续，1,627更新。无中途调参。
- 每代独立1000seed×3座位，最终固定Gen10并封存5000seed×3；共230,000局完整审计。不能用每代现金挑冠军，不自动替换baseline。
- 最终Gen10 anchor−baseline+.2743元95%[−.1413,.7307]，对初始offline−.1923[−.4880,.1200]；raw−19.459。后期未稳定为正，未证明policy improvement，也未证明显著差于起点或Mortal范式失效。
- 本轮已停止，不自动扩大或启动新实验。下一步仅建议先分析MC采样继续策略与greedy部署口径差异及价值/advantage漂移，再预声明单因素验证；未经新指令不执行。
- 完整协议、曲线、结果见docs/training/mortal-style.md；日志/权重本地，重要实现/总结commit并按既有授权push。

## MC continuation单因素对照完成（2026-10-16）

- 重新下载并逐Git blob核验官方固定commit的13个文件；官方online为ε-greedy/Boltzmann/top-p混合，实际动作MC监督用逐kyoku GRP期望点差，并非整场现金；CQL关闭但辅助排名loss可用，BN冻结由配置控制。不可再简写为“官方只有整场terminal cash MC”。
- 固定原offline B-coverage共同起点、baseline、T=.5/λ=.3和MC-only/CQL0。A全程Boltzmann，B首个非HU多选动作后切本代冻结greedy；两组均只训练首根以匹配行数、shuffle和更新预算，各3×5000真实局、14,965根、60更新。全部后续行仅审计。
- Gen1同输入/实际根动作/完整前缀一致，B−A真实标签+7.292元95%[5.242,9.372]；但最终固定Gen3 holdout B−A−.00722[−.07889,.05667]，B−baseline+.050[−.505,.60946]，B−initial−.03611[−.22444,.14778]。未见修正带来稳定actor改善，不把标签差异直接当主瓶颈证据。
- 12万完整对局审计、采样/切换/一次消费/Adam/RNG/冻结SHA复核通过。baseline未换，本轮停止；原始证据本地，重要代码/总结commit并自动push。
- 下一步仅建议单因素LR时间尺度／更新稳定性验证，不同时改BN、探索或target；未经新指令不执行。详细源码参数流、协议、限制和结果见docs/training/mortal-style.md。


## Online LR时间尺度单因素对照完成（2026-10-16）

- 从同一原offline B-coverage起点，各10×5000真实新局；A保持warm20/cos300，B采集前按历史415284行预估1630更新，锁cos1609。全部真实非强制决策单遍训练，MC-only/CQL0，T=.5/λ=.3/BN/optimizer/网络均不改，无continuation变体。
- A/B分别414760/414651行、均1625更新；A的post-warmup floor1305次（80.31%），B未提前到floor且未事后重算。Gen3–9 B/A centered advantage漂移比中位数2.83，旧LR确实限制更新幅度。
- 固定Gen10、独立5000seed×3最终holdout：B−A+.12267[−.084,.33867]，B−initial+.11633[−.19334,.42202]，B−baseline−.01867[−.37067,.32536]元/局。未满足预声明提升门槛；不能用中间代或argmax运动量宣称更强。
- 34万完整对局轨迹审计、独立实际采样/Adam/RNG/LR/参数/BN/共同probe Q漂移复算通过。baseline未换、服务正常结束，本轮停止，不自动扩大或启动下一阶段。
- 仍缺可靠超越baseline的棋力证据；下一步仅建议研究MC信用分配与更新稳定性对相对动作排序的影响，未经新指令不执行。完整曲线/指标见docs/training/mortal-style.md。
- 为磁盘余量，五个已完成历史实验无损归档到Windows盘，原路径保留目录链接；逐文件SHA清单见本地data/generated/storage-archive-20261016.json。历史证据和权重不得按“链接/旧日志”直接删除，访问原明确路径可读取完整内容。

## MC信用分配／advantage可辨识性只读诊断完成（2026-10-17）

- 本轮仅分析long-v1与LR-horizon-v1 A/B各50k既有完整轨迹，共1,244,695真实历史决策；LR A/B同seed分别统计，不冒称150k独立牌局。不新增训练/对局/rollout/loss，不改网络、探索、target、LR、BN、baseline或active指针。
- 每个有决策局约8.31–8.32个不同Observation共享1个terminal label；exact/encoded/fine均无重访；broad＋相同合法集合＋同代producer的跨局多动作覆盖仅0.0306%–0.0395%行。label复用和exact稀疏本身不证明MC理论无效，神经网络仍可泛化。
- 固定5000状态/流：残差std约66.57–68.78元，raw margin中位1.36–1.75元；逐状态abs residual/margin中位long/A/B=7.94/8.07/10.77，整体>10条件并未三流一致通过。discard/chi均约14.6–20.9倍，peng/hu不同，不可一概而论。低chosen-return R²不证明绝对价值已学好，也不能单凭它认定条件期望拟合差；同state-action不可约噪声尚不可识别。
- legal-mean Dueling推导与4096行只读autograd一致：chosen e(1−1/K)，未选−e/K；有chosen-vs-rest信号，无未选动作独立outcome。adv scorer梯度非零，末scalar bias均值抵消是正常gauge；不把gradient norm当信息量或棋力证据。参数/BN/.grad不变。
- 支持把action-relative信用分配作为下一步重点，尚未因果证明是主要瓶颈，也不把本轮诊断当反事实EV真值。下一步仅建议预声明单因素可辨识性验证及独立现金标准；本轮完成后停止，未经新指令不新增loss或开训练。
- 两个诊断入口与完整分族表/公式/数值重现修正见docs/training/mortal-style.md。本机保留输入/aggregate SHA与固定样本，重要代码/总结一次commit并按既有授权自动push，不逐JSON提交。


## Counterfactual Q-ranking benchmark完成（2026-10-17）

- 用户授权诊断用500根、全部2437合法动作×32，共77,984分支；从旧LR-B uniform5000池按200/125/125/50无放回分层抽样，先锁locator/manifest/SHA和预算，无人工挑case、无补抽/自适应追加。四个固定checkpoint都比较raw/anchor及baseline，不选新冠军。
- 现有HistorySampler的合法重放不等于已认证posterior权重，本轮明确fixed-hidden-hands＋uniform shuffled余牌墙，不称information-set EV。共同continuation固定initial offline greedy anchor，两个冻结baseline，四critic对照同一Qπ0参照而非各自continuation；不是Q*。
- 500根有效、0失败，179clear/321ambiguous、43经验并列。B10 raw macro pairwise61.94%、pooled54.08%，baseline65.96%/57.60%、B anchor66.01%/57.71%；raw高于uniform参考，不可说整体随机。
- discard raw全程约52%，B clear discard pairwise46.33%仍差；chi initial54.76%→B63.58%有局部改善。B−initial整体pairwise CI跨0；B−A约+.94pp仅小幅未校正信号，未带来更佳条件EV。不能把所有动作都说没学习，也未证明principal bottleneck唯一是credit assignment。
- B anchor仅5/500根偏离baseline，prior仍主要保护；不把更接近baseline或根上条件EV当完整对局现金晋级。fixed-hidden排序与玩家信息集排序可能不同，本轮未测该差异；不把本批标签输入训练。
- 全部16k shuffle world可达性、77,984分支普通引擎/冻结actor重执行逐步一致，checkpoint/BN及源文件SHA不变、训练0。完整公式/指标/分族表/图/复现见docs/training/mortal-style.md；原始root/seed/每分支suffix与现金留本机。
- 本轮已停止，baseline未换。目标仍缺可靠弃牌EV、权重正确的hidden posterior和独立现金提升；下一步仅建议认证information-set测度并用明确continuation的独立根复核，不自动扩样本、加loss或训练。重要代码/总结一次commit并按既有授权push，不逐JSON提交。

## Information-set隐藏世界采样器认证完成（2026-10-17）

- 本轮只做公开信息条件化采样与完整历史重放；未训练、未新增未来Q分支、未改模型/规则/baseline，不自动进入下一轮benchmark。
- 固定随机从原500根抽20/10/10/10共50根，先锁manifest/config/source/checkpoint SHA，再各采1000世界；1,952,474 proposal取得50,000接受世界，普通RuleEngine独立全prefix重放50,000/50,000通过。
- 真值仅审计：50/50在支持集；50/50不同暗手同public配对输出相同，copy-ID置换通过。两份小空间fixture穷举合法12/4世界全部覆盖；额外统一公共小空间条件下两个真实根穷举293→4、46→9，解释低多样性，无删除/替换根。
- 认证对象为当前Observation契约下的constraint-conditioned current-slot distribution：计数分配质量∏m!/(a!b!d!)，无序暗手＋有序墙质量∝1/∏a!b!，按完整规则历史拒绝。不是均匀hidden-world，不是初始发牌诱导后验，更不是行为条件化exact posterior。
- PublicInformationState与audit-only truth隔离；从候选初始状态重放恢复全部私有字段，不得clone真实root仅替换暗手/墙。未知规则直接失败，不像旧HistorySampler静默拒绝。旧PASS预过滤在50k合法样本未见反例，不冒称全空间证明。
- 当前仍缺初始发牌先验权重认证、现实历史/提示可见性核对和policy likelihood；下一步优先建议前者，未经新指令不执行information-set Q benchmark或训练。完整概率差异、50根熵/边际/拒绝统计和复现见docs/training/mortal-style.md，原始赋值仅本地，不写训练dataset。
- 服务已正常结束MainPID=0。重要代码/总结一次commit并按既有授权push，逐JSON/权重不入Git；本轮完成后停止。

## Mortal fidelity主线收束（2026-10-18最新指令）

- 优先级：Mortal fidelity > independent cash improvement > Q-ranking diagnostics > counterfactual/belief extensions。
- 封存counterfactual-ranking/information-set-sampler、fixed-hidden/information-set rollout、initial-deal posterior/behavior likelihood/belief、pairwise/ranking及新counterfactual训练；不删除代码/证据，不进入主线。覆盖上一轮“继续认证发牌先验”的下一步建议。
- 本轮只重新下载固定commit 0cff2b52982be5b1163aa9a62fb01f03ce91e0d2的15源码并验Git blob，扫描既有415284决策，更新fidelity差异/排序及唯一下一步实验设计；训练/新局/rollout均0，不改训练核心。
- 官方自身Q控制greedy/探索/test，无frozen prior；示例ε=.005不是生产默认。官方gamma1同kyoku也共享target；GRP是独立终场排名预测构造局界utility差，next-rank aux则预测下一边界比分排名，两者不能混同。生产aux/探索/数据密度未知。
- 现有MC每行单遍并不比官方示例更稀疏；官方sample reuse字段无实现路径，不能杜撰高倍replay或量级更快更新。旧加量/探索/LR实验削弱简单解释，未做等效性证明，不说全面排除。
- 仅建议受限actor-center对照：共同小ε环境、固定10%整局probe开关anchor中心/自身Q中心，统一D算子评估与匹配updates；保留现金MC/LR/BN/网络，不直接整条主线raw接管。协议见docs/training/mortal-style.md。reward缺忠实单局现金GRP适配，暂不设计自创shaping。
- 本轮审计/设计后停止，不自动训练或进入任何下一阶段。重要总结/只读脚本commit并按既有授权push；下轮实验需新指令。

## Restricted Q-ownership actor-center A/B完成（2026-10-18）

- 最新用户授权仅执行fidelity audit预定的受限center干预，现已完成并停止。共同offline起点、冻结两对手、55×20/网络/现金MC halfMSE/CQL0/BN/默认warm20+cos300均未改；A全anchor，B仅预锁10%整局probe的非HU用raw Q，HU全anchor；共同ε=.005、T=.5、top-p1，不称官方生产配置或完整Mortal复现。
- 3代×5000局/arm＝30k真实训练；27k中期eval＋36k最终holdout，总93k轨迹、18,500独立seed。A/B相同seed/mask、独立连续optimizer/RNG；匹配每组489 updates、125,184 used行单遍，无replay/补采/按family或cash筛行，993尾行仅审计。
- 最终统一D（eval10%raw/其余anchor/HUanchor/ε0）下B3−A3=−.242元/局，95%CI[−.612,+.098]；B3−D(initial)=−.098[−.494,+.278]；B3−baseline=−2.247[−3.114,−1.407]。未通过三条下界>0，baseline不换；Gen1/2局部正差未持续，不挑冠军。D(initial)本身含10%raw并亏约2.15，不能与旧纯anchor部署结论混淆。
- B实际raw-center12,278决策、父动作偏离约6.2%–6.7%，A约1.5%–1.8%，干预生效但未取得可靠现金证据；不得外推100%ownership或Mortal范式无效。counterfactual/infoset/posterior/belief/pairwise等支线继续封存，不自动扩ownership/generation或新loss。
- 完整独立审计通过：全部真实规则/守恒/MC、seed/mask、实际抽样与概率/RNG、匹配updates、连续Adam、参数/BN/Q漂移、四块eval现金及bootstrap。reader先修复并发完成顺序比较、后修复跨batch浮点核对，均有旧源码/config/SHA及amendment；actor/learner/权重/轨迹/结果不改。251,361训练行重算最大Q偏差.0251元、centered .0035元，raw argmax变化0；不放松真实动作一致性。
- pytest438 passed/1 skipped，ruff及diff检查通过。原始638项artifact封存本机completion-evidence，不逐JSON/权重提交；重要代码/完整总结一次commit并按授权push。结果/协议/边界/审计SHA见docs/training/mortal-style.md。与强AI目标仍缺稳定独立现金提升；本轮停住，下一步等用户决定，不自行新增实验。
