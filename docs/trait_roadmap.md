# 性状/形态演化落地路线图

这份文档回答一个 `docs/trait_evolution.md` 调研清楚、但从未拿出执行顺序的问题：
**怎么把"性状演化"从三个旋钮（`diet`/`invest`/`size`）变成一件用户能看见正在发生的事？**

`docs/trait_evolution.md` 是这条线索的权威参考——现状盘点、OEE 可行性判决、体型基因的
完整证伪记录、`mutation_sigma` 的理论依据与可证伪预测，全部已经在那里。**本文档不
重新论证任何一条结论，只把已经写好的提案排进一个可以直接照着做的执行顺序**，并补
三样那份文档没有给出的东西：(1) 对"要不要先解幼体渴死瓶颈"这个问题的正面判决，
(2) 把候选性状按"是否会被幼体死亡删失"分成两类并逐条判定，(3) `mutation_sigma`
的逐行实现规格。

标记约定与 `docs/biology.md`/`docs/trait_evolution.md` 一致：`[现实]` 已确立的真实
世界科学；`[本世界实测]` 在这个模拟里跑出来的数字；`[对应]` 原则落到了哪个参数/
机制；`[提案，非结论]` 尚未实现、尚未验证的设计构想。

---

# 第一部分：总闸门——交割期检验与幼体渴死瓶颈

## 1. 复述判决依据

`docs/trait_evolution.md` §11 的交割期检验（deferred-payoff test）：**一个新性状,
如果它的收益需要"先活过某个高死亡率阶段"才能兑现,而这个阶段恰好吞掉了绝大多数
个体,那么在选择看来,这个性状只有成本,没有收益。** `docs/mortality.md` §1.2 给出
了这个阶段的具体数字（六种子×20000步，约130万次死亡）：

| 死因 | 占比 | 死亡时平均年龄 |
| --- | --- | --- |
| 渴死 thirst | 83.32 ± 0.66% | 52.5 步 |
| 被捕食 predation | 9.95 ± 0.58% | 170.7 步 |
| 饿死 starvation | 6.65 ± 0.36% | 353.0 步 |
| 老死 senescence | 0.10 ± 0.00% | 3001 |

体型基因是这条原则最干净的反例样本：预测 `mean_size` 漂移到 [1.1, 1.4]，实测
0.725——不是没有漂移，是朝相反方向漂移，因为"更大水箱"这个收益要求先把水箱装满，
而幼体从出生到渴死平均只有 52.5 步，从来没有机会做到这件事（`docs/trait_evolution.md`
§8）。

## 2. 正面判决：这不是一个全称先决条件，是一个分诊判决

**用户问的问题（"形态演化要不要先解幼体渴死瓶颈"）背后藏着一个隐含的全称量词——
"任何形态性状都要先解瓶颈"。这个全称命题是假的。** 正确的表述是分诊，不是排队：

- **对 B 类性状（收益终身兑现、被幼体死亡删失）**：判决是**是**，必须先解除或显著
  缓解瓶颈，否则重演体型基因的故事——不是"效果打折"，是"方向反转"，这已经是
  实测结果，不是推测。
- **对 A 类性状（收益即时兑现、不依赖熬过幼体期）**：判决是**否**，这些不需要
  等瓶颈解除，现在就可以做，而且应该现在就做——第 4 节会说明理由。

第二部分会把每一个具体候选性状过一遍这道判据。这里先把最重要的一条结论提前：
**A 类候选里排第一位的 `vision_radius` 可遗传，恰好和 `docs/mortality.md` §1.4
列出的三条瓶颈缓解路线之一（"更远的水感知"）是同一个机制。** 这意味着对这条具体
的性状，"先修瓶颈再做性状演化"和"用性状演化去修瓶颈"不是两个先后的任务，是**同
一个任务**——不需要额外排一个"修瓶颈"的独立里程碑再回头做性状，第一个性状实验
本身就是在尝试解除瓶颈。这是本文档给出的最主要的排序建议，第 4 部分会展开。

## 3. 与两个并行任务的关系：正交，不是竞争或依赖

写这份路线图的同时，另外两个 agent 分别在做"生态重调降低种群/让捕食者离水"和
"恐惧地景"（`docs/carnivore_riparian.md` 已经是这两项工作的落点文档）。核对这份
文档的内容后，可以给出一个明确、可核实的判断：

**这两项工作与幼体渴死瓶颈是正交的，不构成依赖关系，谁也不能替代谁。** 理由是
死因分解本身给出的年龄结构：渴死平均发生在 52.5 步（幼体期），被捕食平均发生在
170.7 步（成体期，`docs/mortality.md` §1.2）。`docs/carnivore_riparian.md` 整份
文档处理的是捕食者的**空间分布**问题（食肉者密度随离水距离塌缩得比食草者陡得
多，`docs/carnivore_riparian.md` §1.3）和"恐惧地景"（给猎物一条危险记忆，抵消
捕食者贴河而居的行为）——这两件事都发生在"个体已经活过了幼体渴死这第一关"的
前提之下,是捕食维度、成体年龄段的问题，处理的是第二死因和空间分布，不是第一
死因和年龄结构。**它们不会加重也不会缓解渴死瓶颈**：`carn_water_dist`（捕食者
离水距离）和 `death_thirst_frac`（渴死占比）是两个不同的因果链条，前者的调整
不改变新生水量、耗水速率或水感知半径这三个决定渴死的量。

所以：**这份路线图不需要等那两项工作完成，那两项工作也不需要等这份路线图完成
的任何一步**。唯一需要留意协调的一点在第四部分会点出——如果第二部分的
`attack_range`/猎物逃逸性状被实现，它和 `docs/carnivore_riparian.md` 会共享
同一组捕食参数（`attack_range`、`pred_efficiency`、`meat_water_frac`），两边
如果各自在独立 worktree 里调整这些常量，合并时需要互相核对，不要在互不知情
的情况下同时改同一批参数。

## 4. 哪条缓解路线最优先解闸门

`docs/mortality.md` §1.4 给出三条候选缓解路线：亲代照料、社会学习、更远的水
感知。逐条评估哪条最适合排在第一位：

- **社会学习（peer 通道后续）**：`docs/trait_evolution.md` §9 已经测出微弱正
  信号（`mean_age` +29 步，CI 排除零），但"跟随行为的直接证据"（幼体到最近
  成体的距离 vs 随机置换基线）还没算出来。这是**零新代码、零新性状**的分析
  缺口——直接用已有的活体状态在 run 结束时算一次即可，`docs/TODO.md` 已经
  点名。**应该最先做，因为它免费**，但它本身不是一个新机制，无法单独扛起
  "解闸门"的任务，只能确认已有机制（peer 通道）是否真的在起作用。
- **亲代照料 / 强制非繁殖幼年期**：`docs/trait_evolution.md` §22.8 已经指出，
  这条恰好是这份清单里**受瓶颈影响最严重**的一条——`maturity_age` 一旦设成
  有生物学意义的量级，绝大多数个体根本活不到那个年龄，机制本身的验证信号
  会被同一个瓶颈砍到测不出。**这是一个"用来解闸门"的候选里，自己反而最依赖
  闸门先被打开的悖论候选，不适合排第一。**
- **更远的水感知（`vision_radius` 可遗传）**：`docs/mortality.md` §1.3 给出的
  数字最具体——水感知半径 `vision_radius + food_sample_dist = 30`，食草动物
  平均离水距离 **34.7**，缺口只有 **4.7 个世界单位**，是三条路线里唯一一个
  已经被量化到"差多少"精确程度的候选。而且它同时满足：(a) 只需 `trait_dim`，
  不改 `in_dim`，代价最低；(b) 通过交割期检验（第二部分逐条判定，见下）；
  (c) 直接命中已确认的病因（"有腿、有时间、没有方向"）。

**结论**：`vision_radius` 可遗传是这三条里唯一同时满足"便宜、可证伪、直接命中
量化病因、通过交割期检验"四个条件的候选，应该是解闸门的主攻方向；peer 跟随
证据的免费分析应该同批做，用来确认现有机制是否已经在部分工作；亲代照料/
强制幼年期应该**推迟到 `vision_radius` 的效果被测出之后**再决定要不要做——如果
`vision_radius` 已经把 `death_thirst_frac` 压低到让更多个体活过幼体期，届时
做亲代照料实验的统计功效会显著好于现在。

---

# 第二部分：候选性状的 A/B 分类

## 5. 分类判据，加一条新原则：代价货币决定归类，不是性状概念本身

`docs/trait_evolution.md` §11 的判据是——问"携带 X 的个体，从出生到能够从 X
获益的那一刻，中位存活概率是多少"。这个判据对"性状本身"成立，但复核每一条
具体候选的实现方案后，发现一个判据本身没有说清楚、却同样重要的变量：

**[提案，非结论] 代价货币原则**：体型基因失败的具体机制不是"这个性状有代价"，
是"代价货币选在了水这个轴上"——体型越大，耗水速率越高（`size^0.75`），而这
个代价从出生第一步就照付，收益（更大水箱）却需要先把水箱装满,装满需要的时间
恰好被瓶颈删掉了。**任何新性状，只要它的代价项被记在水账本上，就会自动继承
体型基因的失败结构，不论这个性状的概念听起来多有道理。**

这条原则有一个已经在跑、且完全没暴露出这个问题的正面对照组：`dynamics.metabolize`
里的 `carn_cost * diet`（`underworld/dynamics.py:204`）——这也是一个"性状值越
高、持续代价越高"的项，但它记在**能量**账本上（`metabolize`），渴死看的是
**水**账本（`dynamics.thirst`，`underworld/dynamics.py:209-219`：`cost =
base_water_cost + move_water_cost * thrust`），两个账本互不相通，能量不足对应
的死因是饿死（6.65%，平均年龄 353 步，早就过了瓶颈窗口）。`diet` 基因从建群
起就稳定分化，从未表现出体型基因那种"被幼体期反向选择"的迹象，是这条原则最
好的现成反例证据。

**推论**：第 4 节推荐的 `vision_radius`、以及下面要判定的 `attack_range`、速度、
代谢率，只要给它们配的代价项记在 `dynamics.metabolize` 的能量账本上（而不是
`dynamics.thirst` 的水账本上），就能安全地留在 A 类。反过来，如果实现时图省
事把代价写成"耗水速率提高"，哪怕性状概念本身设计得再合理，也会在不知不觉间
把它拖回 B 类，重演体型基因的故事——而且这次更隐蔽，因为设计者会误以为自己
已经吸取了教训（配了代价项），却在货币选择这一步踩了同一个坑。

**`vision_radius` 有一个必须诚实指出的额外风险**：它是这份清单里唯一一个被
提议来**修复**渴死瓶颈的性状。如果它的代价项不慎选在水账本上，效果不是"体型
基因式的中性失败"，而是**主动让它想解决的问题变得更糟**——携带更大视野基因
的幼体反而耗水更快，这是一个比一般性的货币误选更糟的自摆乌龙，必须在实现
时明确避免（具体做法见第 7 节 4 条）。

## 6. 分类表

| 候选性状 | 分类 | 判据要点 |
| --- | --- | --- |
| `mutation_sigma` 可遗传 | **A** | 收益在繁殖事件兑现，不依赖携带者本身活到某个年龄；`docs/trait_evolution.md` §21 已判定"通过，收益不依赖幼体存活" |
| `vision_radius` 可遗传 | **A**（需正确记账） | 收益从出生第一步持续兑现；风险仅在代价货币选错时反转为自摆乌龙，见 §5 |
| `attack_range` 可遗传 | **A** | 捕食收益在捕食资格年龄段（成体，>170步）持续兑现，且该年龄段本就在瓶颈窗口之后；已有的能量货币先例（`carn_cost`）证明这类代价不会被水轴污染 |
| 代谢率可演化（`base_cost`/`move_cost` 乘子） | **A**（需配对代价） | 收益（更低能量消耗）从出生持续兑现，且天然记在能量账本；但若无配对代价会跑向下界，是不可证伪的基因饱和，不是交割期问题 |
| 速度（`max_speed`）可演化 | **A，但有隐藏耦合，需谨慎** | 见 §7 第 6 条：当前的耗水公式按 `thrust`（推力比例）计费而非按 `max_speed` 计费，本身不构成体型式陷阱，但若配的代价项挂错账本，或此性状本身不配代价，会产生"免费换取更多位移"的白嫖 |
| 生活史/幼体期长度（`maturity_age` 门） | **B** | `docs/trait_evolution.md` §22.8 明示：受瓶颈影响最严重，多数个体活不到能表达这个决策的年龄 |
| `repro_threshold` 可遗传 | **条件性 B** | 决策本身不要求先熬过幼体期，但演化信号仍被瓶颈砍到统计功效不足（§22.7），且排在"先看 `invest_frac` 数据"之后 |
| 体型基因（重做版，投资随子代体型等比例放大） | **B** | 已实测证伪的原型就是这条的前身；重做版本身仍要求幼体活着把水箱装满，除非先解瓶颈 |
| 性状间遗传相关（G 矩阵分析） | 不适用 | 是分析，不是新增选择压力，不消耗 `trait_dim`，应该最先做 |

## 7. 逐条展开：文件、代价、可证伪预测、预期方向

**1. `mutation_sigma` 可遗传（A）** —— 完整实现规格见第三部分，此处不重复。

**2. `vision_radius` 可遗传（A，本文档推荐的主攻方向）**

- **改哪些文件**：`underworld/config.py`（新增 `vision_index`、`vision_min`/
  `vision_span` 属性/字段，仿照 `size_min`/`size_span`）；`underworld/state.py`
  （新增 `vision_radius_of(genome, cfg)`）；`underworld/sensors.py:69`（
  `vision = cfg.vision_radius * (1.0 - cfg.forest_occlusion * terrain.forest[own_cell])`
  这一行，`cfg.vision_radius` 替换为按行读取的 `vision_radius_of(genome, cfg)`）；
  `underworld/dynamics.py:204`（`metabolize` 里加一项正比于
  `vision_radius_of(genome, cfg) - baseline` 的**能量**代价，不要碰
  `dynamics.py:219` 的 `thirst` 公式）。
- **`trait_dim` 还是 `in_dim`**：只动 `trait_dim`（+1），不新增任何感觉输入
  维度——它改变的是现有感知通道的衰减半径，不新增通道本身。
- **代价设计的关键约束**（重申 §5 的自摆乌龙风险）：代价必须走
  `dynamics.metabolize` 的能量项，绝不能走 `dynamics.thirst` 的水项，否则
  这条本该缓解瓶颈的性状会主动加重瓶颈。`docs/trait_evolution.md` §22.4 提到
  的第二个选项（让 `forest_occlusion` 对大视野基因的衰减更陡）同样安全，
  因为它不消耗任何资源，只是降低森林下的有效视野，可以和能量代价并用或
  单独使用。
- **必须处理的天花板约束**（`CLAUDE.md` 记录的硬约束）：`vision_min`/
  `vision_span` 的上界必须严格封顶在 `sense_grid` 对应的世界单位格宽以下，
  否则演化出的大视野个体会遭遇一个和基因值无关的静默失败（超出 3×3 邻域
  块的个体对视觉和捕食都不可见）——这必须用 `assert` 或取值范围裁剪在代码
  层面排除，不能指望演化自己避开。
- **可证伪预测**：`death_thirst_frac` 随种群平均 `vision_radius` 基因值上升
  而下降；`herb_water_dist`（食草动物到水的平均距离）同步上升，向地图
  离水距离中位数靠拢；死亡时平均年龄（渴死类）应该从 52.5 步上移。
- **预期选择方向**：基因值从中性起点（对应当前 21.0）向上漂移，直到代价
  项（能量税）与收益（更早发现水源）在某个中间值达到平衡——不应该无约束
  地跑向 `vision_span` 上界，如果跑到上界，说明代价项配得太轻，需要重调，
  不是性状设计本身的问题。

**3. `attack_range` 可遗传（A）**

- **改哪些文件**：`underworld/config.py`（新增 `attack_index`、
  `attack_min`/`attack_span`）；`underworld/state.py`（新增
  `attack_range_of(genome, cfg)`）；`underworld/dynamics.py:154`（
  `predation` 函数的 `eligible = ... & (dist < cfg.attack_range) & ...`，
  `cfg.attack_range` 替换为按攻击方个体读取的基因值）；代价项挂在
  `underworld/dynamics.py:169/180`（`pred_efficiency` 项，让更大攻击范围
  换取更低的转化效率）或 `dynamics.py:204` 的 `metabolize` 加一项正比于
  `attack_range_of - baseline` 的能量税，仿照已经验证过安全的 `carn_cost`
  记账方式。
- **`trait_dim` 还是 `in_dim`**：只动 `trait_dim`（+1，若同时加对称的猎物
  逃逸性状则 +2）。基因对全体个体存在（和 `diet`/`invest`/`size` 一样是
  每个基因组里的一列），食草动物携带者的基因中性漂变，不需要按食性条件化
  存在——`docs/trait_evolution.md` §22.5 已说明理由：`predation` 的
  `eligible` 判定本身就要求食性差超过 `diet_delta`，食草动物几乎永远没有
  攻击资格。
- **与并行任务的协调点**（重申第 3 节）：这条改的常量（`attack_range`、
  `pred_efficiency`）与 `docs/carnivore_riparian.md` 正在讨论的参数
  （`attack_range` 放大方案、`meat_water_frac`）重叠，实现前应该核对
  `docs/carnivore_riparian.md` 当时的实现状态，避免两个 worktree 各自
  改同一批常量后难以合并。
- **可证伪预测**：食肉动物谱系的平均 `attack_range` 应该和食草动物一侧的
  逃逸相关性状（如果同时实现）形成协同演化的轨迹，而不是各自收敛到一个
  静态值后不再变化——这是区分"军备竞赛"和"单边最优化"的直接判据。若只做
  单边（只给捕食者），预期是攻击范围向上漂移直到代价项拉平，此后种群
  `attack_range` 方差应该显著低于协同演化版本。
- **预期选择方向**：单边版本——攻击范围上升直至代价平衡；双边版本——
  攻击范围与逃逸性状交替上升，形成本项目第一次可测量的红皇后动态
  （Van Valen 1973，`docs/trait_evolution.md` §3 已引用）。

**4. 代谢率可演化（A，需配对代价）**

- **改哪些文件**：`underworld/config.py`（新增代谢效率基因的索引/范围）；
  `underworld/state.py`（新增 `metabolic_rate_of`）；`underworld/dynamics.py:204`
  （`cost = (cfg.base_cost + cfg.move_cost * thrust + cfg.carn_cost * diet)`
  这一行整体乘以按个体读取的 `metabolic_rate_of(genome, cfg)`）。
- **`trait_dim` 还是 `in_dim`**：只动 `trait_dim`（+1）。
- **必须配对的代价**：如果只做"更低代谢率越好"这一个方向，没有任何下游
  权衡，基因会无约束地跑向下界，是 `docs/trait_evolution.md` 反复警告的
  "不可证伪的基因饱和"结果，不是交割期问题（它不受幼体死亡删失，因为收益
  从出生持续兑现），而是一个独立的、必须单独解决的设计问题——需要一个
  和代谢率反向关联的下游量（例如让更低代谢率的个体某个感知/运动参数打
  折扣，模拟真实世界"低代谢通常伴随更低的活跃度/反应速度"这条权衡），
  这条配对代价的具体形式本文档不预先指定，留给实现时按当时的生态状态
  设计，但**必须存在**，否则不要实现这条性状。
- **可证伪预测**：`death_starvation_frac` 随种群平均代谢率基因值下降而
  下降；由于饿死本身只占 6.65%、平均死亡年龄 353 步（远在瓶颈窗口之外），
  这条性状的**统计功效受限的原因和体型基因不同**——不是被幼体死亡删失，
  是它调节的风险本身发生率太低，样本量天然小，报告结果时需要把这条讲
  清楚，不能把"效应量小"误读成"性状无效"。

**5. 速度（`max_speed`）可演化（A，但需要谨慎处理一个隐藏耦合）**

- **一个容易被忽略的实现细节，必须先弄清楚再动手**：`underworld/dynamics.py:33-41`
  显示当前的推进链路是 `thrust = 0.5*(outputs[:,1]+1.0)`（脑输出映射到
  [0,1] 的推力比例）→ `speed = thrust * cfg.max_speed * slow`（实际速度）。
  而 `dynamics.py:204` 的能量代价和 `dynamics.py:219` 的水代价**都是按
  `thrust`（推力比例）计费，不是按 `max_speed`（基因/常量）计费**。这意味
  着：如果 `max_speed` 变成基因而不配代价项，效果是纯粹的白嫖——同样的
  `thrust`、同样的代价，换来更多实际位移，对能量和水两个账本同时如此。
  这不是体型基因那种"代价货币选错"的陷阱（体型基因的陷阱是代价与基因值
  绑定但收益绑定不上；这里是代价压根没有与基因值绑定），是一个更直接的
  "跑向上界"风险，`docs/trait_evolution.md` §22.6 已经点名需要一个随速度
  值本身超线性增长的代谢成本。
- **改哪些文件**：`underworld/config.py`（新增 `speed_index`、
  `speed_min`/`speed_span`）；`underworld/state.py`（新增 `speed_of`）；
  `underworld/dynamics.py:41`（`speed = thrust * cfg.max_speed * slow` 里的
  `cfg.max_speed` 替换为按个体读取的 `speed_of(genome, cfg)`）；配对代价
  **必须**加在 `underworld/dynamics.py:204` 的 `metabolize`（能量账本），
  一项正比于 `speed_of(genome, cfg)` 某个超线性次方的能量税——**绝不能**
  加在 `underworld/dynamics.py:219` 的 `thirst`（水账本），否则会把这个
  隐藏耦合从"代价没绑定"变成"代价绑定在错误的账本上"，两个问题叠加，
  是这份清单里风险最高的一条实现细节。
- **`trait_dim` 还是 `in_dim`**：只动 `trait_dim`（+1）。
- **可证伪预测**：`docs/trait_evolution.md` §22.6 已经给出保留意见——
  死因分解的诊断是"有腿、有时间、没有方向"，暗示单独提速对渴死瓶颈的
  边际贡献应显著小于 `vision_radius`。预测：速度性状对 `death_thirst_frac`
  的边际效应应显著小于 `vision_radius` 的效应；速度性状可能对捕食逃逸/
  一般觅食效率有独立正面效应，这是与"缓解渴死"不同的价值主张，应分开
  报告。
- **预期选择方向**：若代价项配得足够重，速度应该在一个中间值达到平衡；
  若配得太轻，会无约束跑向上界——这条更需要在跑长种子之前先做短种子
  探路，确认代价项量级，再进入六配对种子的正式验证，避免像
  `docs/carnivore_riparian.md` §342 记录的教训那样在错误的参数量级上
  浪费一整轮长跑。

**6. 生活史/幼体期长度：`maturity_age` 门（B）**

- **性质**：不是纯 trait 基因提案，是给状态机加一个年龄门（`want = alive
  & (energy > repro_threshold) & (age > maturity_age)`），`maturity_age`
  本身可以是常量也可以是基因。`docs/trait_evolution.md` §22.8 已经论证
  这是让亲代照料真正可演化所必需的前提——缺了它，照料是免费的，权衡消失。
- **为什么排在 B 类**：如果 `maturity_age` 被设成有生物学意义的量级（比如
  几百步），绝大多数个体根本活不到那个年龄——这条性状的演化信号会被瓶颈
  砍到几乎测不到任何东西，是清单里受瓶颈影响最严重的一条。
- **判决**：**必须等 `vision_radius`（或其他瓶颈缓解路线）的效果被测出、
  且 `death_thirst_frac` 有实质下降之后再做**，否则重复"实验够不到问题"
  的教训（正如 `docs/experiments.md` §3 记录的拆反混合装置实验：单峰
  founder 六种子全灭绝，因为起点先崩了，压根测不到想问的问题）。

**7. `repro_threshold` 可遗传（条件性 B）**

- **排序依据**：`invest_frac` 已经是可遗传基因，`docs/trait_evolution.md`
  §22.3 指出一个免费的前置检查——现有六种子×20000步的数据里，`mean_invest`
  相对建群中性值（0.5）是否已经按年龄别死亡率理论的方向（更低投入，
  对冲幼体高死亡率）漂移。**这个检查零新代码，应该先做**；若理论方向
  得到支持，`repro_threshold` 可遗传是这条线索的自然延伸；若不支持，
  应该先搞清楚为什么理论预言在这里失灵，而不是急着加下一个基因。
- **判决**：有条件通过交割期检验——决策本身不要求先熬过幼体期（它作用
  于已经存活到成年的个体），但演化信号仍受瓶颈的竞争性风险删失影响，
  因为渴死已经先一步决定了大多数个体活不到能表达这个决策的年龄。**排在
  数据挖掘（第 7 条本身的前置检查）之后，也建议排在 `vision_radius` 效果
  测出之后**，理由与第 6 条相同：统计功效会因瓶颈缓解而改善。

**8. 体型基因重做版：投资随子代体型等比例放大（B）**

- `docs/trait_evolution.md` §8 末尾已经给出这条修正的具体方向——"如果要
  让体型成为真正的权衡，必须让新生供给随子代自己的体型等比例放大"。
- **判决**：**必须等瓶颈缓解之后再做**。这不是猜测，是已经实测过一次的
  同构失败——原版体型基因的问题根源就是"幼体从来没有机会把水箱装满"，
  重做版本改变的是供给量的计算方式，不改变"幼体要先活着才能体现体型
  优势"这个结构性前提。在瓶颈缓解之前重做这条实验，大概率会得到同样
  方向反转的结果，而且会让人误以为"重做方案本身有问题"，其实问题
  出在时序上，不出在方案设计上。

**9. 性状间遗传相关（G 矩阵）——不适用 A/B 框架，应该最先做**

- `docs/trait_evolution.md` §22.1 已给出零代码方案：把 `underworld/metrics.py`
  已有的 `invest_diet_corr` 计算模式（`metrics.py:95-99`）扩展到
  `corr(size, diet)`、`corr(size, invest)`，只需要在 `Metrics` NamedTuple
  里加两个字段，用已经算好的 `size`、`invest`、`diet` 数组直接算协方差。
  这是分析，不是新增选择压力，不消耗 `trait_dim`，不受交割期检验约束。

---

# 第三部分：`mutation_sigma` 实现规格

这是用户"生物进化的能力"最直接的对应，`docs/trait_evolution.md` §18-20 已经
给出设计与三条可证伪预测（B1/B2/B3）。这一部分把它落成一份可以直接照抄的实现
规格，逐文件、逐行对照 `invest_of`/`size_of` 的既有模式。

## 8. 逐文件改动

**`underworld/config.py`**

- `trait_dim: int = 3` → `trait_dim: int = 4`（当前在 `config.py:54`）。
- 在 `size_min`/`size_span`（`config.py:198-199`）之后新增：
  ```python
  sigma_min: float = 0.01         # floor; a gene of 0 sigmoids to 0.5 ->
  sigma_span: float = 0.08        #   sigma = 0.01 + 0.5*0.08 = 0.05, matching
  #                                  the old fixed `mutation_sigma`, so a fresh
  #                                  population starts at the previous behaviour.
  ```
- 在 `size_mutation_sigma`（`config.py:215`）之后新增：
  ```python
  sigma_mutation_sigma: float = 0.02  # fixed meta-mutation rate on the sigma
  #                                     gene itself -- NOT self-referential, see
  #                                     the reasoning below. Setting this to 0
  #                                     freezes the sigma gene at its founder/
  #                                     crossover value, the control arm for B1-B3.
  ```
- 一个新的消融开关，仿照 `peer_channel_enabled` 的模式（`config.py:247-256`），
  让开/关两个臂共享同一个基因组布局，不需要两次种群作废：
  ```python
  mutation_sigma_heritable: bool = True  # False: brain genes mutate at the
  #                                    constant `cfg.mutation_sigma` regardless
  #                                    of each individual's own sigma gene value
  #                                    -- the sigma gene still exists, drifts,
  #                                    and is reported in Metrics, but has no
  #                                    functional effect. This is the ablation
  #                                    arm for "does making mutation_sigma
  #                                    heritable change anything", genome-
  #                                    compatible with the True arm.
  ```
- 新增派生属性，仿照 `invest_index`/`size_index`（`config.py:486-499`）：
  ```python
  @property
  def sigma_index(self) -> int:
      return self.brain_params + 3
  ```

**`underworld/state.py`**

- 在 `size_of`（`state.py:69-78`）之后新增：
  ```python
  def mutation_sigma_of(genome: jax.Array, cfg: Config) -> jax.Array:
      """Map the sigma gene to [sigma_min, sigma_min+sigma_span]; 0.05 at gene=0.

      Read inside `genome.mutate` as the *offspring's own* pre-mutation value
      (see genome.py) -- this is the modifier-locus mechanism itself, not a
      cached per-agent quantity, so it is recomputed from the genome like
      `invest_of`/`size_of`.
      """
      return cfg.sigma_min + cfg.sigma_span * jax.nn.sigmoid(genome[:, cfg.sigma_index])
  ```

**`underworld/genome.py`**

`crossover()`（`genome.py:38-75`）**不需要改动**：`sigma` 基因刻意不豁免交叉，
参与逐基因独立 50/50 交叉，与 `invest` 同构——理由相同：只在出生时被读取一次，
从不进入感觉-运动回路，没有控制器/身体错配风险，没有理由防止它被交叉稀释。

`mutate()`（`genome.py:16-35`）需要改写，因为 `sigma` 现在是**逐个体**的量，
不再是逐基因位置的固定标量：

```python
from .state import mutation_sigma_of  # new import; no circularity -- state.py
                                        # does not import genome.py

def mutate(genome: jax.Array, key: jax.Array, cfg: Config) -> jax.Array:
    if cfg.mutation_sigma_heritable:
        brain_sigma = mutation_sigma_of(genome, cfg)          # [N]
    else:
        brain_sigma = jnp.full((genome.shape[0],), cfg.mutation_sigma)
    sigma = jnp.broadcast_to(brain_sigma[:, None], genome.shape)
    diet_sigma = cfg.diet_mutation_sigma if cfg.diet_mutation_asymmetric else cfg.mutation_sigma
    sigma = sigma.at[:, cfg.diet_index].set(diet_sigma)
    sigma = sigma.at[:, cfg.invest_index].set(cfg.invest_mutation_sigma)
    sigma = sigma.at[:, cfg.size_index].set(cfg.size_mutation_sigma)
    sigma = sigma.at[:, cfg.sigma_index].set(cfg.sigma_mutation_sigma)
    return genome + jax.random.normal(key, genome.shape) * sigma
```

`genome` 参数在调用点（`underworld/reproduction.py:132-133`）是 `crossed`——
crossover 之后、mutate 之前的子代基因组，也就是子代刚从双亲继承、尚未突变的
`sigma` 基因值。**这正是修饰基因理论要求的机制**：这一步既完成了"子代继承了
一个（经交叉重组的）修饰等位基因"，又用这个值决定"子代自己的脑基因要突变多
少"，一步到位，不需要额外传入父代基因组。

## 9. 自指 vs 固定元突变率：为什么选固定，且必须选固定

`docs/trait_evolution.md` §18 已经论证过这个取舍，这里只重申落进规格的结论：
**`sigma` 基因自己的突变率必须是固定的 `sigma_mutation_sigma`（0.02，与
`size_mutation_sigma` 同量级），不能自指**（不能让 `sigma` 基因的突变幅度
由它自己当前的值决定）。自指版本有一个和适应度无关的病理——偶然突变出较低
`sigma` 值的谱系会同时继承一个较低的自我突变率，形成棘轮式收敛，把"选择在
压低突变率"和"数值机制本身锁死了一个低值"混在一起，污染整个实验的可解释
性。上面的代码里 `sigma.at[:, cfg.sigma_index].set(cfg.sigma_mutation_sigma)`
这一行就是在强制这条约束——`sigma` 基因位自己永远用固定速率突变，不会读取
`mutation_sigma_of` 的结果来决定自己的突变幅度。

## 10. 平凡结果 vs 有意义结果：把判据落进执行细节

`docs/trait_evolution.md` §19-20 已经给出 B1/B2/B3 三条非平凡预测和它们的
判据表。这里只补上让这些预测可以被实际计算出来所需要的具体改动：

**`underworld/metrics.py`**（仿照 `mean_size`/`size_std`，`metrics.py:57-61,
92-94,130-131` 的模式，`append`，不要插在中间）：

```python
mean_mutation_sigma: jax.Array   # population mean of the evolved sigma gene
sigma_std: jax.Array             # spread -- a collapsed value near sigma_min
                                  # with near-zero spread is the "trivial" signature
carn_mutation_sigma: jax.Array   # B2: carnivore-lineage mean, mirrors carn_speed
herb_mutation_sigma: jax.Array   # B2: herbivore-lineage mean, mirrors herb_speed
```

在 `compute()`（`metrics.py:64-132`）里仿照 `size = size_of(...)` 
（`metrics.py:92`）和 `carn_speed`/`herb_speed` 的 `is_carn`/`is_herb` 掩码
（`metrics.py:74-77,109-110`）计算这四个字段。这些字段一旦存在，`server/protocol.py`
不需要任何改动就能通过 `metrics._asdict()` 在需要时暴露给 wire 协议
（`CLAUDE.md`"Cross-file contracts"一节已经说明这条），但**这一步实现暂时
只需要 `scripts/run_headless.py --json` 能读到它们**，不必接入实时协议。

**判据回顾（复制 `docs/trait_evolution.md` §20 的表，执行时直接对照）**：

- **平凡**：`mean_mutation_sigma` 在所有臂里收敛到几乎相同的值，且紧贴
  `sigma_min`；`carn_mutation_sigma` 与 `herb_mutation_sigma` 无显著差异
  （B2 无效）；`mutation_sigma_heritable=True` 臂在种群瓶颈（既有的 `n_init`
  建群下跌）中不比 `False` 臂恢复得更快（B3 无效）；不同 `n_max`/`n_init`
  配置得到的地板高度几乎相同（B1 无效）。
- **有意义**：地板显著高于 `sigma_min` 且随有效种群大小反向缩放（B1 成立）；
  或食肉/食草谱系分化（B2 成立）；或 `True` 臂在建群瓶颈中表现出可测量的
  恢复优势（B3 成立）。

**统计纪律**：六配对种子起，报告每个种子的数字，Mann-Whitney 或配对
Wilcoxon + bootstrap 区间，不做 Bonferroni 校正——与 `CLAUDE.md`/
`docs/trait_evolution.md` 全文一致。`sigma_min` 必须严格大于零：允许突变率
精确归零的种群会在某个演化分支上失去响应任何选择压力的能力，这是数值设计
错误，不是"演化能力被压到最优"的证据。

---

# 第四部分：分阶段执行顺序

## 11. 依赖关系

```
Stage 0（零代码，立即可做，互相独立，也独立于 Stage 1）
  ├─ G 矩阵扩展：corr(size, diet)、corr(size, invest)
  ├─ invest_frac 既有漂移方向的重新分析（六种子×20000步已有数据）
  └─ peer 通道跟随行为直接证据（幼体-最近成体距离 vs 随机置换基线）
        │
        ▼（若支持理论方向，才升高 repro_threshold 的优先级；不阻塞下面任何事）

Stage 1（唯一一次种群作废，trait_dim 3→N，四条候选打包）
  ├─ mutation_sigma 可遗传（第三部分规格）
  ├─ vision_radius 可遗传（本文档推荐的瓶颈主攻手段）
  ├─ attack_range 可遗传（需与 docs/carnivore_riparian.md 协调参数）
  └─ 代谢率可演化（需配对代价，若代价设计不到位可以推迟出这一批）
        │
        ▼（vision_radius 的效果决定 Stage 2 走哪条分支）

Stage 2（分叉，取决于 Stage 1 的 death_thirst_frac 是否实质下降）
  ├─ 若下降：解锁 B 类候选——maturity_age 门、体型基因重做版、repro_threshold
  └─ 若未下降：不做 B 类候选，转攻 mortality.md 另外两条瓶颈缓解路线
        （亲代照料的非 trait 版本 / 社会学习机制的进一步加强）

独立分支（可与 Stage 1/2 并行，不占用同一次种群作废）
  └─ 连接代价驱动的模块化（docs/trait_evolution.md §17 末尾提案）：
     加进 dynamics.metabolize 的代谢成本项，不改变任何张量形状，
     不需要作废种群，随时可以做
```

## 12. 分阶段表：每阶段做什么、能看到什么新现象、是否作废种群

| 阶段 | 做什么 | 是否作废种群 | 做完能观察到什么新现象 |
| --- | --- | --- | --- |
| 0 | G 矩阵扩展 + `invest_frac` 数据挖掘 + peer 跟随证据 | 否 | 三个问题各自有了答案：体型漂移是否与食性绑定；投资策略是否已经按年龄别死亡率理论的方向演化；"活得久一点"和"学会跟随"之间那道缺的证据是否补上。这一步本身不产生新的演化现象，是给后续阶段的判据打底 |
| 1 | `mutation_sigma`/`vision_radius`/`attack_range`/代谢率 打包成一次 `trait_dim` 变更；每条各自配独立的冻结开关（仿照 `size_mutation_sigma=0`）以便单独消融 | 是（一次） | 这是用户第一次能**直接看到**性状在演化：`vision_radius` 均值随代数移动、`death_thirst_frac` 下降；`mutation_sigma` 的种群均值曲线；如果做了 `attack_range`，捕食成功率随代数的变化。这是本文档里"用户要的是看到性状在演化，不只是代码跑通"这条诉求第一次被真正满足的阶段 |
| 2（分支 A：瓶颈已缓解） | `maturity_age` 门、体型基因重做版、`repro_threshold` 可遗传 | 是（一次，把这三条尽量打包） | 第一次能测试真正的生活史权衡——亲代照料是否演化出可测量的投资/存活权衡；体型基因重做版能否表达出原来被删失的耐渴优势；繁殖阈值是否随年龄别死亡率理论的方向移动 |
| 2（分支 B：瓶颈未缓解） | 转向 `mortality.md` 另外两条路线：社会学习机制加强、非 trait 版本的亲代照料（比如让成体主动向附近幼体转移水/能量，不依赖 `maturity_age` 门） | 视具体机制而定 | 验证"性状路线打不开瓶颈"这个假设本身，同时给下一次尝试留下诊断——如果连成体主动照料都测不出效果，说明瓶颈的真正病因可能不在"缺乏机制"，而在别的地方（比如水资源的绝对稀缺度），这本身是一个必须写进 `docs/mortality.md` 的负结果 |
| 独立 | 连接代价驱动的模块化（代谢成本项） | 否 | 这个阶段测的不是"某个性状在朝哪个方向漂移"，是"脑权重矩阵的范数分布是否被系统性压低"，以及压低之后种群在既有建群瓶颈中的恢复速度是否变快——是"演化能力"这条主线的第二个独立验证，服务同一个用户诉求的不同侧面 |

---

## 结语

- **总闸门判决**：不是全称先决条件——B 类性状（生活史/幼体期长度、体型基因
  重做版、条件性的 `repro_threshold`）必须等瓶颈缓解，A 类性状（`mutation_sigma`、
  `vision_radius`、`attack_range`、代谢率、有隐藏耦合但可控的速度）现在就能做，
  且应该现在就做。
- **最重要的排序建议**：`vision_radius` 可遗传应该排第一，因为它同时是 A 类
  性状**和**瓶颈缓解手段本身——做它不是"绕开瓶颈再等瓶颈解决"，是用性状演化
  本身去解瓶颈，一步做两件事。
- **新增的设计原则（本文档在复核已有提案时补上的一条，`docs/trait_evolution.md`
  没有明确写出）**：代价货币决定 A/B 归类，不是性状概念本身决定——任何新性状
  的代价项只要记在水账本（`dynamics.thirst`）上，都会自动继承体型基因的失败
  结构；已经在跑且从未暴露这个问题的能量账本先例（`carn_cost`）证明了这条
  区分不是理论空谈。`vision_radius` 因为身兼"瓶颈解药"和"新性状"两个角色，
  这条原则对它的适用尤其关键——代价记错账本，会把解药变成毒药。
  速度性状还有一层需要额外小心：当前的耗水/耗能公式按 `thrust`（推力比例）
  计费而非按 `max_speed` 本身计费，若不配代价项会产生与货币选错完全不同
  的另一种"白嫖"。
- **与两个并行 agent 的关系**：生态重调/离水和恐惧地景处理的是捕食维度
  （死亡年龄 170.7 步），本路线图处理的是渴死维度（死亡年龄 52.5 步），
  两条链路正交，互不阻塞；唯一需要协调的交点是 `attack_range` 相关参数，
  实现前应核对 `docs/carnivore_riparian.md` 当时的状态。
- **执行顺序**：Stage 0（零代码分析）现在就做；Stage 1（`mutation_sigma`+
  `vision_radius`+`attack_range`+代谢率打包成一次 `trait_dim` 变更）是第一个
  真正让用户"看见性状在演化"的阶段；Stage 2 按 Stage 1 的结果分叉，不预先
  假设瓶颈一定会被解开。
