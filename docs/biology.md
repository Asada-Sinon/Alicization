# 生物学参考

这份文档记录支撑 Underworld 设计决策的**真实生物学与自然科学依据**。写它的理由很实际：这些结论多半是在开发过程中查证的，如果只留在对话历史里，半年后有人问"为什么 `forage_water_frac` 是 0.10 而不是 0.3"就没有答案了。代码注释适合记一句话的理由，这里放完整的论据链。

**新的发现请追加，不要重写既有条目。** 按生物学主题组织，不按时间顺序；条目里的结论如果后来被推翻，就地标注推翻的证据，保留原文——被否掉的假设和被采纳的一样有信息量。

## 阅读约定

每个主题下用三类标记，**不要混用**：

| 标记 | 含义 |
| --- | --- |
| **[现实]** | 已确立的真实世界生物学，附来源。可以据此论证设计。 |
| **[本世界实测]** | 在本模拟里跑出来的数字。这是**我们模型的性质，不是自然界的性质**——它可以证明"我们的模型做到了 X"，不能用来论证"自然界如此"。 |
| **[对应]** | 该原则落到了哪个参数、哪个机制上；或者明确写"尚未实现"。 |

标识符（`forage_water_frac`、`water_dist`、`inland_frac`）、术语（piosphere、path integration、Hamilton's rule）、人名与文献标题一律保持原文，不翻译。

---

## 1. 水点效应（piosphere effect）

### [现实] 放牧压力随离水距离衰减

半干旱牧场周围存在一个以水点为中心的同心圈结构：践踏与采食强度在水点处最高，随离水距离衰减。这个现象在文献里叫 **piosphere**。

机制的关键一句，引自文献：

> "Animals cannot match the distribution of their resources with the distribution of their foraging behaviour, because of their daily drinking requirements."

也就是说，这个梯度不是因为水边的草更好吃，而是因为**动物的活动分布被每日饮水需求钉住了**，无法与资源分布对齐。资源在外面，行为在里面，两者错配，于是水边被过度利用。

第二个要点：当水点附近的可食草料被消耗到一定程度后，**利用率（utilization）的峰值会外移到中等距离**——最近的地方已经没什么可吃的了，实际发生采食最多的地带反而在外圈。所以"离水越近压力越大"描述的是长期的立地存量（standing crop）梯度，而瞬时的采食量峰值位置是会移动的。

来源：
- Thrash & Derry, *The nature and modelling of piospheres: a review*, Koedoe
- *The development of forage production and utilization gradients around livestock watering points*, Landscape Ecology (Springer)
- Karamoja sub-region (Uganda) 研究，*Pastoralism* 期刊

### [本世界实测] 同样的梯度自发出现了，且没有被写进任何规则

这个梯度不是我们编码的。`ecology.regrow` 只做逐格的 logistic 回复；水边格子被啃得更狠，纯粹因为 agent 在那里更密集。按离水距离分带统计的立地存量比 `plant / capacity`：

| 离水距离（世界单位） | 实现存量比 |
| --- | --- |
| 0–8 | 0.293 |
| 8–24 | 0.576 |
| 24–48 | 0.883 |
| 48+ | 0.955 |

河岸带的消耗是内陆的 **3.3 倍**。绝对草料量：河边 0.53，内陆 1.43。

这条测量的实际用途是**否掉了一个改动**。当时怀疑地形本身给了水边容量加成，是它把种群吸在河岸上，计划把这个加成压平。测下来陆地上 `corr(capacity, water_dist) = -0.046`，基本是零——容量与离水距离无关，agent 是在承受 **2.7 倍的食物惩罚**的前提下仍然扎堆在河边。改动因此取消：这证明了病理是**导航性的，不是营养性的**（agent 不敢离开水，因为它离开就找不回来），后续工作应该去修感知/记忆，而不是去调地形。这条推理直接导向了 §4 的两层记忆。

### [对应]
- 梯度本身：`ecology.regrow` 的 logistic 消耗，无需额外机制。
- 被取消的改动：地形容量的水近性加成保留原样，见 `terrain.py` 的 `capacity` 推导。
- 度量：`Metrics` 的 `water_bound_frac` / `inland_frac` / `herb_water_dist` / `carn_water_dist`（提交 `412d108` 引入）。

---

## 2. 森林 vs 草原的可食生物量

### [现实] 森林生物量远高于草原，但对食草动物可食的部分反而更低

这一条是反直觉的，也是本项目里最容易设计错的地方——"森林 = 植被茂密 = 食物更多"是错的。

- 稀树草原（savanna）的植被中 **15–90% 是适口、可消化的草**，其中每年可被消耗的比例高达 **60%**。
- 热带湿润森林由木质生长主导，地面的草本与杂类草只占很小一部分。
- 生长条件差时叶片被保留更久，因而防御更强（粗糙、含单宁、不适口）。
- 闭合树冠遮蔽草本层。

结果就是：**承载大型食草动物生物量的是稀树草原，不是雨林。** 森林真正能提供给动物的高价值资源是**果实**，而果实的分布是斑块状的。

来源：
- Britannica, *Savanna — Biological productivity*
- *Many shades of green: the dynamic tropical forest–savannah transition zones*, Phil. Trans. R. Soc. B
- *Exploring a natural baseline for large herbivore biomass*, bioRxiv

### [对应] Stage 2 的果实层

所以林冠在这里**不是"食物更多"**，而是**低草料底 + 散布其中的高价值例外**：

```
fruit_capacity = fruit_max * patch * forest**2 * (1 - rock)
```

用 `forest**2` 而非 `forest` 是刻意的：平方把边缘稀疏林冠压向零，让果实成为"森林的资源"，而不是"按林冠加权的普遍加成"。`patch` 取互质波数（`fruit_wavenumber_x=7`、`fruit_wavenumber_y=11`）正弦积的上部约 20%，保证环面无缝且拍频足够长。慢回复（`fruit_regrow_rate=0.008`，约为草料的 1/7）也是设计的一部分——回满太快，记忆相对于随手搜索就没有优势，而奖励记忆正是这个资源存在的目的。

### [本世界实测] 诚实的保留意见：果实太小，判决推迟

三个乘性因子复合后，容量均值只有 `fruit_max` 的 6.3%；果实仅占全世界食物能量的 **5.0%**；立地存量更只有容量的 4.7%。种群级指标没有被撬动：

| 指标 | 基线 → 加果实后 |
| --- | --- |
| `forest_frac` | 0.415 → 0.401（−3.4%，未达判据） |
| `inland_frac` | 0.248 → 0.251 |
| `water_bound_frac` | 0.400 → 0.416 |
| population | 751 → 741 |

判决推迟到记忆系统上线之后（提交 `3dff412`）。理由：果实是**为记忆系统设计的资源**（稀疏、慢再生、高价值），在当时感知半径只有 30 单位、零记忆的条件下必然无用，那时考它等于在它注定失效的条件下考它。重测若仍无效，应当删除而不是靠信念保留。

---

## 3. 草料中的预成水（preformed water）

### [现实] 食草动物有三档饮水依赖度，最轻的一档完全不需要喝水

- 食草动物按水依赖度分三类：**完全从草料取水**的、**需要不定期饮水**的、**需要接近每日饮水**的。
- 草料干物质从早季的约 **43%** 升到晚季的约 **68%**，即含水量从 57% 降到 32%。
- 多汁植物（succulents）即便在干燥夏季仍保有 **61–93%** 的含水量。

关键一句：

> "If moisture content of diets is sufficiently high, herbivores can probably obtain all of their exogenous water preformed in their forage, and thereby become independent of permanent water sources."

来源：
- Kihwele et al. 2020, *Quantifying water requirements of African ungulates through a combination of functional traits*, Ecological Monographs
- *Potential Foraging Decisions by a Desert Ungulate to Balance Water and Nutrient Intake*, PLOS One

这条直接反驳了本世界原先的设定：这里的植物是**完全干燥的**，`graze` 只返回能量，河海是食草动物唯一水源。参照 §1，这正是把种群钉死在河岸上的那一条。

### [对应] `forage_water_frac = 0.10`

取值依据（见 `config.py` 该字段的注释）：食草动物平衡态 `mean(last_food) = 0.144`/步，在实测推力 0.65 下水耗 0.0524/步，故 0.10 相当于约 **27% 的水补贴**，把单程射程从约 198 拉到约 272 世界单位（往返 99 → 136，明显越过 35.5 的离水距离中位数），同时不至于让水变成非问题。

实测效果（三种子，4000 步，提交 `1c7c159`）：`water_bound_frac` 0.528 → 0.400，`herb_water_dist` 14.51 → 22.72（+56.6%），`inland_frac` 0.126 → 0.248。意料之外的是 `carnivore_frac` 不降反升（0.088 → 0.130）：食草动物散开后种群更大，猎物总量增加；而食肉动物自身几乎没散开（`carn_water_dist` 仅 +17%），它们靠 `meat_water_frac` 取水并跟着猎物走。

### [现实] 诚实的限制：内陆自给不可能被彻底禁止

设计时想要的强不变量是"只吃草永远无法维生"。**这个不变量在任何有用的取值下都不成立**：一个穿越未开垦地的采食者剥离约 0.37 能量/步，`forage_water_frac` 超过约 **0.14** 即净为正。

但这在生态学上是对的，不是 bug。**内陆自给是低密度特权**——它依赖于内陆的草没被吃过，随着内陆填满、场被 draw down 而消失。所以 `test_forage_water_cannot_replace_drinking` 断言的是**平衡态场下**的不变量，而不是无条件的不变量。这个区分要保住：把它改成无条件断言就等于要求生态学上错误的行为。

---

## 4. 动物的空间记忆

视网膜按设计就是近视的：水通道在 `vision_radius` 处归零，加上扇区采样前置，有效感知上限约 30 世界单位，而地图离水距离的中位数是 **35.5**——超过一半的位置上，8 个水通道恒等于 0。站在内陆的 agent 无法区分"最近的河在北边 40 单位"和"在南边 140 单位"。

这不是应该靠放大感知半径抹平的缺陷。**真实动物同样看不见地平线以外；真实动物有的是记忆。**

### [现实] 短期：path integration（路径积分）

沙漠蚁维护一个持续更新的 **home vector**——回到起点的距离与方向。它们导航超过 **10,000 个体长**的距离，并能准确返回一个毫不显眼的洞口。这是短期、连续更新、随运动而积分的机制。

来源：
- *Spatial Memory in Insect Navigation*, Current Biology
- *Two distance memories in desert ants*, PMC

### [现实] 长期：数十年尺度的水源记忆

硬证据是 1993 年坦桑尼亚 Tarangire 干旱。Foley, Pettorelli & Foley,
*Severe drought and calf survival in elephants*, Biology Letters 4:541–544 (2008)
追踪了 21 个家族群、matriarch 年龄 14–45 岁、81 头幼崽，历时 9 个月：

| 族群 | 幼崽死亡 | 幼崽总数 | 死亡率 | 行为 |
| --- | --- | --- | --- | --- |
| B | 11 | 27 | **≈41%** | **留在**园区北部 |
| A | 4 | 39 | ≈10% | 多数家族**离开**园区 |
| C | 1 | 15 | ≈7% | 多数家族**离开**园区 |
| 合计 | 16 | 81 | **≈20%** | |

非干旱年的基线幼崽死亡率约 **2%/年**，故整体是约 **10 倍**、留守族群约 20 倍。
迁移与否显著（p=0.02），母亲年龄显著（p=0.03，母亲越年轻幼崽存活越低）。
作者的解释是 A、C 两族群的 matriarch 年长到足以记得上一次大旱——Tarangire
上一次同级干旱在约 35 年前，即不足 35 岁的 matriarch 没有那段记忆。

> **本文档曾记录一个错误版本**，称"由 33 岁 matriarch 带领的象群损失 20% 幼崽"。
> 已发表的结果是**族群层面**的（留守 vs 离开）加上**母亲年龄**效应，没有任何
> 已发表数字把 20% 归给某个具体年龄的 matriarch。20% 是全体的合计值。保留这条
> 更正是因为这个错误正是本文档 [现实]/[本世界实测] 分标记想要防的那类问题——
> 一个听起来合理、传播性强、但把统计层级搞错了的数字。

另一条独立证据：McComb, Moss, Durant, Baker & Sayialel, *Matriarchs as
repositories of social knowledge in African elephants*, Science 292:491–494
(2001)。回放实验显示 matriarch 越年长，家族区分熟悉/陌生叫声的能力越强，
防御性聚拢的反应也越恰当；关键在于 **matriarch 年龄可以预测该家族的人均
繁殖成功率**——这条把认知能力和适应度直接连上了。

大象的海马体异常大，约占脑容量的 **0.7%**，对比人类的约 0.5%。

### [现实] 工程上关键的一条：长期空间记忆不需要认知地图

Cruse & Wehner, *No Need for a Cognitive Map: Decentralized Memory for Insect Navigation*, PLOS Computational Biology (2011)：长期空间记忆**不必**实现为完整的认知地图，一组去中心化的向量就够用。

这一条是让整个机制可实现的原因。认知地图意味着某种可变尺寸的图结构，在定形张量的 jit kernel 里没法承载；一组固定数量的向量可以。**生物学在这里不是装饰，它直接决定了实现是否可能。**

### [对应] `underworld/memory.py`

- 短期层 = `brain.py` 里已有的循环隐状态（工作寄存器：刚刚发生了什么、我朝哪走）。
- 长期层 = `[n_max, memory_slots, 3]` 的槽位，每槽 `(dx, dy, strength)`。
- 向量**相对于持有者，不是绝对坐标**：每步减去位移并重新回卷成最短路径向量，环面因此只需被推理一次。永远不要从绝对坐标重算槽位。
- 槽位按位置分区而非打类型标记：`[0, memory_water_slots)` 是水，其余是果实。
- `memory_decay = 0.998` → 半衰期 346 步；到中位数距离水源的往返约 190 步，因此一个槽位大约撑两趟——长到值得拥有，短到不会让陈旧记忆坑死后代。
- `memory_drift = 0.25`：随机游走误差按 sqrt(n) 增长，200 步只累积约 3.7 单位，量级上正对应 ant-grade path integration。

实测效果（三种子 20000 步，提交 `cbe434d`）：`inland_frac` 0.126 → 0.382（+203%），`herb_water_dist` 14.51 → 35.19——几乎正是地图离水距离的中位数 35.46，即食草动物的空间分布相对于水已基本无偏。

---

## 5. 记忆不遗传：Weismann barrier

### [现实] 生殖质 / 体细胞的区分

August Weismann 在 1892 年的 *Das Keimplasma: eine Theorie der Vererbung*（《生殖质：一种遗传理论》）中提出：多细胞动物由**携带并传递遗传信息的生殖细胞系（germ line）**与**执行日常身体功能的体细胞（soma）**构成，前者"不朽"、后者"可弃"。遗传信息的流向是**单向的**：从生殖质流向体细胞，反向不成立。

这个单向性就是 **Weismann barrier**。它的直接推论是：**个体在一生中获得的性状无法传给后代**——这正是对 Lamarckian inheritance（获得性遗传）的否定。Lamarck 主张后天获得的特征（练出来的肌肉、失去的肢体）可以被继承；Weismann barrier 使这条路径在机制上不成立。

来源：
- [Weismann barrier — Wikipedia](https://en.wikipedia.org/wiki/Weismann_barrier)
- [Germ plasm — Wikipedia](https://en.wikipedia.org/wiki/Germ_plasm)
- Nilsson, Sadler-Riggleman & Skinner, *Environmentally Induced Epigenetic Transgenerational Inheritance and the Weismann Barrier: The Dawn of Neo-Lamarckian Theory*（[PMC7768451](https://pmc.ncbi.nlm.nih.gov/articles/PMC7768451/)）——表观遗传跨代传递确实对这条屏障提出了修正，但那讨论的是生殖系表观标记，与"把学到的空间记忆交给子代"完全不是一回事，不能拿来给后者背书。

放到这里：**基因是跨代的通道，记忆在个体生命内习得、随个体消亡。** 出生时把亲代的记忆槽拷贝给新生个体，就是 Lamarckian inheritance，生物学上是错的。

### [本世界实测] 而且它也没用

初版记忆系统确实在出生时按 `memory_inherit_frac` 折扣拷贝亲代槽位。删除它的两条理由同向：

1. 上面的原则。
2. 测量：n=6 种子配对实验，`inland_frac` 平均差 **+0.020**，而配对差的标准差是 **0.031**，6 个种子里还输了 2 个。这是一个 null result——效应量小于噪声，没有任何理由为它付出违反原则的代价。

两条同向时删除是容易的决定；值得记下来的是，如果它们方向相反（原则说不该有，但测量说明显有效），应当先怀疑测量而不是先放弃原则，因为"有效"往往意味着它绕过了某个本该被演化解决的问题。

### [对应] 正确的路线是社会学习，不是出生时移交

跨代传递知识的正确机制是 **social learning**：幼体跟随成体、**自己**喝到水、**自己**写下记忆槽。`memory.write` 的写入路径已经支持这件事——它只要求"此刻在水边"，不关心你是怎么到那儿的。缺的是让幼体有跟随成体的动机与能力，而这被 §7(c) 的感知问题挡住了。

> 注：`CLAUDE.md` 的"Memory is two tiers, and the long one is inherited"一节仍描述着已被删除的继承行为（`memory_inherit_frac`），该段落已过期，以 `memory.py` 与提交 `cbe434d` 为准。

---

## 6. III 型存活曲线（Type III survivorship curve）

### [现实] 三型分类

Deevey 在 *Life tables for natural populations of animals*（The Quarterly Review of Biology 22: 283–314, 1947）中把存活曲线形式化为三种类型（Pearl & Miner 1935 有先行的类似曲线）：

| 类型 | 死亡率模式 | 典型类群 | 生活史策略 |
| --- | --- | --- | --- |
| **Type I** | 幼年与中年死亡率低，死亡集中在老年 | 人类与多数大型哺乳动物（象、鲸） | K-selected：少产、亲代照料充分、单个后代存活率高 |
| **Type II** | 各年龄段死亡率大致恒定 | 多数鸟类、啮齿类、兔、部分龟 | 介于两者之间 |
| **Type III** | 幼年死亡率极高，活过幼年者其后死亡率大幅下降 | 树木、海洋无脊椎动物、鱼类、多数昆虫 | r-selected：多产、几乎不提供照料 |

关键的对应关系：**Type I 与亲代照料是绑定的**——照料少产的后代把幼年死亡率压下去，曲线才成为凸形。Type III 是相反策略：以数量对冲幼年高死亡率，不投资单个后代。

来源：
- Deevey (1947), *Life tables for natural populations of animals*, Q. Rev. Biol. 22:283–314（[PubMed 18921802](https://pubmed.ncbi.nlm.nih.gov/18921802/)）
- [Biology LibreTexts — Life History Characteristics](https://bio.libretexts.org/Courses/Evergreen_Valley_College/Introduction_to_Ecology_(Kappus)/06%3A_Life_History_and_Reproductive_Strategies/6.03%3A_Life_History_Characteristics)
- [Khan Academy — Life tables, survivorship curves & age-sex structure](https://www.khanacademy.org/a/life-tables-survivorship-age-sex-structure)

### [本世界实测] 本世界是教科书式的 Type III

平衡态下 3000 步，按年龄带统计每 1000 agent-步的死亡风险：

| 年龄带（步） | 风险 / 1000 agent-步 |
| --- | --- |
| 0–50 | 5.3 |
| 50–100 | 3.2 |
| 100–200 | 1.8 |
| 200–400 | 1.4 |
| 400–800 | 0.9 |
| 800–1600 | 0.4 |
| 1600–4000 | 0.3 |

幼体风险是老年风险的 **17.7 倍**，且**单调下降**；**全部死亡的 63% 发生在生命的头 50 步。**

### 本世界特有的成因：是记忆系统制造了这个压力

这个曲线的直接原因很可能是：**新生个体的记忆槽是空的，它不知道水在哪里。** 参照 §4——成体靠长期记忆才敢深入内陆，而幼体没有这份地图，出生即处于成体已经脱离的那个困境里。

值得注意的是：**这个劣势在两层记忆加入之前并不存在。** 记忆系统作为副作用，制造了亲代照料的选择压力——它让"幼体"第一次成为一个在适应度上真正不同的阶段。这不是设计出来的，是加了记忆之后测出来的。

---

## 7. 待补：亲代照料为何演化

**这是一个占位条目。** 一项独立调查正在进行中，涵盖：Hamilton's rule、Lack's principle、Trivers 的 parent-offspring conflict、为什么是哺乳类特别发展出照料、以及"照料要在模拟中可演化"所需的最小成分表。结论到位后追加到这一节，不要在别处另起。

已经确定的是三个前置条件——注意判据是**可演化（evolvable）而非被脚本写死（scripted）**，即照料必须是选择的产物，而不是我们直接编码的行为：

**(a) 幼体必须真的处于劣势。** ✅ 已成立，见 §6：幼体风险是老年的 17.7 倍，63% 的死亡发生在头 50 步。没有这一条，照料没有可回收的收益。

**(b) 亲代必须有帮助的手段。** ❌ 目前缺失。后代出生即独立，朝向随机（`reproduction.place`），亲代除了交出 `repro_cost_frac` 的能量之外没有任何可施加的影响。护送、引导、喂食、乃至只是"留在附近"，目前都不是可表达的行为。

**(c) 亲代必须能感知/识别后代。** ❌ 目前**不可能**。`sensors.py` 的 `prey_val` / `pred_val` 是从**食性差**算出来的：

```python
prey_val = closeness * jnp.maximum(di - diet_j, 0.0)   # j 更偏草食
pred_val = closeness * jnp.maximum(diet_j - di, 0.0)   # j 更偏肉食
```

两个食性相近的同种个体在这两个通道上**都恰好返回 0**，即**彼此完全不可见**。

**(c) 是阻塞项，而且阻塞的不只是亲代照料。** 同种个体互相不可见意味着**任何社会行为都不可能演化**：群居、跟随、社会学习（§5 指出的跨代知识传递正路）、领域行为、集群防御，全部依赖"能看见同类"这一前提。修 (c) 是社会性方向上的第一块地基，优先级应当高于 (b)。

修 (c) 的代价要提前知道：向视网膜增加通道会改变 `in_dim` → `genome_size`，**现有演化种群作废、从随机重启**（见 `CLAUDE.md` 的 config 一节）。因此这件事应当与其他需要作废的改动**打包在同一次**里做完，不要分两次付这个代价。
