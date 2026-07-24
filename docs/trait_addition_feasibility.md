# 加入可见形态『防御』性状的可行性判决：工程 + 演化 + 首个落地规格

这份文档回答用户诉求的后半段——**「加入这些性状做性状演化，可行吗？先做哪个、怎么验证？」**——
是 `docs/trait_defense_catalog.md`（候选菜单 + 真实生物学依据）的姊妹判决篇。候选清单在那份，
本文只判**可不可行、该不该做、首个做哪个**。

**一句话判决**：防御性状是**红皇后 `escape` 基因的同轴延伸**，与 escape 共用同一套已验证
安全的管线（能量账、`(1-diet)` 门控、慢突变、不豁免交叉），工程上是「再加一列 trait」，
演化上落在**成体向 A 类**且已有存在性证明——**建议做，且推荐首个落地为厚皮/减伤（armor）**。

**对接说明**：本文不重新论证以下已确立结论，只显式引用——代价货币原则
（`docs/trait_roadmap.md` §5）、交割期检验（`docs/trait_evolution.md` §11 / `docs/mortality.md`
§1.4）、红皇后 attack/escape 实测（`docs/attack_range_redqueen.md`）、`vision_radius` 几何
天花板负结果（`docs/vision_radius_heritable.md`）、通勤证伪（`docs/day_night.md` §6）。标记
体例沿用四标签（`docs/biology.md`）。

**当前状态校对**：`trait_dim=5`（`[0]diet [1]invest [2]size [3]attack [4]escape`，`config.py:54`）；
红皇后 attack/escape 已落地，新防御性状是 `trait_dim` 5→6、`*_index = brain_params + 5`。
协议已到 v7（72 字节头、每 agent 5 个 f32、`STRIDE=5`）。

---

# A. 工程可行性

## A.1 最小钩子集（成立）

[对应] 防御性状是纯 trait，复刻 escape 基因的每一处钩子（以 `armor` 为例）：

- `config.py`：`trait_dim` 5→6；新增 `armor_span`/`armor_cost`/`armor_mutation_sigma`/
  `armor_heritable` 常数；`@property armor_index → brain_params + 5`。
- `state.py`：新增解码器 `armor_of(genome, cfg)`，gene=0 → 0（中性无防御，仿 `escape_of`
  `state.py:113-124` 的 `clip(sigmoid-0.5, 0, None)`）。
- `genome.py`：`mutate` 给 `armor_index` 设 `armor_mutation_sigma`。**crossover 不改**——armor
  喂 `predation` 但**不进大脑感觉-运动回路**（脑不读 armor、armor 不改移动/控制），无控制器/
  身体错配，和 escape 一样**不豁免**（`genome.py:78-81` 只豁免 diet/size），保持 G 矩阵分析干净。
- `dynamics.py`：收益钩子挂 `predation`（减少猎物实际受损），代价税挂 `metabolize` 能量账。
- `step.py`：把 armor 传进 `metabolize`（predation 侧无需 step.py 改动——`predation` 已直接读
  `state.genome`，见 `attack_range_of(state.genome)` `dynamics.py:181`）。

**只动 `trait_dim` 不动 `in_dim`**：防御不新增任何感觉输入通道，`in_dim` 不变、脑权重块
`brain_params` 不变、现有每个脑基因偏移不动（`trait_index` 一律 append 在脑块之后，
`config.py:743-779` 的设计原语）。

## A.2 `trait_dim` vs `in_dim` 成本

[对应] 加 trait 只让 `genome_size` +1，脑拓扑不变、`WorldState` 形状不变（genome 是
`[n_max, genome_size]`，只加一列）。反例代价：若曾想让防御「上屏被脑看见」而加一路 retina
通道，则动 `in_dim` → `brain_params` 变 → 整个脑权重块重排 → 全演化种群作废且脑重新随机。
防御性状**不需要**这条——它是身体属性，不是脑输入。

## A.3 genome 作废与 golden 重 bless（固定契约代价）

[本世界实测/对应] genome 变宽 → `init_state` 的 `jax.random.normal(k_gen, (n, genome_size))`
（`state.py:155`）抽样形状变 → 即便 armor gene=0 逻辑无操作，founder RNG 重排导致**全下游
数值漂移**，golden 十项必然失配。**这是加任何 trait 基因不可躲的固定代价**，须 `--bless`
重录并在 commit 说明「trait_dim 变更导致种群作废，预期失配」，**不是放宽 band 掩盖失败**
（红皇后 trait_dim 3→5、vision trait_dim 3→4 均照此办）。

## A.4 可见性管线（三处改动）+ 真尖刺几何天花板（诚实指出）

上屏是**可选的、与科学正交的**一步，[对应] 三处 append：
1. `server/protocol.py`：`_HEADER` 不动；agent 记录尾部追加 `armor` f32（20→24 字节，`encode`
   的 agents 从 5 列→6 列）。按「append never insert」，追加在每 agent 5 f32 之后，不移动
   x/y/diet/energy/id 偏移。
2. `web/main.js`：`STRIDE = 5 → 6`；头偏移全不动。
3. `web/render.js`：新增顶点属性 `a_armor`，片元着色器把它画成 sprite 的深色/加厚描边。

**真尖刺几何的天花板（诚实）**：当前全种群是**一次 `gl.POINTS` draw call**（`gl_PointSize`
算出屏幕对齐方块）。**一个点无法长出朝外的真尖刺几何**——真尖刺需要 (a) 片元着色器里的程序化
尖刺纹样（alpha mask，便宜、仍在 POINTS 内），或 (b) 另建 instanced-mesh 管线（每 agent 一个
多边形，昂贵、不存在）。**这正是首推 armor 而非 literal spikes 的一条工程理由**：「厚皮」用
片元描边就够，不诱导人去踩 instanced-mesh 死胡同。shader 改动须对着 running server 截图验证
（`docs/conventions.md` §10），不能只读代码判对。

## A.5 要守的测试契约

[对应] 照 escape 的四类各加一条：
- **守恒** `test_predation_energy_not_created`：armor 通过缩小 `removed` 减伤，攻击方赔付
  `scale=removed/wanted` 同比下降，能量既不凭空造也账要平——必须重跑通过。
- **中性起点** `test_armor_gene_neutral_start_and_bounds`：gene=0 → armor=0 → 复现旧 `predation`
  行为；取值有界。
- **代价在能账** `test_armor_tax_hits_energy_not_water`：税在 `metabolize`，`thirst` 水公式
  （`dynamics.py:290`）一字不碰；`armor_heritable=False` 时税为零（干净对照）。
- **边界 + 交叉** `test_armor_recombines`：armor **不**豁免交叉（区别于 size 的 `test_size_gene_*`）。

## A.6 「加一个防御性状要动的文件清单」

| 文件 | 改动 | 作废种群? |
| --- | --- | --- |
| `underworld/config.py` | `trait_dim`+1、四个 `armor_*` 常数、`armor_index` 属性 | 是（`genome_size` 变） |
| `underworld/state.py` | `armor_of()` 解码器 | — |
| `underworld/genome.py` | `mutate` 加 armor 列 sigma（crossover 不改） | — |
| `underworld/dynamics.py` | `predation` 减伤钩子 + `metabolize` 能量税 | — |
| `underworld/step.py` | 把 armor 传进 `metabolize` | — |
| `underworld/metrics.py` | append `mean_armor`/`armor_std`/`herb_armor`/`carn_armor` | — |
| `tests/test_kernel.py` | 上述四类测试 | — |
| `scripts/golden.json` | `--bless` 重录（A.3 契约代价） | — |
| **(可选上屏)** `protocol.py`+`main.js`+`render.js` | 追加 armor f32 + STRIDE + 着色 | — |

---

# B. 演化可行性（判决核心）

## B.1 代价货币原则：防御代价必须记能账

[对应] `docs/trait_roadmap.md` §5：代价记在**水账**（`dynamics.thirst`）的性状会自动继承
体型基因的失败结构（幼体渴死删失收益、方向反转）；记在**能账**（`dynamics.metabolize`）且被
`carn_cost`/`escape_cost` 验证安全。

- **有天然能账代价的防御**：厚皮/尖刺/角/骨甲的合成与维持是代谢开销——[现实] 结构性防御
  组织有真实的能量建造/维持成本。落到 `metabolize` 的一项 `armor_cost * armor * (1-diet)`
  完全自然，[对应] 与 `escape_cost * escape * (1-diet)`（`dynamics.py:261-262`）逐字同构。
- **容易被图省事写进水账的防御**：一旦把「厚皮=更大体重=更多失水」或直接耦合到 `size`
  （size 已按 `size^0.75` 抽水账，`dynamics.py:301`），防御代价就悄悄落回水轴——即便设计者
  自以为「配了代价」也踩了体型基因同一个坑，而且更隐蔽。**判据**：防御税只准出现在
  `metabolize`，`thirst` 公式不得因防御性状增加任何一项。

## B.2 A/B 交割期检验：防御是成体向 A 类（并正面处理与 catalog 的张力）

[本世界实测] 死因年龄结构（`docs/mortality.md` §1.2）：渴死均龄 **52.5 步**（幼体）、被捕食
均龄 **170.7 步**（成体）。

- 防御的收益**在捕食暴露的成体年龄段兑现**（170 步），这一段**本就在幼体渴死瓶颈之后**。
  受益者是「已经活过第一关」的个体，**不像 `vision_radius` 要求从出生第一步就在 52 步窗口内
  直接和渴死竞争、且被几何天花板锁死**（`vision_radius_heritable.md`：0.33 单位头顶空间 ≪
  15.5 单位水缺口，近中性负结果）。
- **区别体型/视野的失败模式**：size/vision 是**方向反转/结构不可用**（B 类或几何证伪）；防御
  只是**竞争性风险删失稀释统计功效**（只有活到成体的 ~45% 表达选择差），**方向不反转**。
- **杀手论据**：escape 基因面对**同一套删失**（渴死仍占 ~55%）却**仍然干净演化出来**（1.91 vs
  0.50，p=0.031，`docs/attack_range_redqueen.md`）——成体捕食这条选择通道已被实测证明有足够
  功效推动一个防御基因。防御不是猜测能过删失，是**同类基因已经过了**。

**⚠ 与 `docs/trait_defense_catalog.md` §1 的张力，正面处理**：那份文档（据真实生物学，
barrett2008）指出护甲的现实代价是**生长/发育预算**、落在幼体期，按分诊会滑入 **B 类**。这不
与本节矛盾——**它恰恰是代价货币原则的活教材**：护甲的 A/B 归类**不是「护甲」这个概念固有的，
取决于实现时把代价记进哪本账**。本文的 A 类判决**成立的前提是采纳 B.1 的设计**（代价记
metabolize 能量税，成体持续付），而**不是**照搬真实生物学的生长账。如果实现时图方便把护甲
代价写成「拖慢生长」或耦合到 size（=水账），它就滑回 B 类、重演体型基因方向反转。**这条张力
是一个必须在实现时明确做出的设计决定，不是已裁决的结论**——落地实验（B.6）才能确证能量账
版本确实留在 A 类。

## B.3 红皇后先例作为存在性证明

[本世界实测] `docs/attack_range_redqueen.md`：escape（猎物防御性状，减少有效攻击距离）在
双侧臂真实演化——逃逸 **1.91 vs 单侧中性漂变 0.50**，6/6 同向，**p=0.031**；逃逸滞后攻击
~3500 步；把有效攻击距离从 6.96 压回 6.28（≈起点）；捕食者占比减半（25%→13.7%）。**这证明
本世界的捕食维度能承载可测量的协同演化**。厚皮/尖刺是**同一维度（猎物抗捕食）的另一条轴**：
escape=让更少的咬命中（减遭遇），armor=让每口咬伤更少（减伤/耐受）。两者正交，合起来把「防御」
分解成「躲开 vs 扛住」。

## B.4 动态选择压对比：防御不会重蹈通勤证伪

[本世界实测] `docs/day_night.md` §6：「演化出空间通勤」证伪（2 尺度、脑 16/24/32、2 种子
一致 NULL），根因**「静态地形→节律可被固定策略吃掉→不奖励内生时钟这种二阶结构→缺动态选择压」**。

对比：**捕食是活的、生物性的、协同演化的选择压**。[现实] Van Valen(1973) 红皇后的定义性质
就是**选择的来源本身在演化**——捕食者的 attack 基因每代都在变，猎物的防御面对的是一个**由
另一个受选择种群不断再生的移动靶子**，而非可被固定策略吃掉的静态环境节律。**这正是通勤 null
所缺的动态选择**。所以防御有真实、持续、内生的选择来源；escape 的成功（B.3）已把这条通道
验证过一遍。

## B.5 runaway 风险与 diet 门控

[对应] 若防御不配代价 → 无下游权衡 → 基因无约束跑向上界，是「不可证伪的基因饱和」（roadmap
§7.4 对代谢率的同一警告），不是交割期问题。**能量税 `armor_cost*armor*(1-diet)` 把它变成
权衡 → 内点均衡**。

**diet 门控**：防御只对**猎物**（食草者，低 diet）有收益——捕食者极少被咬（需 `diet_delta`
食性差，`dynamics.py:195`），其 armor 基因中性漂变、应付 ~0 税。`(1-diet)` 缩放让食肉者的
防御基因免税自由漂变，与 escape 的 `(1-diet)`、attack 的 `diet` 缩放对称（`dynamics.py:260-262`）。

**可证伪预测**：
- (P1) `herb_armor` 均值从中性 0 **上升**，当且仅当 armor 有功能且捕食是真威胁；对照臂
  （`armor_heritable=False`）近 0 漂变——照抄 escape 的 1.91 vs 0.50 存在性判据。
- (P2) 随 `herb_armor` 上升，`death_predation_frac` 和/或 `carnivore_frac` **下降**（捕食者
  在带甲猎物身上净能量下降 → 密度中介减半，与红皇后同签名）；**不必是 `hunt_success` 下降**
  （`attack_range_redqueen.md`：防御效应表达在密度轴，非单次命中率）。
- (P3) `carn_armor` 均值与其税**都近 0**（diet 门控正确）；若 `carn_armor` 也显著上升 = 门控漏了。
- **单侧/双侧协同的方差判据**：单侧（仅猎物 armor、捕食者 reach 固定）→ armor 漂到**税决定的
  平台、低方差**；双侧（armor 与 attack 同演化）→ 第二条红皇后轴。**诚实预告**：按红皇后已
  证伪的教训，线性能量税天然收敛到稳定均衡，**预测防御性状同样不出现「方差越拉越大的失控军备
  竞赛」**——不要重犯 roadmap §7.3「双侧方差显著更高」那条被证伪的预测。

## B.6 6 配对种子测试设计

- **臂**（基因组布局相同、可直接配对）：
  - 主消融：`armor_heritable` **on/off**（armor 有功能+计税 vs 存在但中性漂变、免税）——回答
    「让防御可遗传是否改变任何东西」。
  - 可选交叉：与 `attack_range_heritable` 交叉，得**单侧**（仅 armor 演化）vs **双侧**
    （armor+attack 同演化）= 测第二条红皇后轴与方差判据。
- **报什么指标**：`herb_armor`/`carn_armor`（存在性+门控）、`death_predation_frac`、
  `carnivore_frac`、`population`、`mean_age`、`hunt_success`，给**每种子**数字（`--json`）。
- **护栏**：`carn_frac` 不灭绝不爆炸（照 `attack_range_redqueen.md` §5）。
- **统计纪律**：6 配对种子起（n=6 配对 Wilcoxon 的 p 地板 0.031），配对 Wilcoxon + 10000 次
  bootstrap 95% CI，**报告算过的每个 p**，**不做 Bonferroni**；伪重复诚实标注（`terrain.build`
  无 RNG，6 种子同一张地图 → 结论只对**这一套河系**成立，推广需交叉 `ridge_wavenumber` 等
  地形种子）。

---

# C. 结尾：推荐首个落地防御性状 + 逐文件规格草稿 + 3 条预测

## C.1 推荐：**厚皮/减伤（armor）**，而非 literal 尖刺

[提案，非结论] 三条理由：

1. **最便宜（且最安全）**：是已上线、已实测的 `escape` 基因的**结构孪生**——同能量账、同
   `(1-diet)` 门控、同慢突变、同不豁免交叉、同 `metabolize` 税写法。代价货币判决被先例预先
   裁定，新论证量最小；落地只碰 `predation` 的每猎物减伤 + 一行 `metabolize` 税。
2. **足够可见，且避开几何天花板**：片元着色器给现有 `gl.POINTS` sprite 加深色/加厚描边即可
   读作「带甲」——不逼你去建 instanced-mesh。承认 literal 尖刺更醒目，但真尖刺几何是 A.4
   点名的死胡同。
3. **演化信号最干净**：armor 的收益**直接作用于携带者本身**（自己每口咬伤更少 → 更可能活过
   这次捕食），是**无歧义的个体级选择差**；而 literal 尖刺「反伤」是对捕食者的伤害、携带者
   当场不受保护，收益偏**亲缘/群体级**（靠 `spawn_radius=3` 的 kin 聚集才可演化），更易得到
   「漂到零」的 null 且难解释。armor 给最干净的存在性判据，又与 escape 是**不同轴**（躲开 vs
   扛住）。

## C.2 逐文件规格草稿（仿 `docs/trait_roadmap.md` §8 粒度）

[提案，非结论]

**`underworld/config.py`**
- `trait_dim: int = 5 → 6`。
- 在 escape 常数后新增：
  ```python
  armor_span: float = 1.0    # armor_of maps gene to [0, ~0.5]: fraction of a bite's
  #                            energy damage negated. Neutral (gene 0) = 0 EXACTLY --
  #                            a fresh population has no armor, any armor is evolved,
  #                            the clean baseline (mirrors escape_span's one-sided form).
  armor_cost: float = 0.012  # energy/step per unit armor, scaled by (1-diet) so
  #                            herbivores (the hunted) pay and carnivores' armor gene
  #                            drifts neutrally. Symmetric to escape_cost. NEVER thirst.
  armor_mutation_sigma: float = 0.02   # slow trait rate, same as size/escape.
  armor_heritable: bool = True         # False: predation ignores the armor gene and
  #                            levies no tax; the gene still exists/drifts/reports --
  #                            the clean control arm, genome-compatible with True.
  ```
- `@property armor_index(self) -> int: return self.brain_params + 5`。

**`underworld/state.py`** — 仿 `escape_of`（`state.py:113-124`）新增，gene=0 → 0：
```python
def armor_of(genome, cfg):
    # Damage-reduction fraction in [0, armor_span/2]; 0 at gene=0 (no armor seeded).
    return cfg.armor_span * jnp.clip(jax.nn.sigmoid(genome[:, cfg.armor_index]) - 0.5,
                                     0.0, None)
```

**`underworld/genome.py`** — `mutate`：`sigma = sigma.at[cfg.armor_index].set(cfg.armor_mutation_sigma)`。
`crossover` **不改**（armor 不进感觉-运动回路，同 escape 不豁免）。

**`underworld/dynamics.py`**
- `predation`：在 `wanted`/`removed` 处按**猎物** armor 缩小实际受损（能量账；水账保持不动
  作为「甲挡牙不挡渴」的最小首版，注明这是设计取舍）：
  ```python
  if cfg.armor_heritable:
      armor_slot = jnp.concatenate([1.0 - armor_of(state.genome, cfg), jnp.ones(1)])
      wanted = wanted * armor_slot          # per-prey-slot, dump slot unarmored
  removed = jnp.minimum(wanted, prey_e)     # scale=removed/wanted → payout同比降，守恒
  ```
- `metabolize`：加可选参数 `armor`，`tax += cfg.armor_cost * armor * (1-diet)`，由
  `armor_heritable` 门控，逐字仿 escape 税（`dynamics.py:261-262`）。

**`underworld/step.py`**：计算 `armor = armor_of(...)` 传给 `metabolize`（predation 侧直接读
`state.genome`，无需 step 改动）。

**`underworld/metrics.py`**：append `mean_armor`/`armor_std`/`herb_armor`/`carn_armor`（仿
carn/herb 掩码，`metrics.py:141-185`），wire 经 `_asdict()` 自动可用，首版只需
`run_headless --json` 读到。

**`tests/test_kernel.py`**：A.5 四类。**`scripts/golden.json`**：`--bless` 重录（A.3）。

## C.3 三条可证伪预测

[提案，非结论]
1. **存在性**：`herb_armor` 从中性 0 显著上升（功能臂），`armor_heritable=False` 对照臂近 0
   漂变——量级/判据照抄 escape 1.91 vs 0.50、6/6、p=0.031。
2. **密度签名**：`death_predation_frac`/`carnivore_frac` 随 `herb_armor` 上升而下降，**不必
   伴随 `hunt_success` 下降**；`carn_armor` 及其税近 0（diet 门控成立）。
3. **协同 + 方差判据**：双侧（armor 与 attack 同演化）出现第二条红皇后轴，「每口有效伤害」
   被压回基线；**但线性能量税→稳定均衡，预测无方差发散**（主动规避 roadmap §7.3 已被证伪的
   「双侧方差更高」）。
