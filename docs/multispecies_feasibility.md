# 多物种 / 多营养级 / 种间关系：本世界工程可行性

`docs/multispecies_ecology.md`（生物学依据）的姊妹篇：回答「在本代码库里加物种/营养级/种间机制，
具体动哪些代码、代价多少、哪条最小可行」。标记体例四标签，给 `file:行`。

---

## 1. 物种现在怎么表示（现状 + 头号约束）

**[对应]** 物种**不是离散标签，而是一根连续 `diet` 基因轴** [0,1]（0 纯食草，1 纯食肉）：
- `diet_of = sigmoid(genome[:, diet_index])`（`state.py:69-71`）；`diet_index = brain_params`，即第一个
  trait 基因，append 在脑权重块之后；所有 trait 基因排在脑块后（`config.py` 的 `*_index` 属性）。
- 缓存在 `WorldState.diet`，每步末从 genome 重刷（`step.py`）。

**[对应] 「二分」由六层机制焊住**（不是选择压自然维持，`docs/trait_evolution.md` §2.2、`experiments.md`）：
四层遗传反混合装置（`config.py`，各带 `--set NAME=0` 消融）：①双峰播种 `diet_bimodal_init`
（founder 播成 herb≈0.076 / carn≈0.88 两簇，`carnivore_init_frac=0.05`）；②重组豁免
`diet_crossover_exempt`（diet 恒取亲 A，`genome.py`）；③低突变 `diet_mutation_asymmetric`
（`diet_mutation_sigma=0.015` vs 脑 0.05）；④同型交配 `assortative_mating`（二亲按 diet 排序配对，
`reproduction._assortative_mate`）。两层生态装置（让中间型直接饿死）：⑤采食 `(1-diet)^6` 陡衰 + 硬
截断 `carn_graze_cutoff=0.75`（`dynamics._herbivory`）；⑥捕食阈值 `diet_delta=0.15`。

**[本世界实测] 头号约束——diet 轴中段是生态死区**：只关四层遗传装置（单峰起始）→ 六种子全灭绝
（419–987 步，99% 饿死）：diet=0.5 采食只剩纯食草的 1.56%，单峰全距 ±0.03 无一对满足 `diet_delta`。
**中间型既吃不了草也吃不了肉，直接死。** 这对下面所有「加物种」路线是硬约束。

## 2. 加第三个物种：路线与代价表

**成本原语**（`docs/trait_addition_feasibility.md` §A.2/A.3）：

| 动什么 | 形状后果 | 演化种群 | golden |
| --- | --- | --- | --- |
| **只动 `trait_dim`**（加一基因） | `genome_size +1`，脑拓扑/`WorldState` 形状不变 | 仅 founder RNG 重排致漂移 | 必须 `--bless` 重录 |
| **动 `in_dim`**（加一路 retina 通道） | `brain_params` 变 → 整个脑权重块重排 → **全演化脑作废重随机** | 是，硬作废 | 重 bless |
| **加一个 `[n_cells]` 生态场**（如 carrion） | `WorldState` 多一个 per-cell 场（同 fear/trample），**`reproduction.place` 不用改** | 否（不动 genome） | 重 bless + 生态近阈值重调、6 种子 |

**(a) diet 三峰**：最省形状（零 shape 改动，只改 `init_state` 播种），但第三峰只能落在 diet≈0.5 的
杂食/中营养——§1 已证此处直接致命。要活须先做独立生态重调（放平 `(1-diet)^6`、降 `diet_delta`）。
**判决：shape 最便宜，但被生态焊死；当前是死刑不是生态位。**

**(b) 独立「物种/生态位」基因维度**：便宜（`trait_dim +1`），得一根与营养级正交的物种轴。三个隐藏
代价：(i) 基因不接机制什么都不做；(ii) **retina 看不见与 diet 无关的物种标签**（prey/pred/peer 全由
diet 差/相似构造），要分辨第三物种要么复用 diet 派生交互（省）要么加通道（动 `in_dim` 作废全脑）；
(iii) 同型交配只按 diet 排序，新物种轴拿不到免费生殖隔离。**判决：「正确」的正交物种轴，但隔离与生态
分化都得自己补，不自维持。**

**(c) 新营养级 / 生态第三方**：
- **(c1) 腐食者 + carrion 场**：`[对应]` 本世界**当前没有尸体**（死亡只翻 `alive=False`，槽位立即
  被出生复用）。要喂腐食者须**新加 `carrion` [n_cells] 场**：`cull` 里把死者残余能量散射累加到其 cell
  → 按 decay 腐化 → 腐食者像 graze 取食。**fear/trample 同款 per-cell 场原语，`reproduction.place`
  无需改**；折进现有 food 通道（`sensors.sense` 的 edible）则 `in_dim` 不动。**最便宜的真·新营养级**，
  有真实分解者回路依据。「谁是腐食者」的身份仍需 (a) 生态重调后的 diet 带或 (b) 新基因。
- **(c2) 顶级捕食者吃食肉**：机制近乎免费（apex diet≈0.99 吃 carn 0.88，但 gap 0.11 < `diet_delta`
  0.15 须缩），但 `[本世界实测,推断]` 能量上几乎注定 NULL（carn 本就近灭绝阈值，第四级底座更薄）。
  **判决：快速证伪，不是真提案。**
- **(c3) 第二种食草竞争者**：`[对应]` 竞争已隐含（graze/eat_fruit 的 per-cell demand 池自动争食）。
  关键机会：**世界已有 grass 与 fruit 两个场**，当前被同一条 `_herbivory(diet)` taper 统一取食。一个
  「草效率↔果效率」权衡基因即可制造**资源分割/性状替代**，`trait_dim` 级便宜、不动 `in_dim`、不加场。

## 3. 种间关系机制挂在哪

**[对应]** 现有与可挂点：
- **捕食（已有）**：`dynamics.predation`。红皇后 attack/escape、armor/spike 都挂此。
- **竞争（已隐含，零代码）**：利用性竞争在 graze/eat_fruit 的 per-cell demand 池与 predation 的
  per-prey `wanted` 池天然发生。
- **互利/合作（无现成钩子）**，候选挂点：
  - **报警共享**：`fear` 场当前**只由食肉者沉积**、所有个体经 pred 通道读。让**猎物感知天敌时也
    沉积**到共享场，附近异种即可读危险——挂 `step.py` 沉积块 + `sensors` pred-fold，per-cell、不动
    `in_dim`。对应真实混种鸟群公共信息。
  - **稀释/自私羊群**：`predation` 里按局部（异种）密度缩放个体被捕食风险，复用邻居表。
- **retina 区分异种 vs 同种？** 只有 `peer` = diet 相似度（`sensors.sense`）。「异种」目前只能表达为
  diet 距离；与 diet 正交的物种轴对大脑不可见，除非加通道（动 `in_dim`）。

## 4. 最小可行提案（排序：形状代价低 × 涌现有趣 ÷ 生态风险）

1. **腐食者 + carrion 场（c1）——首推。** 加 `carrion` [n_cells] 场（fear/trample 原语，`place` 不改、
   不动 `in_dim`/genome），死亡沉积残余能量、腐化衰减、折进现有 food 通道取食；腐食身份用新 `scavenge`
   trait 基因或生态重调后的 diet 带。**真正的新能量通路/分解者回路**，把捕食者成功率与腐食者食物耦合。
   风险中（生态重调 + golden 重 bless；须验证不是白送食肉者第二个粮仓、抬高 carn_frac）。
2. **资源分割第二食草者（c3，草↔果权衡基因）——次选。** 最省（仅 `trait_dim`，无新场），吃现成
   plant/fruit 两场。可证伪预测干净（该基因是否双峰化=性状替代）。风险低，涌现中。
3. **顶级捕食者（c2）——只作快速证伪。** 机制近乎免费但按能量学预期 NULL；值得跑一次拿「四营养级
   底座太薄」的实测。
4. **独立物种基因 + 自带隔离 + retina 通道——不推荐做第一步。** 唯一真·正交物种轴，但要动 `in_dim`
   （作废全脑）或让物种隐形，最贵最难先见效。

## 5. 种间合作能涌现 vs 要脚手架（与 `multispecies_ecology.md` §5 同判决）

三条本世界先验（peer 通道无效、加脑证伪、社会学习未证实）一致指向：**纯涌现的跨种互利不现实作为首个
结果**。现实路线是**最小互惠脚手架**（硬编码 affordance，如共享报警场/稀释项），把「用不用」留给演化
的脑（Quinn 模式），**不硬编码合作行为本身**。脚手架必须制造当前不存在的**协调收益**——光加可见性
（peer）已证 NULL。

---

## 6. 三条关键发现（给下一步）

1. **diet 轴中段是生态死区**——任何落在 diet≈0.5 的第三物种当前直接饿死，是加物种的头号约束。
2. **世界无尸体机制**——腐食者需新加 carrion [n_cells] 场，但这是 fear/trample 同款便宜原语、不动
   `in_dim`。这是**最可能出有趣新动力学的最小结构改动**。
3. **retina 只能按 diet 距离看异种**——与 diet 正交的物种轴对大脑不可见，除非付 `in_dim`（作废全脑）
   的代价。故第一步优选「复用 diet 派生交互」的机制（腐食/资源分割），而非「独立物种轴」。
