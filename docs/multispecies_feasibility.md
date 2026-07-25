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

---

## 7. 腐食通路首个落地（carrion + scavenge，2026-07-25，run_id: 20260725-carrion）

按 §4 首推落地了**腐食营养通路**（不是新物种，是给食肉者一条新取食通道——最小结构、最安全）：

- **[对应] 机制**：新增 `WorldState.carrion:[n_cells]` 场（fear/trample 同款 per-cell、`reproduction.place`
  无需改）。`step.py`：cull 后按 `carrion_per_death * size` 把新死者的尸体沉积到其 cell，逐步
  `carrion_decay` 腐化（沉积-下步读取范式）。`dynamics.scavenge`：食肉者（skill=diet）在所在 cell
  取食 carrion（每格公平分池，仿 graze），得能量+水（`carrion_energy`/`meat_water_frac`）。折进现有
  取食阶段，**不加 retina 通道（食肉者踩到才吃、不主动找）→ 不动 `in_dim`、不动 genome**。
- **[对应] 安全性**：`carrion_enabled` 默认 **False** → 整条分支编译期消掉、carrion 恒 0 →
  **bit-exact 旧世界，golden 不动、无需重 bless**（同 L6/armor-off 的纪律）。`--set carrion_enabled=True`
  开启。测试：默认关 carrion 恒 0；开启时食肉者取食 carrion、食草者≈0、carrion 被消耗。
- **[本世界实测] 早期信号**（单种子 4000 步，非结论）：ON vs OFF——carrion_total 累积到 256（机制在
  开火）、**carn_frac 0.043→0.072**（腐食给食肉者第二食源、抬高其数量）、pop 略降。**这可能正好缓解
  本项目长期的"捕食者近灭绝阈值"痛点**（`carnivore_riparian.md`）——腐食是捕食者的抗灭绝缓冲。

**演化验证判据（先写后跑，run_id 20260725-carrion）**：
- **假设**：开启腐食通路给食肉者一条抗饥缓冲，**提高捕食者存活/占比、降低其灭绝风险**，且不破坏
  食草侧或渴死平衡。
- **成功判据**：6 配对种子 ON(`carrion_enabled=True`) vs OFF——`carnivore_frac` ON > OFF，6/6 同向、
  配对 Wilcoxon p≤0.05；且 `carn_frac` 各种子 min 抬高（抗灭绝）；护栏：`population`/`death_thirst_frac`
  不显著恶化。
- **失败/负结果**：若 carn_frac 不升或食草/渴死恶化，记录并判断是"腐食被食草者搭便车"还是"白送食肉
  第二粮仓致爆炸"。
- **结果**（6 配对种子 ×20000 步，出处 `outputs/20260725-carrion/`，统计
  `explorations/20260725-carrion/analyze.py`；n=6 地板 p=0.031）：

  | 指标 | OFF | ON | 配对差 | 同向 | 配对 Wilcoxon |
  | --- | --- | --- | --- | --- | --- |
  | carrion_total | 0 | 204 | +204 | 6/6 | —（机制在开火） |
  | carnivore_frac | 0.127 | 0.168 | +0.041 | **4/6** | W=5.0 **p=0.31** |
  | population | 1849 | 1733 | −116 | 2/6 | p=0.22 |
  | death_thirst_frac | 0.506 | 0.461 | −0.045 | 1/6（ON 更低=更好） | p=0.094 |
  | carn_frac 逐种子 | ON [0.18,0.13,0.215,0.207,0.193,**0.084**] / OFF [0.13,0.091,0.097,0.091,0.203,0.151] | min OFF 0.091 → ON 0.084 | | |

- **结论**：
  - **[本世界实测] 机制验证通过、但抗灭绝假设未证实。** carrion 场累积到 204（6/6，腐食通路确实在
    运转、食肉者在吃尸体）；`carnivore_frac` 均值抬高（0.127→0.168）但**仅 4/6 同向、配对 Wilcoxon
    p=0.31（远未过地板 0.031）**，且**最低种子不升反微降**（0.091→0.084）——**没有兑现"腐食缓冲捕食者
    近灭绝"的成功判据**（既非 6/6、也没抬 min）。
  - **[本世界实测] 护栏守住。** `population` 稳定（p=0.22）、`death_thirst_frac` 甚至略降
    （0.506→0.461，p=0.094）——腐食带的水分（`meat_water_frac`）让食肉者少跑河边，是个小的正面副作用，
    不是恶化。没有"白送第二粮仓致爆炸"。
  - **[对应] 诊断**：效应弱大概率因**食肉者踩到才吃、不主动找尸体**（首版不接 retina）——碰上 carrion
    的概率低，补贴稀薄。carrion_total 204 说明尸体在积压、没被高效消费。
  - **[提案，非结论] 判决与后续**：腐食通路是**可用的、安全的新营养通道**（默认关、bit-exact、护栏
    守住），但**当前形态的捕食者收益弱到不显著，不足以默认开启**。要放大效应的两条路：①把 carrion
    折进 food retina 让食肉者**主动找尸体**（动 `in_dim`、作废全脑，代价大，但直接解决"碰不到"）；
    ②加独立 `scavenge` trait 基因做**真·腐食者物种**（trait_dim+1，专食腐位、与 diet 正交），可能演化
    出专门吃尸体的谱系。二者都需再一轮 6 种子。**当前默认保持关闭。**

**[提案，非结论] 后续**：若验证为正，可考虑（a）把 carrion 折进 food retina 让食肉者**主动找尸体**
（动 `in_dim`、作废全脑，代价大，仅当基础版证明价值后）；（b）加独立 `scavenge` trait 基因做成
**真·腐食者物种**（trait_dim+1，与 diet 正交的专食腐位）。可视化上腐食者聚到死亡点是天然的"新物种"
观感（`server/protocol.py` 可加一路 carrion 场着色，仿昼夜/水层）。

### 第二版：接 retina 主动觅食（2026-07-25，run_id 待定：`20260725-carrion-v2`）

第一版的诊断（§7 上文 line 149）是"效应弱因食肉者**踩到才吃、不主动找**、碰上概率低、补贴稀薄"。
第二版直接修这一点——让食肉者能**沿 food 通道朝尸体转向**。

- **[对应] 关键差异 = 接 retina 主动觅食，且未付 `in_dim` 代价。** 首版曾担心"接 retina 就要动
  `in_dim`、作废全脑"（line 153/157），实测**不必**：`sensors.sense` 的 food 通道本就把 plant 和
  fruit 融在同一路（`edible = plant + fruit_energy*fruit`，再 `/plant_max`），fruit 就是"以边际食用
  价值加权折进 food、不占第六通道"的先例。第二版照抄这条纪律——在 `if cfg.carrion_enabled:` 内把
  `carrion_visible_scale * carrion_energy * diet * carrion[cells]` 加进 `edible`：**carnivore（diet≈1）
  看得见尸体、herbivore（diet≈0）看不见**（diet 加权），于是食肉者能朝前方尸体转向。**不加 retina
  通道、`in_dim` 恒 67、genome_size 恒 1385、不作废演化种群**——比首版设想的"作废全脑"便宜得多。
- **[对应] 安全性同首版**：融合 gate 在**现有** `carrion_enabled`（默认 False）后——关时 food 采样只有
  plant+fruit、逐位复现旧世界、**golden 不动、无需重 bless**。测试
  `test_carrion_food_channel_off_is_bit_exact`（关时 carrion 对 food 通道 bit-exact 无影响）+
  `test_carrion_food_channel_draws_carnivore_not_herbivore`（开时食肉者 food 读数升、食草者≈不变）。
  新增旋钮 `carrion_visible_scale`（默认 1.0，只在开启时活）控制 carrion 在 food 通道里的可见权重。

**演化验证判据（先写后跑，run_id `20260725-carrion-v2`，本会话只写设计、不跑 6 种子）**：
- **假设**：让食肉者**主动找尸体**后，carrion 补贴不再稀薄——食肉者碰上尸体的概率从"随机撞上"升到
  "定向觅食"，于是 §4 的第二食源真正兑现，`carnivore_frac` ON>OFF 比首版**更硬**：
  期望从首版的 **4/6 同向、p=0.31** 抬到 **6/6 同向、配对 Wilcoxon p≤0.031（n=6 地板）**，且
  **最低种子 min 抬高**（首版 0.091→0.084 不升反降，本版期望 min 上移＝真正的抗灭绝缓冲）。
- **成功判据**（照 §7 首版格式）：6 配对种子 ON(`carrion_enabled=True`，含 retina 融合) vs
  OFF（`carrion_enabled=False`）——① `carnivore_frac` ON>OFF，**6/6 同向**、配对 Wilcoxon **p≤0.031**；
  ② `carn_frac` 各种子 **min 抬高**（抗灭绝硬指标）；③ 护栏：`population`/`death_thirst_frac` 不显著
  恶化；④ 机制在开火：`carrion_total` ON>0，且**较首版更快被消耗**（主动觅食→尸体不该像首版那样积压到
  204，应更低＝被更高效吃掉）。
- **失败/负结果预案**（照首版）：
  - 若仍 4/6、p 不过地板 → 判定"主动找也救不了"：说明瓶颈不在觅食效率而在**尸体供给量**（死亡率×
    `carrion_per_death` 太低），下一步该调 `carrion_per_death`/`carrion_energy` 或换第二版失败即
    转 §7(b) 独立 scavenge trait，而非再堆 retina。
  - 若 carn_frac 6/6 升但 `population` 显著跌或 `death_thirst_frac` 恶化 → "白送第二粮仓致食肉爆炸、
    压垮食草基座"，回退默认关。
  - 若 `carrion_total` 不降反升 → 融合权重 `carrion_visible_scale` 太低、食肉者其实没看见，先扫这个旋钮
    再下结论。
- **对比首版的意义**：第一版证明了"机制安全、但被动觅食补贴太稀薄"；第二版是**同一机制的觅食升级**，
  隔离变量恰好是"主动 vs 被动"（首版=对照）。若第二版 6/6 过而首版 4/6 不过，则**"接 retina 主动觅食"
  正是首版弱阳性的成因**，且这一升级**零 `in_dim` 代价**——是本条腐食通路能否默认开启的关键一跳。
- **[本世界实测] 早期点火证据（单种子 seed 0 ×4000 步，非结论，仅证"主动找"在动）**：v2-ON
  `carrion_total=224`、`carnivore_frac=0.052`，对比首版记录的单种子 ON `carrion_total=256`、
  `carn_frac=0.072`（§7 line 118）。**carrion_total 224<256 = 尸体积压更少、被消费更快**，方向与
  "主动觅食→尸体不再堆积"一致（弱证据）；而 `carn_frac` 单种子 0.052<0.072 落在噪声内——CLAUDE.md
  明言捕食者存活近阈值、run-to-run 方差超过多数参数效应，**单种子的 carn_frac 差异毫无判决力**，
  必须 6 种子。此处只证机制在动、方向不反，判决留给上面的 6 种子设计。

**[本世界实测] 结果（6 配对种子 ×20000 步，run_id `20260725-carrion-v2`，出处 `outputs/20260725-carrion-v2/`，
统计 `explorations/20260725-carrion-v2/analyze.py`；HEAD `c25ba60`；n=6 配对 Wilcoxon 地板 p=0.031）：
**弱阳性——比首版进步，但未达"更硬"标准，维持默认关。**

| 指标 | OFF | ON | 配对差 | 95% bootstrap CI | 同向 | 配对 Wilcoxon |
| --- | --- | --- | --- | --- | --- | --- |
| carnivore_frac（末态） | 0.143 | 0.175 | +0.032 | [+0.007, +0.057] | **4/6** | W=3.0 **p=0.156** |
| late_carn（后 1/4） | 0.141 | 0.161 | +0.020 | [−0.002, +0.039] | 5/6 | p=0.156 |
| carn_frac min（抗灭绝） | 0.0975 | **0.1273** | +0.030 | — | — | min **抬高** |
| carrion_total | 0 | 198.2 | +198 | [+180, +218] | 6/6 | 机制在开火 |
| population | 1798 | 1718 | −80 | [−155, +0.3] | 1/6↑ | p=0.094（弱负苗头）|
| death_thirst_frac | 0.498 | 0.443 | −0.055 | [−0.089, −0.019] | 1/6↑（ON 更低=有益）| p=0.063 |

- **预注册的"更硬"（6/6 同向、p≤0.031）没兑现**：carn_frac 仍 4/6、p=0.156，靠 seed0/1/4/5 撑均值，seed2/3 反向。
  差值 +0.032 与种子间 SD（0.03–0.04）同量级——近灭绝阈的高方差正是 4/6 而非 6/6 的原因（CLAUDE.md 警告）。
- **唯一实打实的改进是 min 抬高**：carn_frac 种子间 min 0.0975→0.1273（late_carn min 0.103→0.144）——首版是
  min **反降**（0.091→0.084），本版首次让最差种子也上移，**抗灭绝缓冲这次成立**。p 也从首版 0.31 收到 0.156。
- **"主动觅食→尸体更快消费"基本证伪**：`carrion_total` ON 198 vs 首版 204，仅低 ~3%、实质持平——积压水平没变。
  注意 carrion_total 是**驻留场快照（积压量）非吞吐量**，测不出单位尸体截获率；要判"主动觅食是否真生效"需一个
  消费流量计（累计 `scavenge_gain` 或食肉者 food 通道 carrion 读数），当前指标反推不出。
- **护栏**：death_thirst ON 下降（腐食供水，方向有益）；但 `population` 5/6 降、p=0.094，「第二粮仓压食草基座」
  的弱苗头存在、未显著。
- **[提案，非结论] 判决与后续**：v2 优于 v1（min 首次抬高 + p 收窄 + late_carn 5/6），但**主判据未过、carrion
  消费未加速、population 偏负**，**不够格默认开启，维持默认关**（同 v1）。若要把 p 推过线：按 §5 功效算术需
  ~20+ 配对种子（差值与 SD 同量级），补 seed 6–17 到 ≥12 配对；诊断"主动觅食是否生效"需加消费流量计。

---

## 8. 共享报警场（alarm，2026-07-25，落地未跑，run_id: 20260725-alarm）

按 §5 的判决落地了**用户 #5「种间合作」的最小互惠脚手架**（Quinn 模式）。这不是一个新物种，也不是一条
新能量通路，而是一个**当前不存在的协调收益**：把「某个个体看见了捕食者」变成邻居（含异种）可学习的
空间危险信号，**沉积是硬编码 affordance、响应留给已演化的脑**。

### 8.1 为什么是「报警」而不是别的合作

**[本世界实测/对应] 三条同向先验指向「纯涌现跨种互利不现实作为首个结果」**（§5 与 `multispecies_
ecology.md` §5）：peer 通道近乎无效（3×6×20k，`mean_size` 0.725 vs 0.737）、加脑容量在静态地形下二阶
适应测不出、社会学习本身从未在本世界被证实。故正路是**脚手架**：制造一个协调收益的 affordance，把
「用不用」留给脑。**光加可见性（peer）已证 NULL**，所以脚手架必须提供 peer 之外的东西——**别人的
观察结果**（公共信息），这正是混种鸟群/哨兵系统的可迁移核心（`multispecies_ecology.md` §4）。

### 8.2 机制 [对应]

- **新场**：`WorldState.alarm:[n_cells]`（fear/trample/carrion 同款 per-cell 原语；`reproduction.place`
  无需改——它对 rank-any 场通用）。**刻意独立于 `fear`，不复用**：`fear` 记录「**捕食者站过哪**」（由
  食肉者沉积），`alarm` 记录「**猎物在哪看见了捕食者**」（由受惊猎物沉积）。两者语义不同，混用会把
  「威胁在此」和「有人示警」搅在一起。
- **沉积（`step.py` 7a''' 块，`alarm_rate>0` 编译期分支）**：猎物（`diet<0.5` 且 `alive`）在**感知到
  附近天敌**时，往其所在 cell 沉积 `alarm_rate`，逐步 `alarm_decay` 腐化、clip 到 [0,1]（沉积-下步读取
  范式，完全仿 fear）。「感知到天敌」= 用 post-movement 邻居表（`nbr2/dist2/valid2`，与 predation 同一
  张表）**原始重建** `pred_seen = max_j closeness*(diet_j-diet_i)`——与 sensors 的 `pred_val` 同构，但
  **取自任何 fear/alarm 折叠之前**，故 alarm 不会自我喂食成 runaway；超过 `alarm_pred_threshold` 才沉积。
- **读取（`sensors.sense` pred-fold，`alarm_rate>0` 编译期分支）**：把采样 cell 的 `alarm` 乘
  `alarm_sense_scale`，**max 折进现有 pred 通道**——与 fear 折叠**同一行下方**。于是「别人往那边看见了
  捕食者」成为任何 diet（含异种、含不同 diet 猎物）都能读到的危险信号，**不加 retina 通道、不动
  `in_dim`/genome**，演化种群照常加载。
- **默认关 & bit-exact [对应]**：`alarm_rate=0.0`（默认）→ 沉积块与折叠**同为编译期分支**、整段从 jit
  trace 消失、`alarm` 场恒 0 → **逐位复现旧世界、golden 不动、无需重 bless**（同 fear_rate=0 /
  carrion_enabled=False 的纪律）。开启：`--set alarm_rate=0.05`（配套 `alarm_decay=0.99`、
  `alarm_sense_scale=3.0`、`alarm_pred_threshold=0.05`，均照 fear 同名参数取值）。
- **测试**（`tests/test_kernel.py`）：①默认关 alarm 恒 0，且两组不同 alarm_decay/scale/threshold 但
  rate=0 的运行**逐位相同**（`array_equal` 于 alive/pop/pos/energy，证 off-branch 真从 trace 消失）；
  ②开启时猎物沉积（场非零、capped ≤1、`alarm_total` 指标 >0），且无新沉积时纯 `alarm_decay` 衰减；
  ③开启时 alarm 确实 max 进 pred 通道（同 diet 邻居使 raw pred=0，故 pred 通道激活只能来自 alarm 折叠）。

### 8.3 点火验证 [本世界实测]（单种子 4000 步，非结论）

`run_headless.py 4000 --set alarm_rate=0.05 --json`：`alarm_total` 累到 **1570**（机制在开火、沉积-
衰减达稳态），`carrion_total` 仍 0（互不干扰），世界没崩（pop 1980、carn_frac 3.7%）。check.py 默认 tier
golden 10 指标全过（默认关 bit-exact）；68 项 pytest（含 3 项新 alarm 测）全过。**这只证明机制在运转，
不证明它有生态效应。**

### 8.4 演化验证判据（先写后跑，run_id 20260725-alarm；6 种子由主控跑，此处只写设计）

- **假设 H1（主）**：共享报警场给猎物一个**当前不存在的协调收益**——「用别人的眼睛提前避开捕食者」。
  若脑能学会对 alarm 折进来的 pred 信号做出逃避响应，则**猎物被捕食死亡率下降**、猎物存活改善。
- **假设 H2（响应被选择）**：alarm 折叠只在 `alarm_rate>0` 时给 pred 通道额外信号；若它有用，则对该
  信号敏感的脑权重应被选择出来——**行为层面**表现为猎物在高 alarm cell 的停留/密度下降（避开示警区）。
- **成功判据**：6 配对种子 ON(`alarm_rate=0.05`) vs OFF(`alarm_rate=0.0`) ×20000 步：
  - 主判据：`death_predation_frac` ON < OFF，**6/6 同向、配对 Wilcoxon p≤0.031**（n=6 统计地板，
    `docs/conventions.md` §5）；或等价地 herbivore 谱系存活/占比 ON>OFF 同强度。
  - 护栏：`population`、`death_thirst_frac`/`death_thirst_age`（示警区常与河边重叠，别恶化幼年渴死
    瓶颈——同 fear 的风险，`landscape_of_fear.md` §3.2）、`carnivore_frac`（别把食肉者饿到近灭绝阈下）
    均不显著恶化。
  - 逐种子报告，不只均值；报所有算过的 p，不做 Bonferroni。
- **失败/负结果预案（诚实预期：大概率 NULL）**：三条先验指向 NULL，以下任一即判 NULL 并如实记录，**不
  放宽判据、不重跑找显著**：
  1. **信息冗余**：猎物本已能直接看见近处捕食者（pred 通道原始项），alarm 只是把「已经看得见的东西」
     换个来源再说一遍——若 alarm 的空间/时间外延不比原始视觉多给信息，`death_predation_frac` 不动。
     这是最可能的 NULL 机制（与 carrion「碰不到才吃」弱效应同类：脚手架在，但**边际信息为零**）。
  2. **响应未被选择**：静态地形 + Ne~2000 下二阶行为适应可能测不出（同加脑证伪的排程问题）——
     `death_predation_frac` 不动且高 alarm cell 猎物密度无下降。
  3. **反被捕食者利用**：alarm 折进的是所有 diet 都读的 pred 通道；若食肉者也读它去**回避**猎物聚集
     的空区（而非猎物避险），可能零和甚至负向。需在分析时分 diet 看响应方向。
  4. **护栏破**：若 `death_thirst` 恶化（猎物为避 alarm 弃河）或 `carnivore_frac` 崩，即使主判据动了
     也不算净正。
- **判 NULL 后的价值**：即便 NULL，本落地仍是**安全的、默认关的、bit-exact 的**互惠脚手架原语，和
  carrion 一样为「脚手架式种间合作」这条线留下可复现的阴性结果，阻止同一想法被重试。**当前默认保持
  关闭**，除非 6 种子兑现主判据。

### 8.5 [提案，非结论] 若为正 / 若加强

- 若主判据兑现：可视化加一路 alarm 场着色（仿 fear/水层），看猎物是否真在示警区外绕行。
- 若 NULL 且诊断为「信息冗余」：alarm 的独特价值只在**超视距/跨地形**才成立——需配合 LOS 遮挡
  （`los_occlusion_enabled`）让直接视觉被山挡住、而 alarm 经邻居接力翻过山，才可能制造出视觉给不了的
  信息（**先写后跑的下一轮假设**，非本轮结论）。
- 若要做成真·哨兵互利：需一个付出成本的示警者（如示警拖慢自己/更显眼），当前版本示警零成本，故不含
  「利他代价 vs 群体收益」的张力——那是另一个更贵的设计（动 trait 或 act），非本轮范围。

### 8.6 [本世界实测] 结果：NULL（信息冗余），默认保持关闭

6 配对种子 ×20000 步（run_id `20260725-alarm`，出处 `outputs/20260725-alarm/`，统计
`explorations/20260725-alarm/analyze.py`，HEAD `c25ba60`；n=6 配对 Wilcoxon 地板 p=0.031）。
**主判据未兑现——判 NULL，与 §8.4 诚实预期一致。**

| 指标 | ON | OFF | 配对差 | 95% bootstrap CI | 同向 | 配对 Wilcoxon |
| --- | --- | --- | --- | --- | --- | --- |
| death_predation_frac（主判据）| 0.360 | 0.375 | −0.015 | [−0.060, +0.030] | **4/6** | **p=0.5625** |
| alarm_total（机制点火）| 2637–3497 | 0 | — | — | 6/6 | 开关在动 |
| population | 1661 | 1681 | −20 | [−174, +147] | — | p=1.0 |
| carnivore_frac | 0.163 | 0.196 | −0.033 | [−0.104, +0.037] | — | p=0.5625 |
| death_thirst_frac | 0.533 | 0.500 | +0.033 | [−0.009, +0.079] | 4/6 差 | p=0.3125（不显著）|

- **主判据双重不成立**：`death_predation_frac` 仅 4/6 同向（需 6/6）、p=0.5625（需 ≤0.031），mean diff −0.015、
  CI 跨 0，差值远小于 OFF 臂种子间 SD（0.059）。seed 1、2 反而 ON 更高。
- **不是"没跑起来"**：`alarm_total` ON 6/6 都在 2637–3497（>0）、OFF 恒 0——猎物在沉积、脑能读到，机制确在运转。
  故 NULL 是**边际信息为零**，归 §8.4 **第 1 类「信息冗余」**（最可能项）：alarm 只是把猎物本已能直接看见的近处
  捕食者换个来源再说一遍。与 carrion「碰不到才吃」的弱效应同类——脚手架在，但没造出新信息。
- **无法与第 2 类「响应未被选择」完全区分**：本批 JSON 未记「高-alarm-cell 猎物密度」这一行为读出量，H2 的直接
  检验缺指标。无第 3 类「被捕食者反用」的净负证据（carn_frac 未显著升）。
- **护栏未破**：population/min_pop/carn_frac/thirst_age 配对 p 全在 0.56–1.0、CI 全跨 0。`death_thirst_frac`
  有不显著的轻微偏差（+0.033、p=0.31），方向与 §8.4「示警区与河边重叠」预警一致，n=6 功效不足、记一笔不判负。
- **[提案，非结论] 判决**：默认保持关闭（安全落点，bit-exact 保 golden、无需 bless）。作为「脚手架式种间合作」
  这条线的可复现阴性结果留档，阻止重试。要区分 NULL 第 1/2 类或追 §8.5：需先加「高 alarm cell 猎物密度」读出
  指标，再配 `los_occlusion_enabled` 让直接视觉被山挡住、alarm 经邻居接力翻山造出超视距信息——先写后跑的下一轮。

---

## 9. 资源分割第二食草者（forage_pref，草↔果权衡基因，2026-07-25，run_id 待定）

按 §4 次选（§2(c3)）落地了**草↔果取食权衡基因**——不是新物种，也不是新营养级，而是给现有食草者
一条**生态位分化**的演化通道：世界本就有 grass(`plant`) 与 fruit 两个食草层，此前被同一条
`_herbivory(diet)` taper 统一取食，人人对两层同等（无）能。引入一个权衡基因后，个体可专精一层、
牺牲另一层；若该基因**演化出双峰分布**，就是**性状替代 / character displacement**——两个食草生态位
分占两种资源的证据。

### 9.1 机制（`trait_dim` 7→8）

`[对应]` 逐文件，最省的 trait 级改动（**零 `in_dim` 改动、零新场**，仅 genome 加一列）：

- `config.py`：`trait_dim=8`；`@property forage_pref_index = brain_params + 7`；新增强度旋钮
  `forage_tradeoff: float = 0.0`（**默认关**）与 `forage_pref_mutation_sigma=0.02`（慢档，同
  size/escape/armor）。
- `state.py`：`forage_pref_of = sigmoid(genome[:, forage_pref_index])`，映射 [0,1]，**gene=0 → 0.5
  = 不偏**（两端对称、中点中性，是 diet_of 式的**双边** sigmoid，区别于 escape/armor 的单边
  `clip(sigmoid-0.5,0)`——后者是"买不买"的单向投资，forage_pref 是"两资源之间的拨盘"，两端都有意义）。
- `genome.py`：`mutate` 给该列设慢档 sigma；**crossover 不豁免**（不进感觉-运动回路，同 escape/armor，
  保持 G 矩阵估计干净）。
- `dynamics.py`：`_forage_pref_scale` 返回 `(grass_mult, fruit_mult)`，以 `s=2·pref−1 ∈[−1,1]`
  （gene=0→s=0）给出 `grass_mult = clip(1 + forage_tradeoff·s, 0)`、`fruit_mult = clip(1 −
  forage_tradeoff·s, 0)`——**两乘子恒和为 2**（§9.2）。`graze` 用 `grass_mult`、`eat_fruit` 用
  `fruit_mult` 调制各自 `demand`，**编译期 gate 于 `forage_tradeoff>0`**：默认关时整段不进 trace、
  取食 bit-exact 旧内核（gene 只是加宽 genome）。
- `metrics.py`：append `mean_forage_pref`/`forage_pref_std`/`herb_forage_pref`（herb 是功能载体，
  仿 herb_escape/herb_armor）。**不动 wire/协议**（`encode` 按名 `.get` 取字段，新字段不打包即不影响
  偏移）。

`[对应] 权衡挂点是取食效率、不是能量税`：与 armor/escape 不同，forage_pref **不进 `metabolize`**——
它没有单独的能量账代价，**权衡本身就是代价**（草专家吃果差）。这正是"总效率守恒"的来源。

### 9.2 为什么"总效率守恒"是关键设计（no free lunch）

`[对应]` 两乘子恒和为 2 ⟹ 一个个体在草上多得的效率，**一对一**由它在果上的损失偿付。若不守恒
（比如只给 grass 加成、不扣 fruit），基因就成了**纯增益**，会无约束跑向极值、无法证伪——正如
`docs/trait_roadmap.md` §7.4 对无代价基因饱和的警告。守恒把它变成**真权衡 → 内点/分化均衡**：
单一策略无法通吃两层，专精才有净收益，这是资源分割能被选择的前提。

`[本世界实测]` 单元测试守住三点（`tests/test_kernel.py`）：
- `test_forage_pref_neutral_when_tradeoff_off`：`forage_tradeoff=0` 时草/果基因个体与中性个体取食
  完全相同（gene 行为中性、复现旧内核）。
- `test_forage_pref_partitions_grass_and_fruit_when_on`：`forage_tradeoff=0.5` 时高 pref 个体吃草多、
  吃果少，低 pref 反之。
- `test_forage_pref_tradeoff_conserves_total_efficiency`：归一化总取食
  （grass/eat_rate + fruit_taken/fruit_eat_rate）在草专家/果专家/中性三者间相等——拨盘只在两层间
  **搬移**效率，不抬高总和。

### 9.3 演化验证设计（先写后跑，四标签，**尚未跑**）

- **假设**：开启权衡（`forage_tradeoff=0.5`）后，`forage_pref` 基因在食草谱系里**演化出双峰分布**
  （性状替代）——种群分裂为"草专家"与"果专家"两个食草生态位，而非停在中性 0.5 的单峰。
- **臂**（基因组布局相同、可直接配对）：主消融 `forage_tradeoff` **0.5(ON) vs 0.0(OFF)**——回答
  "让草↔果权衡生效是否驱动生态位分化"。OFF 臂 gene 中性漂变（照 escape/armor 的存在性判据体例）。
- **成功判据**（6 配对种子起，n=6 配对 Wilcoxon 的 p 地板 0.031）：
  1. **分化信号**：`forage_pref_std`（方差）ON > OFF，6/6 同向、配对 Wilcoxon p≤0.05。方差是**双峰性
     的必要下界**——单峰收窄给低方差，两簇分占 0 与 1 给高方差。
  2. **真双峰判定**（比方差更强）：离线读末态 `genome[:, forage_pref_index]` 的 `forage_pref_of`
     分布，做 Hartigan dip test 或双高斯 vs 单高斯 BIC；ON 臂显著双峰、OFF 臂单峰=证实性状替代。
     （`--json` 只给标量方差，双峰需存末态 genome，见 provenance。）
  3. **载体正确**：`herb_forage_pref` 是食草谱系的实际拨盘；分化应主要发生在 `is_herb`（diet<0.35）
     里，`carn` 的 forage_pref 应近中性漂变（食肉者几乎不取食植物，pref 对其近中性）。
- **护栏**（照 §7 carrion / `attack_range_redqueen.md` §5）：`population` 与 `carnivore_frac` 不显著
  恶化、不灭绝不爆炸；`death_thirst_frac` 不显著上升（权衡挂在取食效率、不该动水账，但果层多在林
  内、专精可能改变空间利用，须核）。
- **统计纪律**：配对 Wilcoxon + 10000 次 bootstrap 95% CI，**报告算过的每个 p**，**不做 Bonferroni**；
  **伪重复诚实标注**（`terrain.build` 无 RNG，6 种子同一张地图 → 结论只对**这一套河系/林果格局**成立，
  推广需交叉 `ridge_wavenumber` / `fruit_wavenumber_x/y` 等地形种子）。
- **负结果预案**：
  - **若 `forage_pref_std` ON≈OFF（无分化）**：最可能因**果层太稀/太边缘**——`fruit` 只落在
    canopy² 的 ~小片（`fruit_patch_threshold`），果专精的可及回报不足以支撑一个专家谱系（与 §7
    carrion "碰不到"同类失败：资源太稀 → 专精无利可图 → 拨盘漂回中性）。诊断先看 `fruit_total` 与
    食草者对果层的实际取食占比。
  - **若方差升但非双峰（宽单峰/连续梯度）**：是**性状变异扩大而非离散生态位分化**——分化被
    `assortative_mating` 只按 diet 排序、拿不到按 forage_pref 的生殖隔离所限（§2(b)(iii) 同款约束）。
    记为"变异有、离散分化无"，并指向"若要真双峰可能需按 forage_pref 的隔离脚手架"。
  - **若护栏恶化**（pop/carn_frac 掉）：判断是"果专精挤兑草层削薄食草基座"还是别的耦合，记录并否掉
    默认开启。
- **provenance**：`--set forage_tradeoff=0.5` 开 ON 臂；ON/OFF 各 6 配对种子 ×≥20000 步；存
  `outputs/<run_id>/` 完整 config + git hash + seed 列表 + **末态 genome**（双峰检验需要，标量方差不够）；
  分析脚本写 `explorations/<run_id>/`。**本次只落机制与设计，不跑 6 种子。**

### 9.4 落地状态（本次）

`[本世界实测]` 机制已落地、单元契约三测全绿、`check.py --contracts`/tier2/pytest 全过；**默认关**
（`forage_tradeoff=0.0`）故取食对 gene 中性。**golden 重 bless**：`trait_dim` 7→8 使 `genome_size`
1385→1386，`init_state` 的 `jax.random.normal(k_gen,(n,genome_size))` 抽样形状变 → founder RNG 重排 →
整条混沌轨迹漂移（**即便 gene 是 no-op**）：population 1517→1591、fruit_total 29.18→40.35、
carnivore_frac 0.00989→0.00691。这是加任何 trait 基因不可躲的固定契约代价
（`docs/trait_addition_feasibility.md` §A.3、armor 5→7 同款先例 `docs/trait_defense_landing.md` §3），
**已 `--bless` 重录，非放宽 band、非行为改变**（默认 tradeoff=0 时取食 bit-exact 旧内核，见 §9.2 中性
测试）。**演化是否真出双峰仍待 §9.3 的 6 配对种子验证。**

### 9.5 [本世界实测] 结果：证否（方差反向塌缩），默认保持关闭

6 配对种子 ×20000 步（run_id `20260725-forage`，出处 `outputs/20260725-forage/`，统计
`explorations/20260725-forage/analyze.py`，HEAD `c25ba60`）。**主判据不仅未兑现、而是方向反转——性状替代
的必要条件被证否。**

| 指标 | ON(`tradeoff=0.5`) | OFF(`=0.0`) | 配对差 | 95% bootstrap CI | 同向(ON>OFF) | 配对 Wilcoxon |
| --- | --- | --- | --- | --- | --- | --- |
| forage_pref_std（主判据）| 0.0548 | 0.0813 | **−0.0265** | [−0.048, −0.007] | **1/6**（反向 5/6）| p=0.0625 |
| mean_forage_pref | 0.627 | 0.514 | +0.128 | [+0.044, +0.181] | 6/6 | p=0.0625 |
| herb_forage_pref | 0.632 | 0.515 | +0.118 | — | 6/6 | — |

- **主判据反向**：`forage_pref_std` 不是 ON≈OFF，而是 **OFF>ON**，5/6 种子 OFF 方差更高。开权衡后方差
  **塌缩 32.6%**——这是**定向/稳定化选择把分布收窄到单峰**，恰与双峰（歧化选择）所需相反。
- **走向是整体平移、不是绕中性分裂**：OFF `mean_forage_pref=0.514≈0.5`（确认基因中性漂变）；ON 抬到 0.627
  （偏草端 +0.128），`herb_forage_pref` 同步。种群**整体倒向草**，而非围绕 0.5 裂成"草专家/果专家"两簇。
- **根因：果层太薄，无生态位可专精**。实测 `fruit_total≈2.7–5.4` vs `plant_total≈4900–8000`——**果层仅草层的
  ~0.07%**。专果没有可及回报，权衡的两端不对称，选择只会把所有食草者单向推向草。这正是 §9.3 预写的失败模式
  「无分化最可能因果层太稀/太边缘」被兑现。
- **护栏全部安全**：population/carn_frac/late_carn/death_thirst_frac 无一显著恶化（p 均 ≥0.31，ON min_pop 略低
  属瞬态、无灭绝）。未见「果专精挤兑草基座」的资源恶化（plant_total ON 略降 −331、p=0.84，噪声）。
- **[提案，非结论] 判决**：**不值得**下一轮存 genome 验真双峰——§9.3 自己写明「若方差都不动则性状替代直接证否」，
  本轮方差非但不动、还反向塌缩，**必要条件已否**，dip-test 无意义。保持 `forage_tradeoff=0.0` 默认关（config
  现状）。真要追双峰的可能性：先改变量必须是**加厚/扩散果层**（动 `fruit` 生态位使两层可比），并**交叉 ridge
  地形因子**脱离本河系伪重复——在果层与草层可比之前，权衡两端不对称、选择只能单向塌缩，双峰无从谈起。
- **[对应] 关于保留与否**：机制默认关、gene 中性漂变、无行为代价；但它是**唯一付了永久基因组代价（trait_dim
  7→8、作废演化种群、golden 重 bless）却被证否**的本轮改动。保留理由是回退（8→7）会再churn 一次 golden+作废
  种群，且它是"果层足够厚时可复活"的可复现阴性基座（同 L6/carrion 留档纪律）；留在树上、默认关。
