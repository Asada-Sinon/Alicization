# 可见形态防御性状：首个落地（armor + spike）+ 可视化

这份文档记录 `docs/trait_defense_catalog.md`（候选清单）与 `docs/trait_addition_feasibility.md`
（可行性判决）之后的**实际落地**：把两个防御性状——**厚皮 armor（减伤）** 与 **尖刺 spike
（反伤捕食者）**——真正实现进内核，并接到 dashboard 让**每个个体的性状在身上可见**。

标记体例沿用四标签。**一句话状态**：机制已落地、单元契约已测、可视化已截图验证、不破坏世界；
**但「防御性状是否真的会被选择演化出来」尚未验证**——那是下一步的 6 配对种子实验（见 §5）。

---

## 1. 落地范围（相对 feasibility 篇的偏差）

`docs/trait_addition_feasibility.md` §C 推荐**先只做 armor**（信号最干净）。本轮按用户「多提一点 +
要在可视化里直观看到每个个体的性状变化」的要求，**同时落地 armor 与 spike**，理由：

- 两者共用同一条最小钩子链（predation 收益 + metabolize 能量税），一次 `trait_dim` 变更即可，
  作废种群一次（本来就要付的代价，加两列与加一列同价）。
- 用户最初的两个原话例子就是「长尖刺」「加厚皮肤」——spike 是最具辨识度的可视化，armor 是
  信号最干净的科学主角；一起上，可视化与实验价值兼得。

**未做**：警戒色/毒（catalog §4，需谨慎处理捕食者感知/学习）、诱导型（需风险感知反应函数）、
体型 refuge（与既有决定冲突）、dilution（群属性、不可见）。这些留作后续。

## 2. 内核实现（`trait_dim` 5→7）

`[对应]` 逐文件，全部照红皇后 `escape` 基因的结构孪生：

- `config.py`：`trait_dim=7`；新增 `armor_span/armor_cost/armor_mutation_sigma/armor_heritable`
  与对称的 `spike_*`（外加 `spike_reflect`）；`@property armor_index=brain_params+5`、
  `spike_index=brain_params+6`。
- `state.py`：`armor_of`/`spike_of`，单边映射 `span * clip(sigmoid(gene)-0.5, 0, None)`，
  **gene=0 → 0**（中性无防御，任何防御都是演化出来的）。
- `genome.py`：`mutate` 给两列设慢档 sigma（0.02）；**crossover 不豁免**（不进感觉-运动回路，
  同 escape，保持 G 矩阵估计干净）。
- `dynamics.predation`：
  - **armor** 把 per-prey 需求 `wanted` 乘 `(1-armor[prey])`——缩需求而非只缩伤害，使攻击者
    payout `scale=removed/wanted` 同步下降，`meat_gain ≤ damage` 守恒不破。
  - **spike** 在 `energy = energy - damage + meat_gain` 后，对攻击者按 `spike[target] *
    (dmg*scale[target])`（实际咬出的能量）扣血——纯能量 sink（不转移给猎物），只减总能量，
    predation 仍不凭空造能量。
- `dynamics.metabolize`：新增 `armor`/`spike` 可选参数，`tax += armor_cost*armor*(1-diet) +
  spike_cost*spike*(1-diet)`——**能量账，绝不水账**（`docs/trait_addition_feasibility.md` §B.2）。
- `step.py`：读 `armor_of`/`spike_of` 传给 `metabolize`（predation 侧直接读 `state.genome`）。
- `metrics.py`：append `mean_armor/armor_std/herb_armor/carn_armor` 与 `spike` 四个同构字段
  （herb 是功能载体、carn 是 diet 门控对照，应 ~0）。

## 3. 契约与测试

- **单元测试**（`tests/test_kernel.py`，全绿）：`test_defence_genes_neutral_start_and_bounds`
  （gene=0→0、有界）、`test_defence_taxes_hit_energy_not_water`（税在 metabolize、thirst 不碰、
  diet 门控使肉食付≈0）、`test_armor_reduces_bite_and_spike_hurts_attacker`（armor 减猎物掉血、
  spike 减攻击者能量、两者都守住 `meat_gain≤damage`）、`test_defence_genes_recombine_unlike_size`
  （不豁免交叉）。并更新 `test_trait_gene_indices_are_distinct_and_in_range` 纳入两个新 index。
- **golden 重 bless**（`[本世界实测]`）：`genome_size` 1383→1385，founder RNG 重排使整条混沌
  轨迹漂移（即便 gene=0 armor/spike 是 no-op）：population 1494→1555、plant_total −6.9%、
  fruit_total +7.2%、mean_size 1.003→0.9775。这是加任何 trait 基因不可躲的固定契约代价
  （`docs/trait_addition_feasibility.md` §A.3），已 `--bless` 重录，**非放宽 band**。
- **check.py --contracts / tier2 / pytest** 全过。

## 4. 可视化（wire v8 + shader）：每个个体的性状可见

`[对应]` 目标是「用户直观看到每个个体的性状变化」，落地三处（append 纪律，header 不变）：

- `server/protocol.py`：每 agent 记录 20→32 字节，在 `id` 之后**追加** `size/armor/spike` 三个
  f32（x/y/diet/energy/id 偏移不动，客户端 id-at-16 选择逻辑不受影响）。`server/app.py` 解码
  三者传入，并把 armor/spike 加进 inspector JSON。
- `web/main.js`：`STRIDE` 5→8；inspector 面板新增「厚皮/尖刺」两行（`web/index.html`）。
- `web/render.js`：`POINT_VS` 用 `a_size` 直接缩放 sprite（体型基因→点大小，最直观），spike 加
  一点尺寸余量；`POINT_FS` 用 `a_armor` 画深色去饱和的厚描边（厚皮/甲壳），用 `a_spike` 画 8
  条放射状尖刺（`pow(max(cos(8·ang),0),6)` 锐化成尖）。

**天花板（诚实）**：agent 是屏幕对齐的 point-sprite，尺寸被 `u_sizeRef/w` 设计成近似恒定像素
大小（缩放相机不放大个体），所以尖刺/厚皮细节在整图截图里只有几像素、需要用户在真实高 DPI
屏上凑近看。真正的独立几何（每个体一个网格）需要新建 instanced-mesh 管线（未做）。

**验证（`docs/conventions.md` §10，看过而非只读）**：
- 用 headless chromium（swiftshader）截活体 dashboard：wire v8 正确解析、agents 正常渲染
  （紫食草/红食肉）、昼夜/地形/inspector（含新增厚皮/尖刺行）全部正常、无 JS 报错。
- 用一页独立 WebGL 测试把 `POINT_FS` 原样渲染成大点：**plain 圆盘 / armor 深色厚描边 /
  8 尖刺 / 食肉红+尖刺 / armor+尖刺 / 大体型** 六种形态均按设计正确绘出。截图存于会话
  scratchpad（非仓库）。

## 5. 尚未验证：防御是否真的会演化（下一步）

**这是本轮最重要的诚实边界。** 已落地的是**机制 + 可见性**，不是**演化结论**。短跑（2000 步）
里 `herb_armor≈0.049`、`herb_spike≈0.025`——这只是从 0 起点的近中性漂变，不是选择信号
（红皇后 escape 也要 ~3500+ 步才起来）。要回答「防御性状会不会被选择推上去」，需要
`docs/trait_addition_feasibility.md` §B.5-B.6 的实验：

- **可证伪预测**：(P1) `herb_armor`/`herb_spike` 在功能臂显著高于 `*_heritable=False` 对照臂
  （照 escape 的 1.91 vs 0.50、6/6、p=0.031）；(P2) `death_predation_frac`/`carnivore_frac`
  随防御上升而降，不必伴随 `hunt_success` 降；(P3) `carn_armor`/`carn_spike` 及其税 ~0（门控成立）。
- **设计**：6 配对种子 ×20000 步，主消融 `armor_heritable`/`spike_heritable` on/off，可与
  `attack_range_heritable` 交叉测双侧红皇后；配对 Wilcoxon + bootstrap，报每种子，不 Bonferroni；
  伪重复诚实标注（同一张地图）。
- **调参提醒**：`armor_cost`/`spike_cost`/`spike_reflect` 现为初值（0.012/0.012/1.0），未经探针
  标定；正式长跑前应先短探针确认防御既不 peg 到上界、也不把捕食者压灭（`carn_frac` 护栏）。

**结论**：armor+spike 作为可演化、可见的防御性状已进入默认世界；它们**会不会演化出来**是下一个
实验，不是已成立的结论。

---

## 6. 演化验证实验（2026-07-24，run_id: 20260724-defence-evo）

- **假设**：让 armor/spike 可遗传且有功能（默认臂）会使**食草谱系**的防御基因显著高于「基因存在
  但无功能、无税」的中性漂变对照臂——即防御性状真的被选择演化出来（而非仅从 0 起点漂变）。
- **成功判据**：
  - **P1（主）**：`herb_armor` 与 `herb_spike` 各自 ON > OFF，**6/6 配对同向**，配对 Wilcoxon
    **p ≤ 0.05**（n=6 地板 0.031），并报 bootstrap 95% CI 与效应量。方向一致性是硬判据
    （照红皇后 escape 的 1.91 vs 0.50 先例）；均衡幅度报告但不预设阈值，因 cost 未探针标定。
  - **P2（次）**：`death_predation_frac` 与/或 `carnivore_frac` 在 ON < OFF（防御压低捕食净收益，
    与红皇后同签名）。报方向与 p，不预设必达（可能表达在密度轴而非单次命中）。
  - **P3（对照/门控）**：`carn_armor`/`carn_spike` 在 ON 与 OFF **无显著差异**（`(1-diet)` 税使
    肉食谱系不被选择加防御，其基因中性漂变），与 herb 谱系形成对照。
- **失败判据**：`herb_armor(on)` 与 `herb_spike(on)` 与 OFF 对照**无法区分**（无一致方向、p>0.05）
  → 在当前 cost 值下防御**不演化**。注意这有两种解释（真无选择 / cost 误标定使基因 peg 在 0），
  须在结论区分，并触发 cost 探针作为后续。
- **对照臂**：ON = 默认 = `--set armor_heritable=True --set spike_heritable=True`；
  OFF = `--set armor_heritable=False --set spike_heritable=False`（基因存在、漂变、无功能无税，
  与 ON 基因组布局相同、可直接配对）。
- **种子**：founder 0,1,2,3,4,5（6 配对），各 20000 步。**伪重复注意**：`terrain.build` 无 RNG，
  6 种子同一张地图；本实验是「防御是否被选择演化」的时序/选择结论、非空间结论，伪重复风险较低但
  仍标注——结论不外推到「河流一般」。
- **git hash**: 33bac2b（跑时唯一脏文件是本判据节所在的 doc 本身，代码在 33bac2b 干净）。
- **结果**（12 run 全部无 NaN、无灭绝；逐种子出处 `outputs/20260724-defence-evo/{on,off}_seed{0..5}.log`
  的 `JSON` 行，统计出处 `explorations/20260724-defence-evo-analysis/analyze.py`）：

  **herb_armor（P1 主指标）逐种子**：

  | seed | ON | OFF(漂变) | 差 |
  | --- | --- | --- | --- |
  | 0 | 0.195 | 0.032 | +0.163 |
  | 1 | 0.152 | 0.089 | +0.063 |
  | 2 | 0.162 | 0.012 | +0.150 |
  | 3 | 0.192 | 0.108 | +0.083 |
  | 4 | 0.180 | 0.138 | +0.042 |
  | 5 | 0.138 | 0.024 | +0.114 |
  | **均值** | **0.170** | **0.067** | **+0.102**（6/6 同向） |

  **汇总（配对 Wilcoxon + 10000 次 bootstrap 95% CI，全部算过的 p 都列、不 Bonferroni）**：

  | 指标 | ON | OFF | 配对差 | boot 95% CI | p | 同向 |
  | --- | --- | --- | --- | --- | --- | --- |
  | herb_armor | 0.170 | 0.067 | +0.102 | [+0.068,+0.138] | **0.0312** | **6/6** |
  | carn_armor | 0.091 | 0.050 | +0.042 | [+0.015,+0.071] | **0.0312** | 6/6 |
  | herb_spike | 0.027 | 0.026 | +0.001 | [−0.018,+0.025] | 1.000 | 3/6 |
  | carn_spike | 0.019 | 0.033 | −0.014 | [−0.042,+0.013] | 0.5625 | 2/6 |
  | death_predation_frac | 0.378 | 0.395 | −0.016 | [−0.061,+0.026] | 0.5625 | 2/6 |
  | carnivore_frac | 0.133 | 0.129 | +0.004 | [−0.003,+0.011] | 0.5625 | 3/6 |
  | population | 1822 | 1790 | +33 | [−56,+112] | 0.625 | 4/6 |

- **结论**：
  - **[本世界实测] P1 armor 达成——armor 是本项目第一个真正演化出来的可见形态防御性状。**
    herb_armor ON 0.170 vs 中性漂变 OFF 0.067，**6/6 同向**，配对 Wilcoxon **p=0.0312**（n=6 地板）、
    rank-biserial +1.0、boot CI 全正 [+0.068,+0.138]，ON 最小种子 0.138 远离 0。选择把厚皮从漂变
    水平抬高到约 2.5×。
  - **[本世界实测] P1 spike 未达成，且是「收益太弱」而非「成本压在 0」。** herb_spike ON 0.027 vs
    OFF 0.026，仅 3/6 同向、p=1.0、CI 跨 0、ON 最小 0.0009 贴 0。**关键对照**：同样 0.012 的能量税
    下 armor 能演化、spike 不能——说明不是全局成本误标定，而是 `spike_reflect=1.0` 的反伤收益
    抵不过税。这与 catalog/feasibility 的预判一致：spike 的反伤收益偏群体/亲缘级，个体级选择差弱。
  - **[本世界实测] P2 无信号（非反向）。** death_predation_frac（p=0.5625，2/6）与 carnivore_frac
    （p=0.5625，3/6）方向不定。armor 演化出来了但 20k 步内未把捕食死亡压出可检出下降——可能是
    真无密度效应，也可能功效不足（这两个量种子间方差大）。
  - **[本世界实测] P3 部分证伪。** carn_spike 中性（p=0.5625，符合 `(1-diet)` 门控预期）；但
    carn_armor **也 6/6 ON>OFF、p=0.0312**——肉食谱系 armor 并非中性漂变。最可能是来自强选择
    食草谱系的**基因流/搭车**：ON 下 carn_armor 0.091 仍 6/6 低于 herb 0.170（肉食支受反选择压低，
    却被基因流抬到漂变之上）。seed-99 那个「OFF 里 carn>herb」的诊断在 6 个正式 OFF run 里**不
    复现**（armor 仅 2/6、spike 3/6），故正确解读是「ON 基因流抬高 carn_armor」，非「carn 天生更高」。
  - **[对应]** armor 收益在 `dynamics.predation`（缩 per-prey `removed`）、代价在 `metabolize` 能量
    税（§2-3）；可视化上这意味着**随代数推移 dashboard 上会出现越来越多深色厚描边（带甲）的个体，
    而尖刺个体不会增多**（spike 停在漂变）。
  - **[提案，非结论]** 三条后续：①**spike 救活**——跑受控臂上调 `spike_reflect`（2–3）或下调
    `spike_cost`，看 herb_spike 能否脱离 0（坐实是收益侧太弱）；②**P2/spike 脱离功效不足**——补到
    ~12–21 配对种子（0.02 级效应的功效需求，`docs/conventions.md` §5）；③**carn_armor 基因流假说**
    ——查 `reproduction`/`genome.crossover` 是否跨 diet 交换 armor 位点，可加「diet 内交配隔离」臂验证。
  - **伪重复边界**：6 种子同一张地图；P1/P3 是基因层面性状均值、P2 是全局量，受伪重复影响小，但
    结论不外推到「河流一般」。分析脚本 `explorations/20260724-defence-evo-analysis/`（入库，可复算）。

