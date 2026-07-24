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
