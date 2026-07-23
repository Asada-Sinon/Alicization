# 测试覆盖审计（tests/test_kernel.py）

一次针对 `underworld/` 各模块的覆盖缺口审计，并补上 13 个单元级测试。目标是抓**真回归**
——那些"改坏了但整体 sim 仍然看起来活着、golden 带宽也不一定跳"的沉默失效——而不是凑
覆盖率数字。所有新测试都是确定性的（不依赖 GPU scatter-add 的重排），因此能在既有 sim 级
测试只能断言"带宽"的地方断言**精确相等**。

标注约定同 `docs/conventions.md`：`[对应]` 指向代码，`[本世界实测]` 指在此处跑出来的结果。

## 审计方法

逐模块对照 `underworld/*.py` 的不变量与既有 `test_kernel.py` / `test_terrain.py`，记录哪些
契约没有直接测试守护。既有套件对**性状边界**（invest/size/attack/escape 的 sigmoid 范围、
中性起点、crossover 豁免）、**水/能量双账本的税项落点**、**死因互斥划分**、**记忆不可遗传 +
位置跟踪**、**terrain 派生场边界**已经相当完备。缺口集中在下列六处。

## 找到的覆盖缺口与补上的测试

### 1. 红皇后消融开关的 no-op 等价性 —— 最大的缺口
`attack_range_heritable` / `prey_escape_enabled` 是编译期 flag（`Config` 被 close over 进 jit，
`dynamics.predation`/`metabolize` 上是 Python `if` 分支）。既有测试验证了基因**开启时**的行
为（中性起点、escape 缩短有效射程），但**没有**验证"flag 关闭"这条分支与"中性基因 + flag 开
启"是否产出同一个世界。这正是"沉默基线漂移"最容易钻进来的地方：任何消融臂如果基线本身
就变了，测得的差异就不是机制本身。 [对应] `dynamics.predation` 的 `if cfg.attack_range_heritable`
/`if cfg.prey_escape_enabled`；`dynamics.metabolize` 的 tax 分支。

- `test_attack_flag_off_reproduces_neutral_gene_predation` —— 两个配置（flag 全开 + 中性基因
  reach 6.0/escape 0，对 flag 全关 + 固定 `attack_range`）对同一 2-agent 场景，`predation` 的
  完整 6 元组逐元素 `allclose`；并断言这一咬确实命中（否则等价是空的）。 [本世界实测] 通过。
- `test_metabolize_flags_off_ignore_reach_and_escape_args` —— flag 关闭时，无论传入多大的
  reach/escape，`metabolize` 的能量税恒为 0（off 分支把 tax 编译掉）。 [本世界实测] 通过。

### 2. 有效攻击距离 = attack − escape 的边界算术
既有 `test_prey_escape_shrinks_effective_attack_reach` 只测了两个极端（escape 0 命中、escape 极
大脱靶）。**过零点的具体位置**没有被钉住。

- `test_prey_escape_effective_reach_is_attack_minus_escape` —— 攻击者中性 reach=6.0，猎物携带
  已知 escape 基因（由 `escape_of` 算出精确值），断言距离在 `6.0 - escape` 内侧 0.15 命中、外
  侧 0.15 脱靶。把"减法"本身钉在边界上。 [本世界实测] 通过。

### 3. 捕食的目标选择：最近可食猎物 + diet_delta 阈值
`predation` 用 `argmin` 在**可食**邻居里挑**最近**的一个，且要求 `d_i - d_j > diet_delta`（严格
大于）。这两条选择逻辑此前无直接测试。

- `test_predation_hits_nearest_eligible_prey_only` —— 一个捕食者，两个都在射程内的草食邻居
  （dist 3 与 5），断言只有较近者被咬、较远者毫发无伤。 [本世界实测] 通过。
- `test_predation_respects_diet_delta_threshold` —— diet 差距略小于 `diet_delta` 的邻居不可食，
  零伤害。守住"近似同类不互食"这条。 [本世界实测] 通过。

### 4. spatial 邻居索引的沉默丢弃行为
CLAUDE.md 明说"overflow beyond `k_neighbors` … silently dropped"，且死者进 dump 行——这类"沉
默丢弃"正是最该有回归守护的。此前只有 `test_neighbor_index`（手验聚簇可见/远处不可见），没
有直接测溢出与死者排除。 [对应] `spatial.build_table` 的 `cols = min(rank, K)` 溢出列 + 死者
`cell = where(alive, cell, n_cells)` dump 行，返回时 `table[:n_cells, :K]` 双双切掉。

- `test_neighbor_table_drops_overflow_beyond_k` —— 把 K+4 个 agent 塞进同一格，断言 table 里
  恰好 K 个不同 index 幸存，且正是 cell-rank 最前的 K 个（`set(range(K))`）。 [本世界实测] 通过。
- `test_neighbor_table_excludes_the_dead` —— 死者与活者叠在同一位置（用"是否活着"而非距离来
  区分），断言 table 只含活着的两个。 [本世界实测] 通过。

### 5. 生育 permutation-scatter 的守恒不变量
CLAUDE.md 的核心契约"每个 index 恰好写一次，非生育写回原值（no-op）"此前只被整体 sim 间接
覆盖。 [对应] `reproduction.reproduce` 的 `place()` + `energy.at[parent_idx].add(...)`。

- `test_reproduce_conserves_energy_and_writes_each_slot_once` —— 断言：出生数 = `min(想生, 空位)`；
  活着的 agent 一个都不会被 reproduce 杀掉；新生只落在原空位；活着的**非父母**逐字节不变；
  **总能量精确守恒**（父代付出 == 子代所得，`sum(energy)` 不变）。能量守恒是一条很强的
  "写一次"证据——重复写或错位写都会破坏它。 [本世界实测] 通过。
- `test_reproduce_bounded_by_free_slots_not_wanters` —— 想生者多于空位时只发生"空位数"次出生，
  且无任何活着的 agent 被覆盖（`n_birth = min(...)` clamp）。 [本世界实测] 通过。

### 6. 记忆分区边界 + 编码的自我中心朝向
既有 `test_memory_write_replaces_exactly_one_slot` 只测了水分区 `[0, water_slots)`。跨分区不串扰
（水果写入不动水槽）以及 `encode` 的 egocentric 约定没被测。 [对应] `memory.write` 的
`dynamic_slice_in_dim(..., lo, k)`；`memory.encode` 的 `bearing = arctan2 - heading`。

- `test_memory_write_respects_partition_boundary` —— 向水果分区写入，断言水槽对所有 agent 都未
  被触碰、写者恰好改一个水果槽（最弱者）。 [本世界实测] 通过。
- `test_memory_encode_bearing_is_egocentric` —— 一个正对 agent 自身朝向的槽，无论其绝对方向如
  何，都编码为 sin≈0、cos≈1。守住"记忆与视网膜共用同一朝向语义"。 [本世界实测] 通过。

### 附带：ecology.regrow 与 genome 布局的直接单元测试
- `test_ecology_regrow_clips_and_recovers` —— 把 `regrow` 从整体 sim 里剥出来直接测三件事：超容
  量被夹回、零容量格恒为零（`fruit_max=0` 消融依赖的 0/0 guard）、被吃光但有容量的格靠 baseline
  恢复（logistic 项为 0 时）。 [本世界实测] 通过。
- `test_trait_gene_indices_are_distinct_and_in_range` —— 五个性状基因索引互不相同、都在
  `[brain_params, genome_size)` 内、`genome_size == brain_params + trait_dim`、`trait_dim` 与性状
  基因数一致、`in_dim` 符合文档的通道公式。守住"再加一个性状时索引不会与既有列相撞"——相撞会
  让一个基因静默覆盖另一个而没有任何测试跳。 [本世界实测] 通过。

## 疑似 bug

**未发现被测代码的疑似 bug。** 所有被新测试覆盖的不变量都按文档所述成立。一处值得记录、
但**不是 bug**：`reproduce` 中子代水量在超过自身水箱（`water_max * size`）时会被 `min` 夹掉，
因此**水不守恒**（多出的部分丢失），而能量守恒。这是有意设计（见 `reproduction.py` 注释与
`test_child_water_investment_clamped_to_own_tank`），新测试因此只对**能量**断言精确守恒。

## 仍未覆盖（留给后续，非本次结论）

- `brain.forward` / `split_params` 的张量切分维度正确性（目前仅由整体 sim 的 shape 断言间接
  覆盖）。 [提案，非结论]
- `sensors.sense` 各视网膜通道（food/prey/predator/water/slope）的独立语义——现只有 peer 通道被
  `test_peer_channel_*` 直接测；其余五通道靠整体 sim。 [提案，非结论]
- `metrics.compute` 的相关系数估计量（`invest_diet_corr` 等）的数值正确性，只有有界性被测。
  [提案，非结论]
