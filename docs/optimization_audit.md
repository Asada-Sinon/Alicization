# JAX kernel 性能审计

对 `underworld/` 整个 GPU kernel 做的一次性能审计：找每步的重复计算、多余分配、
次优 JAX 惯用法、可合并的算子、以及编译期常量内联的机会。每个点都用实测
（`scripts/check.py` 默认档的 golden 是行为不变的判据；wall-clock 用 `lax.scan`
计时，见下），标注收益/风险/**是否改数值**。落地纪律：只实现"行为不变"的——同样
的数学更快/更省内存，golden 必须 held；有风险或会改数值的只写成
`[提案，非结论]`。

标注约定同 `docs/conventions.md` §3：`[现实]` / `[本世界实测]` / `[对应]` /
`[提案，非结论]`。

## 0. 测量方法与基线

- 硬件：RTX 4090，`cuda:0`。`[对应]` 全部实测在本 worktree。
- 计时：`scan_fn(state, key, 200)` + `jax.block_until_ready`，取多次 median。
  单进程内 A/B（同一进程里建两个 step 变体交替计时）用来抵消 GPU 时钟漂移与
  **一个正在跑的实时仪表盘**对 GPU 的争用——跨进程测量在本机会有 ±30% 的抖动
  （首个冷测量 465ms，稳定后 ~350ms，就是这个争用造成的，不要用跨进程数字下结论）。
- 规模：真实配置 `n_max=16384, n_init=2000`，`genome_size=1383`，`in_dim=67`，
  邻居候选宽度 `M = 9·k_neighbors = 216`。

**基线画像**（`[本世界实测]`，`jax.jit(step_fn).lower().compile()` 的
`cost_analysis` + `memory_analysis`，单步）：

| 量 | 值 |
|---|---|
| FLOPs / step | 14.09 GFLOP |
| bytes accessed / step | 1.39 GB |
| 达到的算术吞吐 | ~6.0 TFLOP/s（4090 fp32 峰值的 ~7%） |
| 达到的显存带宽 | ~597 GB/s（4090 峰值 ~1 TB/s 的 ~59%） |
| argument / output / temp | 93 / 98 / 154 MiB |
| 单步 wall-clock | ~1.7–1.8 ms（≈ 560–590 steps/s） |

**结论一句话：这个 kernel 是显存带宽受限，不是算力受限。** 优化的杠杆是"减少字节
搬运"（更少的中间量、更少的 gather/scatter、更少的 pytree 重写），而不是"减少
FLOP"。而字节搬运的大头是 `genome`：`[16384, 1383] f32 = 90 MiB`，一个数组就几乎
等于整个 state（argument 93 MiB / output 98 MiB），它每步被读+写一遍，繁殖时还要被
gather/scatter 多遍。

**逐组件耗时**（`[本世界实测]`，各自单独 jit，含各自 kernel-launch 开销，所以**不能
相加**——全步融合后 ~2.3ms，组件和 ~4.3ms，融合省掉 ~45%；此表只用来排序热点）：

| 组件 | median ms |
|---|---|
| `reproduction.reproduce` | **0.99** |
| `reproduction.cull` | 0.46 |
| `dynamics.act` | 0.38 |
| `sensors.sense`（改前） | 0.32 |
| `brain.forward` | 0.30 |
| `genome.mutate` | 0.23 |
| `metrics.compute` | 0.20 |
| `dynamics.predation` | 0.20 |
| `spatial.geometry` | 0.09 |
| `spatial.build_table` | 0.08 |
| `spatial.gather_neighbors` | 0.06 |

热点是 `reproduce`，其次 `cull`。注意 `brain.forward` **不是**瓶颈——每 agent 的
递归网只有 `i·h+h·h+h·o = 67·16+16·16+16·2 = 1360` MAC，全群才 ~45 MFLOP，相对
14 GFLOP 微不足道。14 GFLOP 的大头在繁殖的整基因组算子（mutate 的
`[16384,1383]` 高斯采样、crossover 的 bernoulli+where）和逐格 scatter。

---

## 1. 已落地（行为不变，golden held）

### 1.1 `sensors.sense`：扇区聚合的 Python 循环 → 单次 scatter-max ✅

`[对应]` `underworld/sensors.py`。原代码把候选按扇区聚合用一个 `for s in range(R)`
的 Python 循环（`R = retina_sectors = 8`）：对 prey/pred/peer 三个通道各扫一遍整个
`[n, M]` 候选数组 `R` 次（`jnp.where(sector==s, val, 0).max(axis=1)`），共 `3R = 24`
次对 216 宽的全宽扫描。

改成对每个通道一次 `.at[rows, sector].max(val)`：把 `[n, M]` 候选一次性
scatter-max 进 `[n, R]`。

- **是否改数值：不改，逐位一致。** `[本世界实测]` sense 输出 old vs new
  `max|Δ| = 0.0`，`array_equal = True`。因为 (a) `max` 与归约顺序无关（不像逐格
  `scatter-add` 的浮点重排），(b) prey/pred/peer 全 ≥ 0，空扇区保持 zero-init，与
  `max(where(m,val,0))` 给 0 完全一致。
- **不引入新的不确定性。** `CLAUDE.md` 警告的是逐格 **scatter-add** 的原子重排；
  scatter-**max** 是幂等、顺序无关的，确定性不受影响。
- **收益：** `[本世界实测]` sense 组件 `0.304ms → 0.212ms`（-30%，-0.09ms）。
  单进程 A/B 全步 `360.1ms → 355.4ms`（**-1.3%**）——组件省的 0.09ms 在融合后大
  部分被回收，但净为正。
- **风险：低。** golden 10 项 held（`population=1520`、`deaths=376` 逐位不变）。

为什么全步只 -1.3% 而组件 -30%：sense 只占融合步的一小块，且融合本身已经把循环的
中间量消掉了一部分。这是一个"更省、更干净、零风险"的改动，不是一个大加速。

---

## 2. 查证为"已经免费"的候选（XLA 已处理，不需改代码）

审计任务点名了"每步重复计算的量能不能缓存"。**大部分在 jit 里已经被 XLA 的公共
子表达式消除（CSE）+ 算子融合免费处理了**，手动缓存不会带来收益，只会让代码更难读。

### 2.1 重复的 `pos_to_cell(state.pos)`

`[对应]` 同一个 `state.pos` 上 `pos_to_cell` 被调用多次：`graze`、`eat_fruit`、
`drink`（移动后同一位置，3 次相同调用），`sensors.sense` 的 `own_cell` 与
`dynamics.act` 的 `cell`（移动前同一位置，2 次相同）。

- **XLA CSE 会去重。** 相同 SSA 值上的相同纯逐元素计算，XLA 在优化前必然合并。
  `[本世界实测]` 优化后 HLO 的 opcode 直方图：全步只有 48 个 `convert`、77 个
  `remainder`、163 个 `gather`——远低于"按调用点数×每次的算子数"应有的数量，说明
  移动前/后各自那一批同位置调用已被折叠成一次。
- **结论：不改。** 手动把 `cell` 提出来当参数传进 `graze/eat_fruit/drink` 不会更
  快（只会改函数签名、降低可读性）。`[提案，非结论]` 若纯为可读性可做，但不是性能项。

### 2.2 重复的性状 sigmoid（`size_of` / `attack_range_of` / `escape_of` / `diet_of`）

`[对应]` `size_of` 在 `step.py` 与 `reproduction` 各调一次；`attack_range_of` 在
`step.py`、`dynamics.predation`、`metrics` 三处；均对**同一个** `state.genome` 列做
`sigmoid`。同理被 CSE 折叠。`diet` 已经缓存在 `state.diet`（避免在邻居轴上广播时重
算），每步末从 genome 重算一次 sigmoid（`[n]` 上一次，便宜）——这是既定设计，正确。

### 2.3 `graze` 与 `eat_fruit` 共享的 `herbivory` 与 `demand_per_cell` 结构

`[对应]` 两者都算 `herbivory = where(diet>cutoff, 0, (1-diet)^6)`，输入相同 →
CSE 折叠。两者的 `demand_per_cell` scatter-add 是**不同**的（rate 不同、字段不同），
不可合并且不应合并。

**2. 节总结：这些"重复计算"看起来是优化点，实测是伪优化——编译器已经免费消除。**
把它们记在这里，是为了下一个 session 不再重复"手动缓存 pos_to_cell"这类无收益改动。

---

## 3. 提案（有收益潜力，但会改数值/确定性或需额外改动，未落地）

### 3.1 `donate_argnums` — 收益边际，且被 `init_state` 的 buffer 别名挡住 `[提案，非结论]`

想法：给 `make_scan` 的 `scan_steps` 加 `donate_argnums=(0,1)`，让 XLA 复用输入
`state`/`key` 的 buffer 当输出（省掉每次调用重新分配 ~98 MiB 的 state pytree）。三
个调用点（`server/app.py`、`run_headless.py`、`check.py`）都 `state,key = scan_fn(...)`
重新绑定，语义上安全。

- **是否改数值：不改，`[本世界实测]` 逐位一致**（smoke 与 full 两个规模 pop/energy
  完全相同）。
- **但两个问题让它不值得现在落地：**
  1. **收益边际。** `[本世界实测]` smoke +1.9%，full -0.2%（噪声内）。因为 `lax.scan`
     内部本来就跨迭代复用 carry buffer，donation 只省掉最外层一次输出分配，摊到 200
     步几乎为零。
  2. **`init_state` 的 buffer 别名会让首次调用直接崩。** `[本世界实测]`
     `INVALID_ARGUMENT: Attempt to donate the same buffer twice`——`init_state` 里
     `last_food=last_meat=last_damage=last_drink=generation=zeros` 全指向**同一个**
     `jnp.zeros(n)` buffer，donation 拒绝重复捐同一 buffer。要落地必须先给
     `init_state` 去别名（每个字段独立分配），这本身是第二处改动、需自己的论证。
- **结论：** 收益 0-2% 换一处 state 初始化改动 + 一个语义契约（输入被消费），
  cost/benefit 不划算。留作提案。真要做，先给 `init_state` 去别名，再只在
  `server/app.py` 这个反复调用、持有 state 的路径上开 donation（headless 的 scan
  本就一次性，收益更小）。

### 3.2 `reproduction.reproduce` 的 ~20 次 `place()` scatter — 消除来回 gather `[提案，非结论]`

`reproduce` 是最大热点（~1ms）。它对每个字段调 `place()`：

```python
def place(field, child_vals):
    keep = field[slot_idx]                        # gather：genome 是 90 MiB 读
    new_at_slot = jnp.where(is_birth_exp, child_vals, keep)
    return field.at[slot_idx].set(new_at_slot)    # scatter：90 MiB 写
```

因为 `slot_idx` 是 `[0, n_max)` 的**排列**，非繁殖项会把 `field[slot_idx[i]]` 原样
写回 `slot_idx[i]`（no-op 来回）。等价改写：

```python
born_mask = jnp.zeros(n, bool).at[slot_idx].set(is_birth)      # 每字段共用一次
child_at  = zeros_like(field).at[slot_idx].set(child_vals)     # scatter child
new_field = jnp.where(born_mask, child_at, field)              # 省掉 field[slot_idx] 的 gather
```

- **潜在收益：** 每字段省一次全数组 gather。对 genome（90 MiB）省一次 90 MiB 读，
  对全部 ~20 个字段累计可观。
- **是否改数值：** child 值逐位相同；但 `.at[slot_idx].set` 的 scatter（`slot_idx`
  唯一无碰撞）在 GPU 上的写入顺序会变，理论上仍是确定的（无碰撞 scatter 是纯写，不
  是 add），**但需实测确认 golden 逐位不变**——若浮点/布局差异让 golden 失配，说明
  它动了行为，就必须降级为提案而非 `--bless`。
- **风险：中。** 触碰 `CLAUDE.md` 明确保护的"排列-scatter"繁殖惯用法（"Preserve
  this pattern"）。改写虽保持排列语义，但足够微妙，值得单独一个 PR + 完整
  `test_kernel` + 长程 `run_headless` 验证捕食者不崩，不适合在这次"安全项"里搭车。

### 3.3 `last_input` / `last_output` 不进 scan carry `[提案，非结论]`

`[对应]` `state.last_input` `[n, in_dim=67]`、`last_output` `[n, out_dim=2]` **只**给
`server/app.py` 的 inspector 用，kernel 逻辑从不读。headless `scan` 里它们每步被
`place()`（gather+scatter `[16384,67]≈4.4 MiB`）、并在整个 scan carry 里被携带，却
只有最后一帧被用到。

- **潜在收益：** 小（4.4 MiB/step 的 place + carry），且只在 headless 有意义。
- **为什么不落地：** 它们是 `WorldState` 的字段，server 读它们；从 state 拆出去会改
  pytree 结构、改 server 契约，不是"行为不变"。真要做得给 headless 与 live 两条路
  分别的 state 视图，工程量与收益不成比例。

### 3.4 两遍邻居索引 — 设计使然，不可合 `[提案，非结论]`

`[对应]` `spatial.build_table`+`gather_neighbors`+`geometry` 每步跑两遍（sense 前一
遍、predation 前一遍），~0.47ms。第二遍**必须**看移动后的位置（`CLAUDE.md`：
"predation must see post-movement positions"）。位置变了，`argsort`/gather 结果不能
复用。**不可合并——合并即改行为。** 记此以免下个 session 误当成重复计算去删。

### 3.5 `build_table` 的 `argsort` → 计数排序 `[提案，非结论]`

`[对应]` `build_table` 用 `jnp.argsort(cell)`（`O(n log n)`）把 agent 按 sense-cell
分组。`cell` 值域只有 `[0, n_sense_cells] = [0, 577]`，理论上计数排序 `O(n)` 更快。

- **为什么不做：** (a) `build_table` 只 0.08ms，非热点；(b) 在 JAX 里写**稳定**的
  计数排序（要保持 cell 内 agent 顺序，`cummax` 那套依赖稳定性）复杂且易错；
  (c) GPU 上 `argsort` 已高度优化。收益小、风险不成比例。

---

## 4. 审计点逐条对照（任务清单）

| 任务点 | 结论 |
|---|---|
| 每步重复计算能否缓存 | §2：`pos_to_cell`/性状 sigmoid/`herbivory` 已被 XLA CSE 免费去重，手动缓存无收益 |
| 多余数组分配/拷贝 | §3.1 `donate_argnums`（边际+被别名挡住）；§3.3 `last_input/output`（收益小） |
| Python 循环 → `lax` 原语 | §1.1 sense 扇区循环 → scatter-max **已落地**（-1.3% 全步，逐位不变） |
| Python 侧分支 | 现有 `if cfg.los_occlusion_enabled` 等都是**编译期** Config 常量分支（`CLAUDE.md` 明确），已是最优——off 时整块不进 trace，无运行时代价 |
| step 每步顺序可合并的操作 | §3.4 两遍邻居索引不可合（predation 要移动后位置）；graze/eat_fruit 的 scatter 不同不可合 |
| `spatial` 邻居索引冗余 | §3.5 argsort→计数排序（非热点，不值）；两遍是必需 |
| `sensors` 视网膜采样冗余 | §1.1 扇区聚合已优化；LOS 块默认 off 时编译期消失 |
| `dynamics` 捕食判定冗余 | `predation` 0.20ms 非热点；`argmin` 最近猎物是必需的单遍 |
| `donate_argnums` | §3.1 提案：边际收益 + 需先给 `init_state` 去别名 |
| `jax.checkpoint` | **不适用。** checkpoint 是用重算换显存、服务于**反向传播**；本 kernel 无梯度、无反向、显存峰值仅 918 MiB，无需以算力换显存。 |
| 编译期常量内联 | Config 已 frozen、baked into jit，`in_dim`/`genome_size` 等是 derived property；所有 shape 与开关已是编译期常量，无进一步内联空间 |

---

## 5. 一句话总结给下一个 session

这个 kernel 已经相当紧凑、**显存带宽受限**，热点（`reproduce` 的整基因组 scatter 与
RNG）由固定形状+排列-scatter 设计锁死，无法在"行为不变"前提下大改。唯一落地的安全
优化是 **sense 扇区循环 → scatter-max（-1.3% 全步、逐位不变）**。"手动缓存
`pos_to_cell`" 一类是伪优化（XLA CSE 已免费）。真正有潜力的两个（`reproduce` 消除
来回 gather、`donate_argnums`）都需要单独 PR + 逐位验证，不适合当作免费项搭车——见
§3.1 / §3.2。
