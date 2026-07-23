# 昼夜系统（Day–Night / Diel Cycle）

这份文档承接 `docs/landscape_of_fear.md` §6 留下的那条主线。恐惧场把捕食者从"坐等"
推向"游走"（`carn_speed` 1.5→2.4，6/6 种子，配对 p=0.031），但**给不了"搬离河岸"**
（`carn_water_dist` 仅 +1.4，噪声大）。§6 结论 3 明确点名了缺口：

> 本世界水是硬约束、捕食者又恰驻水边，"水边即最恐惧处"的公共信息无法在不把猎物推向
> 渴死的前提下把捕食者赶离水——真正缺的是**昼夜通勤**（猎物快饮即走），而这个无昼夜
> 时钟的世界给不了。

所以昼夜系统是这条线的**下一步**，不是新方向。

标记体例沿用 `docs/biology.md`：**[现实]** 已发表事实、**[本世界实测]** 在这个代码库
跑出来的数字、**[对应]** 落到哪段代码、**[提案,非结论]** 尚未验证的设计建议。

---

## 1. 设计约束（用户定盘）

两条轴由用户明确定盘：

1. **不给大脑显式时钟输入。** 不加 sin/cos 相位通道——那等于把"现在是几点"直接喂给
   大脑。改为让"感知昼夜"**自己演化出来**：环境节律的感官*后果*（夜里看不清别的 agent、
   正午水分掉得快）就是内生时钟可以 entrain 的 zeitgeber。**[提案]** 用户明确留了后手：
   若现有 16 个 recurrent units 学不出内生时钟，就适当加大 `hidden` 或单独进化一个小的
   "感光"子网络（见 §5 Phase 2）。
2. **昼夜同时驱动热与暗（Combination）。** 正午热→耗水↑ + 视野长；夜里凉→耗水↓ +
   视野短。

**为什么这套能给出通勤：** 正午 = 热（离水代价高）+ 看得远（不易被伏击）；夜里 = 凉
（离水便宜）+ 看不清（伏击占优）。猎物想夜里去内陆觅食（水便宜）但夜里危险，正午能看见
捕食者却离不开水——这正是 §6 要的"错峰饮水"的适应度地形。**[现实]** 与 `landscape_of_fear.md`
§1 引的 Hwange 水坑昼夜通勤（白天觅食、傍晚迁移数公里外过夜）、以及夜间捕食者视觉占优
（同文 §"感官不对称"）一致。

---

## 2. 机制（Phase 1，已落地，默认关）

全部逻辑用 `if cfg.day_length > 0:` 编译期分支包住（`cfg` 被 `build_step` 闭包、不被
trace）。`day_length=0`（默认）时整块从 jit 里消失、`phase` 恒为 0、下游全是逐位 no-op
——与 `fear_rate=0` 完全同一个惯例。**[对应]** 这一相**不动任何 `[n_max]`/genome 形状**
（只给 `WorldState` 加一个标量 `phase`），故 founder RNG 不变、`day_length=0` 逐位复现
旧基线、golden 不用重 bless（实测 `check.py` 41 项过、golden band 10 项 held、population
1549 与旧值一致）。

| 部件 | 代码 | 说明 |
| --- | --- | --- |
| 全局时钟 | `state.py` `WorldState.phase`（0 维 f32）、`step.py` §7a'' | 每步进 `1/day_length`、模 1；`init` 置 0（午夜）。步尾写、下一步步首读，同 trample/fear 的"沉积-次步读"惯例 |
| 光照 | `light = 0.5*(1 - cos(2π·phase))` | 午夜(phase0)=0、正午(phase0.5)=1 |
| 暗→视野 | `sensors.sense`，`vision *= night_vision_floor + (1-floor)·light` | **只缩 `closeness`（看别的 agent：prey/pred/peer）**；`water_ch`/food 通道刻意用 `cfg.vision_radius`/`plant_max` 原值不变暗——夜=纯粹伏击杠杆，绝不更难找水/找食 |
| 热→耗水 | `dynamics.thirst` 可选 `light` 形参，`cost *= 1 + heat_water_amp·light` | 正午耗水抬到 `(1+heat_water_amp)×`；`metabolize` 不动。可选参默认 `None`，保留旧签名与单测（同 `metabolize` 里 attack/escape 可选参惯例） |
| 度量 | `metrics.py` `Metrics.phase`（追加） | 逐步 stack，供相位分箱分析；追加、绝不插入（run_headless 按名读） |

### 参数（`config.py`，仅 `day_length>0` 时读后两个）

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `day_length` | `0`（关） | 一个完整昼夜的步数。0 = 编译期 no-op、逐位旧基线。6 种子拟用工作值 **400**（见 §3） |
| `heat_water_amp` | `0.5` | 正午热峰耗水相对增幅 |
| `night_vision_floor` | `0.4` | 午夜时"看别的 agent"的视野乘子（1.0=不变暗） |

---

## 3. 时标依据

**[本世界实测]** 相关时标：幼体渴死均龄 ~52 步、恐惧/踩踏半衰 ~69、记忆半衰 346、
`max_age` 3000。`day_length=400` 的取值理由：

- 让成体活约 **7.5 个昼夜**，昼夜是它一生反复经历的节律，值得学。
- **远高于 69 步的恐惧半衰**，节律不会被恐惧场自身的时间平滑抹掉。
- 幼体只经历不到一个昼夜，所以昼夜对**幼体渴死瓶颈的好处是间接的**（成体通勤→更好
  供给/更少被捕），不是直接救幼体。这一点要在实测里盯：不能因为暗夜/热午把幼体渴死
  推高（见 §4 头号风险）。

`day_length` 是可 sweep 的常数，400 是拟用起点，不是定论。

---

## 4. 验证协议（6 种子，待跑）

**[提案,非结论]** Phase 1 机制已落地、默认关、逐位可逆。下一步是 6 配对种子实测决定
是否翻默认开启。铁律照 `docs/conventions.md`：

- `run_headless 15000 200 --seed {0..5} --json --set day_length=400` 对
  `--set day_length=0` 控制臂**配对**比较。逐种子报数、配对 Wilcoxon + bootstrap 区间、
  **不做 Bonferroni**、报每个 p。n=6 双侧符号秩 p 地板 0.031（是地板不是强证据）。
- **空间结论需跨地形种子**（`ridge_wavenumber`/`ridge_amplitude`/`ridge_base_y`），
  否则只对这张地图成立、是伪重复。
- **主判据 = 相位分箱通勤**：把整段 run 的 `carn_water_dist`（及 herb）与 `phase` 全序列
  stack，按 `light≷0.5`（昼半/夜半）分组求均值。通勤 = 捕食者在一个半相位离水、另一个
  半相位贴水。静态均值分不清"错峰"（这正是 §6"未做"点出的、静态均值分不清"缩短停留"
  与"完全不去"的同一病），所以必须分箱。
- **头号风险 = 渴死**：`death_thirst_frac`/`death_thirst_age` **不得恶化**。暗只动
  inter-agent 视野、不动找水，就是为压这个风险；若仍恶化，回退或调 `night_vision_floor`
  /`heat_water_amp`。
- **落地纪律**：先默认关合入；**只有** 6 种子拿到通勤且不伤渴死，才把默认翻开（照抄
  恐惧场"验证后才默认开启"的先例）。翻开后 golden 会微动、按预期重 bless。

---

## 5. Phase 2 —— 大脑容量（条件触发，用户点名的后手）

**[提案,非结论]** **触发条件**：Phase 1 用现有 16 units 跑 6 种子，若相位分箱**测不到**
通勤（捕食者昼夜离水距离无差异），才做 Phase 2。而非默认假设需要——16 units 理论上够
放一个振荡器（2–3 units 即可成极限环），真正的问题是**演化**能否找到它，加容量未必有
帮助，所以先测 16 作为干净对照。

**推荐做法（最小）**：`config.py` `hidden: 16 → 24`。`brain.py` 的 `split_params`/
`forward` 全由 `cfg.hidden` 派生、无需改结构；`genome_size` 自动跟着动、`WorldState.hidden`
变 `[n_max,24]` → founder 重排 → **一次性种群作废 + golden 重 bless**（同红皇后 trait_dim
3→5 的作废惯例）。两臂对照仍是 `day_length=0` vs `>0`，在**同一** 24-unit 基底上比，
故循环效应依旧干净隔离；另跑 24-unit / `day_length=0` vs 现 16-unit 核对"单纯加容量是否
改变基线"。

**备选**：若纯加容量仍学不出，`brain.py` 里单独接一个几 units 的小 recurrent"感光"子模块
喂入主网络——结构性改动、更贵，仅在加容量被证伪后再上。

---

## 6. 现状

- **Phase 1 机制**：✅ 已落地，默认关（`day_length=0`），`check.py` 41 项过、golden 逐位
  held、pytest 全过。逐位可逆（`--set day_length=0` 恢复旧行为）。
- **6 种子实测**：⏳ 待跑（§4）。
- **默认是否翻开**：⏳ 待实测判决。
- **Phase 2 大脑容量**：⏳ 仅在 Phase 1 通勤为 null 时触发（§5）。
