# AI 冗余代码审计与清理

一次针对 `underworld/`、`scripts/`、`server/`、`web/` 的"清理而非重构"审计：找死代码、
重复逻辑、过度啰嗦实现、过期/自相矛盾注释、命名不一致，并落地**行为完全不变**的安全清理。

**纪律**：golden 带宽必须 held、全部 pytest 必须过、删任何东西前 grep 全仓确认无引用。
`CLAUDE.md` 记录为"刻意如此"的设计（定形张量、permutation-scatter 生育、记忆不遗传、
按位置分区记忆槽、水/能量双账本、cross-file 契约）不动。注释密度刻意匹配周围——本仓
的长"为什么"注释是资产，只删真正过期/错误/纯复述代码的。

标注沿用本仓词汇：`[对应]` 落在代码哪里、`[本世界实测]` 在此测得。

---

## 1. 死代码：核实结果——没有发现新的

- `[对应]` `ecology.prey_field` 与 sine-stream helpers：`CLAUDE.md` 点名的历史残留，
  grep 全仓 `prey_field` **零命中**，确已删除。`ecology.gradient` 仍活（`terrain.build`
  在 §build 里 `from .ecology import gradient` 局部导入并用于地形坡度），`CLAUDE.md` 的
  "Dead code note" 与现状一致，无需改。
- `[对应]` 未用 import：写了一个 AST 扫描器过一遍 `underworld/`、`scripts/`、`server/`
  的所有 `.py`。除 `from __future__ import annotations`（`__future__` 副作用导入，非死码）
  与 `underworld/__init__.py` 的公共再导出（有 `__all__` 兜底）外，**无真正未用的 import
  或未用局部变量**。这部分代码已经是干净的。
- `[对应]` `Metrics.fruit_total` 在 `web/main.js` 里被 `parse()` 解出并放进 `latest`，
  但 `tick()` 未展示它——这是线协议 header 的固定字段（v6），解出来是契约完整性，不是
  死码，**保留**。

## 2. 重复逻辑：落地一处去重

- `[对应]` `dynamics.graze` 与 `dynamics.eat_fruit` 里**逐字重复**的食草能力锥度：
  `jnp.where(state.diet > cfg.carn_graze_cutoff, 0.0, (1.0 - state.diet) ** 6)`。
  提取为模块私有 `_herbivory(diet, cfg)`，两处改为调用。算术完全相同，golden held。
  解释锥度形状的长注释仍留在 `graze` 的调用点（那是资产），helper 的 docstring 指向它。

- `[对应]` **疑似重复但保守未动**：`graze` 与 `eat_fruit` 的整段"按格需求池 →
  `minimum(demand, field)` → 公平分摊 `frac[cell]` → `gain`"结构也近乎一致，可以合成一个
  `_forage_pool(field, demand, ...)`。未动，两条理由：(1) 两者返回语义有别（`graze` 的
  `gain` 直接是能量，`eat_fruit` 先算 `taken` 再乘 `fruit_energy`，且水返回口径不同），
  合并会牵动返回签名；(2) 这是每步热点，**性能审计 agent 正在并行改 `underworld/`**，
  按协调约定把这块大改让给它，避免抢同一段热点。

## 3. 过期/自相矛盾的注释：修正三处（纯注释，零行为改动）

- `[对应]` `memory.py` 模块 docstring：旧文写"An earlier version copied the parent's
  slots at birth -- Lamarckian, and **measured to do nothing**"。这正是 `docs/conventions.md`
  §4 与 `CLAUDE.md` 记忆章节明确点名并已更正的那次 overclaim——`reproduction.py:210` 早已
  写成诚实版本（n=6 配对、+0.020、p=0.175、25% 功效、TOST 上界 0.05），唯独 `memory.py`
  的 docstring 还带着旧的过硬断言。改为与 §4 一致的"功效不足、非零结果、等价检验上界 0.05"，
  并指向 `reproduction.py` 与 `docs/conventions.md` §4。

- `[对应]` `sensors.sense` 函数 docstring：把返回向量写成
  `[food(R), prey(R), predator(R), water(R), slope(R), energy, diet, own_water]`——
  **漏了实际 concatenate 末尾追加的 `memory(4*slots)` 与 `peer(R)`**（模块 docstring 提到了
  peer，函数 docstring 没跟上）。补全为与函数末尾 `jnp.concatenate` 实际顺序一致，并说明
  "追加在末尾而非插入，保 `server/app.py` 的 retina 切片偏移有效"。

- `[对应]` `spatial.geometry` 里 `half = cfg.world_size / 2.0` 的 torus 半宽字面量：见 §4。

## 4. 命名/常量一致性：一处

- `[对应]` `Config.half_world` 属性存在且 docstring 写明"沿一轴的最大 torus 距离"，
  `memory.advance` 与 `terrain.py` 都用它，唯独 `spatial.geometry` 手写 `cfg.world_size / 2.0`。
  改用 `cfg.half_world`，消除重复的 torus 半宽魔数、与其余最短向量 wrap 的写法统一。值恒等，
  jit 内常量折叠，行为不变。

## 5. 明确未动（设计而非冗余）

- `web/main.js` / `web/render.js` / `web/index.html`：`main.js` 通读后**无死码**；
  三处重复的物种颜色是 `CLAUDE.md` 点名的 cross-file 契约、由 `check.py` 机械守，不是啰嗦。
  shader 改动按 §10 需"用眼睛验"，本次不碰 `render.js`。`index.html` 里 `--plant` 注释与
  已消失的 render.js 常量对不上，`CLAUDE.md` 已记录此事，属已知、非本次清理目标。
- `config.py` 的长参数注释（`carn_cost`/`plant_max`/`n_init`/水系列…）：记录调过的参数与
  否掉的尝试，是资产，一字未动。
- `genome.py` 模块 docstring 的"(asexual-triggered) crossover ... 完整有性繁殖是后续里程碑"：
  措辞可推敲，但严格说仍准确（出生由单亲能量阈值触发；无求偶行为；只单亲付投资能量），
  未改以免把当前语义描述歪。
- `step.py` 模块 docstring 提 `scan_steps` 而公共名是 `make_scan`：`scan_steps` 是 `make_scan`
  返回的内层函数名，非明显错误，未动。

## 6. 验证 `[本世界实测]`

- `check.py`（tier 2）：39 项全过，**golden band held for 10 metrics**（种群 1520 存活）。
- `check.py --full`：40 项全过，含整套 pytest（164.7s）。
- `_herbivory` 提取是逐字相同的算术，golden 未漂移即为行为等价的直接证据。
- 改动文件：`underworld/dynamics.py`（去重）、`underworld/memory.py`、`underworld/sensors.py`
  （过期注释）、`underworld/spatial.py`（命名一致）。均为 `underworld/` 下、与性能审计
  可能重叠——倾向删除/简化类低冲突改动，主 agent 合并时处理重叠。
