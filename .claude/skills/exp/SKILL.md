---
name: exp
description: "跑消融/对比实验并记录 provenance：先写假设与成功判据，按 12 种子 × 2 重复的统计协议设计，用 --set 跑两臂并把 git hash、完整 config、seed 列表落进 outputs/，跑完派 result-analyst 用 scripts/exp_stats.py 读日志，结论按四标签写进 docs/<topic>.md 并在 docs/TODO.md 加指针。任何要跑对比、消融、调参判定的场景都走这里。"
argument-hint: "[实验描述]"
allowed-tools: Read, Write, Edit, Glob, Grep, Task, Bash
---

# 跑实验并记录 provenance

实验：$ARGUMENTS

> 为什么 `allowed-tools` 里是完整的 `Bash`：本 skill 要**真的启动多种子 sweep**，长任务还要
> 后台跑、查日志，命令形态事先不可枚举。`allowed-tools` 是「本轮免批准的范围」，
> **不构成硬限制**。真正的护栏是 `.claude/hooks/`（PREALLOCATE 前缀、protect_paths）
> 和下面的硬约束。

## 第 1 步 先写判据（这是本 skill 存在的全部理由）

**先写判据再跑。** 先跑再解释，人一定会把任何结果讲成故事。本项目已经有一份模板级范例：
`docs/rebalance.md` 就是「用户假设的实测判决」——判据先立，然后数据说了不。

在目标 `docs/<topic>.md` 里**先建一节**（此时实验还没开始），字段照抄：

```text
## <实验名>（<日期>，run_id: 20260723-<slug>）
- 假设: <一句话，可证伪。例：把 fear_rate 提到 0.05 能让 carn_water_dist 上升>
- 成功判据: <具体指标 + 方向 + 统计口径。例：12 种子 × 2 重复，carn_water_dist 的格均值
  配对 Wilcoxon p<=0.05、>=9/12 同向，且中位差 >= +5.0 单位>
- 失败判据: <什么结果算假设被证伪>
- 护栏与容差: <哪些指标不许崩 + 容差。**容差对着 √2·σ̂_W 定**，不许写 ±10%>
- 对照臂: <baseline 是什么 --set 组合；没有对照就写「无对照，仅探索」>
- 种子: <founder 种子列表（默认 0..11）+ 每格重复数；空间性结论还要写地形种子列表>
- git hash: <第 3 步填>
- 结果: <跑完填，逐种子>
- 结论: <跑完填，带四标签>
```

判据写不出具体数字，就说明这个实验还没想清楚——先想清楚再跑。
主题已有文档就追加一节，没有就新建 `docs/<topic>.md`（一个主题一个文件）。

## 第 2 步 统计设计（本项目最贵的教训，全在这一节）

**绝不在单次运行上调生态参数。** 捕食者存活是近阈值的，run-to-run 方差超过大多数参数
效应：一个在单种子上看起来明显最好的配置，在四个种子上得了 0% 捕食者，整个第一轮结论
全是噪声。单种子探针只能用来看"跑不跑得起来"，不能产出任何结论。

**更糟的是：`--seed` 在 20000 步上根本控制不住这个世界。** 同种子、同配置、同代码重跑，
`carnivore_frac` 能跑出 0.060–0.2505。多数生态标量的创始者方差分量实测为 **0**
（`late_carn` / `death_thirst_frac` / `total_flux`），`carnivore_frac` 的 ICC 只有 0.130。
所以**"换种子"和"重跑同一个种子"吵得差不多**——这是 `docs/run_to_run_variance.md` 的核心
发现，也是下面整套协议的由来。

### 2.1 默认协议：**12 种子 × 每格 2 重复**（每臂 24 run）

| 项 | 值 |
| --- | --- |
| 种子数 `s` | **12** |
| 每格重复 `r` | **2** |
| 每臂 run 数 | **24** |
| 双侧 p 下限 | **0.00049** |
| 两臂一轮 | 48 run，约 45 分钟 |

**为什么是加种子而不是加重复**（配对差均值的 `SE = σ_W·√(2/(s·r))`，**只看乘积**，
所以同预算下两者对精度贡献一样，但）：

1. **符号秩 p 下限只由 `s` 决定**，是 `2/2ˢ`：n=3→0.25、**n=6→0.031**、n=12→0.00049。
   本项目结论常年正好落在 0.031，**那不是数据干净，是撞到了 n=6 的天花板**。
2. **伪重复也只由 `s` 决定**：`terrain.build(cfg)` 不吃 RNG，多一个种子多一次创始者
   抽样，多一次重复什么都不多。
3. **`r` 唯一不可替代的作用**：让实验**自己估出自己的噪声**。`r=1` 时只能引用存档值，
   `r≥2` 就能就地估。所以 `r=2` 够了。

预算不够 12×2 时**如实写"本轮功效不足"**，不要假装 6 个种子还是地板。

### 2.2 分析口径（三条，缺一不可）

**别自己重推这段算术——`scripts/exp_stats.py` 是它的唯一实现，import 它。**
（在共享它之前，同一段代码已经在三份 `analyze*.py` 里被抄了三遍。）

1. **先把每格 `r` 次重复取均值，再对 `s` 个格均值做配对检验。**
   不得把 `s×r` 个 run 当成 `s×r` 个独立样本——重复是同种子复跑，只含混沌噪声。
   → `RunSet.cell_means()`
2. **必报「效应量 ÷ 配对差噪声」**，噪声 `= √2·σ̂_W/√r`，`σ̂_W` 从**本次实验**的格内
   散度自估（**不许引用别处的数**）。**比值 <1 一律标注功效不足，哪怕 p 过线。**
   → `PairedResult.ratio` / `.underpowered`
3. **护栏容差必须对着 `√2·σ̂_W` 定**，不许凭直觉写 ±10% / +5pp——那大约就是 1 个噪声 SD，
   零效应也会经常撞线。参考量级：`carnivore_frac` 的 `√2·σ_W ≈ 0.0625`、
   `death_thirst_frac ≈ 0.066`。 → `RunSet.pair_noise()`

用法：

```python
import sys; sys.path.insert(0, 'scripts')
from exp_stats import RunSet, paired, mde_sign_consistent, required_seeds

rs = RunSet.load('outputs/<run_id>', arms=['ctrl', 'treat'], seeds=range(12), reps=(1, 2))
assert not rs.problems, rs.problems          # 载入自检：缺文件/JSON 行数/seed 不符/collapse
assert rs.overrides_diff() == ['<那一个变量>']  # 多于一项 = 多变量混杂，判决作废
print(paired(rs, 'carnivore_frac', 'treat', 'ctrl').format())
```

**报零结果时必须一起报 MDE**（`mde_sign_consistent(observed_sd, s)`）：零结果只能排除
**大于它**的效应，更小的窗口本设计看不见。

**设计阶段先算功效，别事后解释为什么没跑出来**：`required_seeds(effect, sigma_w, reps=2)`
给所需种子数，`power_paired_wilcoxon(effect, pair_noise, s)` 给现有设计的功效（配对
Wilcoxon 的精确零分布 + 蒙特卡洛，小 n 下无正态近似误差）。参考量级：`carnivore_frac`
的 `σ_W = 0.0442`，12×2 对 +0.05 的功效是 0.92，但要检出 +0.04 就得 13 个种子——
**这个世界最吵的那个指标，小效应本来就贵**。

### 2.3 其余四条不变

4. **报每个种子的数字，不只报均值。** `--json` 已经把它们吐出来了，把逐种子表格写进文档。
5. **不做 Bonferroni 校正——把你算过的每一个 p 值都报出来。** 藏起来的比较比未校正的
   p 值更危险。
6. **伪重复警告**：`terrain.build(cfg)` 不吃 RNG，所以**每个种子跑在同一张地图上**——
   种子只变创始者，不变世界。任何空间性 claim 只能推广到**这一条河系**，不是河流一般。
   要做空间结论就交叉地形因子：`ridge_wavenumber` / `ridge_amplitude` / `ridge_base_y`
   加河源，和创始者种子当两个因子交叉（刚性平移不算——世界是环面，整体挪一下什么都没变）。
7. **空间指标先算零假设**：`inland_frac = 0.30` 在你知道随机放置给出 **0.556–0.675**
   之前不叫"低"。种群坐在比随机低 0.35 的位置，那才是发现；一个 +0.02 的效应是那个
   差距的 5.8%——这才是诚实的量级表述（`docs/conventions.md` §7）。

## 第 3 步 跑之前：可复现性

```bash
git status --short
git rev-parse HEAD
```

工作区不干净时，**先警告用户**：

> 工作区有未提交改动，本次结果将无法复现（跑的代码和任何 commit 都不对应）。
> 建议先 commit 再跑。要继续吗？

用户坚持就继续，但必须在记录里标 `dirty: true` 并列出脏文件。

## 第 4 步 跑：两臂、多种子、provenance 落盘

**消融一律用 `--set FIELD=VALUE`（可重复传参），不要改 `config.py`。** 改 config 会让两个臂
落在两棵不同的工作树上，事后无法配对；而覆盖项会被写进 `--json` 行，自带 provenance。

产物目录用 **`outputs/<run_id>/`**（已在 `.gitignore` 里，不进仓库）。每个 run 一个新目录，
**绝不复用别的 run 的目录**——覆盖等于毁掉那次实验的证据。

跑之前先往新目录里写一个 `provenance.txt`：`git rev-parse HEAD` 和是否 dirty、
**完整的 resolved config**（展开后的最终配置内容，不是配置文件路径，也不是命令行片段）、
seed 列表、完整启动命令，以及环境：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -c \
  "import sys, jax; print(sys.version); print(jax.__version__, jax.devices())"
```

因为每个进程都带 `XLA_PYTHON_CLIENT_PREALLOCATE=false`（实测真实峰值 918 MiB），
**多种子可以并行**——但**并发上限 6**：瓶颈不是显存也不是 GPU，是**主机 CPU 上的 XLA
编译**，压满了整套会一起变慢（`MEMORY.md` `[LEARN:env]`）。

**sweep 一律写成脚本文件再 `bash` 执行，公共臂用数组传**——本项目的 shell 是 zsh，
它**不做单词分割**，内联的 `$COMMON` 会被当成一个参数整体传进去（`MEMORY.md`
`[LEARN:shell]`）。附带好处：脚本文件本身就是可复跑的 provenance。

文件名必须是 **`{arm}_s{seed}_r{rep}.log`**——`exp_stats.RunSet.load` 默认按这个解析。

```bash
# 写进 outputs/<run_id>/sweep.sh，然后 bash outputs/<run_id>/sweep.sh
RUN=outputs/20260723-fear-rate
mkdir -p "$RUN"
COMMON=(20000 500 --json)                  # 数组，不是字符串
declare -A ARM=( [ctrl]="fear_rate=0.0" [treat]="fear_rate=0.05" )

for s in $(seq 0 11); do                   # 12 种子
  for r in 1 2; do                         # 每格 2 重复
    for a in ctrl treat; do
      while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done   # 并发上限 6
      XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/run_headless.py \
        "${COMMON[@]}" --seed "$s" --set "${ARM[$a]}" \
        > "$RUN/${a}_s${s}_r${r}.log" 2>&1 &
    done
  done
done
wait
```

**重复怎么变得不一样**：同种子复跑的差异来自 GPU 原子重排本身，不需要（也不该）改
`--seed`——`_r1`/`_r2` 传的是同一个 `--seed`，这正是 `σ̂_W` 要估的那份噪声。

长任务后台跑并说明怎么看日志，不要傻等；48 个 run 在并发 6 下是**约 45 分钟**量级，
给足超时。**输出一律重定向到文件，不要让几万行日志流经终端**——那会把终端模拟器刷成
一个跑满一核的进程，反过来拖慢 XLA 编译（实测把 `check.py --full` 拖慢 2.7 倍）。

## 第 5 步 跑完：派 result-analyst 读日志，再写结论

**派 `result-analyst` subagent 去读日志和 JSON 行，不要自己读。** 几十个 run 的日志拉进
主 context 就再也清不掉了；而且它是 fresh eyes，不知道你希望看到什么结果。
派给它的 prompt 要写明：`outputs/<run_id>/` 路径、**成功判据与护栏容差原文**、两臂的
`--set` 定义、种子与重复列表，以及一句硬要求：

> **统计走 `scripts/exp_stats.py`，不要自己重写这段算术**（口径见 `docs/conventions.md`
> §5.2）。分析脚本写进 `explorations/<run_id>/analyze.py` 并 import 它。

要求它回报：**逐种子的格均值** + 每个数字来自哪个文件、`rs.problems` 与
`rs.overrides_diff()` 的自检结果、检验统计量与 p 值（每一个都要，不做 Bonferroni）、
**效应/噪声比及是否 <1**、零结果时的 MDE、判据逐条「达成 / 未达成」。

## 第 6 步 写回 `docs/<topic>.md`

把结论补进第 1 步那一节，**四标签逐条标注**：

```text
- git hash: eca17e5（dirty: false）
- 结果: 逐种子表（seed / 基线 / 处理 / 差值），出处 outputs/20260723-fear-rate/*.log
- 结论:
  [本世界实测] 12 种子 × 2 重复，carn_speed 格均值 1.5→2.4（11/12 同向），
    配对 Wilcoxon p=0.0015；效应/噪声比 1.8（自估 σ̂_W=0.31，配对差噪声 0.31）。
  [本世界实测] carn_water_dist 仅 +1.4，效应/噪声比 0.4——**功效不足，p 即使过线也不算结论**；
    本设计 80% 功效的 MDE 是 +3.6，所以只排除掉「大于 +3.6」的效应。
    成功判据「>= +5.0」未达成。
  [对应] 机制在 underworld/ecology.py 的 fear 场散射沉积，折入 sensors 的 pred 通道。
  [现实] 恐惧地景在真实系统里的证据见 docs/biology.md 对应小节。
  [提案，非结论] 要真正实现"搬离河岸"可能需要昼夜通勤，未测。
```

然后：

- **负结果照样完整写**——它们是重点，不是丢人的事，是唯一能阻止同一个想法被重试的东西。
  纯负结果的机制实验，把它加进 `docs/experiments.md`（失败实验档案）。
- **在 `docs/TODO.md` 加一行指针**（文档地图表 + 需要时的判决表）。
- 提醒用户：**结论没落进 `docs/` 并提交，按 `CLAUDE.md` 就等于没做过。**

## 硬约束（违反即本次实验作废）

1. **不得口算或估算任何指标。** 不许从日志里心算平均值、不许目测报数字、不许把
   「大概 0.85」写成 0.85。
2. **不得报告没有落盘文件支撑的数值。** 每个数字要能指到 `outputs/<run_id>/` 下的某个文件。
3. **不得用单次运行下生态参数的结论。** 少于 12 种子 × 2 重复就明确写「功效不足，
   不构成结论」——不要假装 6 个种子还是地板。
4. **不得只报均值。** 逐种子格均值必须在文档里。
4b. **不得把 `s×r` 个 run 当独立样本做检验**（重复是同种子复跑，是伪重复）。
    **不得省略「效应量 ÷ √2·σ̂_W/√r」这个比值**；比值 <1 而 p 过线时，
    必须写「p 过线但功效不足」，**不许只报 p**。报零结果必须同时报 MDE。
5. **不得复用别的 run 的输出目录**，也不得手写数字进去——数字只能由脚本产生。
6. **不得在工作区脏的情况下静默开跑**，必须先警告。
7. **绝不为了让实验"成立"而放宽 `scripts/golden.json` 的 band。** 改动本就该移动 golden
   数字时，用 `--bless` 重录并在 commit message 说明理由；band 为了掩盖失败而放宽，
   等于把这个检查删了。
8. **不得把 `[提案，非结论]` 悄悄升级成 `[本世界实测]`。** 只有跑出来的数字能当实测。

### 需要一个还没有的指标时怎么办（合法路径）

不要绕过约束 1 自己算。正确做法：写一个脚本来算，从 `outputs/<run_id>/` 的产物读入、
把结果写成文件；脚本进版本控制（放 `scripts/`，这样这个数字以后可以被重新算一遍）；
跑脚本，报告它输出的文件里的数字。临时试探性脚本放会话 scratchpad，不要进仓库
（`CLAUDE.md`：截图、scratch 脚本、`outputs/` 不进仓库）。
