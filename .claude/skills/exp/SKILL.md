---
name: exp
description: "跑消融/对比实验并记录 provenance：先写假设与成功判据，按 6 配对种子的统计地板设计，用 --set 跑两臂并把 git hash、完整 config、seed 列表落进 outputs/，跑完派 result-analyst 读日志，结论按四标签写进 docs/<topic>.md 并在 docs/TODO.md 加指针。任何要跑对比、消融、调参判定的场景都走这里。"
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
- 成功判据: <具体指标 + 方向 + 统计口径。例：6 配对种子 carn_water_dist 配对 Wilcoxon
  p<=0.05 且中位差 >= +5.0 单位>
- 失败判据: <什么结果算假设被证伪>
- 对照臂: <baseline 是什么 --set 组合；没有对照就写「无对照，仅探索」>
- 种子: <founder 种子列表；空间性结论还要写地形种子列表>
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

设计时逐条过：

1. **样本量地板：6 配对种子，或每臂 5 个不配对。3 个在地板以下**——3v3 的 Mann-Whitney
   最小双侧 p 是 0.10、3 对符号秩是 0.25，**无论数据长什么样都到不了 0.05**。
   n=6 配对符号秩能达到的最小双侧 p 就是 **0.031**，所以 0.031 是地板，不是强证据。
2. **报每个种子的数字，不只报均值。** `--json` 已经把它们吐出来了，把逐种子表格写进文档。
3. **检验用 Mann-Whitney（不配对）或配对 Wilcoxon（配对）**，加**效应量**和 **bootstrap
   区间**。均值 ± SD 不够。
4. **不做 Bonferroni 校正——把你算过的每一个 p 值都报出来。** 藏起来的比较比未校正的
   p 值更危险。
5. **功效要诚实**：种子间 SD 实测是 `inland_frac` ±0.027、`carnivore_frac` ±0.012。
   要在 80% 功效下检出 `inland_frac` 0.02 的移动**需要约 21 个配对种子**——照此编预算，
   或者直说这次功效不足（`docs/conventions.md` §5）。
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
**多种子可以直接并行**，十几个进程共存没问题：

```bash
mkdir -p outputs/20260723-fear-rate
for s in 0 1 2 3 4 5; do
  for arm in "fear_rate=0.0" "fear_rate=0.05"; do
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/run_headless.py 20000 500 \
      --seed "$s" --json --set "$arm" > "outputs/20260723-fear-rate/${arm}_seed${s}.log" 2>&1 &
  done
done
wait
```

长任务后台跑并说明怎么看日志，不要傻等；20000 步 ×12 是几十分钟量级，给足超时。

## 第 5 步 跑完：派 result-analyst 读日志，再写结论

**派 `result-analyst` subagent 去读日志和 JSON 行，不要自己读。** 十几个 run 的日志拉进
主 context 就再也清不掉了；而且它是 fresh eyes，不知道你希望看到什么结果。
派给它的 prompt 要写明：`outputs/<run_id>/` 路径、**成功判据原文**、两臂的 `--set` 定义、
种子列表、要求的检验（配对 Wilcoxon / Mann-Whitney + 效应量 + bootstrap CI）。

要求它回报：**逐种子**的指标值 + 每个数字来自哪个文件、检验统计量与 p 值（每一个都要，
不做 Bonferroni）、判据逐条「达成 / 未达成」。

## 第 6 步 写回 `docs/<topic>.md`

把结论补进第 1 步那一节，**四标签逐条标注**：

```text
- git hash: eca17e5（dirty: false）
- 结果: 逐种子表（seed / 基线 / 处理 / 差值），出处 outputs/20260723-fear-rate/*.log
- 结论:
  [本世界实测] 6 配对种子 carn_speed 1.5→2.4（6/6 同向），配对 Wilcoxon p=0.031（地板）。
  [本世界实测] carn_water_dist 仅 +1.4，成功判据「>= +5.0」未达成。
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
3. **不得用单次运行下生态参数的结论。** 少于统计地板就明确写「功效不足，不构成结论」。
4. **不得只报均值。** 逐种子数字必须在文档里。
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
