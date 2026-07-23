---
name: plan
description: "Research + Plan 两段式：先查失败实验档案确认这个想法没被否过，再派 subagent 调研写 research.md，然后写精确到文件和验证判据的 plan.md，最后派 plan-critic 对抗式审查。用于根因不明、涉及多文件或有方案权衡的改动。"
argument-hint: "[要做的事]"
disable-model-invocation: true
allowed-tools: Read, Write, Glob, Grep, Task, Bash(git status:*), Bash(git log:*), Bash(git diff:*)
---

# Research + Plan

任务：$ARGUMENTS

两段式。Research 段只读不写代码，Plan 段只写文档。**全程不改任何源文件。**

## 第 0 步 先劝退（最先做，不满足就直接停）

判据：**如果这个改动一句话就能描述完 diff，就不该走 plan。**

比如：改个常量、加条 log、重命名变量、修 typo、往已知位置加一个字段。

这种情况直接输出：

> 这个不值得走 /plan——它一句话就能说完（把要改的那一句写在这里）。
> 直接说「改吧」我就动手。

然后 **return，本次 skill 到此结束**。不要建目录、不要写 research.md、不要「顺便先调研一下」。

放最前面的理由：反模式的成本发生在建目录那一刻。一旦开始写 research.md，
后面每一步都会为「已经开了头」自我合理化，最后用 200 行文档描述一个 3 行的改动。

满足下列**至少一条**才继续：

- 不知道该改哪些文件，或涉及 3 个以上文件
- 根因不明确，需要先查
- 有多个可选方案要权衡
- 会动生态参数、`in_dim`/`trait_dim`/wire format，或影响已有实验结果的可比性

## 第 1 步 查这个想法有没有被否过（本项目专属，最高杠杆，不能跳）

`CLAUDE.md` 写着：想到一个"显然该 work"的机制时，先查这里有没有被否过。**这一步比任何
调研都省时间**——已经有果实层、被动踩踏、拆反混合装置、踩踏正反馈、`vision_radius` 可遗传、
`mutation_sigma` 可遗传、"减慢植物生长赶捕食者"整批被实测否掉过。

按顺序查三处，把命中的结论**原样引用进 research.md**：

1. `docs/experiments.md` —— 失败实验档案（一节一个曾经"显然正确"的设计 + 它错在哪）。
2. `docs/TODO.md` 的「## 不做的事」和「已实现、已验证、结论已归档」表 —— 一眼能看出
   这条是不是已经有判决。
3. `docs/TODO.md` 的**文档地图表** —— 每行的「什么时候读它」直接告诉你该翻哪份专题文档
   （`rebalance.md` 管资源总量、`landscape_of_fear.md` 管捕食者离水、`trait_roadmap.md`
   管新性状代价货币、`mortality.md` 是所有性状实验的分母、`conventions.md` 管规则的论证）。

命中已被否过的方案 → **先停下告诉用户**，引用那份文档的判决和数据，问「是要换方向，还是
你认为当时的实验有缺陷、要重做」。不要装作没看见然后重新提一遍。

## 第 2 步 Research

### 派 subagent 并行调研

**不要自己一个个读文件。** 探索会读进大量最终用不上的内容，那些不该留在主 context 里。
派 Explore subagent 并行去查，每个 subagent 一个明确问题，只回报结论和 `文件:行号`：

- subagent A：这条链路的入口和调用链在哪（`underworld/step.py` 的每步顺序是硬约束）
- subagent B：数据怎么流的，形状/单位在哪一步变，会不会碰到跨文件契约
- subagent C：项目里已有的同类实现怎么做的（`config.py` 的长注释记着试过什么）

需要真实文献背景走 /lit。需要历史实验数值时**去 `docs/` 下对应的专题文档查，不要猜**。

主 context 只保留 subagent 的回报，不要把原始文件内容整段拉进来。

### 写 research.md

路径 `docs/plans/<YYYYMMDD>-<slug>/research.md`（slug 用小写英文加短横线）。固定五节：

1. **先例检查** —— 第 1 步查到的东西：这个想法在 `experiments.md` / TODO 判决表里
   有没有前科，命中就写清判决和数据；确实没有也要写「查过，无前科」。
2. **相关文件位置** —— `路径:行号` + 一句话职责。只列真正相关的，不是目录树。
3. **数据如何流动** —— 从输入到输出经过哪些函数，哪一步改变了形状/单位/量纲。
4. **根因假设** —— 至少 1 条，最好 2-3 条互相竞争，每条写明**怎么才能证伪**。
5. **既有模式与约束** —— 定形张量与 permutation-scatter、terrain 闭包不进 state、
   config 烘进 jit（改 `in_dim`/`genome_size` 会作废整个演化种群）、三处重复的物种颜色、
   wire format 只能 append。

每条断言按项目四标签标注来源：`[现实]` 已发表事实、`[本世界实测]` 本项目测出（附文档出处）、
`[对应]` 落在代码哪一行、`[提案，非结论]` 我的猜测。**核不出来的就明说核不出来**，
不要洗成自信散文。

## 第 3 步 Plan

写 `docs/plans/<YYYYMMDD>-<slug>/plan.md`。

### 精度要求（最硬的格式要求）

每一步必须回答四个问题，缺一个这步就是废的：

1. **改哪个文件** —— 精确到路径，能到函数/行号更好
2. **改什么** —— 「把 A 换成 B」「新增函数 f，签名是 …」。不要写「优化 X」「重构 Y」
3. **怎么验证** —— 一条可直接粘贴执行的命令（本项目的三档见下）
4. **成功判据** —— 看到什么算过。「检查通过」不够，要写清哪一项、期望什么输出

验证命令按代价从轻到重选，**别一上来就挂 `--full`**：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --contracts  # 0.2s 跨文件契约
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py              # 14s +golden band
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --full       # 3min +pytest，提交前
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/run_headless.py 20000 500 --json
```

改了 shader 或 `web/` 的 phase，验证必须包含**截图**（`docs/conventions.md` §10），
不能只写 code review。

### 结构

````markdown
# Plan: <标题>
关联 research: ./research.md

## 目标
一句话：做完之后什么变成了可能，或什么问题消失了。

## 不在范围内
- 显式列出这次**不做**的事，至少 2 条
- 尤其是那些「顺手就能做」但会让 diff 变大的事
- 这一节是实现阶段用来拒绝范围蔓延的依据，必须写

## Phase 1: <名字>
- 改 `underworld/dynamics.py:120` 的 `predation()`：<改什么>
- 验证：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py`
- 判据：golden band 十项全过，`carn_frac` 落在 <具体区间>

## Phase 2: <名字>
...

## 端到端验证
- 命令：<一条把整条链路跑通的命令>
- 判据：<具体到数字或具体输出>

## 发现但未做
（实现阶段往这里追加，本阶段留空）
````

分 phase 的原则：**每个 phase 结束时代码必须可运行、可验证、可提交**，而且
**一个 phase = 一个"变更理由"**——本项目按理由拆 commit，phase 边界就该是 commit 边界。
一个 phase 大到需要超过一轮对话就拆开。

### 涉及生态参数或空间结论的 plan，必须把实验独立成步

不要在 plan 里写「跑一遍看看效果」。生态参数改动的判定要走 /exp：**6 配对种子起**
（3 个在地板以下），报每种子数字，空间性结论还要跨地形种子。plan 里就写清
「Phase N：按 /exp 的统计纪律跑 6 配对种子消融，判据 = <指标> 的配对 Wilcoxon p 与效应量」。

### 结尾必须有端到端验证

phase 级验证只证明局部没坏。plan 末尾一定要有一条把整条链路走通的命令和判据。
没有这一节的 plan 不算写完。

## 第 4 步 对抗式审查（不能跳）

plan.md 写完后，**派 `plan-critic` subagent** 审这份 plan，明确要求它挑刺：

- 哪一步的成功判据含糊（「能跑通」「效果更好」这类）
- 哪一步的验证命令其实验证不了这一步声称的东西
- phase 之间有没有隐藏依赖，导致中间状态不可运行
- research.md 里的**假设**是否被 plan 悄悄当成了既成事实
- 有没有哪个 phase 会动 `in_dim`/`genome_size` 却没写「种群作废」的代价
- 有没有更简单的方案被漏掉

逐条处理：采纳就改 plan.md；不采纳就在 plan.md 里写一行为什么不采纳。
**不要只是把意见转述给用户。**

## 收尾

输出：

1. 两个文件的路径
2. 先例检查结论（有没有前科）
3. phase 列表，每个一行
4. plan-critic 提了什么、怎么处理的
5. 原话告知用户：**「这是你唯一需要认真 review 的东西。看完 plan.md 再 /impl —— 我现在不会开始写代码。」**

## 硬约束

- 全程不改源文件，产物只有那两个 `.md`。
- **不要自动开始实现。**
- 不确定的地方写「不确定」并说明怎么才能确定，别用流畅措辞盖过去。
- plan.md 里出现的任何数值（现状指标、baseline）必须来自 `docs/` 下某份文档的实测或本次
  实测，标 `[本世界实测]` 并注明出处文件。**不要凭印象写数字。**
- 若这次 plan 的产出其实是结论而非代码，那它不该走 plan——按 `CLAUDE.md` 落成
  `docs/<topic>.md` 并在 `docs/TODO.md` 加指针。
