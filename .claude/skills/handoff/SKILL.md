---
name: handoff
description: "Session 收尾：更新 HANDOFF.md、把本次纠正以 [LEARN:tag] 追加进 MEMORY.md、把结论性调查落进 docs/<topic>.md 并在 docs/TODO.md 加指针、覆写 current-focus.md。context 快满或准备结束 session 时用。"
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Glob, Grep, Task, Bash(git status:*), Bash(git log:*), Bash(git diff:*)
---

# Session 收尾

把本次 session 的状态落到磁盘，让下一个 agent 能接上。

先跑 `git status --short` 和 `git log --oneline -5`，用实际状态写，不要凭记忆。

## 1. 调研结论先落进 `docs/`（本项目的第一优先级）

`CLAUDE.md`：**「Research lands in `docs/`, or it did not happen」**。本次如果产出了
**结论而不是代码**——消融判决、机制诊断、可行性分析、文献综述、审计——那么在写任何交接
文件之前，先把它写成 `docs/<topic>.md` 并提醒用户提交。只存在于对话里的报告会在下一次
context 压缩时消失，这件事已经让这个项目丢过一份完整的三维化可行性研究
（`docs/conventions.md` §3）。

规矩，一条都不能省：

- **一个主题一个文件**，`docs/<topic>.md`。长是好事——它该读起来像文献综述（完整散文、
  真实引用、表格、推导、死胡同），不像 prompt。
- **每条 claim 标注它是怎么确立的**：`[现实]` 已发表事实 / `[本世界实测]` 本项目测出 /
  `[对应]` 落在代码哪里 / `[提案，非结论]` 提案。来源核不到就明说核不到，
  **不要洗成自信的散文**。
- 在 `docs/TODO.md` 的文档地图表加一行指针（文档 / 内容 / 什么时候读它），
  否则下一个 session 找不到它。
- **负结果和不确定结果正是重点，不是丢人的事。** 它们是唯一能阻止同一个想法被重试的东西；
  该进 `docs/experiments.md`（失败实验档案）的就进那里，并在 TODO 的判决表里加一行。

跑过实验的，数值细节归 `docs/<topic>.md`，**不要**另起流水账文件——本项目没有
`research-log.md` 这类东西，也不该有：事实底账就是 `docs/` 加 git 历史。

### 本次读过论文的，核一遍笔记与台账

`notes/papers/` 和 `lit/literature-log.md` 是通往 `docs/` 的**上游**，不是它的替代：
逐篇笔记记「这篇说了什么、统计层级是什么」，台账记「读过没有、并进了哪份 `docs/`」。
本次如果读了新论文（自己读的，或派 `lit-reviewer` / 走 `/lit` 读的），收尾前确认三件事：

- 每篇入选论文都有 `notes/papers/<citekey>.md`，正文六节没有半张模板放着；
- `lit/literature-log.md` 有对应行，且**本次检索的 query 记进了「检索记录」表**——
  没有命中的 query 同样要记，否则下个 session 会把它再跑一遍；
- 已经并进 `docs/` 的，那一行的「已并入 docs/」回填到小节级。**只写了笔记而结论没进
  `docs/`，等于那条结论没做过**（除非它确实不相关，那要在台账里写清理由）。

```bash
ls -lt notes/papers/ | head          # 本次新增/改动的笔记
grep -c "未并入" lit/literature-log.md   # 还欠着的账，PENDING 里提一句
```

### 本次动过代码时，核一遍旧结论有没有被推翻

改动**可能让某份 `docs/` 文档里的结论失效**时（改了指标计算、config 默认值、
重构了被结论引用的模块、动了地形或水系统参数），派 `plan-critic` 或一个 Explore subagent
把相关文档里的量化 claim 逐条映射回今天的代码，判定是否仍然成立。

它只判定、不改文档。拿到回报后：**不要偷偷改旧结论的措辞**，在该文档里补一节写明推翻了
哪条、依据是什么、什么时候测的；判定不了的在 HANDOFF 的 PENDING 里留一条。

## 2. HANDOFF.md（本次状态）

在文件**顶部**追加一节（新的在上，老的往下沉），格式固定，**照抄字段名**：

```text
## Session 2026-07-23
- 完成: <一条一件事，带 文件:行号 或 commit hash。可以多行，一行一条>
- PENDING: <下次第一件事，具体到「打开哪个文件、跑哪条命令」的程度>
- 坑: <具体现象 + 怎么绕过去的。没踩到就写「无」>
```

**这是写给下一个 agent 的一封信，不是工作日志。** 判据：

- 短。整节 20 行以内。
- 具体。「修了水系统」没用；「`underworld/dynamics.py:88` 的 `drink` 在幼体上按满箱算，
  已改成按当前容量」才有用。
- 只写下次用得上的。这次想过但放弃的思路，除非会被重新想一遍，否则不写。
- PENDING 写成可执行的下一步，不是愿望清单。
- **结论不写在这里**，写在 `docs/<topic>.md`，这里只留一行指针。

**只保留最近 3 节**，更早的直接删掉——HANDOFF 是易过期文件，越长越误导（历史在 git 里）。

## 3. MEMORY.md（跨 session 教训的缓冲区）

把本次**用户对我的纠正**追加到文件**末尾**（累积式，新的在后），一条一个多行块：

```text
### [LEARN:scope] 不要顺手改 plan 之外的文件，即使只有一行
- 现象: 顺手把相邻函数重命名了，validate 阶段没法对照 plan 逐条核。
- 原因: 「反正只有一行」把范围蔓延合理化了；plan 的「不在范围内」当时没读。
- 对策: 看到的问题一律追加到 plan.md 的「发现但未做」，本轮不动。
- 来源: Session 2026-07-23
```

四行缺一不可，**「原因」是一条 LEARN 里最值钱的部分**——只有它能让下次真的避开，
「现象 + 对策」没有原因就退化成一条不知道为什么要守的规矩，早晚被绕过去。
tag 用短英文词（scope / stats / repro / env / contracts / commit / render ...），便于 grep。

**禁止为了填表而编造 LEARN 条目。** 本次用户没纠正过我，这一节就不动。
空着的成本是零，编造出来的条目会在后面每个 session 被当成真教训遵守，成本是永久的。
判据：一条 LEARN 必须能指到本次对话里一句真实的用户纠正。指不到就不写。

### MEMORY.md 与 `docs/conventions.md` 的分工

两者都放「规则」，但重量级完全不同，**不要写重**：

| | `MEMORY.md` | `docs/conventions.md` |
|---|---|---|
| 是什么 | 轻量级 `[LEARN:tag]` 缓冲区 | `CLAUDE.md` 每条约定背后的**完整论证** |
| 门槛 | 一次真实的用户纠正就够 | 要有实测数字和算术（918 MiB、p=0.031 地板、随机放置 0.556–0.675） |
| 生命周期 | 攒着，反复出现且够分量就**毕业** | 长期文档，改它约等于改项目宪法 |

毕业动作：一条 LEARN 反复出现、或已经值得写进 `CLAUDE.md` 时，把它扩写成
`docs/conventions.md` 的一节（补上实测和论证），必要时在 `CLAUDE.md` 加一条指令行，
然后从 `MEMORY.md` 删掉那条。**毕业需要用户点头，不要自己改 `CLAUDE.md`。**

## 4. .context/current-focus.md（当下焦点）

**覆写**（不是追加）。内容就三行以内：现在在攻什么问题、卡在哪、下一动作。
它是给 /kick 快速定位用的，长了就失去意义。

## 防重叠规则（写死，不要越界）

| 文件 | 只放 | 不放 |
|---|---|---|
| `HANDOFF.md` | 本次 session 的状态、PENDING | 通用规则、实验数值、结论 |
| `MEMORY.md` | 跨 session 的教训（轻量缓冲） | 本次进度、实验结果、已毕业的约定 |
| `docs/<topic>.md` | 结论、实验数值、论证、四标签 claim | 计划、待办、session 状态 |
| `docs/TODO.md` | 下一步做什么 + 指向哪份文档 | 论证本身 |
| `.context/current-focus.md` | 当下一句话焦点 | 历史、细节 |
| `notes/papers/<citekey>.md` | 单篇论文的原始笔记：方法、统计层级、数字出处 | 本项目的设计论证（那是 `docs/` 的）、实测数字 |
| `lit/literature-log.md` | 文献索引与检索台账：状态、关系、并入了哪份 `docs/` | 论文内容、结论 |
| `explorations/<dir>/` | 产生某个数字的一次性分析脚本（可重跑） | 结论、实验产物（产物进 `outputs/`） |

同一条信息只写进一个文件。写重了，下次读的时候两份会不一致，然后没人知道信哪个。

## 收尾输出

1. 列出改了哪几个文件、各加了几条
2. 本次有没有结论落进 `docs/`；有就给出文件路径 + TODO 指针那一行
3. 本次有没有读新论文；有就给出新增的 `notes/papers/` 文件与台账行数，并说明「已并入」列
   是否都回填了。没读就明写「本次无新文献」
4. 明确说「MEMORY.md 本次未新增（无用户纠正）」——如果确实没有
5. 提示用户：**「这些是记忆与文档文件，和代码是不同的变更理由，建议单独一个中文 commit；
   `check.py --full` 过了就可以直接 push。」**（不要自己发起这个 commit。）
