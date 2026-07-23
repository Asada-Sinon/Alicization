---
name: kick
description: "Session 开场：读 HANDOFF/MEMORY/current-focus 与 docs/TODO.md 的任务队列，用 git 实际状态核验，输出「我在哪 / 上次遗留 / 建议下一步」。新开 session 或 /clear 之后的第一件事。"
disable-model-invocation: true
allowed-tools: Read, Glob, Grep, Bash(git status:*), Bash(git log:*), Bash(git diff:*)
---

# Session 开场

目标：几分钟内和用户对齐状态，而不是从零重扫代码库。

> `allowed-tools` 声明的是**本轮免批准的范围**，不是硬限制。超出它的命令（比如下面那条
> `check.py`）仍然能跑，只是会多一次权限确认——这是刻意的：开场不该悄悄执行任意命令。

## 第 1 步 读状态

并行读，别一个个来。**SessionStart 钩子可能已经把其中一部分注入进来了，注入过的就不要
再读一遍。**

- `HANDOFF.md` —— 上次 session 的状态，**只看最上面那一节**（最新在上）。
  **会过期，当线索不当事实。**
- `.context/current-focus.md` —— 当下焦点。
- `MEMORY.md` —— 跨 session 的 `[LEARN:tag]` 教训，全读。
- `docs/TODO.md` 的**任务队列部分** —— 这是本项目真正的接续入口，比 HANDOFF 更权威
  （它进过 commit，被多个 session 校对过）。

`docs/TODO.md` 有 150+ 行，前半是文档地图大表，**开场不要整份读**。定位后只读队列：

```bash
grep -n '^## 队列\|^## 不做的事' docs/TODO.md
```

用得到的行号 `Read(offset=..., limit=...)` 只取「## 队列」到「## 不做的事」之间那段；
「## 不做的事」本身也值得扫一眼，它会直接毙掉一批候选下一步。文档地图表按需再查。

## 第 2 步 核验（不能省）

交接文档是上一个 agent 写的，可能写错、也可能已被后续操作推翻。用实际状态对账：

```bash
git status --short
git log --oneline -5
```

对照检查：

- HANDOFF 说「已完成 X」，但 X 涉及的文件在 `git status` 里还是脏的、或 `git log` 里没有
  对应提交 → 存疑。本项目的规矩是**检查通过就 push**，所以"做完了但没提交"本身就是异常信号。
- HANDOFF 说「结论是 …」但 `docs/` 下找不到对应的 `<topic>.md` → 按项目规矩那份调研**不算做过**，
  标成「未落盘」。
- HANDOFF 说「检查通过」→ 需要时自己跑一次实测，不要转述（会弹一次权限确认，正常）：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --contracts
```

- 最近一次提交时间距今很久 → 整份 HANDOFF 都要打折。
- 有未提交改动 → 先讲清楚这些改动是什么，再谈下一步。

**文档与实际不一致时明确指出来，不要悄悄以文档为准。**

## 第 3 步 输出（固定三段）

### 我在哪
2-4 行：分支、工作区是否干净、上次做到哪。与 HANDOFF 有出入的地方标出来。

### 上次遗留
HANDOFF 的 PENDING 逐条列出，每条注明「已确认 / 存疑 / 已失效」。`docs/TODO.md` 队列里
标着「未做」的主线，挑与 PENDING 相关的补一行。

### 建议下一步
给 **2-3 个**选项，每个一行，说清代价和入口。例：

- A) 接着做 PENDING 里的 X —— 改动小，直接 /impl
- B) 先 /plan 查 Y 的根因 —— 需要调研，约一轮
- C) 先 /exp 把上次那个单种子探针补成 6 配对种子 —— 先有数据再决定

## 硬约束

- **不要开始改代码。** 本 skill 只负责对齐状态 + 给选项，等用户选。
- 不要提出 HANDOFF、current-focus、`docs/TODO.md` 队列里都没有的新方向。
- 选项不得与 `docs/TODO.md` 的「## 不做的事」冲突（减慢植物生长赶捕食者、重加记忆遗传、
  再调踩踏参数……那些都已经测过并否掉了）。
- 读不到某个文件就说读不到，不要脑补内容。
- 本 skill 每个 session 都加载，输出要短：三段合计控制在 30 行以内。
