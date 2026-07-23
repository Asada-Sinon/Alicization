---
name: validate
description: "对照 plan.md 对 git diff 做对抗式验收：派 plan-critic subagent 逐条核对需求实现、跨文件契约、以及有没有 plan 之外的改动，输出「必须修 / 建议修 / 仅记录」三档。实现完成后用。"
disable-model-invocation: true
allowed-tools: Read, Glob, Grep, Task, Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py:*)
---

# 对照 plan 的对抗式验收

## 第 1 步 取材料

```bash
git status --short
git diff
git diff --stat
```

读对应的 `docs/plans/<...>/plan.md`。

## 第 2 步 跨文件契约必须**跑过**，不能靠读

本项目有三类会**静默**损坏的东西，code review 对它们无效——一个错的偏移量产出的是
看起来合理的错误数字，不是异常（`docs/conventions.md` §10）：

- **wire format**：`server/protocol.py` 的 `_HEADER` 与 `encode()`、`web/main.js` 的
  `HEADER_BYTES`、以及插入点**之后**的每一个 `dv.getFloat32(offset)` 必须同时改。
  只能 append，不能 insert。
- **物种颜色重复三处**：`web/render.js` 的 shader `vec3` 字面量、`web/index.html` 的
  `:root` 变量、`web/main.js` 的 `C` 对象。
- **config 缩放规则**：世界尺度长度按 `world_size` 取比例、`sense_grid` 必须随
  `world_size` 缩放（小了看不见邻居、大了溢出 `k_neighbors` 被静默丢弃）。

所以本步是**跑一条命令并把输出贴进验收报告**，不是勾一个复选框：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --contracts
```

diff 碰了 kernel、`config.py` 或任何生态参数，再补一次默认档（14s，含 golden band）：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py
```

额外核两件事：

- diff 里如果 `scripts/golden.json` 变了 → 确认是**有意 `--bless`**、且 commit message
  写了理由。**为了让失败消失而放宽 band 的，一律进「必须修」。**
- diff 碰了 shader 或 `web/` → 在报告里原话提醒：
  **「按 `docs/conventions.md` §10 用截图验证，不能只靠 code review。」**
  没有截图证据就不算验收通过。

## 第 3 步 派 plan-critic 审

**必须派 `plan-critic` subagent，不要自己审。** 写代码的那个 context 已经被自己的实现思路
污染了，它会倾向于认为自己写的是对的。`plan-critic` 是对抗式审查专用、且没看过实现过程，
只看 diff 和 plan。派给它的 prompt 必须带上：plan.md 的路径、`git diff` 的范围、
以及下面第 4 步的**整份黑名单**（不带黑名单它一定会给你凑一堆命名和注释建议）。

给它的任务，只有三条：

1. **逐条核对 plan 的每个需求是否真的实现了。** 按 plan 的顺序过，每条给结论 +
   `文件:行号` 证据。plan 说改 A 结果改了 B，或者只改了一半，都要报。
2. **边界情况有没有测试。** 只针对 plan 里明确要求的行为。空输入、越界、异常路径——
   有没有测试覆盖。没有就报。
3. **有没有 plan 之外的改动。** diff 里出现 plan 没提的文件或改动，逐个列出来问「为什么」。
   这是最重要的一条：范围蔓延会让「按 plan 验收」这件事本身失效，也会把两个变更理由
   混进同一个 commit。

## 第 4 步 反过度工程（比正面要求更重要，请完整执行）

官方给过明确警告：**reviewer 一旦被要求「找 gap」，它总能找出来。**
追着 reviewer 的每条意见改，最终产物就是过度工程。

所以这是整套流程里**唯一一个宁可漏报也不要多报**的 skill。

### 显式黑名单——以下一律不报，一条都不报

- 命名（变量名、函数名不够好、不够语义化）
- 注释（缺注释、注释可以更详细、缺 docstring）
- 函数长度、文件长度、嵌套层数
- 是否「pythonic」/ 是否符合某种风格惯例
- 类型标注不全（除非 plan 明确要求了类型标注）
- 「可以抽成一个函数」「这里可以更通用一点」
- 「建议加日志」「建议加异常处理」——除非 plan 要求了，或缺它会导致明确的失效
- 格式、空行、import 顺序（本项目**没有** linter/formatter，别去当人肉 linter）

### 判据：说不出具体失效场景的，不算 finding

每条 finding 必须能写出：**什么输入 / 什么状态 → 会发生什么错误结果。**

写不出来就删掉这条。「这样写不太好」「将来可能有问题」「不够健壮」——都不是失效场景，删掉。

例外：第 2 步那三类跨文件契约、以及 golden band 被放宽，**即使当下看不到失效现象也要报**——
它们的失效场景就是"静默产出合理的错误数字"，本来就看不见。

### 三档都空就直接说通过

如果没有任何 finding，输出就是一句「对照 plan 逐条核过，`--contracts` 21 项全过，验收通过」，
外加核过的条目清单和命令输出。**不要为了显得尽职而硬凑一条。** 硬凑出来的 finding 会被
真的去改，那是纯粹的成本。

## 第 5 步 输出三档

### 必须修
会导致错误结果、plan 需求未实现、引入了 plan 之外的行为改动、碰坏跨文件契约、
或放宽了 golden band。每条：`文件:行号` + 失效场景 + 建议怎么改。

### 建议修
真实存在但不阻塞的问题（如某个边界没测到）。每条同样要有失效场景。

### 仅记录
不用现在动，但值得写进 plan.md 的「发现但未做」。

三档都可以为空。为空就写「无」。

## 硬约束

- 不要在本 skill 里动手改代码。验收和修复分开，混在一起就没人能审了。
- **不做数值判断。** 「这个改动让捕食者占比变差了」这种结论必须来自 /exp 的 6 配对种子
  实测，不能靠读 diff 推断，也不能靠一次 `run_headless.py` ——单次运行的方差超过大多数
  参数效应（`CLAUDE.md`「绝不在单次运行上调生态参数」）。
- 用户明确说过「不管这个」的，不要再报第二次（去 `MEMORY.md` 查 `[LEARN:tag]`）。
