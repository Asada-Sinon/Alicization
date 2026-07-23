---
name: impl
description: "按 plan.md 实现：一轮只做一个 phase，跑该 phase 自己写的 check.py 验证命令并贴真实输出，然后停下、给出按变更理由拟好的中文 commit message 等用户确认。用于 /plan 产出 plan.md 之后的实现阶段。"
argument-hint: "[plan 目录，留空则用最新的]"
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# 按 plan 实现

> 为什么 `allowed-tools` 里是完整的 `Bash`：本 skill 的核心动作就是**跑 plan 为这个 phase
> 写的那条验证命令**，那条命令是什么事先不知道，收窄成 git 只读只会让每次验证都卡在权限确认上。
> 另外注意：`allowed-tools` 是「本轮免批准的范围」，**不构成硬限制**，不要把它当安全机制。
> 真正的护栏是 `.claude/hooks/`（PREALLOCATE 前缀、编辑后自动跑 `--contracts`）和下面的硬约束。

目标 plan：$ARGUMENTS（可以是目录，也可以直接是 `plan.md` 路径；留空则取 `docs/plans/`
下日期最新的那个目录）

## 第 1 步 定位并读 plan

读 `docs/plans/<...>/plan.md`。**只读 plan.md**，research.md 除非 plan 里有指向否则不读——
它的作用在 plan 写完那刻就结束了，再读只是烧 context。

确认三件事，缺任何一件就停下问用户：

- 有明确的 phase 划分
- 每个 phase 有验证命令和成功判据
- 有「不在范围内」小节

然后报告：本 plan 共几个 phase，本轮准备做哪一个（默认第一个未完成的）。

## 第 2 步 只做一个 phase

**一轮只做一个 phase。** 做完不要接着做下一个，哪怕它看起来很小。

实现时：

- 严格按 plan 写的「改哪个文件、改什么」来。
- plan 写错了（比如行号对不上、函数已经改名）→ **说出来，问用户**，不要自行发挥。
- 需求在 plan 里没写清楚 → 问，不要挑一个自己觉得对的解释往下做。

本项目在写代码时容易踩的三处（plan 没提也要守）：

- **定形张量**：`WorldState` 的数组永不 resize，生死只是 `alive` 掩码。不要引入布尔索引或
  动态形状；生育走 `reproduction.reproduce` 的 permutation-scatter 惯用法。
- **`step.py` 的每步顺序是语义**：`memory.advance` 在移动之后、写入之前；邻居索引在捕食前
  重建。挪动顺序就是改行为，不是重构。
- **改 `in_dim` / `genome_size`（加视网膜通道、改 `hidden`、加记忆槽）会作废整个演化种群**，
  没有 checkpoint 迁移。plan 没写这条代价就停下确认。

## 第 3 步 验证并贴证据

跑 **plan 里为这个 phase 写的那条验证命令**，不要换成别的。所有 python 调用都要带
`XLA_PYTHON_CLIENT_PREALLOCATE=false` 前缀和 `.venv/bin/python`（缺了会被 hook 直接 deny）。

三档的分工，plan 没指定时按这个选：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --contracts  # 0.2s，动了 wire/颜色/config 缩放就跑
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py              # 14s，改了 kernel/参数就跑，含 golden band
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --full       # 3min，提交前那一次
```

`--full` 要几分钟，**给长超时，不要以为它挂了**。

然后贴**真实输出**：

- 原样粘贴命令和它的输出（长就截关键段落，并说明截了）。
- 逐条对照该 phase 的成功判据，写「达成 / 未达成」。
- **不要用「检查通过了」「看起来正常」代替真实输出。** 没有输出就等于没验证。
- 验证失败就如实说失败，然后停下。不要一边说「基本没问题」一边悄悄改判据。
- **golden band 失败绝不靠放宽 band 解决。** 如果这次改动本就该移动 golden 数字，
  说明理由、征得用户同意后再 `--bless`，并把理由写进 commit message。
- 改了 shader 或 `web/` → 必须**看**截图（headless chromium 加
  `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`），
  `docs/conventions.md` §10。code review 抓不到渲染 bug。

## 第 4 步 停下，并把 commit 提议摆出来

本轮到此为止。输出：

1. 本 phase 改了哪些文件（`git status --short` 的结果）
2. 验证命令 + 真实输出 + 判据对照
3. 「发现但未做」里新增了什么（若有）
4. **拟好的中文 commit message**（subject + body，body 说清为什么改、量到了什么），
   外加这句：**「这个 phase 的改动只有一个变更理由，可以单独提交。说『提交』我就跑
   `check.py --full` 然后 commit + push；要继续下一个 phase 就说继续。」**

等用户决定。

## 关于 commit：为什么这里要停一下

`CLAUDE.md` 说「检查通过后直接 push，不用问」——那条针对的是**范围已经谈定的一件完整工作**。
走 plan 的多 phase 实现不同：只有用户知道 phase N 是不是一个完整的变更理由，还是要和
phase N+1 合起来才算一个。而本项目按**变更理由**拆 commit，判据是「message 里要不要写
『顺便』」。所以：

- **不要自己发起 commit**，在 phase 边界停下、给出提议，让用户一句话确认。
- 用户说提交 → 跑 `check.py --full`，贴输出，用中文 message commit，**然后直接 push 到
  `origin main`，不用再问一次**（这一步遵守 `CLAUDE.md`）。
- **绝不把两个 phase 塞进一个 commit**，那正是「也/顺便」判据要拦的东西。
- `Co-Authored-By:` trailer 和代码标识符（`carn_cost`、`world_step`、`UNTR`、文件路径）
  保持原样，其余用中文。

## 禁止扩大范围

实现过程中一定会看到别的问题：命名不好、有个 bug 不在 plan 里、这里可以顺手重构一下。

一律**不要顺手改**。把它追加到 plan.md 的「## 发现但未做」小节，一行一条，写清楚
`文件:行号` 和问题是什么，然后继续做当前 phase。

例外只有一个：不改它当前 phase 就无法完成。这时先说明，再改，并在输出里单独标出来。

理由：diff 一旦超出 plan，validate 阶段就没法做「对照 plan 逐条核对」了，
整个 R-P-I-V 的验收环节会失效；顺带也会把两个理由混进同一个 commit。

## 硬约束

- 一轮一个 phase。
- 不改 plan.md 里「不在范围内」列出的东西。
- **不产生任何数值结论。** 指标必须由受版本控制的脚本产生，需要跑对比实验走 /exp
  （6 配对种子起，单次运行不作数）。
- **绝不为了让检查变绿而放宽 golden band 或删测试。**
- 同一处被用户纠正 2 次仍不对 → 停下来建议 /clear 重开，不要继续试第 3 次。
