#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SessionStart hook：把项目持久状态注入会话 context。

CLAUDE.md 的「Research lands in `docs/`, or it did not happen」一节已经把
这个 hook 存在的理由写完了：

    "A report that exists only in conversation is lost at the next context
     compaction, and that has already cost this project a full 3D feasibility
     study once"

那一节要求把结论写进文件，这个 hook 负责另外半边：**让写进文件的东西自动回到
context**。否则接续仍然依赖 agent 自觉去 `docs/TODO.md` 读一遍，而 compact
之后它连"该去读"这件事都不记得了。

SessionStart 的 stdout 会被原样加进 context，所以这里输出的是**纯文本**（不是
JSON）。source 枚举有四个：startup / resume / clear / compact，settings.json 的
matcher 必须四个都覆盖（matcher 是正则），漏掉 clear 等于最常走的那条路上没有
状态注入。

任何异常都静默 exit 0 且不输出：注入失败顶多少点上下文，
但 hook 报错会让会话启动就带一堆噪音。
"""

import json
import os
import re
import subprocess
import sys

# 各文件注入上限（字符）与超限时保留哪一头。
#
# 截断方向必须跟着**每个文件自己的书写约定**走，方向反了就等于精准切掉最有用的部分：
#
#   HANDOFF.md      —— 保留 head。约定是「新会话结束时在最上面加一条，最新的在前」。
#                      保留尾部会把最新那次交接连同 PENDING 一起切掉，正好切反。
#   MEMORY.md       —— 保留 tail。MEMORY 是累积式的，新条目**追加在后**；
#                      而且文件顶部是一大段格式说明（不是内容），保留 head 会
#                      注入一堆"怎么写 MEMORY"的元信息、一条真教训都留不下。
#   current-focus.md—— 保留 head。它不是时间序列，是一份「当前目标 / 为什么做 /
#                      完成判据 / 不做什么」的快照，重要度自上而下递减。
#
# 值为 (上限字符数, "head" | "tail", 截断说明里附的一句提示或 None)。
LIMITS = {
    "HANDOFF.md": (3000, "head", "本文件最新的交接写在最上面，保留的就是最新几次"),
    os.path.join(".context", "current-focus.md"): (1000, "head", None),
    "MEMORY.md": (2000, "tail", "本文件新教训追加在最后，保留的就是最新几条"),
}

# docs/TODO.md 是本项目**已有的、真实在用的**接续入口（它自己第一句就写着
# "新开一个对话时读这一份就能接上"），所以它比模板那三份文件更重要。
# 但整篇 21 KB，绝不能整篇注入，见下面 todo_queue() 的切法说明。
TODO_REL = os.path.join("docs", "TODO.md")
TODO_LIMIT = 1500

GIT_STATUS_LINES = 20   # git status --short 最多显示几行
GIT_LOG_COUNT = 3       # 显示最近几条 commit
GIT_TIMEOUT = 10


def project_root(data):
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    return os.path.abspath(root)


def cap(text, limit, keep="tail", hint=None):
    """把 text 截断到 limit 字符，并在截断处留一行说明。

    keep="head" 保留开头（最新的写在最上面、或重要度自上而下递减的文件）；
    keep="tail" 保留结尾（新内容追加在后的文件）。
    截断处一定要留说明，否则模型会把残缺内容当成全文。
    """
    if len(text) <= limit:
        return text
    tail_hint = ("；" + hint) if hint else ""
    if keep == "head":
        return text[:limit].rstrip() + "\n...(以上为开头 %d 字符，后文已截断%s)..." % (
            limit,
            tail_hint,
        )
    return "...(前文已截断，以下为最后 %d 字符%s)...\n" % (
        limit,
        tail_hint,
    ) + text[-limit:].lstrip()


def read_capped(path, limit, keep="tail", hint=None):
    """读文件并截断。文件不存在/空返回 None。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None
    return cap(text, limit, keep, hint)


# ----------------------------------------------------------------------
# docs/TODO.md 的切法
# ----------------------------------------------------------------------
# 这个文件有 21 KB，结构是：
#     # 任务队列与文档索引
#     ## 文档地图        <- 一张 20+ 行的大表，每行一份 docs/ 专题文档的摘要
#     ## 硬约束（每次动手前确认）
#     ## 队列            <- 真正的"下一步做什么"
#     ## 不做的事
#
# 只取「## 队列」那一节，理由三条：
#   1. 「文档地图」占了全文一半以上，而且它是**索引**不是状态——agent 需要它的
#      时候会自己去读 TODO.md，注进 context 只是把 20 份文档的摘要塞进每一次
#      会话开头，纯噪音，还会把真正的接续信息挤出 1500 字符的预算。
#   2. 「硬约束」四条里有三条已经由别处保证：PREALLOCATE 有 PreToolUse hook 硬拦、
#      统计地板和"结论落 docs/"都在 CLAUDE.md 正文里（CLAUDE.md 每次都加载）。
#      重复注入等于花预算买冗余。
#   3. 「队列」的第一小节就是「当前主线」，字面回答"接下来做什么"——这正是
#      SessionStart 唯一该抢的位置。
#
# 节内再做一次删减：丢掉所有 markdown 表格行。「已实现、已验证、结论已归档」
# 是一张 12 行的历史归档表，每行只是"某事已完成 → 见某文档"，属于索引而非待办；
# 删掉它可以让 1500 字符的预算全部落在「当前主线」和「未做」上。删除处留一行
# 计数标记，免得模型以为那一节本来就是空的。
#
# 最后保留 head：节内顺序就是重要度顺序（当前主线 → 已归档 → 未做）。
# 找不到「## 队列」时退回取全文尾部——TODO.md 的新内容通常追加在后半部分，
# 尾部比头部（文档地图）有用得多。

QUEUE_HEADING = re.compile(r"^##\s*(?:任务)?队列\s*$")


def todo_queue(root):
    path = os.path.join(root, TODO_REL)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception:
        return None

    start = None
    for i, line in enumerate(lines):
        if QUEUE_HEADING.match(line.strip()):
            start = i
            break
    if start is None:
        text = "\n".join(lines).strip()
        if not text:
            return None
        return cap(
            text,
            TODO_LIMIT,
            "tail",
            "未找到「## 队列」小节，这里取的是 docs/TODO.md 的尾部",
        )

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    kept = []
    table_run = 0
    for line in lines[start:end]:
        if line.lstrip().startswith("|"):
            table_run += 1
            continue
        if table_run:
            kept.append(
                "（此处 %d 行表格已省略：都是「某事已完成 → 见 docs/xxx.md」的索引，"
                "需要时直接读 docs/TODO.md）" % table_run
            )
            table_run = 0
        kept.append(line)
    if table_run:
        kept.append(
            "（此处 %d 行表格已省略：都是「某事已完成 → 见 docs/xxx.md」的索引，"
            "需要时直接读 docs/TODO.md）" % table_run
        )

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    if not text:
        return None
    return cap(text, TODO_LIMIT, "head", "完整队列见 docs/TODO.md")


def git(root, args):
    """跑一条 git 命令，失败返回 None（非 git 仓库属于正常情况）。"""
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").rstrip()


def git_section(root):
    """组装版本库状态：分支 + 工作区变更 + 最近提交。"""
    if git(root, ["rev-parse", "--is-inside-work-tree"]) is None:
        return None

    parts = []

    branch = git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        parts.append("当前分支：%s" % branch)

    status = git(root, ["status", "--short"])
    if status is not None:
        lines = status.splitlines()
        if lines:
            shown = lines[:GIT_STATUS_LINES]
            block = "\n".join(shown)
            if len(lines) > GIT_STATUS_LINES:
                block += "\n...(还有 %d 处变更未显示)" % (len(lines) - GIT_STATUS_LINES)
            parts.append("工作区变更（git status --short）：\n" + block)
        else:
            parts.append("工作区变更（git status --short）：\n(干净)")

    log = git(root, ["log", "--oneline", "-n", str(GIT_LOG_COUNT)])
    if log:
        parts.append("最近 %d 条提交：\n%s" % (GIT_LOG_COUNT, log))

    if not parts:
        return None
    return "\n\n".join(parts)


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    root = project_root(data)
    source = data.get("source") or ""

    sections = []  # [(标题, 正文)]

    # compact 场景专属提醒：压缩刚发生，最需要重申的规则放这里。
    # 只在 compact 时读 —— startup/clear/resume 时 CLAUDE.md 还会正常加载，
    # 不需要重申；只有 compact 会把它揉成有损摘要。
    if source == "compact":
        note = read_capped(
            os.path.join(root, ".claude", "compact-reminder.txt"), 2000, "head"
        )
        if note:
            sections.append(("压缩后提醒 .claude/compact-reminder.txt", note))

    handoff = read_capped(os.path.join(root, "HANDOFF.md"), *LIMITS["HANDOFF.md"])
    if handoff:
        sections.append(("交接文档 HANDOFF.md", handoff))

    focus_rel = os.path.join(".context", "current-focus.md")
    focus = read_capped(os.path.join(root, focus_rel), *LIMITS[focus_rel])
    if focus:
        sections.append(("当前焦点 .context/current-focus.md", focus))

    memory = read_capped(os.path.join(root, "MEMORY.md"), *LIMITS["MEMORY.md"])
    if memory:
        sections.append(("长期记忆 MEMORY.md", memory))

    todo = todo_queue(root)
    if todo:
        sections.append(("任务队列 docs/TODO.md（仅「队列」一节）", todo))

    repo = git_section(root)
    if repo:
        sections.append(("版本库状态", repo))

    # 什么都没有就彻底闭嘴，别往 context 里塞空壳
    if not sections:
        return

    out = ["===== 项目持久状态（自动注入）====="]
    out.append(
        "以下为项目持久状态，由 SessionStart hook 自动注入。"
        "每次 /clear、恢复会话或压缩(compact)之后都会重新注入，"
        "可信度高于对话历史。"
    )
    for title, body in sections:
        out.append("")
        out.append("----- %s -----" % title)
        out.append(body)
    out.append("")
    out.append("===== 持久状态结束 =====")

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
