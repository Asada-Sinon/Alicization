#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PreToolUse(Edit|Write) hook：受保护路径护栏。

本项目最该守的东西不是产物目录（`outputs/ checkpoints/ runs/` 全部 gitignore，
删了重跑就有），而是 **`scripts/golden.json`**。CLAUDE.md 说得很直白：

    "If a change is meant to move the golden numbers, re-record with `--bless`
     and say why in the commit message. **Never widen the bands to make a
     failure go away** -- a band widened for that reason is a check deleted."

用 Edit/Write 直接改 golden.json 正好是"悄悄放宽 band 让失败消失"的那条操作
路径；合法路径是 `scripts/check.py --bless`，它走 Bash，本 hook 不拦。
换句话说这条规则不是禁止改 golden，是**把改 golden 逼回那条会留下痕迹的路**。

两类规则：

  普通规则       —— 命中后**再看目标文件是否已存在**：
                    已存在 → exit 2 阻断；不存在 → 放行（允许新建）。

  `!` 前缀规则   —— **无条件阻断**，存不存在都拦。
                    用于「新建它本身就危险」的东西：golden.json / .env / *.lock。
                    注意这个 `!` **不是 gitignore 的取反语义**，是"更严格"。

规则来源：项目根 .claude/protected-paths.txt（每行一个 glob，可带 `!` 前缀，
# 注释）。该文件不存在时使用下面的内置默认值。
"""

import fnmatch
import json
import os
import sys

# 内置默认保护规则。当 .claude/protected-paths.txt 不存在时生效。
# `!` 前缀 = 无条件阻断；无前缀 = 仅当文件已存在时阻断。
# 必须与 .claude/protected-paths.txt 的内容保持一致。
DEFAULT_PATTERNS = [
    "!scripts/golden.json",
    "!.env",
    "!.env.*",
    "!*.lock",
    "!uv.lock",
    "!.git/**",
]

# 某些规则值得给一段专属说明，泛泛的"这是受保护路径"帮不到人。
# 键是规则原文（含 `!`）。规则被用户改写后落回泛用说明，不会出错。
RULE_NOTES = {
    "!scripts/golden.json": (
        "golden band 是 tier-2 检查的全部内容——它是那种「config 悄悄挪动了种群」\n"
        "的唯一探测器（3% 的 `eat_rate` 改动就会触发）。手改这个文件等于删掉这个\n"
        "探测器，而且删得没有痕迹。\n"
        "\n"
        "  想让 golden 反映一次**有意**的行为改变，只有一条路：\n"
        "      XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --bless\n"
        "  然后在 commit message 里写清楚为什么这次改动应该挪动这些数字。\n"
        "\n"
        "  **绝不要为了让失败消失而放宽 band**——CLAUDE.md 原话：\n"
        "  「a band widened for that reason is a check deleted」。\n"
        "  如果 band 挂了而你不认为是自己改坏的，先怀疑改动，不要怀疑 band；\n"
        "  带宽是怎么定标的见 docs/conventions.md §9。"
    ),
}

RULES_FILE = os.path.join(".claude", "protected-paths.txt")


def project_root(data):
    """确定项目根：优先 CLAUDE_PROJECT_DIR，其次 hook 输入里的 cwd，最后当前目录。"""
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    return os.path.abspath(root)


def load_patterns(root):
    """返回 (规则列表, 来源说明)。规则保持原始写法，`!` 前缀在 match 里解析。"""
    path = os.path.join(root, RULES_FILE)
    if not os.path.isfile(path):
        return DEFAULT_PATTERNS, "内置默认"

    patterns = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
    except Exception:
        return DEFAULT_PATTERNS, "内置默认（protected-paths.txt 读取失败）"

    if not patterns:
        # 文件存在但全是注释/空行 —— 视为用户主动清空，尊重它，不再兜回默认
        return [], RULES_FILE
    return patterns, RULES_FILE


def parse_rule(raw):
    """拆出 (glob, 是否无条件阻断)。`!` 前缀表示无条件，不是 gitignore 的取反。"""
    if raw.startswith("!"):
        return raw[1:].strip(), True
    return raw, False


def match(rel_path, patterns):
    """命中返回 (原始规则行, glob, 是否无条件)；没命中返回 None。

    fnmatch 的 * 会跨越 /，所以 `docs/**` 这类规则能覆盖任意深度；
    纯文件名匹配则让 `*.lock`、`.env` 这类规则不依赖它在哪一层目录。

    无条件规则优先：先扫一遍 `!` 规则，命中就直接判无条件阻断，
    免得同一个文件先被某条普通规则匹上、又因为"文件不存在"被放行。
    """
    base = os.path.basename(rel_path)
    parsed = [(raw,) + parse_rule(raw) for raw in patterns]

    for unconditional_first in (True, False):
        for raw, glob, unconditional in parsed:
            if unconditional is not unconditional_first:
                continue
            if not glob:
                continue
            if fnmatch.fnmatch(rel_path, glob) or fnmatch.fnmatch(base, glob):
                return raw, glob, unconditional
    return None


def block(rel_path, rule, source, unconditional):
    """stderr 输出中文说明并以 exit 2 阻断（stderr 内容会回给 Claude）。"""
    note = RULE_NOTES.get(rule)
    if note:
        why = note
        outs = """两条合法出路：
  1. 走这条规则说明里指定的那条命令/流程（首选）。
  2. 若确属误伤：请用户手动改，或临时把 .claude/protected-paths.txt 里那条规则
     注释掉，改完立刻恢复，不要长期留着。"""
    elif unconditional:
        why = (
            "命中的是一条**无条件规则**（`!` 前缀）：这类文件不论新建还是修改都被禁止 ——\n"
            "凭空造一个 .env 会让密钥出现在仓库里，手写一个 lock 文件同样会污染\n"
            "环境可复现性。"
        )
        outs = """三条合法出路：
  1. 由用户手动创建/编辑（推荐）—— 保留一次人工确认，最安全。
  2. 锁文件请让包管理器自己生成（`uv lock` 等），不要手写。
  3. 若确属误伤：临时把 .claude/protected-paths.txt 里那条规则注释掉，
     改完后立刻恢复，不要长期留着。"""
    else:
        why = (
            "命中的是一条**仅保护已有文件**的规则：该文件**已经存在**。\n"
            "覆盖/删除它往往不可撤销。\n"
            "（注意：往这些路径里**新建**文件是允许的，护栏不会拦。）"
        )
        outs = """三条合法出路：
  1. 换一个**新的**输出路径（推荐）。
  2. 由用户手动编辑 —— 保留一次人工确认。
  3. 若确属误伤：临时把 .claude/protected-paths.txt 里那条规则注释掉，
     改完后立刻恢复，不要长期留着。"""

    msg = f"""[受保护路径] 本次写入被工作流护栏阻断。

  命中文件：{rel_path}
  命中规则：{rule}
  规则来源：{source}

{why}

{outs}

请不要绕过此护栏，先向用户说明情况并征求确认。"""
    sys.stderr.write(msg + "\n")
    sys.exit(2)


def main():
    raw = sys.stdin.read()
    data = json.loads(raw)

    file_path = (data.get("tool_input") or {}).get("file_path") or ""
    if not file_path:
        return

    root = project_root(data)
    abs_path = os.path.abspath(os.path.join(root, file_path))
    try:
        rel_path = os.path.relpath(abs_path, root)
    except ValueError:
        rel_path = file_path
    # 统一成 posix 分隔符，规则写法才不用管平台
    rel_path = rel_path.replace(os.sep, "/")

    patterns, source = load_patterns(root)
    hit = match(rel_path, patterns)
    if not hit:
        return  # 未命中：静默放行

    rule, _glob, unconditional = hit
    if unconditional:
        block(rel_path, rule, source, True)

    # 普通规则：只有目标已经存在时才是"覆盖已有产物"，才阻断。
    # 用绝对路径判断，避免 hook 的工作目录不是项目根时判错。
    if os.path.exists(abs_path):
        block(rel_path, rule, source, False)
    # 新建文件：放行


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise  # exit 2 必须原样传出去，不能被下面的兜底吞掉
    except Exception:
        # 护栏自身出错时选择放行而不是误杀：这是 PreToolUse，
        # 误杀会让 agent 完全无法工作，而漏判只是回到没有护栏的状态。
        pass
    sys.exit(0)
