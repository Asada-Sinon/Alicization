#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stop hook：完成门禁。

Claude 认为自己干完了要停下来时触发。这里跑一遍 .claude/verify.sh；
不通过就 exit 2 把它打回去继续修。

本项目的 verify.sh 只跑 `scripts/check.py --contracts`（tier 1，约 0.3 秒）——
取舍的完整理由写在 .claude/verify.sh 里。

.claude/verify.sh 不存在时静默放行。
"""

import json
import os
import subprocess
import sys

# verify.sh 的总超时（秒）。超过就当没跑，放行。
# settings.json 里这个 hook 的 timeout 是 310，比这里大 10 秒，
# 这样超时永远由本脚本自己处理（能给出解释），而不是被 harness 硬砍。
TIMEOUT = 300
# 失败时回给 Claude 的日志行数：太多会淹没 context，60 行够定位问题。
TAIL_LINES = 60

VERIFY_REL = os.path.join(".claude", "verify.sh")


def project_root(data):
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    return os.path.abspath(root)


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    # ★ 必须第一件事就检查 stop_hook_active ★
    # 这个字段为 true 表示：本次 Stop 就是被上一次 Stop hook 的 exit 2 逼出来的。
    # 如果这里不短路，就会形成
    #   Claude 想停 → hook exit 2 打回 → Claude 再想停 → hook 再 exit 2 → ...
    # 的无限循环：既永远停不下来，又一直在烧 token，直到 session 超时或余额耗尽。
    # 任何 Stop hook 都必须先做这一步判断。
    if data.get("stop_hook_active"):
        sys.exit(0)

    # 逃生阀：卡在门禁上出不去时，用 SKIP_VERIFY=1 临时跳过。
    if os.environ.get("SKIP_VERIFY") == "1":
        sys.exit(0)

    root = project_root(data)
    verify = os.path.join(root, VERIFY_REL)

    # 没配门禁 —— 静默放行，不打扰
    if not os.path.isfile(verify):
        sys.exit(0)
    if not os.access(verify, os.X_OK):
        # 存在但没有执行权限：提示一次，但不阻断（否则用户会莫名其妙被卡住）
        sys.stderr.write(
            f"提示：{VERIFY_REL} 存在但没有执行权限，已跳过门禁。"
            f"执行 chmod +x {VERIFY_REL} 后生效。\n"
        )
        sys.exit(0)

    try:
        proc = subprocess.run(
            [verify],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"提示：{VERIFY_REL} 运行超过 {TIMEOUT} 秒被终止，本次门禁跳过。\n"
        )
        sys.exit(0)
    except Exception:
        sys.exit(0)

    if proc.returncode == 0:
        sys.exit(0)

    output = ((proc.stdout or "") + (proc.stderr or "")).rstrip().splitlines()
    tail = "\n".join(output[-TAIL_LINES:])
    sys.stderr.write(
        f"[完成门禁未通过] {VERIFY_REL} 退出码 {proc.returncode}。\n"
        f"以下是最后 {TAIL_LINES} 行输出：\n\n{tail}\n\n"
        "这是 scripts/check.py 的 tier-1 跨文件契约检查，全是会静默失败的东西"
        "（wire 偏移、三处重复的物种颜色、config 缩放规则、语法）。"
        "请先修复再结束本轮，**不要为了让它通过而弱化检查**。"
        "若确认是门禁脚本本身的问题（而非代码问题），请向用户说明，"
        "不要擅自修改或删除 verify.sh。\n"
    )
    sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
