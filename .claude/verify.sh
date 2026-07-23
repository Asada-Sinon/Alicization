#!/usr/bin/env bash
# ============================================================
# 完成门禁 —— 由 .claude/hooks/verify_stop.py 在每次 Stop 时调用
# ============================================================
# 约定：
#   * 退出码 0   = 通过，Claude 正常结束
#   * 退出码 非0 = 不通过，最后 60 行输出回给 Claude，让它去修
#   * 临时跳过：SKIP_VERIFY=1 claude
#
# ------------------------------------------------------------
# 为什么**只跑 tier 1**（这是本文件唯一需要解释的决定）
# ------------------------------------------------------------
# CLAUDE.md 的 Verification 一节把验证分成三档：
#
#   tier 1  --contracts   ~0.3s   跨文件契约，不加载 JAX
#   tier 2  默认           ~14s    2048 agent × 200 步冒烟世界 + golden band
#   tier 3  --full        ~3min   以上全部 + pytest
#
# 这个脚本在 Claude **每一轮**打算结束时都会跑一次。一轮 14 秒还能忍，
# 一轮 3 分钟不可能忍——而一个让人无法忍受的门禁不会被优化，会被整个关掉
# （SKIP_VERIFY=1 一挂就是一整天），那样连 0.3 秒的契约检查也一起没了。
# 所以这里选择"必然会被留着开"的那一档，而不是"覆盖最全"的那一档。
#
# 挡住的东西也确实值这 0.3 秒：wire 偏移与 web/main.js 对不上、三个文件里
# 重复的物种颜色漂移、config 缩放规则被违反、语法错误——这几类的共同点是
# **静默失败**，它们产出看起来合理的错数字或错画面，而不是抛异常。
#
# tier 2 / tier 3 不是不跑，是换个地方跑：
#   * tier 2 的 golden band 靠 config 改动触发，改完 config.py 手动跑一次即可；
#   * tier 3 是 **commit 前的那一次**，CLAUDE.md 的 Git 一节已经明确要求：
#         XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --full
#     它要几分钟，给它一个长超时，别以为它挂了。
#
# 另外 tier 1 已经有一个 PostToolUse hook（contracts-after-edit.sh）在每次
# 源码编辑后跑了。这里的重复是**刻意的**：那个 hook 只在 Edit/Write 之后触发，
# 而一轮工作也可能通过 Bash（脚本改文件、git checkout、apply patch）改动代码，
# 那条路径上没有任何检查。Stop 是最后一道，成本 0.3 秒，重复得起。
# ------------------------------------------------------------
set -uo pipefail    # 故意不加 -e：我们要自己控制哪一步失败才算失败

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 0

fail=0

# .venv 不存在就放行——门禁的作用是拦回归，不是拦"环境还没装好"。
if [ ! -x .venv/bin/python ]; then
  echo "跳过：.venv/bin/python 不存在或不可执行"
  exit 0
fi

# PREALLOCATE 前缀是 CLAUDE.md 的硬性要求（见其 Commands 一节）。--contracts
# 这一档其实不加载 JAX，但前缀一律带上：这个脚本迟早会有人往里加东西，
# 那时忘记加前缀就是一次假 OOM。
echo "==> scripts/check.py --contracts (tier 1, ~0.3s)"
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --contracts || fail=1

exit "$fail"
