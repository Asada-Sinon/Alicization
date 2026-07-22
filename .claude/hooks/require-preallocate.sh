#!/usr/bin/env bash
# PreToolUse(Bash): refuse to launch a JAX process without
# XLA_PYTHON_CLIENT_PREALLOCATE=false.
#
# CLAUDE.md has said this for a long time, and it kept being forgotten -- which
# is the difference between an instruction and a guarantee. JAX grabs 75% of the
# card at startup, so the *second* process to start dies with a fake
# CUDA_ERROR_OUT_OF_MEMORY even though the real peak is under 1 GiB. The cost of
# forgetting is a serialised sweep or a mystifying OOM; the cost of the guard is
# a prefix.
#
# Deny rather than ask: the fix is mechanical and there is no case where the
# right answer is "run it without the variable".
#
# Note the program is passed with -c, not on stdin: a heredoc would consume the
# stdin the hook payload arrives on, and the check would silently pass forever.

set -uo pipefail
PY=/usr/bin/python3
[ -x "$PY" ] || PY=python3

exec "$PY" -c '
import json, re, sys

try:
    cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
except Exception:
    sys.exit(0)                      # never break the tool over a parse failure

if "XLA_PYTHON_CLIENT_PREALLOCATE" in cmd:
    sys.exit(0)

# An interpreter launch in *command position* -- start of the line, or after a
# separator, or after do/then/else. Matching on any surrounding whitespace was
# the first attempt and it denied `grep -rn python docs/`, which is a normal
# thing to want to run in a repo whose docs talk about python.
LAUNCH = re.compile(r"""
    (?: ^ | [\n;&|(] | \b(?: do | then | else ) \s )   # command position
    \s* (?: \w+=\S* \s+ )*                             # inline env assignments
    (?: [\w./-]*/ )? python [\d.]* (?: \s | $ )
""", re.X)
if not LAUNCH.search(cmd):
    sys.exit(0)

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            "This repo requires XLA_PYTHON_CLIENT_PREALLOCATE=false on every "
            "python invocation (see CLAUDE.md). Without it JAX preallocates 75% "
            "of the GPU and the next process to start fails with a fake "
            "CUDA_ERROR_OUT_OF_MEMORY -- the real peak is 918 MiB.\n\n"
            "Re-run it with XLA_PYTHON_CLIENT_PREALLOCATE=false in front of the "
            "python invocation itself, e.g.\n"
            "  XLA_PYTHON_CLIENT_PREALLOCATE=false " + cmd.strip()
        ),
    }
}))
'
