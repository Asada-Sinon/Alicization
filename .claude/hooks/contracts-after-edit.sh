#!/usr/bin/env bash
# PostToolUse(Edit|Write): run the cross-file contract checks after every source
# edit, and feed any failure straight back to Claude.
#
# This is the tier-1 half of `scripts/check.py` only -- no JAX, no world, ~0.2s.
# The expensive half stays out of here deliberately: a 14-second check on every
# one of the several dozen edits in a session would cost more wall-clock than
# the bugs it catches, and the smoke world is what the pre-commit run is for.
#
# What it does catch, immediately: a wire-format offset that no longer matches
# `web/main.js`, a species colour that drifted out of sync across the three
# files that duplicate it, a config scaling rule violated, a syntax error. All
# of those are silent failures -- they produce plausible wrong numbers or a
# wrong-looking canvas rather than an exception -- which is exactly the class
# worth spending 0.2s per edit on.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/usr/bin/python3
[ -x "$PY" ] || PY=python3

# Only for source files inside this repo. Editing docs/*.md cannot break a
# contract, and paying for a check there just trains you to ignore the output.
path="$("$PY" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
print(d.get("tool_response", {}).get("filePath")
      or d.get("tool_input", {}).get("file_path") or "")
')"
case "$path" in
  "$ROOT"/*.py|"$ROOT"/*.js|"$ROOT"/*.html) ;;
  *) exit 0 ;;
esac
case "$path" in
  "$ROOT"/.claude/*|"$ROOT"/.venv/*) exit 0 ;;
esac

out="$(cd "$ROOT" && .venv/bin/python scripts/check.py --contracts 2>&1)" && exit 0

# Strip the ANSI colouring -- this text goes into a transcript, not a terminal.
out="$("$PY" -c '
import re, sys
sys.stdout.write(re.sub(r"\x1b\[[0-9;]*m", "", sys.stdin.read()))
' <<<"$out")"

"$PY" -c '
import json, sys
out = sys.stdin.read()
print(json.dumps({"decision": "block", "reason":
    "scripts/check.py --contracts failed after this edit. These are the "
    "cross-file contracts that break silently -- fix them before moving on, "
    "and do not weaken the check to make it pass.\n\n" + out}))
' <<<"$out"
exit 0
