#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PostToolUse(Edit|Write) hook：单文件的即时语法 / 格式检查。

## 为什么这个项目需要它，以及它和 contracts-after-edit.sh 的分工

CLAUDE.md 写得很清楚：「There is no linter, formatter, or JS toolchain
configured」。所以模板原版那套「探测 ruff/black/prettier，有就自动重排」在这里
装上去等于装了个永远静默的空壳。真正有价值的适配是**先搞清楚哪些文件此刻没有
任何人在看**，然后只补那一块。

已有的 `contracts-after-edit.sh` 在每次 Edit/Write 之后跑 `scripts/check.py
--contracts`，而 `check.py::check_syntax()` 已经做了两件事：

    - 全仓 `**/*.py` 的 py_compile（跳过 .venv / .claude / __pycache__）
    - `web/*.js` 的 `node --check`

也就是说 **repo 内的 .py 和 web/*.js 语法已经有人管了**，再 py_compile 一遍只会
让同一个 SyntaxError 在 transcript 里出现两次。那个 shell hook 的触发条件是
`$ROOT/*.py|*.js|*.html`，且显式 `exit 0` 掉 `$ROOT/.claude/*` 与 `$ROOT/.venv/*`。

两边的排除项一叠加，就露出了真正的窟窿——**下面这些文件目前没有任何检查**：

  1. `.claude/**/*.py` —— hooks 自己。protect_paths.py / verify_stop.py /
     session_context.py 全部以 `except: pass; sys.exit(0)` 收尾（本文件也是），
     这是 hook 的正确写法，但代价是**改坏了完全无声**：一个缩进错误会让
     protect_paths.py 从「护栏」变成「什么都不拦」，而你不会收到任何信号。
     这正好是 CLAUDE.md 反复强调的那类 silent failure。
  2. `*.json` —— `.claude/settings.json` 一旦不合法，**整套 hook 会静默失效**；
     `scripts/golden.json` 不合法则 tier-2 直接崩。两者都没人校验。
  3. `*.sh` —— `.claude/hooks/*.sh`、`.claude/verify.sh`，同 (1)。`bash -n` 免费。
  4. `*.yaml` —— pyyaml 是声明依赖、`configs/` 是空目录，将来放进去的配置同样
     只会在运行时才炸。
  5. 仓库之外的 .py/.js（临时脚本），contracts hook 对它们直接 exit 0。

所以本 hook 的规则是：**contracts 管得着的，一个字都不重复说**
（见 `covered_by_contracts()`，它逐条镜像了那个 shell 脚本 + check_syntax 的
排除逻辑）；contracts 管不着的，这里补上。两者严格互补、零交集——顺带地，这也
意味着两个 hook 在同一事件下并发执行时**谁先谁后完全不影响输出**。

`covered_by_contracts()` 还额外要求 `contracts-after-edit.sh` 确实存在：哪天它被
删了，本 hook 会自动接管全部语法检查，而不是跟着一起失明。

## 关于「格式化」

模板原版会 `ruff format` / `black` / `prettier --write` **直接改写文件**。这里
两处改动：

  - **只在项目真正配置了该工具时才启用**（pyproject 的 [tool.ruff] / ruff.toml /
    .prettierrc ...）。光是「这台机器上碰巧装了 ruff」不足以构成理由——本项目的
    代码是手工排版的，拿一个没有配置文件的 formatter 去重排它，制造的 diff 比
    抓到的问题多。CLAUDE.md 那句「no linter, formatter configured」是事实陈述，
    也是一条应当被尊重的项目决定。
  - **一律 check 模式，绝不 --write**。PostToolUse 阶段在 Claude 背后改写它刚写
    完的文件，会让它后续的 Edit 因 old_string 对不上而失败。

结果：今天这几个工具全都不存在也没有配置，这一段静默跳过；将来谁在 pyproject
里加了 [tool.ruff]，它自动生效。这才是模板「优雅降级」设计的正确用法。

刻意不做的：mypy（需要全项目上下文，单文件跑只会在 JAX 代码上刷屏，且那属于
`check.py` 的职责范围）、HTML 校验（无可用工具，且 index.html 已被 contracts
的颜色一致性检查覆盖）、TOML（本机 python 是 3.10，没有 tomllib）。

## 硬性约束

  - **永远 exit 0，绝不阻断**。问题通过 hookSpecificOutput.additionalContext 注入
    context 让 Claude 自己看见。阻断权留给 protect_paths.py(PreToolUse)、
    contracts-after-edit.sh 和 verify_stop.py(Stop)。
  - 只用 Python 3 标准库；任何异常兜底 exit 0——hook 挂掉不能拖垮主流程。
  - 每个子进程 timeout=30；注入文本截断到 2000 字符。
  - python 子进程一律带 XLA_PYTHON_CLIENT_PREALLOCATE=false 前缀，与
    require-preallocate.sh 保持一致（本 hook 不走那个 PreToolUse hook，但没有
    理由在这里破例）。
"""

import json
import os
import subprocess
import shutil
import sys

# 单个子进程超时（秒）。这里跑的都是 compile / node --check，毫秒级；
# 超过 30s 只可能是卡住了。
TIMEOUT = 30
# 注入 context 的文本上限，防止一个刷屏的输出吃掉上下文预算。
MAX_CONTEXT_CHARS = 2000
# 文件大小上限：超过就不做纯 Python 的解析（json/yaml），避免 hook 变慢。
MAX_PARSE_BYTES = 10 * 1024 * 1024

# CLAUDE.md：Every python invocation needs XLA_PYTHON_CLIENT_PREALLOCATE=false.
PY_ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}

# 走 prettier 的扩展名（仅在项目配置了 prettier 时才会真的跑）。
PRETTIER_EXTS = {".js", ".ts", ".tsx", ".jsx", ".json", ".css", ".md"}

# 子进程里跑的语法检查：用内建 compile() 而不是 py_compile，
# 因为 compile() 只解析不写盘——不会在工作区留下 __pycache__ / .pyc 垃圾。
# （check.py 用的是 py_compile + 手动 unlink，那是它自己的选择。）
_COMPILE_SNIPPET = r"""
import sys
p = sys.argv[1]
try:
    with open(p, 'rb') as fh:
        src = fh.read()
    compile(src, p, 'exec')
except SyntaxError as e:
    sys.stderr.write('%s:%s:%s: SyntaxError: %s\n'
                     % (p, e.lineno, e.offset, e.msg))
    sys.exit(1)
except Exception as e:
    sys.stderr.write('%s: %s: %s\n' % (p, type(e).__name__, e))
    sys.exit(1)
"""

_YAML_SNIPPET = r"""
import sys
try:
    import yaml
except ImportError:
    sys.exit(0)          # 没装 pyyaml —— 属于"工具不存在"，静默跳过
p = sys.argv[1]
try:
    with open(p, 'rb') as fh:
        yaml.safe_load(fh)
except Exception as e:
    sys.stderr.write('%s: %s: %s\n' % (p, type(e).__name__, e))
    sys.exit(1)
"""


# --------------------------------------------------------------------------
# 基础设施
# --------------------------------------------------------------------------

def run(cmd, env_extra=None):
    """跑一个子进程，返回 (returncode, 合并输出)。任何异常都当作"跳过"。"""
    try:
        env = None
        if env_extra:
            env = dict(os.environ)
            env.update(env_extra)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=env,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        # 超时 / 工具突然消失 / 编码问题 —— 一律当没发生过
        return 0, ""


def project_root():
    """项目根。优先 CLAUDE_PROJECT_DIR，退化到本文件位置往上三层。"""
    root = os.environ.get("CLAUDE_PROJECT_DIR")
    if not root:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
    try:
        return os.path.realpath(root)
    except Exception:
        return root


def venv_python(root):
    """项目的解释器。CLAUDE.md 通篇用 .venv/bin/python，语法检查也该用它——
    「这个文件能不能被将要运行它的那个解释器解析」才是真正要问的问题。"""
    p = os.path.join(root, ".venv", "bin", "python")
    return p if os.path.isfile(p) and os.access(p, os.X_OK) else None


def under(path, directory):
    """path 是否在 directory 之内（含 directory 本身）。"""
    try:
        return os.path.commonpath([path, directory]) == directory
    except Exception:
        return False


def covered_by_contracts(path, root):
    """这个文件的语法是否已经被 contracts-after-edit.sh 检过了？

    逐条镜像那个 shell 脚本的触发条件与 check.py::check_syntax 的排除项。
    命中就不重复报——同一个 SyntaxError 说两遍不会让它更容易修。
    """
    if not os.path.isfile(os.path.join(
            root, ".claude", "hooks", "contracts-after-edit.sh")):
        return False                                   # 它没了 -> 我们接管
    if not under(path, root):
        return False                                   # 脚本只管仓库内
    if os.path.splitext(path)[1].lower() not in (".py", ".js", ".html"):
        return False                                   # 脚本的 case 只列了这三种
    for excluded in (".claude", ".venv"):
        if under(path, os.path.join(root, excluded)):
            return False                               # 脚本显式 exit 0
    if "__pycache__" in path.split(os.sep):
        return False                                   # check_syntax 跳过
    return True


# --------------------------------------------------------------------------
# 语法检查（本 hook 的主要价值）
# --------------------------------------------------------------------------

def check_python(path, root):
    py = venv_python(root) or sys.executable
    code, out = run([py, "-c", _COMPILE_SNIPPET, path], PY_ENV)
    if code != 0 and out.strip():
        return [out.strip()]
    return []


def check_js(path):
    """node --check —— CLAUDE.md 命令区里自己列着的那条，无需构建。

    web/ 是 FastAPI 静态托管的裸 ES5，没有 bundler、没有 node_modules，
    所以除了打开浏览器之外这是唯一能发现 JS 写坏了的手段。
    """
    if not shutil.which("node"):
        return []
    code, out = run(["node", "--check", path])
    if code == 0:
        return []
    # node 的报错尾部挂着 node:internal 的调用栈，对读者零价值，去掉。
    lines = [ln for ln in out.splitlines()
             if "node:internal" not in ln and not ln.startswith("Node.js v")]
    text = "\n".join(lines).strip()
    return [text] if text else []


def check_json(path):
    """纯标准库，无子进程。`.claude/settings.json` 坏掉会静默停掉所有 hook。"""
    try:
        if os.path.getsize(path) > MAX_PARSE_BYTES:
            return []
        with open(path, "rb") as fh:
            json.load(fh)
    except ValueError as e:                            # 含 JSONDecodeError
        return ["%s: invalid JSON: %s" % (path, e)]
    except Exception:
        return []
    return []


def check_yaml(path, root):
    py = venv_python(root)
    if not py:
        return []
    try:
        if os.path.getsize(path) > MAX_PARSE_BYTES:
            return []
    except Exception:
        return []
    code, out = run([py, "-c", _YAML_SNIPPET, path], PY_ENV)
    if code != 0 and out.strip():
        return [out.strip()]
    return []


def check_shell(path):
    """bash -n：只解析不执行。hooks 目录里的 .sh 同样没有别人在看。"""
    if not shutil.which("bash"):
        return []
    code, out = run(["bash", "-n", path])
    if code != 0 and out.strip():
        return [out.strip()]
    return []


# --------------------------------------------------------------------------
# 可选工具：装了**并且项目配置了**才跑，且一律 check 模式不改写文件
# --------------------------------------------------------------------------

def has_config(root, filenames=(), pyproject_sections=()):
    for name in filenames:
        if os.path.exists(os.path.join(root, name)):
            return True
    pyproject = os.path.join(root, "pyproject.toml")
    if pyproject_sections and os.path.isfile(pyproject):
        try:
            with open(pyproject, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except Exception:
            return False
        for section in pyproject_sections:
            if ("[%s]" % section) in text or ("[%s." % section) in text:
                return True
    return False


def optional_python_tools(path, root):
    problems = []
    if shutil.which("ruff") and has_config(
            root, ("ruff.toml", ".ruff.toml"), ("tool.ruff",)):
        for cmd in (["ruff", "check", "--no-fix", path],
                    ["ruff", "format", "--check", path]):
            code, out = run(cmd)
            if code != 0 and out.strip():
                problems.append(out.strip())
    elif shutil.which("black") and has_config(root, (), ("tool.black",)):
        code, out = run(["black", "--check", "--quiet", path])
        if code != 0 and out.strip():
            problems.append(out.strip())
    if shutil.which("flake8") and has_config(
            root, (".flake8", "tox.ini", "setup.cfg"), ()):
        code, out = run(["flake8", path])
        if code != 0 and out.strip():
            problems.append(out.strip())
    return problems


def optional_prettier(path, root):
    if not has_config(root, (".prettierrc", ".prettierrc.json", ".prettierrc.yml",
                             ".prettierrc.yaml", ".prettierrc.js",
                             "prettier.config.js", "prettier.config.mjs"), ()):
        return []
    exe = shutil.which("prettier")
    cmd = [exe, "--check", path] if exe else None
    if cmd is None and shutil.which("npx"):
        # --no-install 关键：否则 npx 会联网下载，离线环境下卡满超时。
        cmd = ["npx", "--no-install", "prettier", "--check", path]
    if cmd is None:
        return []
    code, out = run(cmd)
    if code == 0:
        return []
    low = out.lower()
    if "could not determine executable" in low or "not found" in low:
        return []                                       # 工具不存在，不是问题
    return [out.strip()] if out.strip() else []


def optional_shfmt(path):
    if not shutil.which("shfmt"):
        return []
    code, out = run(["shfmt", "-d", path])              # -d = diff，不写盘
    if code != 0 and out.strip():
        return [out.strip()]
    return []


# --------------------------------------------------------------------------

def emit_context(text):
    """把问题注入 Claude 的 context（不阻断，只提醒）。"""
    text = text.strip()
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS] + "\n...(输出过长已截断)"
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                "刚写入的文件没有通过语法/格式检查（未阻断，请自行判断是否修复）：\n"
                + text
            ),
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


def main():
    data = json.loads(sys.stdin.read())   # 畸形输入交给外层 except 兜底

    # 和 contracts-after-edit.sh 取同一个字段，优先 tool_response.filePath。
    file_path = ((data.get("tool_response") or {}).get("filePath")
                 or (data.get("tool_input") or {}).get("file_path") or "")
    if not file_path or not os.path.isfile(file_path):
        return                            # 文件可能已被后续操作删掉/移走
    file_path = os.path.realpath(file_path)

    root = project_root()
    ext = os.path.splitext(file_path)[1].lower()
    problems = []

    # 1) 语法：只做 contracts-after-edit.sh 覆盖不到的那部分。
    if not covered_by_contracts(file_path, root):
        if ext == ".py":
            problems += check_python(file_path, root)
        elif ext == ".js":
            problems += check_js(file_path)
        elif ext == ".json":
            problems += check_json(file_path)
        elif ext in (".yaml", ".yml"):
            problems += check_yaml(file_path, root)
        elif ext == ".sh":
            problems += check_shell(file_path)

    # 2) 格式/lint：与语法覆盖无关，contracts 完全不管这一层。
    #    今天全部静默跳过（项目未配置任何一个），将来配了自动生效。
    if ext == ".py":
        problems += optional_python_tools(file_path, root)
    if ext in PRETTIER_EXTS:
        problems += optional_prettier(file_path, root)
    if ext == ".sh":
        problems += optional_shfmt(file_path)

    if problems:
        emit_context("\n".join(problems))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 兜底：hook 出任何意外都必须安静退出，绝不打断主流程。
        pass
    sys.exit(0)
