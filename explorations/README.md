# explorations/

**一次性分析脚本的沙箱**：需要复现、但还不够格进 `scripts/` 的代码放这里。

这个目录的范围很窄，先说清它**不是**什么。`CLAUDE.md` 的「scratch work belongs in
the session scratchpad, not here」仍然成立——这里不是 scratch 的新家。它解决的是另一
件事：`result-analyst` 为了算一个统计量临时写的脚本，如果只活在 session scratchpad
里，那个数字下一次就再也算不出来了。本项目的统计纪律（6 配对种子起、报逐种子数字、
bootstrap 区间、不做 Bonferroni、报出算过的每一个 p 值）全都要求**产出数字的那段代码
本身可追溯**——一个说不清怎么算出来的 p=0.031，和一个凭空写下的 p=0.031 在证据强度上
没有区别。

## 三个去处，别放错

| 放哪 | 是什么 | 进版本库？ |
| --- | --- | --- |
| session scratchpad | 完全一次性：试一条命令、看一眼某个 JSON 字段长什么样、不需要再看第二眼的东西 | 否，也不该 |
| `outputs/ checkpoints/ runs/` | 实验**产物**：run 日志、`--json` 行、权重。体积大、可重跑 | 否（`.gitignore` 已排除） |
| `explorations/` | 产生结论的**分析脚本**：算效应量、拼多种子对比表、画一次性诊断图 | **是**（脚本本身进；它自己的输出不进） |
| `scripts/` | 正式工具：`check.py`、`run_headless.py`、`run_live.py`。有人会依赖，坏了要修 | 是 |

一句话判据：**这段代码的输出会不会被当成数字写进 `docs/`？** 会，就写进
`explorations/`；不会，就留在 scratchpad。

## 布局

一个问题一个子目录：`explorations/<YYYYMMDD>-<slug>/`。

```text
explorations/
  20260723-fear-rate-effect-size/
    README.md          # 三行：在回答什么问题、读了哪些文件、结论去了 docs/ 的哪一节
    cmp_fear.py        # 分析脚本本身（进版本库）
    output/            # 它跑出来的图与中间 CSV（已 gitignore）
```

脚本开头用注释写明：**它在回答什么问题、读了哪几个文件、输出该怎么读**。三行就够，
但不能没有——半年后没人记得 `cmp_fear.py` 当初比的是哪两个臂。

## 命令形态（照抄，不要自己拼）

**每一次 python 调用都要 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 前缀 +
`.venv/bin/python`。** 一个 `PreToolUse` hook 会直接 deny 缺前缀的命令，理由见
`CLAUDE.md`「Commands」一节（JAX 预分配 75% 显存，实测真实峰值只有 918 MiB）。
分析脚本通常根本不加载 JAX，但 hook 拦的是「python 启动」这件事，前缀照加。

```bash
# 跑一个分析脚本
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  explorations/20260723-fear-rate-effect-size/cmp_fear.py

# 先生成它要读的产物（多种子一个臂，并行靠的就是这个前缀）
for s in 0 1 2 3 4 5; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/run_headless.py 20000 500 \
    --seed $s --json --set fear_rate=0 > outputs/fear0_seed$s.log &
done; wait

# 动过被 scripts/ 依赖的东西之后
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --contracts
```

## 规矩

- **这里的代码可以脏**：写死路径、复制粘贴、没有测试、没有类型标注，都行。目的是尽快
  拿到一个可核对的数字，不是留下资产。
- **但必须能重跑。** 写死路径可以，写死一个只存在于你脑子里的中间结果不行；随机过程要
  固定 seed。判据：换一个人、隔一周，照抄上面那条
  `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python <脚本>` 应该给出同样的数字。
- **输出一律写 `explorations/<dir>/output/`**（已 gitignore），不要写到子目录根上，
  否则下次 `git status` 会一片红。
- **不要在这里放任何别处会 import 的东西。** `underworld/`、`server/`、`scripts/`、
  `tests/` 都不许 import `explorations/`。
- **不改产品代码。** 分析脚本只读 `outputs/`、`docs/`、`scripts/golden.json`（只读！）
  和源码。消融一律走 `run_headless.py --set FIELD=VALUE`，不改 `underworld/config.py`。
- **结论不留在这里。** 脚本的 stdout 是证据，结论要按四标签写进 `docs/<topic>.md` 并在
  `docs/TODO.md` 加一行指针——`CLAUDE.md`：「Research lands in `docs/`, or it did not
  happen」。`explorations/` 保存的是**这个数字是怎么算出来的**，不是这个数字意味着什么。

## 毕业路径：`explorations/` → `scripts/`

一个脚本第三次被翻出来用（或者别人开始问「那个算效应量的脚本在哪」），就该升进
`scripts/`。**升级是重写，不是移动**：去掉写死路径、加参数、走统一入口、补一个最小测试。
探索脚本本身**留在原地**当记录——它是当初那个数字的出处，删了就等于把 `docs/` 里的引用
悬空。

反过来的判据同样重要：一个脚本只会被用一次、而且它算出来的数字已经进了 `docs/`，那它就
该一直待在 `explorations/`，不要为了「整洁」搬进 `scripts/`。`scripts/` 里每多一个文件，
就多一样坏了要修的东西。

## 清理

`explorations/` 里的东西**不随便删**——`docs/` 里的数字可能正指着它。真要删，先
`grep -rn "explorations/" docs/` 确认没有引用，再在 commit message 里写清删的是哪个、
为什么不再需要。`output/` 子目录随时可删，那是可重跑的。
