# 当前焦点

只写此刻在做的那一件事。方向变了就整个重写，不要往下堆。超过一屏说明该拆了。

## 当前目标（2026-08-03，用户已拍板的第 2 件）

**把果层挪出共居带 —— `docs/multispecies_program.md` §12 的 C 阶段。**

第 1 件（升级统计协议）**已完成并推送**，不要再回头做。
§12 的 A、B 两阶段**已完成**（判决写在 §12.3.A'、落地写在 §12.3.B）。

## 此刻要做的那一件事

**C 阶段：单种子定标探针**，在 **`niche`（厚果层）配置上**扫
`fruit_dry_weight × fruit_dry_d0`，找**「分离了但果实还吃得到」**的工作点。

- 完成判据：给出一个可以进 D 阶段的 `(w, d0)` 工作点，或**给出「这条曲线上没有可用点」
  这个负结论**——§12.3 C 明写「若确实没有，本轮就此为负结果收口，不要硬调」。
- 读数**五个**（A 判决把三个扩成了五个）：`sel_ratio_water`、`frugivory_frac`、
  `fruit_cells_eff`、`herb_water_dist`、`death_thirst_frac`。
- 判据容差直接抄 §12.3.A' 那张表，**不要重新拍脑袋定**。

## 为什么基线是 `niche` 而不是默认配置

A 判决实测：默认世界里果层与种群的重叠是**纯几何**的（`sel_ratio_water` 1.009，
扣掉水距后落回零模型），厚果层世界里则是**行为性且很强**的（3.712，12/12，p=0.00049）。
`base` 的「可下降空间 ÷ MDE」只有 0.52——**手术做对了也测不出下降**；`niche` 是 6.62。

## 非目标

- 不动 `in_dim`（作废全部演化脑）。
- 不为让实验「成立」而放宽 `scripts/golden.json` 的 band，也不在看到数据后改判据读法。
- **不再调果层的供给量参数**。`fruit_dry_weight` 已做成容量中性（等量再分配），
  **不许为了让 C 好看而破坏这条**——破坏了 D 就归因不了。
- 三条已关闭别再捡：加厚食物层、等量再分配、生殖隔离（`assortative_mating` on `forage_pref`）。

## 纪律

- **sweep 跑的时候绝不能改 `underworld/`**——逐批启动的进程会加载到不同代码，且无报错。
- 并行跑实验**最多 6 个并发**；多参数 sweep **一律写 bash 脚本 + 数组传参**（zsh 不词分割）。
- **单种子探针不能读 `carnivore_frac` / `population` 的好坏**（2×噪声占基线 71% / `σ_B=0`），
  只能读「有没有塌到 0」。
- 统计一律走 `scripts/exp_stats.py`，改了它必须重跑
  `explorations/20260803-partition/verify_exp_stats.py`。
- 空间结论必须交叉地形种子（§12.3 E **是必须的**：A 的 48 个 run 全在同一张地图上）。
- 跑实验前先 commit；判决派 `result-analyst`，主控不自己读几十份日志。
- 长输出一律重定向到文件——终端刷屏会把 `ghostty` 刷到 80% CPU，把 `--full` 从 256s
  拖到 10 分钟以上（套件是**主机 CPU 编译**瓶颈，不是 GPU）。
- 提交身份 `Asada-Sinon <weibinkong.research@gmail.com>`（已设 repo-local）；
  push 用 `GIT_SSH_COMMAND="ssh -o BatchMode=yes" git push`。
