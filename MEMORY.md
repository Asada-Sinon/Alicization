# MEMORY

累积教训。**当前没有任何条目——这是正常状态，不是待填的空白。**

## 和别的文件的分工

- **`HANDOFF.md` 是会过期的当前状态，`MEMORY.md` 是不会过期的教训。**
  「恐惧场那轮改到一半」属于 HANDOFF；「这套生态在单种子上调出来的信号一律不可信」
  属于 MEMORY。
- **`docs/conventions.md` 是重量级，`MEMORY.md` 是它前面的轻量级缓冲区。**
  `conventions.md` 承载「`CLAUDE.md` 里每条约定背后的完整论证」——要有实测数字、算术、
  教训的出处，一条就是一节。踩了坑、也定位到了真实原因、但还不够格写成一整节论证的
  东西，先落在这里；攒够分量（有实测、有可复述的算术、有明确该改的规则）再毕业进
  `conventions.md`，毕业时把这里的条目删掉，不要两边各留一份。

## 规矩

- 累积式，**新条目追加在最后面**。
- 一条一组，用 `### [LEARN:tag]` 起头。tag 是自起的短分类（env / jax / ecology /
  stats / tooling / git …）。
- 写入时机：**踩坑并真正定位到原因之后**。只有现象、原因还在猜的，不要写进来——写
  `HANDOFF.md` 的「坑」，那里允许「暂时还不知道为什么」。
- 发现旧条目是错的，就地改掉或删掉，不要叠加。

## 格式

```markdown
### [LEARN:tag] 一句话标题
- 现象: 当时看到了什么
- 原因: 真实原因（查证过的，不是猜的）
- 对策: 下次怎么做
- 来源: Session YYYY-MM-DD
```

**禁止为了填表而编造条目。空着比编造便宜得多——一条假教训会被后面每一个 agent 当真**
（`SessionStart` hook 会把本文件全文注入每个新会话）。没踩到坑就是没踩到坑，这个文件
长期只有两三条是完全正常的。

---

<!-- 真实的 [LEARN:tag] 条目从这一行下面开始，新的追加在最后。 -->

### [LEARN:env] GPU 驱动会话中途被更新 → CUDA 降级 → JAX 悄悄回落 CPU
- 现象: 探针跑了一个多小时不出结果，不是死锁。`nvidia-smi` 报 `Failed to initialize NVML:
  Driver/library version mismatch`；`ps` 里探针进程累计 100+ 分钟 CPU 时间（GPU 作业不会这样，
  说明在 CPU 上跑）。当天更早的同样探针在 `cuda:0` 上 2 分钟就跑完。
- 原因: 后台包更新把 on-disk NVIDIA 驱动从 580.159→580.173，但**运行内核仍加载旧模块**
  （`cat /proc/driver/nvidia/version` 显示旧版、`modinfo nvidia` 显示新版），userspace
  libnvidia-ml 已是新版 → 版本不匹配 → CUDA 不可用 → JAX 回落 CPU（~100× 慢），不报错、只是慢。
- 对策: 诊断三连——`nvidia-smi`（看 NVML mismatch）、`cat /proc/driver/nvidia/version` vs
  `modinfo nvidia`（内核 vs on-disk）、`ps aux | grep probe`（看 CPU 时间是否暴涨）。修复要 sudo：
  最省事是**重启**（新模块开机加载）；`rmmod nvidia*` 会被 GNOME 桌面（gdm/Xorg/gnome-shell）占住、
  杀不动，别硬刚。**副作用**：驱动版本会改变 XLA 确定性算法选择，本混沌世界 smoke population 随之
  翻动（半更新态跑出 1474、正常态 1549）——**别在坏驱动窗口 re-bless golden**（那次误 bless 又还原了）。
- 来源: Session 2026-07-24

### [LEARN:tooling] 跑实验前先提交，别在脏树上跑
- 现象: Experiment D（L6 密度制约）的 provenance.txt 记 git HEAD=f952f35，但当时 L6 代码在
  工作树里尚未提交（现 bd8107f）。result-analyst 核对时发现该 HEAD 无 `density_repro_penalty`
  字段，provenance 与实际运行码不符。
- 原因: 先实现→先跑实验→后提交。run_headless 读的是工作树里的脏码，但 `git rev-parse HEAD`
  记的是上一个 commit，两者不一致 → 结果无法从记录的 HEAD 单独复现。
- 对策: **跑任何要落 provenance 的实验之前先 commit**（哪怕默认关的机制也先提交）。三臂同码时
  实验内归因仍干净，但可复现性打折。/exp 第 3 步「跑之前 git status 干净」就是防这个。
- 来源: Session 2026-07-25

### [LEARN:stats] 分布形状检验的绝对 p 值在演化种群上无效——中性对照同样打到地板
- 现象: 给 `forage_pref` 基因分布做 1-vs-2 高斯成分自助似然比检验，`forage_tradeoff=0.5`
  臂 LR=212.8、p=0.0200（n_boot=49 的地板），看起来「显著双峰」。但把 `forage_tradeoff=0.0`
  （编译期断开，基因对世界毫无作用）当中性对照跑，**同样是 p=0.0200 地板**，LR=65.9。
- 原因: 所有形状检验（dip / BLRT / 正态性）都假设 iid 抽样，而**演化种群不是样本，是谱系**。
  亲缘个体共享基因值，分布天然呈家系团块；有效样本量是独立世系数，不是个体数（这里 n≈2000
  个体可能只对应几十个有效世系）。按 iid 重采样构造的自助零分布因此窄得离谱，任何真实分布
  都被判为显著偏离。这与生态无关，纯是伪重复。
- 对策: **永远不要报这类检验在单臂上的绝对 p 值。** 改报统计量本身在**配对两臂之间**的差
  （亲缘结构对两臂同样起作用，差值里它约掉），再用 6 配对种子 Wilcoxon。统计量要除以 n
  再比——`2(ll2−ll1)` 随样本量线性增长，两臂种群大小不同时直接比 LR 等于在比种群大小。
  同理适用于任何「拿全部存活个体当独立观测」的检验（性状相关、G 矩阵、HWE）。
- 来源: Session 2026-08-02

### [LEARN:tooling] Bash 工具跑的是 zsh，未加引号的变量不做词分割——多参数臂会被整串塞给 argparse
- 现象: 探针 6 个臂里 5 个秒退，日志只有 `run_headless.py: error: unrecognized arguments:
  --set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40`。
  三个 `--set` 对被当成**一个**参数报出来，而同一行末尾单独写的第四个 `--set` 反而被识别。
- 原因: 本环境 `Shell: zsh`。**zsh 默认不对未加引号的参数展开做词分割**（这是它与 bash 的
  经典差异）。把公共臂放进 `D="--set a=1 --set b=2"` 再写 `run x $D` 时，bash 会拆成 4 个词，
  zsh 只给 1 个词，argparse 收到一个既不是选项也不是已知位置参数的长字符串 → unrecognized。
- 对策: 多参数的 sweep **一律写成 `bash <script>.sh` 执行的脚本文件**，并用**数组**传公共臂
  （`D=(--set a=1 --set b=2)`，调用写 `"${D[@]}"`）。别在 Bash 工具里直接写依赖词分割的内联
  循环。附带好处：脚本文件本身就是可复跑的 provenance。
  （之前几次 sweep 没踩到，是因为它们要么把参数写全、要么已经走 `bash script.sh`。）
- 来源: Session 2026-08-03

### [LEARN:stats] `--seed` 在 20000 步上控制不住这个世界——重跑同一种子比换种子还吵
- 现象: 同种子、同配置、同代码重跑，`carnivore_frac` 跑出 0.060–0.2505（4.2 倍跨度），
  `population` 跑出 1499–2032。18 个默认配置 run 做方差分解（3 重复 × 6 种子）：
  `late_carn` / `death_thirst_frac` / `total_flux` 的创始者方差分量估计为 **0**，
  `carnivore_frac` 的 ICC 只有 0.130、`population` 0.259。
- 原因: per-cell scatter-add 是原子操作会重排（外加 XLA 自动调优的算子选择），
  单步差异微乎其微，但 20000 步的混沌把它放大成轨迹分叉。`conventions.md §9` 原有的
  「连跑五次漂移 0.000%」测的是 **200 步**，不覆盖判决用的 20000 步。
- 对策: ①**配对差的噪声是 `√2 × 同种子 SD`**，不是跨种子 SD——`carnivore_frac` 是 0.0625 而非
  ±0.012，`conventions.md §5` 的功效算术低估 5.2 倍。②护栏容差要对着这个数定，别凭直觉写
  ±10% / +5pp（那大约就是 1 个噪声 SD，零效应也会经常撞线）。③想压噪声就**每格重复 k 次取
  均值**（降 `√k`）；**确定性 XLA 算子实测慢一个数量级以上**（20000 步 57 分钟没跑完 vs 4.5
  分钟），本项目预算下不可用。④性状/基因类指标 ICC 高（`forage_pref_std` 0.547、`carn_attack`
  0.389），比种群比例类结论站得稳——别一竿子打翻所有旧结论，按「效应量/噪声」比值逐条重估。
- 来源: Session 2026-08-03，详见 `docs/run_to_run_variance.md`

### [LEARN:env] 测试套件是**主机 CPU 编译瓶颈**，不是 GPU——终端刷屏能把 `--full` 拖慢 2.7 倍
- 现象: `check.py --full` 从整夜稳定的 233–250s 突然变成 **635.9s**，且冷却后重跑仍是
  **635.8s**（逐秒复现，不是瞬态）。测试集在这两次之间**一个字都没改**。
- 我最初的两个猜测都错了: ①「是我新加的 5 个测试拖慢的」——`pytest --durations` 显示
  前 7 名全是**既有**测试（各约 30s），新增的排第 8（21s）；②「笔记本 GPU 满载一夜后热降频」
  ——冷却到 51°C、`nvidia-smi -q -d PERFORMANCE` 显示所有 throttle reason 均为 Not Active，
  重跑仍是 635.8s。
- 原因: **终端模拟器 `ghostty` 占着 95.2% CPU、load average 5.92。** 整夜把几万行 sweep 日志
  刷进终端，把它刷成了一个跑满一核的进程。而 **XLA 是在主机 CPU 上编译的**——套件里最慢的
  那批测试（`test_determinism`、各种 `*_default_is_truly_off`）耗时几乎全是 jit 编译，
  不是 GPU 算力。主机 CPU 被抢，编译就慢，整套跟着翻倍。
- 对策: 套件变慢时**先看 `uptime` 和 `ps --sort=-pcpu`，再怀疑代码**。`nvidia-smi` 一切正常
  ≠ 没有性能问题，因为这个套件根本不是 GPU-bound。长 sweep 的输出一律重定向到文件
  （`> outputs/.../x.log 2>&1`，本项目已经这么做），**不要让它流经终端**；诊断脚本的输出也
  尽量 `tail`/`grep` 而不是整段打印。
- 来源: Session 2026-08-03
