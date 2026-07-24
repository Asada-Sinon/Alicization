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
