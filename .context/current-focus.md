# 当前焦点

只写此刻在做的那一件事。方向变了就整个重写，不要往下堆。超过一屏说明该拆了。

## 当前目标（2026-07-25 夜，自主授权收尾）

用户睡前自主授权的整批工作已基本完成并推送。收尾中：等 `20260725-carrion` 6 种子重跑
（第一次因 GPU 超订 OOM，改 2 波×6 重跑），填 `multispecies_feasibility.md` §7 结果，做最后
一次状态提交，然后本轮结束。

## 本轮已完成（全部已 push 到 main）

- **防御性状**：armor 演化验证 ✅（6 种子 p=0.031）、尖刺重设计（进攻侧盘活/防御侧未活）、
  可视化 wire v8/v9（厚皮/尖刺/中毒染绿 + inspector）。
- **食草过多判决**：四条降密度杠杆全部失败（`herbivore_overpopulation.md §6`），根因水限+尺度
  不变，诚实建议承认刻意压缩、不改默认。
- **两个默认关新机制**：L6 密度制约繁殖（`bd8107f`）、腐食通路 carrion/scavenge（`37dda74`）。
- **三份调研报告** + MEMORY 教训（跑实验前先提交）。

## PENDING（下个 session 第一件事）

1. **看 `20260725-carrion` 6 种子结果**：carn_frac ON>OFF？min 抬高（抗灭绝）？渴死不恶化？
   → 填 `multispecies_feasibility.md` §7、判正/负、决定是否默认开或加 scavenge trait。
2. 用户 #4（多性状）、#5（种间合作脚手架=共享报警场）**尚未动手**——研究已在
   `multispecies_ecology.md §5`/`multispecies_feasibility.md §5`，是下一批实现候选。

## 纪律

- 并行跑实验**最多 6 个并发**（12 个会 GPU OOM）。
- ecology 改动 6 种子、默认关新机制保 golden bit-exact、跑实验前先提交。
- push 用 `GIT_SSH_COMMAND="ssh -o BatchMode=yes" git push`（ssh-agent 死了）。
