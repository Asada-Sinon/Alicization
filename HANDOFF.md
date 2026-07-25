# HANDOFF

**这不是文档，是上一个 agent 写给下一个 agent 的信。要短、要具体、只写下次用得上的。**
不写背景（背景在 `CLAUDE.md`），不写论证（论证在 `docs/`）。

**与 `docs/TODO.md` 的分工——两者不重复：**

- `docs/TODO.md` = **任务队列 + 文档地图**。长期、面向「下一步做什么」和「去哪找论证」。
  结论一旦稳定就写进它，它是权威来源。
- `HANDOFF.md` = **会话交接**。短期、面向「上次做到哪、什么半途而废、此刻有什么坑」。
  一条信息一旦稳定到值得进 `docs/TODO.md`，就搬过去，别留在这里。

规矩：

- 新会话结束时加一节，**最新的在最上面**。
- 只保留最近 3 节，更旧的直接删掉（历史在 git 里，不用囤在这）。
- `PENDING` 是下一个 agent 开工的第一件事，必须是可执行的动作，不是「继续优化」。
- 教训写 `MEMORY.md`，不写这里：HANDOFF 会过期，教训不会。

格式：

```markdown
## Session YYYY-MM-DD
- 完成: ...
- PENDING: ...        ← 下次第一件事
- 坑: ...
```

---

## Session 2026-07-25（白天，用户在场：三通路落地 + 6 种子判决）

- 承接夜间 PENDING（用户 #4 多性状 / #5 种间合作），用户拍板**三个方向全做**、分 worktree agent 并行实现。
- 完成（三方向已并入 main 三独立提交、已 push）:
  - **实现**：`cbce9fc` forage_pref（草↔果权衡，trait_dim 7→8）、`1964949` 腐食接 food retina（不动 in_dim）、
    `c25ba60` 共享报警场 alarm（独立场、折进 pred）。三者均默认关（forage 付 golden 重 bless、A/C bit-exact）。
    整合冲突手工解（doc 章节 §7→§8→§9 重排、metrics/tests keep-both）。check.py/contracts/pytest 全绿。
  - **6 种子判决**（36 runs、6 并发、20000 步，各派 result-analyst）——**三者全维持默认关**：
    - **腐食 v2 = 弱阳性**：carn_frac ON>OFF 4/6、p=0.156（未达 6/6 硬标准）；**min 首次抬高 0.098→0.127
      （抗灭绝成立，胜首版 min 反降）**；carrion_total 198≈首版 204（主动觅食**没**加速消费）；pop −80 弱负。
      → 不够默认开。`feasibility.md §7 第二版结果`。
    - **alarm = NULL**：death_predation_frac 4/6、p=0.56、CI 跨 0；机制点火（alarm_total>0）但**信息冗余**
      （§8.4 第 1 类）。`feasibility.md §8.6`。
    - **forage = 证否**：forage_pref_std **OFF>ON**（反向 5/6，方差塌缩 32.6%），种群整体倒向草；**根因果层
      仅草层 0.07%、无生态位可专精**。`feasibility.md §9.5`。dip-test 不必做（必要条件已否）。
- PENDING（下个 session 第一件事）: **方向待用户定盘**。三通路都收口、默认关。候选（研究已备、判决已给出下一步）：
  ①救腐食 v2——补 seed 6–17 到 ≥12 配对把 p 推过线（差值与 SD 同量级，§5 功效算术需 ~20+ 种子），或加消费
  流量计（累计 `scavenge_gain`）诊断"主动觅食是否真生效"；②救 forage——先**加厚果层**（动 `fruit` 生态位）+
  交叉 ridge 地形，否则权衡永远单向塌缩；③救 alarm——加「高 alarm cell 猎物密度」读出 + `los_occlusion` 造
  超视距信息。**不要自己挑一个开跑**，都要 6 种子。
- 坑:
  - **worktree agent 无独立 `.venv`**：用主 checkout 的 `.venv/bin/python`、以 worktree 为 cwd 跑（`sys.path` 的
    `.` 指向 worktree 码），全程带 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
  - **6 并发 sweep ~27min**（36 run×~4.5min/6 并发）——不是卡住。诊断 CPU 回落坑三连：`nvidia-smi` 看 util/NVML、
    `ps` 看 CPU 时间是否暴涨、log 看 steps/s。本轮 GPU util 100%、驱动 580.173 一致，正常。
  - forage 是**唯一付永久 trait_dim 代价却被证否**的改动，选择留档（回退 8→7 会再 churn golden+作废种群）。

## Session 2026-07-25（夜间自主，用户睡前授权）

- 授权: 用户睡前给自主授权（见 `memory/overnight-autonomous-mandate.md`），一切调研+实验落报告、
  选最优拍板提交推送。做了两大块：**防御性状**（先）+ **食草过多/多物种**（后）。全部已 push。
- 完成（防御性状线）:
  - **armor 演化验证 ✅**（6 种子）：herb_armor ON 0.170 vs 漂变 0.067，6/6，p=0.031——本项目第一个
    真正演化出来的可见形态防御。可视化：厚皮深描边/尖刺放射纹/中毒染绿 + inspector 行（wire v8/v9）。
  - **尖刺重设计**（进攻/防御双用 + venom 场，`04d18e7`）：6 种子判决——**进攻侧盘活**（carn_spike
    ON 0.202 vs OFF 0.052，6/6，p=0.031，相对旧设计 10.6×）；**防御侧仍未活**（herb_spike 贴漂变，
    venom 毒发时猎物已死、收益回退亲缘）。
- 完成（食草过多/多物种线）:
  - **食草"满山遍野"判决**（`45fd1d2`）：**四条降密度杠杆全部失败**（repro_threshold/L6 密度制约=
    水限补偿；紧水=灭捕食者+渴死回潮；carn_cost↓=食草水限压不动）。根因水限+密度尺度不变。诚实建议
    **承认刻意压缩微缩世界、不改默认**。`docs/herbivore_overpopulation.md`。
  - **L6 密度制约繁殖**（`bd8107f`，默认关）+ **腐食通路 carrion/scavenge**（`37dda74`，默认关）落地。
    **腐食 6 种子验证（run_id 20260725-carrion）：机制通过但抗灭绝假设未证实**——carrion_total 0→204
    （6/6），carn_frac 均值 0.127→0.168 但仅 4/6、p=0.31、min 不升；护栏守住（渴死略降）。诊断：食肉者
    踩到才吃、不主动找尸体（首版不接 retina），补贴太稀薄。**默认保持关闭**（`multispecies_feasibility.md`
    §7）。
  - **三份调研报告**（`3fbf660`）：`herbivore_overpopulation.md`、`multispecies_ecology.md`（13 篇真实
    引用）、`multispecies_feasibility.md`。
- PENDING（下个 session 第一件事）: 用户列的 **#4 多性状** 与 **#5 种间合作** 尚未动手（只调研了）。
  候选（研究已备）：①放大腐食效应——把 carrion 折进 food retina 让食肉者**主动找尸体**（动 in_dim、
  作废全脑）或加 `scavenge` trait 做真·腐食者物种；②#5 种间合作脚手架=**共享报警场**（让猎物也往 fear
  沉积、附近异种可读，Quinn 模式，见 `multispecies_ecology.md §5`）；③#3 资源分割第二食草者（草↔果
  权衡基因，最省）。**都要 6 种子、默认关保 golden。**
- 坑:
  - **教训**：跑实验前先提交（Experiment D 在脏树上跑，provenance HEAD 与实际码不符）。已进 MEMORY。
  - **ssh-agent 死了**：`git push` 默认失败，用 `GIT_SSH_COMMAND="ssh -o BatchMode=yes" git push` 绕过
    （config 的 IdentityFile 直接可用）。
  - 新增默认关旋钮：`density_repro_penalty=0`（L6）、`carrion_enabled=False`（腐食）——都 bit-exact 旧
    世界、golden 未动。armor/spike/venom 是 trait_dim 5→7 + venom 场，golden 已按那些重 bless。
  - `.venv` 外的 `python3` 也被 PREALLOCATE hook 拦——纯 JSON 解析脚本也要加 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 前缀。

## Session 2026-07-24

- 完成: 昼夜系统整条线落地，8 次 commit 全部 push（`main...origin/main` 干净）。
  - 先同步 `docs/TODO.md` 队列（`9fa7893`）。
  - **昼夜 Phase 1**（`d9ae171`）：全局标量 `phase` 时钟 + 暗→视野 + move-only 热，默认关、
    逐位可逆。加相位分箱探针 `scripts/probe_diel.py`。
  - **五杠杆并行探索**（workflow，`c1d1477`）：move_heat/forage/activity_energy/pred_nocturnal/
    torpor。判决：**"捕食者搬离河岸"结构不可达**（水太硬）；碰水杠杆靠剔除破渴死；水中性杠杆
    安全但空间惰性；**pred_nocturnal 唯一干净阳性**（捕食风险昼夜错峰、还降渴死）。
  - **pred_nocturnal 落地 + 验证 + 默认开启**（`9332323`/`271980d`/`8096bda`）：夜间捕食者射程↑。
    amp=1.0 6 种子：hunt_success 夜−昼 +0.245（6/6）、thirst −13pp（6/6）、carn_frac +3.3pp（6/6）。
    **已默认开启**：`day_length=400` + `pred_night_amp=1.0`，热/暗/觅食默认关（只跑这一条已验证
    杠杆）。golden 重 bless。`--set day_length=0` 逐位回退。
  - **Phase 2（演化空间通勤）证伪**（`45f482f`/`32c6a1a`）：`forage_heat` 底座落地（默认关），
    `--set hidden=24/32` 加大脑。40k/24 与 100k/32/组合底座/2 种子一致：**没长出空间通勤、加大脑
    不帮忙**。根因大概率缺动态选择压（静态地形）。同 `mutation_sigma` 类负结果。
- PENDING: **下一个方向待用户定盘**——昼夜整条线已收口。候选：①密度 D（压 carn_frac，但注意
  pred_nocturnal 刚把它抬了 +3.3pp、基线变了）；②别的。不要自己挑一个开跑。若继续碰昼夜：唯一
  剩的实验是"演化通勤需要动态环境"（超出昼夜范围、大工程），按现有证据不建议。
- 坑:
  - **`day_length` 现在默认 400（昼夜默认开）**：所有实验默认带昼夜捕食。要昼夜前的旧基线做对照，
    一律 `--set day_length=0`（编译期分支、逐位复现）。别拿旧分支数字直接比。
  - **golden 已按昼夜默认开重 bless**（population 1549→1494、carn_frac 0.005→0.017 等）。
  - **GPU 驱动会话中途被更新过一次**（580.159→580.173），当时半更新态让 CUDA 降级、JAX 悄悄回落
    CPU、探针跑了一个多小时不出结果；`nvidia-smi` 报 NVML 版本不匹配是信号。重启修复。教训已进
    `MEMORY.md`。当时慌乱中把 golden 误 bless 成坏驱动下的 1474（`c2dbefe`），重启后证实正确值是
    1549、已还原（`8e29ada`）。
  - 四个未选中的昼夜杠杆（forage 已入库作 Phase 2 底座；activity_energy/torpor/heat 只在 workflow
    临时 worktree 跑过、**未入库**）。`.claude/worktrees/` 下有一堆死 agent 的孤儿 worktree，可清。
  - `scripts/probe_diel.py` 已入库（相位分箱探针，通用）；Phase 2 探针 `phase2_probe.py` 在
    scratchpad、未入库（一次性）。

