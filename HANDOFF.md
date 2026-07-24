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

## Session 2026-07-23

- 完成: 三次 commit，一条线（种群密度校验 → 密度杠杆证伪 → 恐惧地景落地），全部已 push
  （`main...origin/main` 无 ahead/behind）。
  - `8c57c33` 定标世界比例尺 + 校验种群密度，落 `docs/scale_and_density.md`（纯文档）：
    1 世界单位≈1 m、全图≈0.26 km²；食草密度≈现实天花板 66×、食肉≈现实极值 3800×；
    补上 `carnivore_riparian.md:286` 留的 [推断,未测] 空档。
  - `bf9e836` 两轮单种子探针证伪「扩世界降密度」，写进 §5.3（纯文档）：三个世界尺寸、多种
    河数下食草密度一律钉在 ~7600–8400 头/km²——**密度是尺度不变量**，选项 B 出局；砍
    `plant_max` 也不动密度却饿死捕食者，选项 C 走这条路出局。只剩 A（承认压缩）或 C′（深
    度重调水经济，高风险），D（修比例）独立低风险。
  - `eca17e5` 落地恐惧地景设计二并**默认开启**：`WorldState` 新增 `fear:[n_cells]`（仿
    `trample` 散射沉积 + 衰减），`sensors.sense` 把前方采样的 fear 与瞬时 `pred_val` 取 max
    折进 `pred` 通道；`in_dim`/`genome_size` 不变，不作废种群。6 配对种子（seed0–5，15k 步）：
    `carn_speed` 1.5→2.4（6/6，p=0.031，最硬信号）、`carn_frac` 12.1%→10.2%（5/6）、
    `death_thirst_age` +4.9（未恶化幼体渴死）；但 `carn_water_dist` 只 +1.4（弱，噪声大）——
    恐惧场能让捕食者动起来、少一点，给不了「搬离河岸」（缺昼夜通勤）。full check + pytest 过。
- PENDING: **先修 `docs/TODO.md` 的「队列」一节——它已经过期，而它是下一个会话的权威入口。**
  两处具体的过期（对照 `git log` 可核）：
  1. §「当前主线」仍写「下一步是托管任务的第三步：开子 agent 找优化点、清理 AI 写的冗余
     代码」，但这一步已在 `27c01cc`/`33d5baf`/`6f5ec55` 做完，三份审计文档（`test_coverage_audit`
     /`cleanup_audit`/`optimization_audit`）已经在同文件的文档地图里。
  2. §「主线 A —— 捕食者离水」仍把恐惧地景写成未做的「正路 / 头号方案」，但 `eca17e5` 已
     落地且默认开启，实测结论已在 `landscape_of_fear.md` §6。
  改完再定下一个方向——**方向本身待用户定盘**，候选与依据见 `.context/current-focus.md`，
  不要自己挑一个就开跑（这两条候选都会动世界行为或 golden 带）。
- 坑:
  - **`fear_rate` 是编译期分支**：设 0 时沉积与折叠从 jit trace 消失、逐位复现旧基线。做
    消融/对照一律 `scripts/run_headless.py --set fear_rate=0`，不要改 `config.py`。
  - **golden 已按恐惧场默认开启重新 bless**（200 步 smoke 里 population 1520→1549 等微动）。
    拿旧 `scripts/golden.json` 或旧分支的数字对比会误报「配置被改坏了」。
  - 仓库根目录躺着未跟踪的 `oa1.json`（199 KB，OpenAlex `/works` 查询的原始 JSON 响应，
    query = "Desiccation resistance water balance Circellium bacchus"，7月21 文献检索的残留）。
    grep 全仓无任何代码/文档引用它。建议直接删或挪进 scratchpad。**没有加进 `.gitignore`**：
    CLAUDE.md 的规矩是 scratch 根本不该进仓库，为一个一次性文件名加 ignore 只会把它永久藏起来。
