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

## Session 2026-08-02→03（夜间自主，用户睡前授权：局部多物种生态箱 + 可视化）

- 授权: 用户定盘长期目标「**局部的多物种生态箱演化 + 可视化结果**」，要求**先把工作流设计成
  一个循环**再一直跑下去、别干一半停，并授权分 agent 做测评。也交代了 push 走 `asada-sinon`
  （已设 repo-local 身份 `Asada-Sinon <weibinkong.research@gmail.com>`，与全部历史提交一致）。
- **工作流已定死**: `docs/multispecies_program.md`——纲领 §1 / **七道闸 §2** / 回合台账 §3 /
  backlog §4。下个 session **从 §4 backlog 顶部取题**，不要另起炉灶。
- 完成（生态线，5 个回合全部落档 + push）:
  - **纲领**: 多物种三次失败是同一根因的三个投影——**这个世界只有一个饭碗（河岸的草）**。
    顺序定为「先造生态位，再谈物种」。
  - **R1 判据自身被证伪**（§5）: 现存量量的是「剩多少」不是「供多少」，把生态位低估约一个
    数量级。产出 = `Metrics` 三个流量观测量 `graze_gain`/`fruit_gain`/`frugivory_frac`。
  - **R1b 不通过**（§7）: 份额做到 0.22（6/6、p=0.031）但渴死护栏破 2–3/6。顺带更正自己一句
    过度解读（林冠移动对着零模型读几乎可忽略）。
  - **R1c 不通过且我的诊断被推翻**（§8）: 真因是**承载力**不是水口径——种群超调与渴死增量
    ρ=+0.90，`mean_water` 三臂对基线全无显著变化。
  - **§10 跨回合命题**: **水限世界里加食物只会变成更多身体。** 「加厚食物层造生态位」整条路
    证伪，负结果已进 `docs/experiments.md §5`。
  - **R3 判为「无法判定」**（§11）: 字面上不通过，但**判据容差 ≈ 1 个复现噪声 SD**，通过/不通过
    都没有信息量。**资源轴挂起，不宣布关闭也不宣布成功。**
- 完成（**方法学，本轮真正的头号产出**）: `docs/run_to_run_variance.md`
  - **`--seed` 在 20000 步上控制不住这个世界。** 18 个默认配置 run（3 重复 × 6 种子）方差分解：
    `late_carn`/`death_thirst_frac`/`total_flux` 的**创始者方差分量为 0**，`carnivore_frac`
    ICC 仅 0.130、`population` 0.259。同一个种子能跑出 `carnivore_frac` 0.060–0.2505。
    seed 3 复现（n=6）同量级或更大——**普适，不是某颗种子的特例**。
  - 后果：**配对差噪声 = `√2 × 同种子 SD`**，`carn_frac` 是 0.0625，而 `conventions.md §5` 用的
    跨种子 SD ±0.012 **低估 5.2 倍、样本量低估约 27 倍**。ICC≈0 的指标**配对完全无用**。
    `conventions.md` 已补 §9.1（这是事实更正，不是规则变更）。
  - §6 给每条旧结论标了「效应量/噪声」比值：**红皇后 2.4、armor 1.19 站得住；恐惧场 0.32、
    pred_nocturnal 0.53 降级待重测**。**已发表的 p 值本身没错**（算的是实测配对差）。
- 完成（可视化交付，用户点名要的）: Artifact
  **https://claude.ai/code/artifact/285960f6-abfd-465f-8d82-e50cdc61bf6d**（默认私有）——
  五个回合的判决、基线 vs 再分配的**两张世界对照图**、三张图表（剂量响应 / 方差分解 / 旧结论
  重估）。另外**仪表盘现在能接 `--set`** 了（`UNDERWORLD_SET` 环境变量），
  例：`.venv/bin/python scripts/run_live.py --no-open --set fruit_energy=4.0`。
  - **R2 完成，P2 不成立且失败方式不是预设的那种**（§9.5–§9.9）: 在果层真供 28% 能量的世界里
    把权衡强度开到最大，`forage_pref` 的方差**比中性漂变对照低 38.3%**（6/6、p=0.031、
    效应/噪声 −1.42），两臂直方图都是单峰同峰位。**基因流只能把处理臂压回等于中性、不能压到
    低于中性——所以这是稳定化选择主动收窄，不是基因流被动压平。** 同时证否了
    `feasibility.md §9.5` 的归因（不是「没饭碗」，是「中间型最优」）。
    **[提案，非结论] 为什么中间最优**：果层长在 `forest²` 门控的中海拔近水带，
    正是所有个体本来待的地方，两种资源空间交织 → 「两样都能吃」严格优于专精。
    **教训改写为：资源分割需要的不是「第二个饭碗」，而是「两个饭碗离得够远」。**
- **PENDING（下个 session 第一件事）**: **把下面三件拿给用户拍板，不要自己挑一个开跑。**
  五个回合全部走完、全部落档，`docs/multispecies_program.md §4` 的 backlog 已按判决重排完毕。
- **要用户拍板（都会改默认世界行为或改变后续所有工作的成本）**:
  1. **先定统计协议**（`run_to_run_variance.md §7`）：每格 k=3 重复。代价 run 数 ×3
     （一轮 30 分钟 → 90 分钟）。**这是先决条件**——不定它，后面任何判决继续是抛硬币。
  2. **把两个饭碗在空间上分开**（§9.9 直接推出的唯一活路）：动 `terrain.py:207` 的
     `fruit_capacity = fruit_max × patch × forest² × (1−rock)` 门控，把果层挪到高海拔或远水带。
     代价：改地形派生量 → golden 重 bless；且会与水限约束正面相撞。
  3. **或走时间轴**（同一份资源、不同取用时段，零新增承载力）。`day_length` 已默认开、
     `scripts/probe_diel.py` 可复用。比空间轴便宜，但 `day_night.md` 已判「空间通勤不可达」，
     时间生态位是否可达未知。
- **⚠️ 已划掉、不要再提的三条**（前提都被数据推翻，别让下一个 agent 重捡）:
  - ~~加厚食物层造生态位~~（证伪，`experiments.md §5`）
  - ~~等量再分配~~（被 R2 从上游解决：供给量根本不是瓶颈）
  - ~~生殖隔离 `assortative_mating` on `forage_pref`~~ —— **它对抗基因流，而实测是稳定化选择。
    这条是我在 §9.4 预设的失败子句，已被 §9.7 推翻。**
- 坑:
  - **Bash 工具跑的是 zsh，未加引号的变量不做词分割**。多参数 sweep 一律写成
    `bash <script>.sh` 并用数组传公共臂（`D=(--set a=1 --set b=2)` + `"${D[@]}"`），
    否则整串被塞给 argparse、秒退。已进 `MEMORY.md [LEARN:tooling]`。
  - **确定性 XLA 算子（`--xla_gpu_deterministic_ops=true`）在 20000 步上 >20× 慢**，
    100 分钟跑不完，别再试。「它能否消除散度」**仍未测**，别写成已知。
  - **`fruit_water_frac` 是本轮新增的默认关旋钮**（默认 0.10 = `forage_water_frac`，逐位等价，
    golden 未动）。`fruit_energy` 抬高时果实每卡路里会变干，要用它补。
  - `outputs/20260803-*` 与 `20260802-*` 下有本轮全部产物（gitignored）；分析脚本在
    `explorations/20260802-fruit-flux/`、`20260803-fruit-water/`、`20260803-repartition/`。

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
