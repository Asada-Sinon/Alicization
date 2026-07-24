# 可见形态『防御』性状目录：候选清单、真实生物学依据与本世界落地锚点

这份文档回答用户诉求的前半段——**「沙盒里能加哪些看得见的有利性状？」**——并把范围
收敛到一个现有 docs **完全没有覆盖**的新维度：**可见的形态『防御』性状**（尖刺、厚皮/
护甲、警戒色、隐蔽、集群）。它是一份候选菜单 + 每条候选的真实生物学依据 + 在本世界代码里
的落地锚点；**「值不值得做、先做哪个、怎么验证」的判决在姊妹篇 `docs/trait_addition_feasibility.md`**，
本文不重复那半边论证。

**为什么是「防御」这个维度**：现有性状线（`docs/trait_evolution.md`/`docs/trait_roadmap.md`）
已密集覆盖觅食、耐渴、代谢、体型，红皇后 `attack_range`↔`escape` 对
（`docs/attack_range_redqueen.md`）覆盖了「攻击距离」与「逃逸=减少被咬命中」。但**没有一个
现存性状改变外观**，也没有「尖刺反伤捕食者」「厚皮减伤」「警戒色/毒」这类**作用在捕食处理
阶段、且天然可视化**的形态防御。这正是用户「长尖刺/加厚皮肤」落点最自然的空档，且因为作用
在成体捕食维度、最「看得见」。

标记体例沿用 `docs/biology.md`：**[现实]** 已发表事实、**[本世界实测]** 本代码库跑出的数字、
**[对应]** 落到哪段代码、**[提案，非结论]** 尚未验证的设计建议。真实引用附 DOI，逐篇细读
笔记在 `notes/papers/<citekey>.md`，台账在 `lit/literature-log.md`。

---

## 0. 共用落地模板（所有候选照抄红皇后 escape 基因）

[对应] 一个纯 trait 基因的最小落地面已经被 `attack_range`/`escape` 走通一遍
（`docs/attack_range_redqueen.md`、`docs/trait_roadmap.md` §8），任何新防御性状照抄这条链：

- **基因位**：`config.py:54` `trait_dim` +1；新增派生 `@property *_index → brain_params + N`
  （追加在脑段之后，`config.py:758-779` 的设计原语）。**只动 `trait_dim` 不动 `in_dim`** →
  便宜，但 `genome_size` 变宽会**作废整群一次**（founder RNG 重排）。
- **解码器**：`state.py` 追加 `*_of(genome, cfg)`（仿 `escape_of` `state.py:113-124` 的
  `escape_span * clip(sigmoid(gene)-0.5, 0, None)` 单边映射），**gene=0 → 中性无防御**——
  任何防御都是演化出来的，红皇后干净基线。
- **变异 sigma**：`genome.py` `mutate` 给该基因位设一个慢档 `*_mutation_sigma`（0.02）；
  是否豁免 crossover 取决于它是否进入感觉-运动回路（防御性状挂 predation、不进脑，**不豁免**，
  与 escape 同）。
- **收益钩子**：见每条（防御性状挂 `dynamics.predation`）。
- **代价账**：`dynamics.metabolize` 的 tax 块（`dynamics.py:258-262`）加一项
  `cfg.*_cost * <gene> * <diet门控>`。**能量账，绝不水账**——记水账会栽进幼体渴死瓶颈被
  反向选择（体型基因的死法，`docs/trait_roadmap.md` §5）。
- **可见性**：每 agent 现在只传 5 个 f32（`x,y,diet,energy,id`，`server/protocol.py`、
  `web/main.js` `STRIDE=5`），size/attack/escape **都没上屏**。要让性状可见需在 wire
  protocol 每 agent 记录**段尾追加** f32 → `main.js` STRIDE → `render.js` shader。诚实指出：
  当前渲染全是 `gl.POINTS`（point-sprite），**真正的尖刺几何超出它的能力**，只能在片元
  着色器画放射状纹样，或另建 instanced-mesh 管线（不存在，昂贵）。

---

## 1. 棘刺 / 骨板护甲（armor / spines）

**[现实]** 三刺鱼（*Gasterosteus aculeatus*）的侧骨板由 *Ectodysplasin* (*Eda*) 位点单个
增强子变异控制：海洋型携带 complete 等位基因（全副骨板），反复独立入侵淡水后又反复固定
low 等位基因（骨板退化）。Colosimo 等把这套平行演化定位到同一个古老 low 单倍型，说明护甲
的得失是可被选择反复调用的**单基因开关**。关键在于护甲**不是免费的**：Barrett、Rogers &
Schluter 把两种等位基因的海洋鱼移入淡水池塘、追踪一个世代内基因型频率，发现 low 等位基因
在骨板发育之后频率上升，**最可能机制是生长优势**——造骨板消耗钙/磷、拖慢生长，淡水中失去
大型鱼捕食压力后这份代价立刻显现。因果链可直接搬：护甲降低被捕食死亡，但抽走生长/发育预算。
（Colosimo et al. 2005, Science, DOI:10.1126/science.1107239；Barrett, Rogers & Schluter
2008, Science, DOI:10.1126/science.1159978）

**[对应] 减伤钩子**：`dynamics.py:205-208`（`predation`）。在 per-prey 实际掉血 `removed`
上乘 `(1 - hide)`：`hide = concat([hide_of(genome), zeros(1)])`（dump 位为 0），
`removed = removed * (1 - hide[prey])`。副作用需写明：`scale = removed/wanted` 随之下降 →
攻击者 `meat_gain` 也减少（物理上一致——咬出的肉也少）；若只想减伤不减攻击者收益，须把护甲
乘子只作用在猎物侧 `damage`，另留未削的 removed 给 payout。代价挂 `metabolize` tax
`hide_cost * hide * (1-diet)`（能账，护甲=重量→代谢/负重代价，自洽）。

**[提案，非结论] A 还是 B 类？——这正是代价货币原则的活教材**：真实生物学里护甲的代价是
**生长/发育预算**（钙/磷、拖慢生长），落在幼体期，按 `docs/trait_roadmap.md` §5 的分诊会
滑入 **B 类**、有重演体型基因方向反转的风险。但若照红皇后先例把代价记成 **metabolize 能量
税**（成体持续付、不进水账），它就留在 **A 类**（收益在成体捕食维度兑现，均龄 170 步 >
幼体渴死均龄 52.5 步）。**分类不是「护甲」这个概念固有的，取决于实现时把代价记进哪本账**——
详细论证与判决见 `docs/trait_addition_feasibility.md` §B.2。gene=0 → 0 减伤、bit 复现旧行为。
runaway 风险**中**：护甲是单向收益、无频率依赖负反馈，只有 `hide_cost` 与捕食者 attack 的
军备竞赛压得住，须映射到有限区间封顶 + 短探针调 cost 量级（仿 `attack_cost=0.012`）。

**可见性**：shader 便宜——`POINT_FS` 按 hide 值加深色/加厚描边或压暗，或 `POINT_VS` 微增
点大小（「更厚」）。这是本目录里**避开尖刺几何天花板的一条**（描边即可读作带甲）。

---

## 2. 反击型防御 / 尖刺（spike，捕食得手但捕食者受伤）

**[现实]** 防御不止于「别被抓到」——猎物的结构可以让**吞咽本身**对捕食者有害。三刺鱼的
经典实验（Hoogland, Morris & Tinbergen）把有棘/去棘的刺鱼喂给梭鱼（*Esox*）和鲈鱼
（*Perca*）：竖起的背棘与腹棘卡住捕食者口腔/咽喉，捕食者常把刺鱼吐出、并在有替代猎物时
主动回避；去棘后保护消失，证明保护主要来自棘刺本身而非行为。这是「猎物对捕食者施加成本」
最早的定量证据之一：防御收益不只体现在自己存活，还体现在**降低捕食者净收益**、从而压低
可维持的捕食者密度。（Hoogland, Morris & Tinbergen 1956, Behaviour, DOI:10.1163/156853956X00156
`[DOI 后缀未独立确认]`）

**[对应] 反伤钩子**：`dynamics.py:199-212`（`predation`）。攻击者 `i` 选定 `target[i]` 后，
从攻击者 energy 扣被咬猎物的尖刺值：`spike = concat([spike_of(genome), zeros(1)])`，
`reflect = where(dmg>0, spike[target], 0)`，在 `energy = energy - damage + meat_gain` 后
再 `- reflect`（能账）。代价挂 `metabolize` tax `spike_cost * spike * (1-diet)`。

**[提案，非结论]** A 类（成体捕食维度）。gene=0 → spike=0。runaway 风险**低**——反伤是
**频率依赖负反馈**：尖刺普及→捕食者被反伤致死→捕食压下降→尖刺选择压下降，自限。**但一个
关键演化学弱点**：尖刺「反伤」保护的不是携带者本人（它当场仍被咬），收益偏**亲缘/群体级**，
需靠 `spawn_radius` 的 kin 聚集才好演化，更易得到「漂到零」的 null 且难解释（是没收益还是
收益在群体级？）。这使它作为**首个**落地性状不如护甲干净（`docs/trait_addition_feasibility.md`
的判决因此选护甲，而非 literal 尖刺）。

**可见性**：shader 便宜——`POINT_FS` 里按 spike 值在 `gl_PointCoord` 上画放射状纹样
（`sin(atan2(d.y,d.x)*N)` 阈值出尖齿轮廓）。诚实指出：这是「纹样示意」，非真尖刺几何。

---

## 3. 诱导型防御（inducible defense，遇险才表达）

**[现实]** 护甲、棘刺这类结构在很多类群里是**诱导型**的——遇到捕食者化学线索才表达。
Harvell 的综述把诱导划算的条件收敛成一组判据：防御必须**有代价**（否则不如常备）、捕食
风险必须**时空波动**、必须存在**可靠预警线索**、且防御可逆。经典实证是水蚤 *Daphnia pulex*：
捕食性 *Chaoborus* 幼虫释放的 kairomone 诱导幼体长出 neckteeth（颈齿），形成量随浓度
**连续标定**（风险越高、防御越重，把代价压到最低），代价以繁殖延迟、首次繁殖体长变小、
内禀增长率下降的形式反复测到。（Harvell 1990, QRB, DOI:10.1086/416841；Tollrian 1993,
J. Plankton Res., DOI:10.1093/plankt/15.11.1309；Tollrian & Harvell 1999, Princeton UP,
ISBN 9780691004945）

**[提案，非结论] 诱导型是 A 类，且与本项目昼夜系统天然契合**：诱导型不预付生长代价、只在
遇险时短暂表达，属即时兑现的 A 类，绕开幼体删失。更妙的是 Harvell 第二条判据「风险时空
波动」——本项目 `pred_nocturnal` 昼夜错峰（`docs/day_night.md`）恰好制造了周期性波动的
捕食风险，理论上正是诱导型比常备型更划算的场景。可作为「先上诱导型、缓上组成型」的排序
依据。**[对应] 尚未实现**：诱导需要「捕食风险感知 → 防御表达」的条件反应函数，比常备型
（一个静态基因值）多一层机制；本项目已有 fear/pred 感知通道（`sensors.sense`），原则上
可承载，但落地成本高于常备护甲，宜排在护甲之后。

---

## 4. 警戒色 + 化学防御（aposematism + toxin）

**[现实]** aposematism 指用醒目、易学、易记的信号（鲜色、条纹）广告自己带毒或不可食，让
捕食者一次学习后长期回避。Mappes、Marples & Endler 的综述梳理核心难题：警戒信号靠捕食者的
**学习与强化**起效，因此初始稀有的醒目型如何越过「更显眼→更易被试探」的门槛是关键，答案
涉及 dilution、新奇回避、以及信号与毒素的协同演化。可见性是这条性状的**定义特征**——它靠
被看见起效，与 crypsis 正相反；毒素本身有代谢/获取代价，构成权衡。（Mappes, Marples &
Endler 2005, TREE, DOI:10.1016/j.tree.2005.07.011）

**[对应] 毒钩子 + 告警色**：毒伤与 spike 同位（`dynamics.py:205-212`）但触发于**成功取食**
（毒伤 ∝ 实际吃下量）。**[提案，非结论]** 只做**选择版**（吃毒谱系被毒死→后代减少）便宜、
不改 `in_dim`；真正的「学习劝退」需给捕食者一条「这只有毒」的感知输入 → **改 `in_dim` =
贵 = 作废种群且加脑输入**，建议不做感知版并诚实标注。A 类（捕食维度），但毒对个体是
**利他/亲缘**收益（自己仍死），演化上比护甲更弱、更依赖谱系聚集。runaway **低-中**（毒有
频率依赖）。

**可见性**：**shader 最便宜、告警色是全目录最自然的可视化旗舰**——`POINT_FS` 已按 diet 在
紫↔红间插值，再按 poison 值向高饱和黄/橙偏移（`mix(c, vec3(1.0,0.85,0.0), poison)`）即可，
现实警戒色就是黄黑/红黑，视觉说服力强。（搭便车风险：拟态者可蹭色不产毒，Batesian mimicry，
日后加拟态需注意。）

---

## 5. 拟态 / 隐蔽（crypsis，降低被检测）

**[现实]** crypsis 通过 background matching、disruptive coloration（破坏轮廓的高对比色斑）、
countershading 降低被**发现**概率，作用点在捕食序列最前端（检测），与护甲/棘刺（捕获/处理
阶段）互补。Stevens & Merilaita 给出机制分类；Cuthill 等用野外实验坐实破坏性色斑——把带
不同色斑、以死黄粉虫作「身体」的人造蛾形靶暴露给野生鸟类，**破坏性色斑靶存活显著更高、且
效果独立于单纯背景匹配**，是对非人类捕食者的直接实验证据。（Stevens & Merilaita 2009,
Phil Trans R Soc B, DOI:10.1098/rstb.2008.0217；Cuthill et al. 2005, Nature,
DOI:10.1038/nature03312）

**[对应] 检测端钩子**：`sensors.py` 的 `prey_val`（捕食者视网膜里猎物的显著度）乘一个隐蔽
因子 `prey_val * (1 - crypsis_of(genome)[邻居])`，降低捕食者转向该猎物的概率。**[提案，非
结论]** 与本项目 retina 感知天然对应，A 类。**诚实指出两个矛盾**：①与 escape 语义部分重叠
（都降低被捕食概率，但 crypsis 在感知端下调转向、escape 在命中端缩 reach），需想清差异化；
②**可视化语义矛盾**——隐蔽的可视化就是「更难看见」（降 alpha/压对比），与「让性状上屏被
玩家看见」目标相反，只能折中（高隐蔽画半透明/边缘虚化，让玩家看出「这只在隐身」）。runaway
**中-高**：隐蔽若廉价普及可能让捕食者集体饿死（carn_frac→0，`docs/conventions.md` §8 反复
警告的肉食灭绝），须 6 种子长跑盯 carn_frac。

---

## 6. 集群稀释 / 自私羊群（dilution / selfish herd）

**[现实]** Hamilton 的 selfish herd 指出，个体挤向群体中心以缩小自己的 **domain of danger**
（危险域，最近邻的 Voronoi 邻域），聚群因此降低**个体**被捕食概率（dilution effect）——不
需要任何利他，纯自私行为即可解释聚集。代价是聚群带来的资源竞争与被发现概率上升。（Hamilton
1971, J. Theor. Biol., DOI:10.1016/0022-5193(71)90189-5）

**[对应] 邻居数可算**：固定容量张量范式下，`spatial.geometry` 的 `valid` 掩码
`(valid2 & (|diet_i-diet_j|<δ)).sum(axis=1)` 即每 agent 同类邻居数，无需动态形状。稀释
钩子可在 `dynamics.py:204` 按目标同类群大小稀释 `dmg`。**[提案，非结论] 不推荐作为「可见
形态防御」**，三条理由：①稀释效应**已部分自然涌现**（攻击者选最近猎物，扎堆时一咬分摊到
多只，个体风险本就下降），显式基因会双重计数；②它是**群体/行为属性、非形态可见**，没有单体
可画的纹样；③无自然成本上限，易 runaway 到全员抱团→空间结构坍塌。**建议先用现有 run 测
「局部密度 vs 个体被捕食风险」是否已呈稀释关系（result-analyst 的活），再决定要不要显式
加基因**，而不是先实现。

---

## 7. 速度 / 体型作为反捕食手段（refuge，证据较散）

**[现实]** 两条独立机制。**体型**：对 gape-limited predator（受口裂限制，多数鱼、蛇、部分鸟）
而言，猎物长过某阈值就吞不下，构成 **size refuge**（Kastner 等：入侵 stoat 重度捕食 kiwi
幼体，直到长到体型庇护之上才脱险）。**速度**：sprint speed 直接降低被捕获概率，但维持高逃逸
速度有氧化损伤代价；二者常协同。**但未检索到统摄「体型/速度 refuge」的高被引综述锚点**，本节
由零散实证拼成，强度弱于前六节，标为 **gap**。（Kastner et al. 2024, Ecol. Evol.,
DOI:10.1002/ece3.11598 `[仅作机制佐证，非综述]`）

**[对应/提案，非结论]** 这两条在本项目里**都已被处理或证伪过一次，勿简单重试**：
- **体型 refuge**：`size` 已是基因但**刻意不碰 predation**（`config.py:275-286`）。原始
  gape-limited refuge 提案已死——体型对幼体渴死无用（收益在成体、代价贯穿全龄），且 size
  耦合任何单向收益而不配新成本就会 runaway 到上界。要做须显著抬 size 代价或把 gape 建成
  「攻击者相对」引入频率依赖，且**与 `config.py:275-286` 已归档的设计决定正面冲突，动手前
  须在 `docs/experiments.md` 核对是否已被否**。可见性最高（点大小天然=体型），但演化可行度
  最低。
- **速度**：`dynamics.py:61` 把 `cfg.max_speed` 换成 `speed_of` 即可，但**隐藏耦合陷阱**——
  当前耗能/耗水按 `thrust`（推力比例）计费、不按 `max_speed`，不配对代价 = 「免费换位移」直接
  peg 到上限。且速度**不是纯防御**（逃逸利猎物、追击利捕食者，两生态位都想要），diet 门控
  方向暧昧，runaway 风险**高**（双向军备）。逃逸维度本项目已由 escape 基因承载。

---

## 8. 小结：候选按「落地便宜度 × 可见度 × 演化可行度」排序

| # | 性状 | 落地便宜度 | 可见度 | 演化可行度 | runaway | 综合 |
|---|------|-----------|--------|-----------|---------|------|
| 1 | **厚皮/护甲 armor** | 高（`removed` 乘子 + 能量税） | 中（压暗/加厚描边，shader，避开尖刺几何） | 高（个体级收益最干净，escape 结构孪生） | 中 | **★ 首推** |
| 2 | **尖刺/反击 spike** | 高（predation 一处 + 税） | 中（放射纹样，shader） | 中（频率依赖自限，但收益偏亲缘/群体级） | 低 | **★ 次推** |
| 3 | **警戒色+毒 aposematism** | 中（毒同 spike + 告警色；选择版免动 `in_dim`） | **高（告警色最自然，可视化旗舰）** | 中（亲缘/频率依赖） | 低-中 | **★ 可视化旗舰** |
| 4 | 诱导型防御 inducible | 中（需风险感知反应函数） | 随宿主性状 | 中-高（A 类，契合昼夜波动） | 低 | 值得，排护甲之后 |
| 5 | 拟态/隐蔽 crypsis | 中（sensors 一处） | 语义矛盾（越隐越难看） | 中-高（可能拖垮肉食） | 中-高 | 慎（盯 carn_frac） |
| 6 | 速度 speed | 中（act 一处，隐藏成本坑） | 弱（拖尾） | 中（非纯防御，双向军备） | 高 | 次要 |
| 7 | 体型 refuge | 高（size 已存在，`gl_PointSize` 直乘） | **高（点大小天然）** | **低（与既有决定冲突，peg 风险）** | 高 | 慎（先查 experiments） |
| 8 | 集群稀释 dilution | 中（邻居数可算） | **无（群属性，非形态）** | 低（与涌现效应重叠、难记成本） | 中-高 | 不推荐 |

**建议头 3 个：护甲（1）+ 尖刺（2）+ 警戒色/毒（3）。** 三者都只动 `dynamics.predation` 一处
收益钩子 + 一条 `metabolize` 能量税，完美复用红皇后模板与「gene=0 中性起点」约定，全部 A 类
（作用在成体捕食维度、不被幼体渴死瓶颈删失），全部 shader 可见且不改 `in_dim`。三者可组成
一个「防御红皇后」实验组：尖刺/毒（反伤，频率依赖自限）对照护甲（单向减伤，靠 cost 封顶），
观察哪种防御在本世界能稳定演化出来而不压垮肉食。**首个落地性状的判决（护甲）及其逐文件规格
在 `docs/trait_addition_feasibility.md`。**

## 9. 真实军备竞赛范例：给 `docs/attack_range_redqueen.md` 补生物学背景

**[现实]** Vermeij 的 escalation 框架区分**单侧升级**（对「敌人整体」的军备升级，化石记录里
贝壳随压碎型捕食者增强而增厚）与严格的**双侧 coevolution**（物种对物种）。后者最干净的活体
范例是 newt（*Taricha*）与 garter snake（*Thamnophis*）：newt 用 tetrodotoxin (TTX) 防御，
snake 反复独立演化出 Na_V1.4 钠通道抗性，TTX 成为协同演化的**表型界面**。抗性带代价：Hague
等测得携带大效应抗性等位基因的加州谱系蛇**爬行速度显著下降**——升级不是免费的，这正是红皇后
「拼命奔跑只为留在原地」的分子级证据。（Vermeij 1994, Annu. Rev. Ecol. Syst.,
DOI:10.1146/annurev.es.25.110194.001251；Brodie et al. 2005, J. Chem. Ecol.,
DOI:10.1007/s10886-005-1345-x；Hague et al. 2018, Evol. Lett., DOI:10.1002/evl3.76）

**[对应] 已部分实现，本节补背景**：本项目已实测双侧红皇后（可遗传 `attack_range`↔`escape`），
测出「攻击/逃逸各自爬上平台、有效距离被压回基线、捕食者占比减半」的协同演化瞬态
（`docs/attack_range_redqueen.md`）。newt-snake 正是这套动态的真实对应；Hague 的「抗性↔运动
速度」权衡对应本项目应给防御性状配代价货币。**建议在 `docs/attack_range_redqueen.md` 红皇后
小节交叉引用这三篇**，不重写既有条目。

---

## 10. 研究缺口与外推风险（诚实清单）

1. **护甲的 A/B 归类是一个悬而未决的设计决定**，不是已确立结论——取决于代价记能量账（A 类，
   `docs/trait_addition_feasibility.md` 的选择）还是照搬真实生物学的生长账（B 类，
   barrett2008 的现实）。两份报告在此有意保留张力，落地实验才能裁决。
2. **所有护甲代价数据来自水生系统**（stickleback、Daphnia），代价货币是钙/磷/生长；本项目
   无矿物质代谢，只有能量与生长时间。「护甲拖慢生长」这条**方向**可靠，**具体数值不可直接搬**。
3. **aposematism 与 inducible defense 都依赖「捕食者能学习/感知线索」**：本项目捕食者是演化脑，
   原则上可承载，但当前捕食判定里是否有这类学习/线索通路尚未核（本轮未深挖脑侧）；落地前需
   确认机制存在，否则这两条只能停在 `[提案，非结论]`。
4. **稀释效应可能已涌现**：Hamilton 是纯理论，本项目「局部密度↔个体被捕食风险」很可能已可
   测量，宜先测再实现，避免加冗余性状。
5. **「体型/速度 refuge」一节强度不足**：无统摄性高被引综述锚点，只有零散实证，若要正式并入
   需再补一轮（关键词 `escape performance review Domenici`、`prey body size predation risk
   meta-analysis`）。
6. **待读升级优先级**：warning-color / crypsis 两节目前各只一个精读锚点，若要加码优先精读
   mappes2005 与 cuthill2005。
7. **伪重复**：任何落地实验的空间结论只对**这一套河系**成立（`terrain.build` 无 RNG，多种子
   同一张地图），推广需交叉 `ridge_wavenumber` 等地形种子（`docs/conventions.md` §6）。
