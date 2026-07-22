# 恐惧地景:文献基础与机制设计

这份文档回答 `docs/carnivore_riparian.md` 留下的一条主线:该文档已经确立"捕食者贴河
驻留是当前规则下的正确解",并且用 6 配对种子的消融**证伪**了最直觉的便宜修法
(`meat_water_frac=0`——捕食者反而贴得更近)。它把真正缺失的东西点了名:**一批现实
里会让捕食者离开水源的机制,在本世界完全不存在**,其中排第一的候选是 landscape of
fear——本世界的食草者对捕食者的感知是瞬时的、没有记忆、没有空间学习,所以猎物不会
学着回避危险区域,捕食者也就没有被推开的压力。

`docs/carnivore_riparian.md` §2.2(a) 已经就 landscape of fear 概念本身做了大量
一手核实的文献工作(概念三篇奠基文献、黄石正反双方证据链、Hwange 水坑研究、播放/
嗅觉操纵实验、方法论批评),本文档**不重复那部分已经做完的核实**,而是:(1) 把那部
分工作里跟"这是不是一个好设计依据"直接相关的结论**原样引用并标注来源**,(2) 补上
`carnivore_riparian.md` 尚未覆盖的几块——风险敏感觅食的经典理论根基、饮水本身作为
高风险行为的独立文献、非消耗性效应大于直接捕食的量化证据,(3) 把文献对照到
`sensors.py`/`memory.py`/`brain.py` 的具体代码,(4) 给出三个可实现的机制设计,每
个都过一遍 `docs/trait_evolution.md` §11 的交割期检验(deferred-payoff test)。

**这是一份调研+设计文档,不改任何 `underworld/` 下的代码。** 标记体例沿用
`docs/biology.md`:**[现实]** 已发表事实、**[本世界实测]** 在这个代码库跑出来的数
字、**[对应]** 落到哪段代码、**[提案,非结论]** 尚未验证的设计建议。

---

## 1. 恐惧地景的实证史

### 1.1 风险敏感觅食:landscape of fear 之前的理论根基

Landscape of fear 不是凭空提出的概念,它是嫁接在一套更早、更基础的最优觅食理论上
的——这套理论解释的是个体如何在单次觅食决策里权衡能量收益与被捕食风险,而 landscape
of fear 只是把这套决策论铺到了空间上。

**[现实] Brown, J.S. (1988), "Patch use as an indicator of habitat preference,
predation risk, and competition," *Behavioral Ecology and Sociobiology* 22:37–47。**
这篇提出了 **giving-up density(GUD,弃食密度)**——一个可测量的觅食经济学工具:
把最优觅食理论(边际值定理)延伸到包含捕食成本与错失机会成本,一个觅食者放弃一个
食物斑块时残留的食物密度,直接反映它把"继续吃"这个行为的代价评估到了多高。GUD 高
说明该地点被判定为危险(或竞争激烈)而提前放弃。这是 landscape of fear 后来能够被
**测量**而不只是被观察到的方法论基础。

**[现实] Lima, S.L. & Dill, L.M. (1990), "Behavioral decisions made under the
risk of predation: a review and prospectus," *Canadian Journal of Zoology*
68:619–640(DOI 10.1139/z90-092)。** 系统综述了捕食风险在整个"遭遇→侦测→攻击→
捕获"链条的每一环上如何塑造行为,论证动物会在觅食之外的几乎所有行为维度上(社会性、
逃跑时机、甚至鱼类的呼吸行为)主动评估和管理风险——这是把"风险塑造行为"从觅食经济
学的一个特例,推广成一个跨行为域的一般命题。

**[现实] Lima, S.L. (1998), "Stress and decision making under the risk of
predation: recent developments from behavioral, reproductive, and ecological
perspectives," *Advances in the Study of Behavior* 27:215–290。** 把反捕食决策
进一步延伸到繁殖行为与长期适合度,并把捕食风险明确框成一种**应激源**,带有种群/群
落层面的后果——这是概念上最贴近一年后 Brown, Laundré & Gurung (1999) 提出
"landscape of fear" 一词的前奏之作。

**[现实] Brown, J.S. (1999), "Vigilance, patch use and habitat selection:
Foraging under predation risk," *Evolutionary Ecology Research* 1(1):49–71。**
需要明确的是:这**不是**同年由 Brown, Laundré & Gurung 发表在 *Journal of
Mammalogy* 上、提出 N-driven/μ-driven 区分的那篇(`docs/carnivore_riparian.md`
§2.2(a) 已核实过后者)——两篇是同一年、部分作者重叠、但发表在不同期刊、内容不同的
独立论文,混为一谈是引用错误。这一篇的具体贡献是把**警戒本身**(而不仅是 GUD/斑块
停留时间)形式化进同一个框架,让警戒、GUD、生境选择三者被统一处理——是同年"恐惧生
态学"概念文章的警戒力学配套篇。

**对本世界的意义**:这套理论建立的核心机制是——风险不需要真的杀死猎物才能改变猎
物的行为与适合度,只需要猎物能够**感知并记住**"这里危险"。这正是本世界当前完全缺
失的一环(§2 详述):`sensors.py` 里有瞬时的风险感知(`pred_val`),但没有任何机制
能把它转化成"这个地点的 GUD 应该更高"这种持久的空间决策依据。

### 1.2 Landscape of fear 概念本身与黄石争议——核实工作已在别处完成,这里只引用结论

概念的三篇奠基文献(Brown, Laundré & Gurung 1999 的理论框架;Laundré, Hernández &
Altendorf 2001 *Canadian Journal of Zoology* 79(8):1401–1409,"landscape of
fear"一词的真正出处,带黄石狼-麋鹿系统的原始警戒行为数据;Laundré, Hernández &
Ripple 2010 的综述/命名文章)、术语滥用的方法论批评(Bleicher 2017)、以及黄石营养
级联争议的完整正反双方证据链(主张方 Ripple & Beschta 2004/2012、Fortin et al.
2005;反驳方 Kauffman, Brodie & Jules 2010/2013、Middleton et al. 2013、
Marshall, Hobbs & Cooper 2013;最新一轮 Ripple et al. 2025 与 MacNulty et al.
2025 的循环论证质疑),**已经由 `docs/carnivore_riparian.md` §2.2(a) 一手核实并逐条
标注置信度,这里不重复那份工作,只摘录对本设计文档有直接意义的结论**:

> **黄石确实发生了某种真实的植被变化,麋鹿确实会对附近的狼做出行为响应——这两条底
> 层观测本身站得住。但"恐惧驱动的行为改变,独立于种群数量,解释了景观尺度的植被恢
> 复"这条具体因果链,在多次专门设计用来检验它的景观尺度研究里都没有获得支持
> (Kauffman 2010、Middleton 2013、Marshall/Hobbs/Cooper 2013,以及 2025 年最新一
> 轮的循环论证质疑)。准确的措辞是"反复未能在景观尺度复现,且最新的强效应论文被独
> 立指出存在方法论循环",而不是"被证伪"(Ripple 等人仍在持续为此辩护,植被数据本身
> 也是真的),也不是"已被证实"(流行版本的传播已经远超已核实证据的边界)。
> ——`docs/carnivore_riparian.md` §2.2(a)

**这条争议对本设计有一个关键的、已经在源文档里被挑出来的限定条件,本文档在此再次
强调,因为它是三个设计方案能否成立的分水岭**:Middleton (2013) 指出狼对麋鹿的风险
结构是**稀疏脉冲式**的(遭遇平均九天一次)——这是**主动追猎型(cursorial)**捕食者
的特征。而狮子、鳄鱼这类**伏击型**捕食者的风险结构完全不同:常驻固定地点,风险在
空间上**静态、持续**。`docs/carnivore_riparian.md` §1.4 已经实测确认本世界的捕食
者是伏击型(`carn_speed` 稳态 1.0–1.9,远低于 `herb_speed` 的 6.9–7.2,且这是演化
收敛而非未演化完全)。**这意味着"连续空间恐惧场"这个假设,对黄石的狼-麋鹿系统可能
确实不成立(这正是它反复测不出效应的原因之一),但对本世界的捕食者结构反而是一个更
贴切的风险模型。** 换句话说:黄石案例的负面证据不构成对本文档设计方案的反驳,因为
本世界的捕食者类型恰好落在黄石案例被证明不适用的那一侧之外。

**更贴题的证据来自水源场景本身**(`docs/carnivore_riparian.md` §2.2(a) 已核实):
Valeix et al. (2009a, *Behavioral Ecology and Sociobiology* 63(10):1483–1494;
2009b, *Ecology* 90(1):23–30)在 Hwange 国家公园水坑测得食草动物对狮子的长期/短期
风险都有可测的回避响应;Courbin et al. (2019, *Journal of Animal Ecology*)记录斑
马的昼夜通勤——白天在水坑觅食、傍晚主动迁移数公里外过夜,是"缩短停留时间/错峰"而
非"放弃水源"的精确行为模式,这是本文档 §3 设计可证伪预测的直接参照系;Zanette et
al. (2023, *Current Biology* 33)在南非稀树草原水坑做的大规模声学播放实验
(n=4238)是"感知风险本身驱动水源使用行为"最强的因果(非相关性)证据,几何结构与本
世界几乎一致。反例同样重要:Davies et al. (2016, *Ecology and Evolution*
6(16):5728–5748)在 Addo 大象国家公园测不到同样的空间响应,提醒这个效应是系统依赖
的,不是普适定律;Boiseau et al. (2024, *Animal Behaviour* 210:139–152)用相机陷
阱在沙漠水源直接检验风险分配假说,**结果不支持**——这条负结果值得如实记录,因为它
警告"猎物会理性地围绕捕食风险重新分配饮水时间"这个直觉本身不是没有反例的。

### 1.3 饮水作为高风险行为:独立于 Hwange 水坑研究的补充证据

`docs/carnivore_riparian.md` 已经核实了 Valeix 系列的 Hwange 水坑研究,这里补充三
组独立的、聚焦"饮水本身是一个警戒-觅食权衡的高风险时刻"的文献:

**[现实] Périquet, S., Valeix, M., Loveridge, A.J., Madzikanda, H., Macdonald,
D.W. & Fritz, H. (2010), "Individual vigilance of African herbivores while
drinking: the role of immediate predation risk and context," *Animal Behaviour*
79:665–671。** 把 GPS 项圈狮子轨迹与长颈鹿、扭角林羚、斑马在 Hwange 水坑的警戒扫
描直接对照:扭角林羚(该系统里狮子最主要的猎物之一)在狮子出现时警戒显著提高,长颈
鹿和斑马没有。**这条证据的价值在于它证明饮水时的警戒响应是物种/风险特异的,不是一
个通用反射**——真正处于捕食窗口内的物种才会为饮水付出警戒代价,这对判断本世界的
食草者是否"应该"演化出对应行为提供了一个更精确的现实基准:不是所有猎物都该同等回
避,而是与捕食者食性重叠程度高的才该。

**[现实] Mooring, M.S., Fitzpatrick, T.A., Nishihira, T.T. & Reisig, D.D.
(2004), "Vigilance, predation risk, and the Allee effect in desert bighorn
sheep," *Journal of Wildlife Management* 68:519–532。** 沙漠大角羊的警戒水平随群
体规模缩小(<5 只)和个体间距增大而急剧上升,与水源利用直接相关——这是一个 Allee
效应论证(小群体本身更危险),不是纯粹的水坑研究,但直接支持"孤立个体/小群体饮水风
险更高"这条与本世界高度相关的推论(本世界没有群体聚集机制,每个个体的饮水决策都是
独立的"孤身一人"决策)。

**[现实] Doody, J.S. et al. (2007), "Environmental Manipulation to Avoid a
Unique Predator: Drinking Hole Excavation in the Agile Wallaby, *Macropus
agilis*," *Ethology*;Steer, D. & Doody, J.S. (2009), "Dichotomies in perceived
predation risk of drinking wallabies in response to predatory crocodiles,"
*Animal Behaviour* 78(5):1071–1078。** 这是本文档要找的、真正意义上"伏击型水生捕
食者塑造猎物饮水行为"的一手证据链,来自澳大利亚而非非洲(马拉河角马渡河被鳄鱼捕食
这个流行意象,`docs/carnivore_riparian.md` 已专门核查过,**没有找到可核实的同行评
审一手文献**,本文档不重复这次徒劳检索)。敏捷沙袋鼠会**主动挖掘**自己的安全饮水坑,
夜间(湾鳄活跃、袋鼠自身视觉劣势的时段)优先使用挖掘的侧坑而非主河道,白天则更自由
地使用河道本身——这是一个比"回避"更极端、也更有意思的应对策略:**改造环境而非单纯
避开**,与感官不对称(捕食者夜视占优)直接绑定。孤身个体比成群个体警戒更高、也更倾
向回避河道。这两篇未能取得全文(付费墙),数字与结论经二手来源交叉印证,置信度低于
本节其余引用,但结论方向本身经两次独立检索确认一致。子任务专门检索后**没有找到与
之对应的非洲鳄鱼-有蹄类饮水行为一手研究**——这是一个值得如实标注的检索空白,而不是
拿澳洲的证据硬套非洲叙事。

### 1.4 非消耗性效应常常大于直接捕食本身

`docs/carnivore_riparian.md` 已经核实 Preisser, Bolnick & Benard (2005,
*Ecology* 86(2):501–509,trait-mediated 效应均值约占总效应 63%)与 Preisser,
Orrock & Schmitz (2007, *Ecology* 88(11):2744–2751,静止型捕食者线索比游走型更能
触发强烈恐惧反应——对本世界的伏击型捕食者是一个正面信号)。补充三篇:

**[现实] Werner, E.E. & Peacor, S.D. (2003), "A review of trait-mediated
indirect interactions in ecological communities," *Ecology* 84(5):1083–1100。**
系统论证标准的密度中介(谁被吃了)交互模型会漏掉一整类效应——猎物/捕食者仅仅因为对
方"在场"就发生的性状响应(trait-mediated indirect interactions),这类效应在结构
化食物网时可以和密度中介效应一样强、甚至更强。

**[现实] Peckarsky, B.L., Abrams, P.A., Bolnick, D.I., Dill, L.M., Grabowski,
J.H., Luttbeg, B., Orrock, J.L., Peacor, S.D., Preisser, E.L., Schmitz, O.J. &
Trussell, G.C. (2008), "Revisiting the classics: considering nonconsumptive
effects in textbook examples of predator–prey interactions," *Ecology*
89(9):2416–2425。** 重新审视四个教科书级捕食者-猎物系统(猞猁-野兔周期、威斯康星
湖泊营养级联、水獭/虎鲸/海带),论证非消耗性效应是这些系统动态里此前被单纯归因于直
接捕杀的**必要组成部分**,不是可以忽略的二阶修正。

**[现实] Creel, S., Christianson, D., Liley, S. & Winnie, J. Jr. (2007),
"Predation Risk Affects Reproductive Physiology and Demography of Elk,"
*Science* 315(5814):960。** 这是本节最硬的大型哺乳动物证据:跨大黄石地区 5 个麋
鹿种群、16 个种群-年、1489 份粪便样本,孕酮水平(怀孕关联激素)随"麋鹿:狼"比例下降
而下降,并预测次年的幼崽补充率——**独立于实际被杀数量的纯恐惧生理代价,直接影响种
群统计学**,是"恐惧效应可以与直接捕食效应相当甚至独立起作用"在大型有蹄类身上最扎
实的单篇证据。

**对本世界的意义**:这条文献线索支持的是一个具体的度量学教训——`docs/carnivore_
riparian.md` §2.2(a) 已经引用 Creel & Christianson (2008) 指出"风险效应与捕杀效
应不一定共变,不能从一个推出另一个"。这对 §3 的可证伪预测有直接含义:如果本文档的
设计实现后 `death_predation_frac` 没有显著下降,**不能**据此认为机制无效——它完全
可能只是把捕食压力从"直接死亡"转移成了"用于警戒/回避的行为预算",这是本世界目前完
全没有工具测量的一类效应,需要新的度量(§3.2 会提出一个候选)。

### 1.5 小结:三层论证,层层收紧到本世界

1. **理论根基**(§1.1):风险不需要真的杀死猎物才能塑造其行为与适合度,只需要猎物
   能感知并**持久化**这个风险信息——这正是本世界目前缺的那一环。
2. **概念与黄石争议**(§1.2,由 `docs/carnivore_riparian.md` 核实):连续空间恐惧
   场对**追击型**捕食者(黄石的狼)证据薄弱、反复测不出景观尺度效应;但对**伏击
   型**捕食者(本世界已实测确认的类型)证据链(Hwange 水坑、Zanette 播放实验)明显
   更强、更直接适用。
3. **本世界最贴题的两块补充证据**(§1.3/§1.4):饮水本身是一个真实存在的高警戒时
   刻,且这种警戒是物种/风险特异的而非通用反射;非消耗性(恐惧)效应在多个系统里都
   被测出与直接捕食相当或更强,包括一个独立于捕食数量的大型有蹄类生理学证据
   (Creel et al. 2007)。

三层论证共同指向:**给本世界的食草者加一条能把"这片区域危险"持久化的机制,文献支
持的强度不是笼统的"landscape of fear 成立",而是精确匹配到"伏击型捕食者+水源场景"
这个具体组合——这恰好是本世界的确切设定。**

---

## 2. 现实机制对照本世界代码:缺什么、能不能省着改

### 2.1 [对应] `sensors.py`:`pred_val` 是纯瞬时函数,没有任何持久化

`sensors.sense` 里:

```python
prey_val = closeness * jnp.maximum(di - diet_j, 0.0)       # j more herbivorous
pred_val = closeness * jnp.maximum(diet_j - di, 0.0)       # j more carnivorous
```

`closeness` 由 `dist`(当前帧从 `spatial.geometry` 算出的邻居距离)决定,`pred_val`
逐 sector 取 `max` 汇总成 `[n, R]` 的 `pred` 通道。**这是一个纯粹的、无状态的当帧函
数**:上一步邻居表里有没有捕食者、这个格子五十步前是否死过同类,`pred_val` 一概不
知。`brain.forward` 的循环隐状态 `hidden` 理论上可以携带任意信息跨步传递,但它是一
个 `[n, hidden]` 的稠密向量,没有为"记住某个方位危险"这件事提供任何结构化的存储位
置——要不要用它来记方位是演化要解的问题,而演化目前缺乏一个可以稳定读写方位信息的
接口,详见 §2.3。

**这条对照本身不需要跑实验验证,是读代码就能确认的事实**,`docs/carnivore_
riparian.md` §2.3 已经这样验证过一次,本文档复用同一个结论。

### 2.2 [对应] `memory.py`:槽位机制是通用的相对向量存储,分区靠位置不靠类型标签

`memory.py` 的槽位是 `(dx, dy, strength)` 三元组,`[0, memory_water_slots)` 是水,
其余是果实——**分区纯粹靠槽位在数组里的位置,不靠任何类型标签**。这意味着"危险"作
为第三种分区在**架构上**是完全对称的:`memory.write(memory, lo, hi, should_write,
cfg)` 已经是一个通用函数,任何满足"给一个 `[lo, hi)` 区间和一个触发条件"的资源都可
以复用它,不需要新写一套写入逻辑。`memory.advance` 对整个 `[n_max, slots, 3]` 张量
统一处理位移、衰减、漂移,同样不关心某个槽属于哪个分区。

**但复用不是零成本的**,原因有二,`docs/carnivore_riparian.md` §4.1 已经点出第一
条,这里补充第二条:

1. **语义漂移**:如果不新增槽位而是把一个现有分区(比如把 `memory_fruit_slots` 从
   2 降到 1、腾出的槽位划给危险)重新定义用途,总槽位数、`in_dim`、`genome_size`
   都不变,**技术上不需要作废种群**,但已演化的大脑权重是针对旧语义训练的,读取新
   语义需要重新收敛——收敛时间大概率和真正作废种群相近,只是旧存档不会报形状不匹
   配的错误。
2. **危险与水/果实的写入触发条件性质不同,这是复用时容易被忽略的一点**:水/果实槽
   位的写入条件是"此刻获得了正向收益"(`drink_gain > 0`/`fruit_gain > 0`),是一个
   **离散的、瞬间发生**的事件,而且发生地点就是 agent 自己脚下(`memory.advance`
   已经把位移移到写入之前,保证写入记录的是"~0 偏移")。危险信号如果用
   `last_damage > 0`(被咬过)触发,同样是离散瞬间事件,记录自己脚下,复用完全对
   称。**但如果用 `pred_val` 超过某阈值触发**(更接近真实的"察觉威胁"而非"已经受
   伤"),这个信号本身来自邻居的**相对方位**而非自己的位置——写入的应该是"捕食者所
   在的方向"而不是"我自己脚下",这需要额外把最近捕食者邻居的 `delta` 传进
   `memory.write` 的调用点,而不能像水/果实那样直接写"此地"。这是一个真实的实现复
   杂度,而不只是复制粘贴现有调用。

### 2.3 [对应] `brain.py`:循环隐状态不天然携带"区域"信息,这正是为什么长期记忆槽位
存在

`brain.forward` 的 `hidden` 是一个 `[n_max, hidden]` 的稠密向量,每步用
`tanh(w_in @ inputs + w_rec @ hidden + b_h)` 更新——它可以携带任意演化出来的内部
表征,但**没有任何架构约束逼迫它编码"世界坐标系里的某个方位"**,而且它和 `memory`
一样在出生时被清零(`reproduction.reproduce` 里 `hidden` 与 `memory` 同时置零,
`docs/biology.md` §5 已经论证过这是 Weismann barrier 的正确做法,不应该被撤销)。
这正是为什么 `memory.py` 的长期槽位机制存在——它是专门为"把方位信息用一个固定、可
解释的结构长期保存"这个需求设计的接口,而不是指望循环隐状态自己演化出等价的东西。
**任何"危险要不要持久化"的设计,如果绕开 `memory.py` 指望 `hidden` 自己学会,是在
要求演化重新发明一遍已经写好的轮子,没有理由这样做。**

### 2.4 最小改动的设计空间:三条路径,复杂度递增

把"某片区域危险"变成可学习的空间信息,按改动量从小到大有三条独立路径,分别对应
§3 的三个设计:

1. **个体习得的记忆(§3.1)**:复用 `memory.py` 现成的槽位机制,危险信息**个体私
   有**,靠亲身经历(被咬/看到捕食者)写入,和水/果实记忆走同一套架构。
2. **环境公共信息场(§3.2)**:不经过 `memory`,而是像 `trample` 一样开一个
   `[n_cells]` 的场,由捕食者的存在被动沉积,任何个体(包括刚出生、`memory`/
   `hidden` 全空的新生儿)路过都能立刻感知——这条路径完全不经过个体学习,是"环境
   记得,不是个体记得"。
3. **无感知的选择压力(§3.3)**:根本不新增任何感知通道,只是让"和同类捕食者挤在一
   起"本身变贵,靠已有的 `peer` 通道(instant 的同食性邻近信号)让演化自己去决定要
   不要用这个通道来避开拥挤。

三条路径分别对应"个体学习"、"生态位构建/环境标记"、"纯粹的适应度地形改变"——现实
生态学里这三种机制都有独立的先例(个体学习对应本节 §1 的全部证据;环境标记对应领
域标记/scent-marking 的地盘行为文献;纯选择压力对应干扰竞争本身不需要任何认知机
制)。

---

## 3. 三个可实现的机制设计

每个设计给出:机制、生物学依据、改动的文件、是否作废种群、**交割期检验**(应用
`docs/trait_evolution.md` §11 的 deferred-payoff test)、可证伪预测、团灭风险。

### 3.1 设计一:危险记忆槽(个体习得,复用 `memory.py`)

**机制**:新增一个"危险"分区,写入触发条件为 `last_damage > 0`(被咬过,离散事
件,记录自己脚下,完全对称于水/果实的写入方式)。是否额外用 `pred_val` 超阈值触发
"看到但没被咬"作为第二触发条件是一个独立的复杂度决策(§2.2 已指出这需要额外传递
邻居方位,不能照抄水/果实的调用模式),本设计先只考虑最简单的 `last_damage` 触发
版本。

**生物学依据**:§1.1 的风险敏感觅食理论(个体需要能够评估并记住"这个地点让我付出
了代价")与 §1.2/§1.3 的伏击型捕食者+水源场景证据链——尤其是这条机制天生对应的现
实类比是 `docs/biology.md` 已经在用的大象母系水源记忆先例(数十年尺度的空间记忆,
本世界的记忆系统本来就是照这个先例设计的),把同一套"记住有价值的地点"的架构扩展
到"记住有危险的地点",是概念上最自然的延伸。

**改动范围与代价——两个版本**:

- **版本 A1(便宜)**:`memory_fruit_slots` 从 2 降到 1,新增
  `memory_danger_slots=1`,总槽位数不变,`in_dim`/`genome_size` **不变,技术上不
  作废种群**。改动文件:`config.py`(槽位数字段)、`step.py`(在
  `dynamics.predation` 之后加一次 `memory.write` 调用,触发条件用
  `damage > 0`)、`memory.py` 本身不需要改(`write`/`advance`/`encode` 都是分区
  无关的)。**这个"便宜"版本能省钱,恰恰是因为 `docs/experiments.md` §1 已经测
  过果实层只贡献全图能量通量的 0.35%——牺牲一个果实槽位的机会成本本身接近于
  零**,这是一个可以直接引用的既有负结果,不需要重新论证。
- **版本 A2(彻底)**:新增独立的 `memory_danger_slots` 字段,总槽位数增加,
  `in_dim`/`genome_size` 随之改变,**严格作废种群**。好处是不用赌"果实槽位真的
  没用"这个判断——万一未来 §4 的果实层扩容提案(`docs/experiments.md` §1.4(a))
  被采纳,A1 会和它正面冲突。

**交割期检验**:这个设计**基本通过**检验,但方式和体型基因的失败模式不一样,值得
把差异讲清楚。体型基因失败是因为"收益需要活到某个阶段才能兑现,而 83% 的死亡发生
在那个阶段之前",是一个纯粹的**净成本**——因为体型基因从出生起就要缴水箱容量的代
谢税,不管这个个体活不活得到用上它的那一天。危险记忆槽**没有这个问题**:空槽位对
应输入值为 0(和水/果实槽位出生即空是同一件事),不产生任何前置成本,新生儿即使一
辈子用不上这个槽位也不会因为"槽位存在"而多付出什么。**但它确实有一个更温和的第二
序问题**:这个槽位的信息只能靠"亲身被咬过一次并且活了下来"来获得(否则就是空
的),而 `docs/mortality.md` 记录捕食致死的平均死亡年龄是 170.7 步,远高于平均寿
命 92 步和渴死的平均年龄 52.5 步——**这意味着能够写下这个槽位的个体,本来就是已
经活过了渴死瓶颈的那一小撮幸存者**,不是体型基因那种"收益人群是空集"的失败模式,
而是"收益人群被过滤到一个较小、偏年长的子样本"。这限制了这条机制的选择信号强
度,但不构成体型基因那种致命缺陷。

**可证伪预测**:
- `death_predation_frac` 预期下降,但按 §1.4 的度量学教训,**不下降不代表机制无
  效**——需要新增一个衡量"警戒行为预算"的指标(比如捕食者进入攻击范围但未命中的
  次数、或者食草者在 `pred_val>0` 时的转向幅度)。
- `carn_water_dist` 预期温和上升(理由同 `docs/carnivore_riparian.md` §4.1:河
  道带的猎物密度优势是地形决定的结构性梯度,危险记忆只能让个体在被咬过一次后减少
  返回,不能消灭梯度本身)。
- 因为只有"被咬过并存活"的个体才携带这个槽位,**种群整体的空间分布指标
  (`inland_frac`、`water_bound_frac`)预期变化幅度小于设计二**——这是这个设计
  与设计二之间最重要的一条区分性预测,值得在实现后对照检验。

**团灭风险**:改动范围局限在食草者一侧的信息获取,不直接改变捕食者的能量/水收
支,团灭风险相对较低,但如果配合方案让捕食致死率下降过多(叠加效应),仍需按
`docs/carnivore_riparian.md` 的惯例先用极短种子数探路。

### 3.2 设计二:捕食者留痕——恐惧场折入现有 `pred` 通道(环境公共信息,不经过个体
记忆)

**机制**:仿照 `trample` 场的实现模式,在 `WorldState` 新增一个 `[n_cells]` 的
`fear` 场。每步用捕食者(`diet > 0.5` 或某阈值)的位置做散射累加沉积,按
`fear_decay` 衰减(复用 `dynamics.graze`/`step.py` 里 `trample` 已经用过的
`jnp.zeros(cfg.n_cells).at[cell].add(...)` 原语,是同一个已经在代码里跑过很多次
的模式,不是新发明)。**关键的折叠技巧**:不新增一个 retina 通道,而是像 `sensors.
sense` 里 `food` 通道折叠 `fruit` 那样——`food = (plant + fruit_energy*fruit) /
plant_max` 把两种资源塞进一个通道——把 `fear` 场在每个 sector 方向前方采样出的值
(和 `water_ch`/`slope` 用同一段 `cells = pos_to_cell(sample...)` 代码)与瞬时
`pred_val` 取 `max` 合并进现有的 `pred` 通道:

```python
# 伪代码,未实现
fear_ahead = terrain_like_field.fear[cells]          # [n, R],采样代码与 water_ch 相同
pred = jnp.maximum(pred_instant, fear_ahead * cfg.fear_sense_scale)
```

**这样 `in_dim` 完全不变**——`pred` 通道原本只能回答"此刻此地有没有威胁",现在同
一个通道还能回答"这个方向最近是不是有威胁常驻过",是同一个语义维度上的信息增
量,不是语义替换(对照设计一版本 A1 是把"果实方向"整个替换成"危险方向",是一次
180 度的语义置换;这里是把"预测方向"的瞬时值和滞后值取 max,是同一个物理量的时
间窗放宽,冲击小得多)。

**生物学依据**:§1.1–1.4 的全部证据链,外加一个更具体的类比——真实的地盘标记
(scent-marking)本身就是"环境携带持久风险信息,不需要每个个体单独学习"的现实机
制,§2.2(b)(`docs/carnivore_riparian.md`)已经引用 Durant (2000) 记录猎豹主动回
避狮/鬣狗的**声音信号**热点,这正是"环境里留存的第三方信号驱动回避"而非"个体记
忆"的现实先例。

**这条设计是否会重蹈踩踏场的覆辙?正面回答。** `docs/experiments.md` §2/§4 记录
了被动踩踏场两次尝试(负反馈侵蚀承载力、正反馈降低通行成本)都没能形成兽径,根因
被归结为**缺少"两个固定端点"**——真实兽径连接巢穴与水源这样的固定地点,重复通行
同一条线路才是正反馈的前提,而本世界的 agent 没有巢、没有需要往返的两点,踩踏场
因此只是复制了一份已经存在的种群密度图,没有沿线聚集。

**这个风险对恐惧场不适用,但理由不是"恐惧场天生没有这个问题",而是恐惧场要做的
事情本质上更容易**:踩踏失败的任务是**形成新的线状结构**(路径),这需要合成此
前不存在的空间模式;恐惧场要做的只是**给一个已经存在、而且高度集中的空间模式打
上标签**——`docs/carnivore_riparian.md` §1.2 已经实测捕食者的空间分布本身极度集
中(河道带 0–8 单位内捕食/食草密度比高达 0.5–0.8,过了 24 单位骤降到 4%–12%,这
比一般种群密度梯度陡峭得多),恐惧场只需要对这个已经存在的强梯度做一次时间上的
低通滤波(衰减而非瞬时),不需要凭空创造聚集,聚集早就在那里了。**换句话说,踩踏
场的失败模式("被动沉积 + 无固定端点 = 只是把种群密度图复制到另一个场上")对恐惧
场反而是一个好消息而不是坏消息**——恐惧场想要的效果,恰恰就是"复制一份已经存在
的密度图",不多不少。

**但确实存在一个方向不同、来自本世界死亡结构的独立新风险,是 `docs/carnivore_
riparian.md` 未讨论过的**:恐惧场是**环境公共信息**,新生儿(`memory`/`hidden`
全空)一出生就能感知到它,不需要像设计一那样先亲身经历一次危险。这本来是设计二
相对设计一的优势(见下文交割期检验),但反过来看,`docs/mortality.md` 记录 83%
的死亡是幼体渴死(平均死亡年龄 52.5 步),而幼体渴死的根因是"有腿、有时间、没有
方向"——出生时记忆槽为空,不知道水在哪。**如果恐惧场让新生儿在瞬时/滞后混合的
`pred` 通道读数偏高时更倾向转向离开,而河道带恰好是密度最高、因而恐惧场读数最
高、但同时也是水源本身所在的地方,这个机制有可能在新生儿最缺水的时候,恰好增加
一个让它们更晚抵达水源的信号。** 这不是"收益被推迟到活过某阶段才能兑现"那种经
典交割期失败(那需要个体先经历某事才能受益,这里恐惧场对新生儿是免费的),而是
**成本恰好砸在死亡率最高的窗口**——是同一条设计原则的镜像版本,值得在文档里明确
记录为一个新发现的风险,而不是简单复用体型基因那套论证。

**交割期检验**:这条设计比设计一更彻底地**规避**了经典的交割期问题——恐惧场不需
要个体活过任何阶段才能受益,出生第一步就能读到。但如上所述,它引入了一个方向相
反的新风险(在死亡率最高的窗口增加一个可能延误饮水的信号),需要作为独立于交割
期检验之外的第二项风险单独跟踪,而不能因为它"通过了"交割期检验就默认安全。

**改动文件**:`state.py`(新增 `fear: jax.Array # f32 [n_cells]`,和 `trample`
一样是逐格场,不是逐 agent,`reproduction.place()` 不需要改)、`step.py`(散射累
加+衰减,紧邻现有 `trample` 那段代码)、`sensors.py`(`pred` 通道的 max 折叠)、
`config.py`(`fear_decay`、`fear_rate`、`fear_sense_scale`、一个类似
`trample_impact` 默认 0 的 `fear_enabled`/`fear_rate=0` 开关,保证默认配置下这
个机制**不存在**而不只是**被抵消**,和 `trample_impact` 的既有约定一致)。
`in_dim`/`genome_size` **不变**,不严格作废种群;但和设计一版本 A1 一样,"形状不
变"不等于"免费"——已有存档的 `pred` 通道语义会突然多出一个此前不存在的滞后分
量,不过因为默认 `fear_rate=0` 完全复现旧行为,这个风险只在显式打开消融开关时才
出现,比设计一 A1 的"果实→危险"整体语义置换温和得多。

**可证伪预测**:
- `carn_water_dist` 预期上升,幅度预计**大于**设计一(因为公共信息覆盖全体食草
  者而非仅"被咬过的幸存者"这一子样本),但仍然是温和上升而非反转(理由同设计
  一,地形梯度本身没有被触碰)。
- `herb_water_dist` 应当只小幅上升,若观察到大幅飙升(比如翻倍)需要被当作意
  外结果检查,而不是理所当然的成功信号(`docs/carnivore_riparian.md` §4.1 已给
  出同样的告诫,此处对设计二同样适用,且因为设计二没有"个体亲历"这道门槛,过度
  反应的风险理论上比设计一更高)。
- 需要新增一个"河道带停留时长/到访频率"指标(§1.2 的 Courbin (2019) 类比目
  标),因为静态快照下的距离均值无法区分"缩短停留"和"完全不去"两种不同的行为响
  应,而这两者的死亡率含义完全不同。
- `death_thirst_age`(渴死平均年龄)和 `death_thirst_frac` 应该被专门监控,而不
  只是看空间分布指标——如果这两个数字在打开 `fear_rate` 后恶化,直接支持上面提
  出的"成本砸在死亡率最高窗口"这条新风险。

**团灭风险**:比设计一更需要谨慎——因为公共信息即时生效、覆盖全体个体(包括新
生儿),任何过强的回避倾向都可能在渴死瓶颈本已很窄的情况下把它收得更窄,是四个
方案里最需要先看 `death_thirst_frac` 而不是先看捕食相关指标的一个。**强烈建议
先用 `fear_sense_scale` 设一个很小的默认值做极短种子数探路,重点观察渴死结构有
没有恶化,再决定要不要放大到能测出捕食空间效应的量级。**

### 3.3 设计三:捕食者局部密度代谢附加成本(无感知的选择压力,复用 `peer` 通道)

**机制**:让捕食者密集聚集本身变贵,而不新增任何感知通道。`spatial.gather_
neighbors` 已经为每个 agent 算出了邻居表,`metabolize` 里加一项:统计每个捕食者
(`diet` 超过某阈值)邻居里同为捕食者的数量(或密度),按这个局部密度给 `carn_
cost` 或独立的一项代谢开销加一个附加项。**不需要新增感知通道,因为惩罚不需要被
"看见"才能起作用——它只需要被"感受到"(代谢账单变高),而 agent 是否学会用已有的
`peer` 通道(`sensors.py` 里现成的、diet 相似度的瞬时信号,原本是为食草者的社会学
习设计的)去主动避开高密度区域,是留给演化自己决定的事**,和 `docs/carnivore_
riparian.md` §4.1 里"是否学会回避是演化的事,不是硬编码的"是同一条设计哲学。

**生物学依据**:`docs/carnivore_riparian.md` §2.2(b) 已经核实的干扰竞争/地盘行
为文献——Palomares & Caro (1999) 记录食肉动物间种间杀戮相当常见、Barker et al.
(2023) 记录狮/鬣狗即便活动范围重叠也靠时间生态位错峰缓解干扰竞争。本设计不实现
真正的攻击行为(那需要新的 `out_dim` 输出,复杂度显著上升),只实现干扰竞争最简
化的经济学后果:拥挤本身有代谢代价。

**改动文件**:`dynamics.py`(`metabolize` 增加一项局部捕食者密度代谢开销,复用
已经传入的邻居表,不需要新的邻居查询)、`config.py`(新增密度惩罚系数,默认 0,
和 `trample_impact` 一样是"不显式打开就不存在"的约定)。`in_dim`/`genome_size`
**完全不变**,**不作废种群**,是三个设计里改动量最小的一个——`peer` 通道已经存
在,不需要新建任何感知路径。

**交割期检验**:这个设计只影响捕食者(`diet` 高的个体)的代谢账单,不直接触碰食
草者的渴死瓶颈(83% 的死亡),所以按 §1 的经典表述,这个设计本身**不属于**交割
期检验要拦截的那类失败——它的成本从捕食者出生起就按局部密度实时计费,不存在"要
先活过某阶段才能兑现"的收益结构,是一个即时反馈的选择压力,不是一个延迟兑现的
性状。**但有一个需要如实标注的数据空白**:`docs/mortality.md` 的死因分解是全种
群口径,没有按 `diet` 拆分——本世状态目前不知道捕食者内部是否也有类似"幼体先要
闯过某个瓶颈才能活到能表达这个策略的年龄"的结构。这是实现本设计前值得先补一个
`Metrics` 字段(按 diet 分层的死因/死亡年龄)来确认的空白,而不是假设捕食者的死
亡结构和食草者一样。

**可证伪预测**:
- 河道带捕食/食草密度比(`docs/carnivore_riparian.md` §1.2 测到的 0.5–0.8)应
  当下降,捕食者空间分布的 Moran's I 或类似聚集度量应当下降(和踩踏场用过的空间
  自相关统计量同一套工具)。
- `carn_water_dist` 预期上升,但**这条预测的风险在于它和 `docs/carnivore_
  riparian.md` 方案 D 的风险评估完全相通**:真实的地盘行为伴随的是"被排挤的个体
  去边缘地带讨生活",而不是"种群整体分散变好"。§1.2 已经测过内陆(24–48 单位带)
  的猎物密度只有河道带的 1/4 到 1/5,被排挤出河道带的捕食者很可能只是死得更快,
  不会自动形成一个新的稳定内陆亚群——这条风险在本设计和 `docs/carnivore_
  riparian.md` 方案 D 之间是完全共享的,不是独立发现,值得明确标注避免重复计数
  为两条不同的证据。

**团灭风险**:这是三个设计里对捕食者种群最直接施压的一个——它不像设计一/二那样
只是"给猎物提供信息、由演化决定要不要用",而是直接对捕食者的生存加税。如果内陆
猎物密度梯度没有被设计二或其他机制拉平,单独实施这个设计有较高概率重演
`docs/carnivore_riparian.md` 记录的"看似合理的改动全灭捕食者"模式。**不建议单独
先做这个设计**,应该等设计二(或其他能拉平猎物密度梯度的机制)有阳性结果之后再
上,和 `docs/carnivore_riparian.md` 方案 D 的建议顺序一致。

---

## 4. 推荐顺序

按"每单位实现风险能买到多少捕食者离水"排序,同时把作废种群的改动单独摘出来处理:

1. **设计二(恐惧场折入 `pred` 通道)排第一**:不改变 `in_dim`/`genome_size`,复
   用已经写过多次的 `[n_cells]` 场散射累加原语(`trample`)与已经写过多次的方
   向采样原语(`food`/`water_ch`/`slope`),文献支持最直接(§1.2/§1.3 的水源伏
   击型捕食者证据链),而且已经正面回答了"会不会重蹈踩踏场覆辙"的疑虑(它要做
   的是标记已存在的强梯度,不是从零合成路径)。**唯一需要认真对待的是本文档新
   提出的风险**:公共信息覆盖新生儿,可能在渴死瓶颈最窄的窗口增加一个延误饮水
   的信号——这条必须先用极短种子数专门盯着 `death_thirst_frac`/`death_thirst_
   age` 探路,不能只看空间分布指标就判定成功。
2. **设计一(危险记忆槽,版本 A1)排第二**:同样不改变 `in_dim`,生物学类比最
   干净(直接延伸本世界已有的水源记忆设计哲学),交割期检验通过得比设计二更彻
   底(没有"信息公开可能伤害新生儿"这条反向风险),代价是只能覆盖"被咬过还活下
   来"的幸存者子样本,选择信号强度预计弱于设计二。**版本 A2(新增独立槽位,作
   废种群)不建议单独做**——应该和 `docs/experiments.md` §1.4(a)/(b) 的果实层
   扩容或删除决策、以及 §4.2 提出的"给 agent 一个家"(trample 路径形成所需的固
   定端点,同样作废种群)打包成一次统一的 `in_dim` 变更,而不是分三次分别作废种
   群。
3. **设计三(捕食者局部密度代谢附加成本)排第三,且明确设了前置条件**:改动量最
   小(不新增任何感知通道),但对捕食者种群的压力最直接、最没有缓冲——只有在
   设计二(或其他能改善内陆猎物密度梯度的机制)已经跑出阳性结果之后再上,理由
   与 `docs/carnivore_riparian.md` 方案 D 完全相同:内陆本来就吃不饱,不先解决
   这个前提,单纯把捕食者从河道带挤出去大概率只是换一种方式的团灭。
4. **`docs/carnivore_riparian.md` 方案 C(放大 `attack_range`)与方案 B(削弱
   `meat_water_frac`,已被数据证伪)不在本文档重复讨论**,它们分别是纯参数改
   动和已经被否决的方案,不属于本文档"设计新的空间信息机制"的范围。

**如果要做 A2(作废种群的危险记忆槽版本),该和谁打包**:目前文档集里已经记录了
至少两处等待作废种群的 `in_dim` 变更——`docs/experiments.md` §1.4(b)(删除果
实层,还回 2 个记忆槽和 8 个输入)与 §4.2(给 agent 一个不随消耗漂移的"家"槽
位,用于兽径路径形成的固定端点)。**A2 应该和这两处一起打包成一次统一的种群作
废**,而不是三次独立的重新演化——这既是工程上的效率考虑,也是方法论上的考虑:
连续三次分别作废种群会让"这次种群表现的变化是哪个改动造成的"难以归因,一次性打
包、6 配对种子对照基线,才能干净地分离每个改动各自的贡献。

---

## 5. 方法论附注

- **本文档的文献工作分两部分**:黄石争议与 landscape of fear 概念三篇奠基文献
  的一手核实**由 `docs/carnivore_riparian.md` 完成,本文档只引用其结论**,不重
  复检索;§1.1(风险敏感觅食理论根基)、§1.3(饮水风险的独立补充文献)、§1.4
  (非消耗性效应量化证据的补充)由本次会话的一个并行子任务专门检索核实,子任务
  报告明确标注了置信度分级:多数条目(Brown 1988、Lima & Dill 1990、Lima
  1998、Brown 1999 EER、Werner & Peacor 2003、Peckarsky et al. 2008、Creel et
  al. 2007)经过至少两个独立文献来源的题目/期刊/卷页交叉确认,置信度较高;
  Périquet (2010)、Mooring (2004)同样经交叉确认;**Doody et al. (2007)与
  Steer & Doody (2009)未能取得全文(付费墙),仅通过摘要级二手来源确认**,置信
  度低于本文档其余引用,标注见正文对应位置。子任务专门检索后**没有找到与马拉河
  角马-鳄鱼意象对应的一手文献,也没有找到非洲鳄鱼-有蹄类饮水行为的对应一手研
  究**——这是一处明确的检索空白,如实记录而非用二手意象填补。
- **本文档没有运行任何 JAX 实验**——所有"可证伪预测"都是尚未验证的提案,与
  `docs/carnivore_riparian.md` 已经跑过 6 配对种子实测的 `meat_water_frac=0` 消
  融不同,本文档三个设计的 falsifiable prediction 全部标注为 [提案,非结论],
  需要未来的会话实际实现代码并跑消融才能升级为 [本世界实测]。
- **交割期检验的应用本身也是一次方法论尝试**:`docs/trait_evolution.md` §11 提
  出这条原则时的样本(体型基因)是一个"收益人群是空集"的极端失败案例。本文档三
  个设计没有一个落在那么极端的失败模式里,但应用这条检验的过程中发现了两类它原
  始表述没有覆盖的情形——"收益人群被过滤到一个较小的幸存者子样本"(设计一)和
  "成本(而非收益)恰好砸在死亡率最高的窗口"(设计二)。这两条本身可能值得在未
  来某次修订 `docs/trait_evolution.md` §11 时补充为这条通用原则的两个变体,但按
  本次任务范围,本文档不修改其他 docs 文件,只在此记录这个观察。

---

## 6. 参考文献总表

除已由 `docs/carnivore_riparian.md` 核实、本文档只引用结论的条目外(黄石争议双方
全部文献、landscape of fear 概念三篇奠基文献、Bleicher 2017、Hwange 水坑系列、
Zanette 播放实验系列、Preisser 系列、Middleton 2013——完整列表见该文档 §7),本
文档新增核实的条目:

- Brown, J.S. (1988). Patch use as an indicator of habitat preference,
  predation risk, and competition. *Behavioral Ecology and Sociobiology*
  22:37–47.
- Brown, J.S. (1999). Vigilance, patch use and habitat selection: Foraging
  under predation risk. *Evolutionary Ecology Research* 1(1):49–71.(与同年
  Brown, Laundré & Gurung *J. Mammalogy* 一文是两篇独立论文,勿混淆)
- Creel, S., Christianson, D., Liley, S. & Winnie, J. Jr. (2007). Predation
  risk affects reproductive physiology and demography of elk. *Science*
  315(5814):960.
- Doody, J.S. et al. (2007). Environmental manipulation to avoid a unique
  predator: drinking hole excavation in the agile wallaby, *Macropus agilis*.
  *Ethology*.(全文未取得,二手来源交叉印证)
- Lima, S.L. (1998). Stress and decision making under the risk of predation:
  recent developments from behavioral, reproductive, and ecological
  perspectives. *Advances in the Study of Behavior* 27:215–290.
- Lima, S.L. & Dill, L.M. (1990). Behavioral decisions made under the risk of
  predation: a review and prospectus. *Canadian Journal of Zoology*
  68:619–640.
- Mooring, M.S., Fitzpatrick, T.A., Nishihira, T.T. & Reisig, D.D. (2004).
  Vigilance, predation risk, and the Allee effect in desert bighorn sheep.
  *Journal of Wildlife Management* 68:519–532.
- Peckarsky, B.L., Abrams, P.A., Bolnick, D.I., Dill, L.M., Grabowski, J.H.,
  Luttbeg, B., Orrock, J.L., Peacor, S.D., Preisser, E.L., Schmitz, O.J. &
  Trussell, G.C. (2008). Revisiting the classics: considering nonconsumptive
  effects in textbook examples of predator–prey interactions. *Ecology*
  89(9):2416–2425.
- Périquet, S., Valeix, M., Loveridge, A.J., Madzikanda, H., Macdonald, D.W. &
  Fritz, H. (2010). Individual vigilance of African herbivores while
  drinking: the role of immediate predation risk and context. *Animal
  Behaviour* 79:665–671.
- Steer, D. & Doody, J.S. (2009). Dichotomies in perceived predation risk of
  drinking wallabies in response to predatory crocodiles. *Animal Behaviour*
  78(5):1071–1078.(全文未取得,二手来源交叉印证)
- Werner, E.E. & Peacor, S.D. (2003). A review of trait-mediated indirect
  interactions in ecological communities. *Ecology* 84(5):1083–1100.

**未采信的检索线索**:非洲鳄鱼-有蹄类饮水行为的一手研究(区别于已知查无实据的马
拉河角马渡河意象)——专门检索后未找到,记录在此避免未来重复徒劳检索。
