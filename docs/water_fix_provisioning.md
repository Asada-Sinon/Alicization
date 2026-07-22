# 解耦新生供水：泌乳式水分地板能不能逃脱"演化压回地板"

这份文档回答 `docs/water_system.md` 排出的头号不合理设置——**新生个体的水箱起点只有满
箱的约 21%**——但走一条与该文档已经证伪的路径不同的修法。`docs/water_system.md`
§3.3 的 arm_B 已经测过最直觉的修法（直接调高 `invest_min`），结果几乎无效：
`death_thirst_frac` 原地不动（0.827 vs 基线 0.825），尽管 `mean_invest` 确实从 0.352
涨到了 0.531。根因写在该文档 §3.3：`invest_frac` 是一个**同时决定能量和水两笔投资**
的单一基因（`reproduction.reproduce`："Energy and water use the same fraction:
provisioning is provisioning"），调高它的下限连带调高了能量投资，而能量投资更贵，
演化没有理由把 `mean_invest` 进一步推高——地板被调高了，但地板"绑定"的那个东西
（亲代付出的总成本）也一起变贵，净效应互相抵消。

本文档的任务是设计并实测一个**真正解耦**的版本：水的供给不再是"同一个基因乘两遍"，
而是给水单独设一个不参与演化、不影响能量投资的下限——对应真实哺乳动物的**泌乳
（lactation）**：母体给幼崽的水分不是产卵数量-质量权衡的副产品，而是一条独立的、
母体承担的生理供给通道。

标记约定沿用 `docs/biology.md`/`docs/water_system.md`：`[现实]` 已发表事实、
`[本世界实测]` 在这个代码库跑出来的数字、`[对应]` 落到哪段代码、`[提案，非结论]`
尚未验证的设计建议。

---

## 0. 一句话结论

1. **[本世界实测，6 配对种子] 逃脱了。** 用同一个数值(0.4)直接对照:
   `docs/water_system.md` arm_B 把 `invest_min`(基因自己的下限)调到 0.4,种群
   `mean_invest` 收敛到 **0.531**——落在新地板之上、被"钉"在附近。本文档的
   `water_lactation_floor_frac=0.4`(只加在水这一侧,不碰基因合法范围)下,
   `mean_invest` 收敛到 **0.372**——**低于**这个数值本身。`floor07`(地板
   0.7)下 `mean_invest` 涨到 0.419,依然只是地板的 60%。基因从未被压到地板
   附近,解耦本身按设计生效。
2. **[本世界实测] 幼体续航变长了,但死因结构没有反转。** `death_thirst_age`
   随地板强度单调上升:48.4 → 51.0(floor04)→ 56.3 步(floor07,+16.4%,配
   对 Wilcoxon p=0.031,95% bootstrap CI [+6.0, +9.6]),说明幼体确实撑得更
   久。但 `death_thirst_frac` **没有下降**,floor04 下反而以 p=0.031 显著
   *上升*了 1 个百分点(0.823→0.833),floor07 方向相同但不显著(p=0.062)。
   渴死仍是压倒性死因,只是死者死得稍晚,而不是少死。
3. **[本世界实测] 没有重演 arm_A/arm_D 那种把捕食者推向失控的副作用——方向
   甚至相反。** `population` 在两个地板强度下都与基线在噪声范围内(+2.0%,
   -1.0%,均不显著),`carnivore_frac` **反而**从 0.236 降到 0.199/0.197
   (两臂都是 -16% 量级,p=0.062,配对方向一致但未达显著),`death_predation_
   frac` 显著下降(两臂都 p=0.031)。这与 `docs/water_system.md` 的 arm_A/
   arm_D/arm_AD 全部推高 `carnivore_frac`(+31% 到 +78%)方向相反。
4. **[提案，非结论] 净判决:这是一个安全但温和的机制,不是 arm_A/arm_D 的替
   代品。** 它干净地做到了"不被演化压回地板"、"不引爆种群/捕食者比例"这两
   件事,但幼体存活的实际收益(仅 `death_thirst_age` 一侧改善,`death_thirst_
   frac` 未改善)明显小于 arm_D(水箱翻倍,同时改善两侧)。更值得的定位是:
   与 `docs/water_system.md` §3.4 提出但未实测的方案 C(失水耐受缓冲)或
   `vision_radius` 可遗传叠加使用,作为一个几乎零副作用、可以放心与其他机制
   组合的底层小补丁,而不是单独解决渴死瓶颈的答案。

---

## 1. 为什么"调高 invest_min"失败，以及这次设计哪里不同

### 1.1 [对应] arm_B 失败的精确机制

`reproduction.reproduce` 里，出生前的写法是：

```python
invest_frac = invest_of(state.genome, cfg)[parent_idx]
invest = state.energy[parent_idx] * invest_frac
water_invest = state.water[parent_idx] * invest_frac
```

`invest_frac` 是**同一个数**，同时喂给 `invest`（能量）和 `water_invest`（水）两行。
`invest_min`/`invest_span` 决定的是这一个基因的取值范围，不是"水的下限"或"能量的
下限"分开设置。`docs/water_system.md` arm_B 把 `invest_min` 从 0.2 调到 0.4，做的事
情是把这一个基因的**整个下界**上移——那一步棋同时、且以完全相同的比例，抬高了新生
个体拿到的能量。而能量投资对亲代是有实打实机会成本的：`repro_threshold=16.0`，把
更大比例的能量给孩子意味着亲代需要更久才能再次达到繁殖阈值，直接压低繁殖频率。
**演化面对的是一个耦合的优化问题**：地板调高了，但地板背后"多给一点水"这件事的
真实价格（多给的能量）也一起涨了，两股力量大致抵消，净效应在死因结构上测不出来
（`docs/water_system.md` §3.3 的原话）。

这解释了一个初看令人意外的现象：`mean_invest` 确实涨到了 0.531（远高于新地板
0.4），说明演化并没有"贴着地板走"——地板调高后种群甚至进一步上浮了一点，但这个
上浮换来的水增量被同时增加的能量支出摊薄，对幼体存活率没有留下可测的净效应。

### 1.2 [提案，非结论] 这次的设计：只解耦水，不碰能量

本文档实现的机制（`water_lactation_floor_frac`，见 §2）做一件更窄的事：

```python
invest_frac = invest_of(state.genome, cfg)[parent_idx]
invest = state.energy[parent_idx] * invest_frac                              # 不变
water_frac = jnp.clip(jnp.maximum(invest_frac, cfg.water_lactation_floor_frac), 0.0, 1.0)
water_invest = state.water[parent_idx] * water_frac                          # 新
```

关键区别：**能量那一行完全没有改动**，`invest_frac` 依然可以自由演化到它的旧下限
附近，亲代的能量成本不受这个新字段影响。水的下限 `water_lactation_floor_frac` 是
`Config` 的一个编译期常量，**不是基因、种群里没有它的遗传变异，选择无法压缩一个
不存在遗传变异的量**。这不是"调高了 `invest_min` 的水这一半"，而是把水的下限从
`invest_frac` 这条基因上完全摘下来，挂到一个演化够不着的地方。

**这正是 arm_B 和这次设计的决定性差别**：arm_B 抬高的是"能量与水共用的那个下限"，
这次抬高的是"只有水才看得到的一个独立下限"。前者连带撬动了对亲代真实有代价的能
量支出，给了演化一个"抵消"的杠杆；后者没有——亲代付出的水分成本是真实的（水箱
真的会被多抽走），但没有连带的能量代偿路径可以让"少给水"变成一个能省下能量的
策略，因为能量这条账目已经和水这条账目彻底分开算了。

### 1.3 [现实] 生物学上，这不是巧合而是历史事实：泌乳本来就是独立起源的通道

`docs/biology.md` §7.9 已经引用过这条证据，这里是它第一次被直接用来支撑一个具体的
代码设计决策：**Oftedal, O.T. (2002), "The origin of lactation and the evolution of
milk: a review with new hypotheses", *Journal of Mammary Gland Biology and
Neoplasia* 7(3): 225–252（及同期姊妹论文 7(3): 253–266）**。[摘要核实，
`docs/biology.md` 已交叉引用] 该综述系统发生学论证的核心结论是**泌乳早于胎生**：
乳腺可追溯到约 3.1 亿年前合弓纲祖先毛囊相关的顶浆分泌腺，最初的功能是往**透水的
羊皮纸质卵壳**上分泌水分与抗菌物质，营养功能是后来才被"征用"（co-opted）上去
的。现存的单孔类（鸭嘴兽、针鼹）是活证据：它们**既产卵又泌乳**——先用一套完全
独立于卵本身营养含量的分泌系统给已经产下的卵/幼体追加水分和保护,这套机制从一开
始就不是"卵里投资了多少"这道数量-质量权衡算式的一部分,而是产后追加的另一条通
道。这与本文档的设计直接同构:`invest_frac`(基因,决定"一次性打包进卵/新生儿体
内的资源比例")对应的是**产卵/妊娠**那条历史更晚的通道;`water_lactation_floor_
frac`(Config 常量,产后由亲代追加,不参与演化)对应的是**泌乳**那条历史更早、
在演化谱系上从来就没有和前者共享同一个基因位点的通道。

**这条通道的真实代价同样有据可查**,不是免费的:**Künkele, J. (2000), "Effects of
maternal food restriction on energy and time budgets of Cavia magna: evidence for
a limitation of milk energy output", *Journal of Zoology* 250(4): 533–539.**
[摘要核实] 豚鼠(*Cavia magna*)实测,妊娠期峰值摄入达基础代谢率(BMR)的
**2.4 倍**,而**哺乳期峰值达到 3.7 倍**——平均摄入增幅分别是 +16% 和 **+92%**。
**泌乳在能量上系统性地压过妊娠**,这与本文档设计里"水的追加供给对亲代是真实成
本"这一点完全吻合:提高 `water_lactation_floor_frac` 不是在凭空变出水,是在要求
亲代实打实地多付出水箱存量,这个成本不会被绕开,只是不再和能量投资绑在同一个基因
上。另见 **Clutton-Brock, T.H., Albon, S.D. & Guinness, F.E. (1989), "Fitness
costs of gestation and lactation in wild mammals", *Nature* 337: 260–262.**
[标题/作者/年份/期刊直接检索确认] 野生马鹿实测:妊娠对后续存活与繁殖的代价"与哺
乳的代价相比是轻微的"——泌乳,不是妊娠,才是哺乳类生活史里那笔真正沉重的账。

### 1.4 [提案，非结论] 安全性论证：为什么地板不会把母体自己逼死

`want = alive & (state.energy > cfg.repro_threshold)`——现有的繁殖资格判定完全不看
水量,一个水量很低的个体理论上仍可能触发繁殖。如果 `water_frac` 没有上界,一个
误配置的高地板可能让亲代把水交光,自身随即渴死。代码里做了两层防护:

1. `water_frac = jnp.clip(jnp.maximum(invest_frac, floor), 0.0, 1.0)` 把水的转移
   比例硬夹在 `[0, 1]`,亲代**永远至少保留自己当前水量的 `(1 - water_frac)`**,
   不可能被抽成负数。
2. 本次实测把地板的扫描范围限制在 `invest_min + invest_span = 0.8`(默认基因的
   理论上限)以内——0.4 和 0.7 两档都在这个范围内。这意味着这个机制施加给亲代的
   "最坏情况"转移比例,**从来没有超过现有基因本来就允许出现的取值**,亲代在
   `water_frac=0.7` 时保留 30% 自身水量,并不比一个天生 `invest_frac≈0.7` 的个体
   更危险——那种个体在基线世界里本来就存在(只是罕见,因为演化把种群压向低值)。
   换句话说,这个机制没有打开一个从未被内核验证过的新状态空间,只是让一个**已经
   被基因上限覆盖、但演化很少主动去的**区域,变得更常被访问。

---

## 2. [对应] 具体实现

`underworld/config.py` 新增字段:

```python
water_lactation_floor_frac: float = 0.0
```

默认 0.0 是严格的 no-op:`invest_frac` 恒 `>= invest_min > 0`(sigmoid 映射保证),
所以 `max(invest_frac, 0.0) == invest_frac` 对任何基因值都成立,`water_invest` 和
这个字段存在之前逐位相同。这是本项目"默认关闭的开关"惯例(`peer_channel_
enabled`、`trample_impact` 同一模式)的直接延续。

`underworld/reproduction.py::reproduce` 的改动只有两行(完整 diff 见提交记录):

```python
invest_frac = invest_of(state.genome, cfg)[parent_idx]
invest = state.energy[parent_idx] * invest_frac
water_frac = jnp.clip(jnp.maximum(invest_frac, cfg.water_lactation_floor_frac), 0.0, 1.0)
water_invest = state.water[parent_idx] * water_frac
```

不改 `in_dim`、不改 `genome_size`、不改任何数组形状——`trait_dim` 仍是 3,现有演
化种群的基因组布局不受影响,不需要重开种群。`tests/test_kernel.py` 新增两个测试:

- `test_water_lactation_floor_is_a_noop_at_default`:默认值下,`water_invest` 与
  改动前的公式(`water * invest_frac`)逐位相同。
- `test_water_lactation_floor_decouples_water_from_energy_investment`:把地板设到
  0.6、`invest_frac` 强制在 `invest_min`(约 0.2)附近时,子代拿到的**能量**依然
  精确等于 `parent_energy * invest_frac`(不受地板影响),而**水**被抬高到
  `parent_water * 0.6`,远高于 `parent_water * invest_frac` 单独给出的量——直接
  断言"两条账目已经解耦"这件事,而不只是断言"水变多了"。

---

## 3. 实测方法

6 配对种子(seed 0–5),20000 步,`scripts/run_headless.py --json`。三臂:

- **baseline**:默认配置(`water_lactation_floor_frac=0.0`,即当前代码的行为)。
- **floor04**:`water_lactation_floor_frac=0.4`——与 arm_B 的新 `invest_min`(0.4)
  同一个数值刻度,方便直接对照"同样名义强度、解耦 vs 不解耦"的差异。
- **floor07**:`water_lactation_floor_frac=0.7`——接近 §1.4 论证的安全上限
  (0.8),测试"强供给"档位是否重演 `docs/water_system.md` arm_A/arm_D 那种把
  `carnivore_frac` 推向失控的副作用。

与 `docs/water_system.md`/`docs/rebalance.md`/`docs/carnivore_riparian.md` 同一条
限制:`terrain.build(cfg)` 不吃随机数,所有种子共享同一张地图,空间性结论只对这
张河流地形成立。运行环境:RTX 4090,`XLA_PYTHON_CLIENT_PREALLOCATE=false`,18 个
`run_headless.py` 进程节流并行(本次会话额外与其他并发任务共享同一张卡,GPU 利
用率在采集期间持续 100%,单进程 20000 步耗时因此显著长于空载测得的基准,但不影
响结果的正确性,只影响墙钟时间)。

---

## 4. [本世界实测] 逐臂结果(6 配对种子,20000 步)

三臂共享同一组 founder 种子(0–5),同一张地形(见 §6.2 的限制)。下表是每臂 6 个
种子的均值±总体标准差;完整逐种子数字见 `aggregated.json`(会话 scratchpad,按
`CLAUDE.md` 约定不入库)。

| 指标 | baseline | floor04(地板 0.4) | floor07(地板 0.7) |
| --- | --- | --- | --- |
| population | 1166.3 ± 62.9 | 1190.0 ± 49.0 | 1154.2 ± 73.1 |
| min_pop | 834.8 ± 36.0 | 864.0 ± 37.8 | 818.8 ± 30.8 |
| carnivore_frac | 0.236 ± 0.027 | 0.199 ± 0.019 | 0.197 ± 0.030 |
| mean_invest | 0.358 ± 0.024 | 0.372 ± 0.022 | 0.419 ± 0.035 |
| invest_std | 0.035 ± 0.006 | 0.038 ± 0.008 | 0.046 ± 0.026 |
| mean_water | 4.237 ± 0.245 | 4.408 ± 0.360 | 4.395 ± 0.499 |
| death_thirst_frac | 0.823 ± 0.006 | 0.833 ± 0.005 | 0.835 ± 0.012 |
| death_thirst_age | 48.41 ± 1.55 | 51.03 ± 1.46 | 56.33 ± 2.79 |
| death_predation_frac | 0.123 ± 0.006 | 0.112 ± 0.005 | 0.108 ± 0.015 |
| death_predation_age | 131.7 ± 9.5 | 142.0 ± 5.3 | 130.6 ± 16.5 |

配对 Wilcoxon(vs baseline,n=6)与 95% bootstrap CI(百分位法,20000 次重抽,
差值 = 处理臂 − baseline):

| 指标 | floor04 diff [95% CI] | p | floor07 diff [95% CI] | p |
| --- | --- | --- | --- | --- |
| mean_invest | +0.013 [−0.011, +0.040] | 0.438 | **+0.060 [+0.034, +0.083]** | **0.031** |
| death_thirst_age | **+2.62 [+0.55, +5.00]** | 0.094 | **+7.92 [+6.04, +9.64]** | **0.031** |
| death_thirst_frac | **+0.010 [+0.003, +0.018]** | **0.031** | +0.013 [+0.004, +0.022] | 0.062 |
| death_predation_frac | **−0.011 [−0.019, −0.004]** | **0.031** | **−0.014 [−0.023, −0.006]** | **0.031** |
| carnivore_frac | −0.037 [−0.062, −0.014] | 0.062 | −0.039 [−0.062, −0.016] | 0.062 |
| population | +23.7 [−7.8, +57.8] | 0.312 | −12.2 [−90.3, +60.3] | 0.844 |

(n=6 配对 Wilcoxon 能达到的最小双侧 p 是 0.031——出现的 0.031 都在地板上,是
"检验能给出的最强信号",不是任意小的强证据,见 §6.1。)

### 4.1 判决 1:它逃脱了"演化压回地板"吗?

**是,而且有一个可以直接数值对照的证据。** `docs/water_system.md` arm_B 把
`invest_min`(基因自己的合法下限)从 0.2 调到 0.4,产生的 `mean_invest` 是
**0.531**——比新地板本身高 33%,是"被压回、钉在地板附近"的教科书信号。本文档
的 `floor04` 用**同一个数值** 0.4 作为水的地板(但不改 `invest_min`,基因合法
范围仍是 `[0.2, 0.8]`),结果 `mean_invest` 只有 **0.372**——**比地板本身低
7%**。同一个数字,一个是"钉在上面",一个是"仍然在下面",这正是 §1.2 论证的
"能量与水解耦"应该造成的差异。`floor07`(地板抬到 0.7,几乎是原有基因上限
0.8 的水位)下 `mean_invest` 涨到 0.419,仍然只是地板值的 60%,远没有被"拉"
到 0.7 附近。

**但确实存在一个更弱、方向相同的漂移**,不应该被掩盖:`mean_invest` 相对基线
从 0.358 涨到 0.372(floor04,不显著,p=0.438)、涨到 0.419(floor07,显著,
p=0.031)。§1.4 之外这里补一条解释:地板对**低于地板的个体**是一笔"强制多付
但不受自己基因支配"的水税——`invest_frac` 低于地板的个体,水的实际转移比例被
拉到地板,但换不来任何好处(不像自己选择高 `invest_frac` 那样,至少对应自己
"要不要多生几个更弱的孩子"这个可优化的权衡)。这创造了一个新的、比 arm_B 弱得
多的选择梯度:远低于地板的基因型要多付"白付"的水,把 `invest_frac` 提到地板
附近能省掉这笔白付——但地板抬得越高,这笔"省下的税"相对亲代总收益的分量也越
大,所以 `floor07` 的漂移(+0.060,显著)比 `floor04` 的漂移(+0.013,不显著)
更明显。**这个梯度是真实的,但和 arm_B"直接压缩基因合法范围"是两种不同强度、
不同机制的效应**——量级上,floor07 让 `mean_invest` 相对基线只涨了 17%,而
arm_B 让它涨了 51%,且 arm_B 的终点就在新地板正上方、本文档两臂的终点都明显
低于各自地板。

### 4.2 判决 2:救幼体了吗?

**部分救了——续航变长,但死因构成没有改善,是本次最值得记录的反直觉发现。**
`death_thirst_age`(渴死时的平均年龄)随地板强度单调、显著上升:48.4 → 51.0
(floor04,+5.4%,p=0.094)→ 56.3 步(floor07,+16.4%,p=0.031,95% CI 排除
零)。这是真实的个体级收益,量级与 `docs/water_system.md` arm_D(水箱翻倍,
+21%)接近,好于 arm_B(+9.5%),但明显小于 arm_A(耗水减半,+55%)。

但 `death_thirst_frac`(渴死占全部死亡的比例)**没有下降**——floor04 下以
p=0.031 的显著性**上升**了 1 个百分点(0.823→0.833),floor07 方向相同但不显
著(0.835,p=0.062)。这与 arm_A/arm_D 形成鲜明对比:那两臂**同时**压低了
`death_thirst_frac`(降到 0.44/0.73)和抬高 `death_thirst_age`,而本文档的机
制只撬动了后者。

**机制解释,呼应 `docs/mortality.md` 的竞争性风险框架**:`death_predation_
frac` 在两臂下都显著下降(p=0.031),说明捕食致死的份额在缩小。渴死和捕食致
死的份额是同一个整体的两块,总和恒为 1(减去饿死和老死的小份额);当捕食致死
的份额缩水时,即使渴死的绝对严重程度(致死年龄)在改善,渴死的**份额**仍可能
不降反升,因为"分母"里其他原因让出的比例被渴死接住了,而不是被其他改善接
住。**这次的水供给增量还不足以把足够多的个体从"52 岁死于渴"这个数量级推过
"130+ 岁死于捕食/饥饿"这个数量级**——只是把渴死本身往后推了几步,没有把整
条竞争性风险曲线的重心真正移出幼体窗口。这是一个诚实的负面/部分结果,不应该
被"age 涨了"这一个指标掩盖。

### 4.3 判决 3:有没有把 carnivore_frac/population 推失控?

**没有,方向甚至相反。** `population` 在两臂下都与基线在噪声范围内(floor04
+2.0%、floor07 −1.0%,均不显著,95% CI 均跨零),`min_pop` 同样平稳(818–864
vs 基线 835),没有任何种子出现接近崩溃的迹象。`carnivore_frac` 从 0.236 降到
0.199/0.197(两臂都约 −16%,p=0.062,配对方向一致但按 6 配对的地板未达
0.05),**没有重演** `docs/water_system.md` arm_A(+45%)、arm_D(+31%)、
arm_AD(+78%)那种把捕食者比例推向失控的副作用——如果这个 −16% 的方向性漂移
是真实效应(需要更多种子确认,见 §6.1),它甚至是在向"更接近现实基准"的方向
移动,而不是像其他三个有效杠杆那样进一步远离。`death_predation_frac` 的显著
下降(两臂都 p=0.031)与此方向一致,提示这不是纯噪声,而是种群结构一个小而
真实的偏移。

---

## 5. 与 `docs/water_system.md` 已测四臂的对照

| 臂 | 机制 | death_thirst_frac | death_thirst_age | population | carnivore_frac | 是否被压回地板 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | — | 0.825(3 种子)/ 0.823(本文档 6 种子) | 49.5 / 48.4 | 1198.7 / 1166.3 | 0.230 / 0.236 | — |
| arm_A(耗水减半) | 全体总量杠杆 | 0.444(−46%) | 76.8(+55%) | +52% | +45% | 不适用(无地板基因) |
| arm_D(水箱翻倍) | 全体总量杠杆 | 0.729(−12%) | 60.1(+21%) | +27% | +31% | 不适用 |
| arm_B(`invest_min`=0.4) | 基因下限直接调高 | 0.827(≈0%) | 54.2(+9.5%) | +3.5% | −9.6% | **是,`mean_invest`=0.531 钉在地板上方** |
| arm_AD(A+D 叠加) | 全体总量杠杆叠加 | 0.336(−59%) | 86.3(+74%) | +72% | +78% | 不适用 |
| **floor04(本文档)** | **解耦水地板 0.4** | 0.833(+1%) | 51.0(+5.4%) | +2.0% | −16% | **否,`mean_invest`=0.372 低于地板** |
| **floor07(本文档)** | **解耦水地板 0.7** | 0.835(+1.5%) | 56.3(+16.4%) | −1.0% | −16% | **否,`mean_invest`=0.419 远低于地板** |

**定位判断**:本文档的机制在"救幼体"这一栏上,效果弱于 arm_D(只改善 age 不
改善 frac,而 arm_D 两者都改善),但在"副作用"这一栏上明显更干净——是六个
可比条目里唯一让 `carnivore_frac` 方向性下降而非上升的。这符合 §1.4 的安全性
论证预期:这个机制只在个体"本来就会给得比地板少"时才追加成本,不像 arm_A/
arm_D 那样让**全体**个体(包括原本活得很好的成体)全程都更不缺水——所以它撬
动的是一个更窄的群体(低投资基因型的后代),连带效应也相应更小,不论是好的
(救幼体)还是坏的(推高捕食者占比)。

`docs/water_system.md` §4.3 曾预测"方案 C(失水耐受缓冲)可能是四选一里精确
打击幼体瓶颈、副作用最小的候选"。本文档的机制是一个不同的候选,同样以"精确
打击"为目标,但走的是供给端而非死亡判定端;两者理论上正交,**§1.2/§4.3 提示
的自然下一步是把两者叠加实测**,看这个供给地板 + 失水耐受缓冲能否互补(前者
让幼体撑得更久,后者让"归零"不再是瞬时死刑),但那需要先实现方案 C 的代码,
超出本文档范围。

---

## 6. 方法论附注

### 6.1 统计学诚实声明

6 配对种子是 `CLAUDE.md` 规定的正式结论门槛(六配对符号秩检验能达到的最小双侧
p 是 0.031,报告里出现的 0.031 是在地板上,不是强证据)。本次每臂 6 个种子、
baseline/floor04/floor07 三臂互相配对(同一组种子编号),可以做配对 Wilcoxon,但
n=6 配对功效仍然有限——`docs/conventions.md` §5 已经算过,检出 `inland_frac`
0.02 的偏移需要约 21 配对种子。本文档对 `death_thirst_frac`、`mean_invest` 这类
效应量远大于该量级的指标给出的方向性判断可信度更高,对接近噪声量级的次要指标
(如 `carn_water_dist` 的小数点后一位差异)应谨慎解读。

### 6.2 只跨了 founder 种子,没有跨地形种子

与 `docs/water_system.md` §5.2 同一条限制:6 个种子共享同一张地图,空间性结论
(`inland_frac`、`carn_water_dist` 等具体数值)严格只对这张地形成立。

### 6.3 复现

原始日志(`baseline_s{0..5}.log`、`floor04_s{0..5}.log`、`floor07_s{0..5}.log`)
在会话 scratchpad 内(`lactation/` 子目录),按 `CLAUDE.md` 约定不进仓库。

---

## 7. 参考文献

- Clutton-Brock, T.H., Albon, S.D. & Guinness, F.E. (1989). Fitness costs of
  gestation and lactation in wild mammals. *Nature* 337: 260–262.（标题/作者/
  年份/期刊直接检索确认)
- Künkele, J. (2000). Effects of maternal food restriction on energy and time
  budgets of *Cavia magna*: evidence for a limitation of milk energy output.
  *Journal of Zoology* 250(4): 533–539.（摘要核实)
- Oftedal, O.T. (2002). The origin of lactation and the evolution of milk: a
  review with new hypotheses. *Journal of Mammary Gland Biology and Neoplasia*
  7(3): 225–252（及姊妹论文 7(3): 253–266)。（摘要核实,`docs/biology.md` §7.9
  已交叉引用)

（本文档复用 `docs/water_system.md` 已核实的水系统生物学谱系文献,不重复列出;
关于自然界补水频率、失水致死阈值的完整文献列表见该文档 §1、§6。）
