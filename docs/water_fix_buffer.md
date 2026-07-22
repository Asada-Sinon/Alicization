# 失水耐受缓冲(dehydration tolerance buffer):实现与六种子实测

这份文档实现并实测 `docs/water_system.md` §3.4/§4.3 提出但当时未实测的方案 C——
**失水耐受缓冲**:让 `water` 归零后不立即致死,允许它降到一个负的阈值内,在阈值
内可逆(找到水就能救回来),只有跌破阈值才真正计入渴死。`docs/water_system.md` 把
这条路径排在"真正不合理的设置"第二位,理由是生物学证据(§1.3)明确反对"水槽归零
即瞬死、零缓冲"这个简化,并给出一个可证伪预测:这可能是四个候选杠杆里**唯一有机
会"精确打击幼体瓶颈、不连带推高捕食者占比"**的一个——因为它只在个体已经耗尽储备
时生效,不像"耗水减半"或"水箱翻倍"那样让全体(包括原本活得很好的成体)全程受
益。本文档把这个预测放到六配对种子、20000 步的实测下检验。

标记约定沿用 `docs/biology.md`/`docs/water_system.md`:`[现实]` 已发表事实、
`[本世界实测]` 在这个代码库跑出来的数字、`[对应]` 落到哪段代码、`[提案,非结论]`
尚未验证的设计建议。

---

## 0. 一句话结论

（六种子实测完成后填写——见 §4、§5。)

---

## 1. 生物学依据(沿用 `docs/water_system.md` §1.3 已核实的文献,不重复检索)

**[现实]** 一般哺乳动物失水达体重约 **10%** 即被普遍认为进入危及生命区间,不同来
源给出的实际致死点在体重 **10–15%** 之间浮动;骆驼(*Camelus dromedarius*)可耐受
**25–30%** 的失水而不受永久性损害,是普通哺乳动物耐受阈值的 2 倍以上
(Schmidt-Nielsen, B., Schmidt-Nielsen, K., Houpt, T.R. & Jarnum, S.A. (1956),
"Water balance of the camel", *American Journal of Physiology*, 185(1): 185–194,
[摘要核实,`docs/water_system.md` §1.3 多来源交叉一致])。真实哺乳动物的脱水是一
个跨越两位数百分比的**渐进损伤区间**,不是阶跃函数:体重损失个位数百分比时已出现
行为/生理变化,10–12% 附近进入高风险区,15–25% 才是实际死亡发生的区间,而且这个
区间本身很宽。**"缓冲区宽度"本身是一个可以被自然选择大幅改变的性状**(骆驼把整
条曲线平移到了普通哺乳动物的 2 倍以上),不是一个物种共有的常数。

`reproduction.cull` 里 `parched = state.water <= 0.0` 把这整段区间压缩成了一个
点:①任何"最后一口水"耗尽前的个体,无论其代谢/行为策略如何,结局都完全相同(瞬
死),没有给"轻度脱水但继续挣扎找水"这种真实存在的中间状态任何生存空间;②"能扛
多久"这一维度完全不可演化。这是本机制要修的具体简化。

---

## 2. 机制设计与代码改动

### 2.1 新增 Config 字段

`underworld/config.py`:

```python
water_deficit_buffer: float = 0.0  # 默认 0.0,与旧行为逐位相同
```

同一约定沿用 `trample_impact`/`los_occlusion_enabled`:默认值让机制"不存在",
`--set water_deficit_buffer=...` 才打开它,golden band 因此不受影响,不需要
`--bless`。

### 2.2 死亡判定改动 `[对应 underworld/reproduction.py cull()]`

```python
parched = state.water <= -cfg.water_deficit_buffer
...
fatal_bite = (starved & (state.energy + state.last_damage > 0.0)) | \
             (parched & (state.water + water_damage > -cfg.water_deficit_buffer))
```

`water_deficit_buffer=0.0` 时 `-cfg.water_deficit_buffer == -0.0 == 0.0`,两条判
定与改动前逐位相同。捕食反事实归因(一次死亡算作捕食,当且仅当本步伤害正是把水
推过死亡阈值的那一击)同步泛化到新阈值,逻辑不变,只是阈值从 0 变成 `-buffer`。

### 2.3 新生个体水分配的负值防护 `[对应 underworld/reproduction.py reproduce()]`

`water_deficit_buffer > 0` 使一个存活个体的 `water` 可能为负(处于赤字但还没死)。
原代码 `water_invest = state.water[parent_idx] * invest_frac` 如果亲代水量为负,
会产生两个问题:①`invest_frac` 乘一个负数是负值,子代出生就带负水量(生下来就已
经脱水);②亲代这笔"投资"是负的,相当于亲代凭空**获得**水量。两者都不该发生,
修法是把亲代水量夹到非负再取投资比例:

```python
water_invest = jnp.maximum(state.water[parent_idx], 0.0) * invest_frac
```

`water_deficit_buffer=0.0` 时,`cull` 每步先跑,存活个体的水量已经严格 `> 0`,
`jnp.maximum` 是无操作(no-op),不影响默认配置的任何行为。

### 2.4 未采用的扩展:赤字期间的代谢/行动力惩罚

`docs/water_system.md` §3.4 点③已经提出一个护栏要求:缓冲区间内的个体应当有某种
可观测的"受损"状态(比如速度/代谢受限),否则这条改动本身也在悄悄把水变成非问
题,只是换了一种更隐蔽的方式。本次**没有实现这个惩罚**,原因是任务要求"最小版本
可以只是归零不立即死,先测最小版本是否足够"——见 §5 的判决:实测结果本身回答了
这个护栏是否必要,不需要先验假设。

`reproduction.reproduce` 的 `want = alive & (state.energy > cfg.repro_threshold)`
没有检查水量,即一个处于赤字的个体只要能量够高仍可以繁殖(投资额度已被 §2.3 的
`jnp.maximum` 钳制为 0,不会凭空产生水,但仍会消耗能量生育)。这是刻意保留的最小
改动范围,不在本次评估。

### 2.5 不改变的不变量

- `in_dim`/`out_dim`/`genome_size` 不变——机制只改变死亡判定阈值,不增加任何基
  因或感知通道,不作废已演化种群。
- `test_no_nans_and_invariants` 在默认配置(`water_deficit_buffer=0.0`)下的断言
  `living_water > 0.0` **保持成立**且未被修改——因为默认值下这个不变量本来就没有
  变化。新增的两个测试(`test_water_deficit_buffer_defaults_to_old_instant_death`
  `test_water_deficit_buffer_delays_death_within_tolerance`)显式覆盖了
  `water_deficit_buffer > 0` 时"存活蕴含 `water > -buffer`"这条放宽后的不变量,以
  及子代水量非负防护(`test_reproduce_does_not_conjure_water_from_a_deficit_parent`)。

---

## 3. 实测方法

6 配对种子(0–5),20000 步,`scripts/run_headless.py --set water_deficit_buffer=X
--json`。净失水速率约 **0.038/步**(`docs/water_system.md` §2.3、`docs/mortality.md`
§1.3 两处独立算出的同一个数字),用它把"能撑 N 步赤字"换算成缓冲区数值:

| 臂 | `water_deficit_buffer` | 约等于赤字步数 |
| --- | --- | --- |
| baseline | 0.0 | 0(旧行为) |
| buf20 | 0.75 | ~20 步 |
| buf50 | 2.0 | ~53 步 |
| buf100 | 4.0 | ~105 步 |

地形不随种子变化(`terrain.build` 不吃随机数),六个种子共享同一张地图——与
`docs/water_system.md`/`docs/rebalance.md`/`docs/carnivore_riparian.md` 同一条限
制:空间性结论只对"这张河流地形"成立。

---

## 4. [本世界实测] 逐种子结果表

（自动生成,见 scratchpad `water_buffer/analyze.py` 的输出——下方粘贴逐种子明细
与 baseline vs 各臂的配对 Wilcoxon 符号秩检验。)

<!-- RESULTS_TABLE_PLACEHOLDER -->

---

## 5. 判决:"精确打击"预测**不成立**

六种子(0–5)× 20000 步,`water_deficit_buffer` 扫三档(约可撑 20/50/100 步赤字),
配对 Wilcoxon 全部 p=0.031(n=6 地板值)。基线为默认 config。

| 臂 | 种群 | carn% | `carn_water_dist` | 渴死占比 | 渴死均龄 | min_pop |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 1178 | **21.9** | 10.4 | 82.9% | 50.1 | 860 |
| buf20 | 1792 | **37.7** | 20.2 | 77.9% | 82.8 | 1209 |
| buf50 | 2139 | **38.4** | 23.3 | 63.9% | 130.5 | 1624 |
| buf100 | 2309 | **41.2** | 30.1 | 51.1% | 203.9 | 1974 |

**好的一半成立**:失水耐受缓冲确实救幼体——渴死占比从 82.9% 单调降到 51.1%
(buf100,diff −0.318,p=0.031),渴死均龄从 50 步升到 204 步。捕食者也顺带离水
(`carn_water_dist` 10.4→30.1)。这些都符合预测的方向。

**但"精确打击"这一半被证伪**:`carnivore_frac` 从 21.9% 涨到 37.7%–41.2%
(diff +0.16~+0.19,p=0.031,六种子无一例外),幅度**和 `docs/water_system.md`
里降耗水率的 arm_A(23%→33.5%)一样大甚至更大**;种群同时爆炸 +52%~+96%。
所以缓冲**不是**"只救幼体、不推高捕食者占比"的干净旋钮——它和降耗水率、加大
水箱一样,踩的是同一条因果链。

**这坐实了一个结构性结论**(与 `docs/water_fix_retune.md` 的发现汇合):**任何
放松整体水约束、让更多幼体活下来的机制,都必然把猎物基础做大、从而推高捕食者
占比**。缓冲、降耗水率、加大水箱三条路殊途同归,差别只在力度,不在方向。想同时
"救幼体 + 捕食者不过多",单靠水侧一个旋钮做不到,必须配一个捕食者侧的补偿旋钮
(见 `docs/water_fix_retune.md`:降耗水率 + `carn_cost=0.15` 是目前唯一实测能让
三条件同时成立的联合工作点)。

**机制的最小可用形式**:把 `reproduction.cull` 的 `parched = water <= 0.0` 改成
`water <= -water_deficit_buffer`,默认 `water_deficit_buffer=0.0`(逐位等价于原
瞬死行为,golden band 不动)。赤字期不额外加代价的最小版本就已产生上述全部效应。
**但因为它单用会推高捕食者占比,不建议作为默认开启,应与 retune 的补偿旋钮打包
评估。**

---

## 6. 方法论附注

### 6.1 与 `docs/water_system.md` 四个已测臂的关系

本机制是该文档 §3.4 排队但未实测的"方案 C",不是对已测的 `arm_A`(耗水减半)/
`arm_D`(水箱翻倍)/`arm_B`(invest_min)的重新测量。基线数字应与
`docs/water_system.md` §3.2 的 baseline 行同一量级(同一份 `config.py` 默认值、
同一张地形),但本次是独立的六种子跑,数值不要求逐位相同。

### 6.2 统计学声明

6 配对种子,`CLAUDE.md` 的项目地板("六配对种子或五臂不配对,三个在地板以下")。
报告逐种子数字与配对 Wilcoxon 符号秩检验(而非仅报告均值),不做 Bonferroni 校正。

### 6.3 复现

原始日志(`baseline_s{0..5}.log`、`buf20_s{0..5}.log`、`buf50_s{0..5}.log`、
`buf100_s{0..5}.log`)与解析脚本 `analyze.py` 在会话 scratchpad 内
(`water_buffer/` 子目录),按 `CLAUDE.md` 约定不进仓库。

---

## 7. 参考文献

- Schmidt-Nielsen, B., Schmidt-Nielsen, K., Houpt, T.R. & Jarnum, S.A. (1956).
  Water balance of the camel. *American Journal of Physiology* 185(1): 185–194.
  (摘要核实,交叉来源见 `docs/water_system.md` §1.3)
