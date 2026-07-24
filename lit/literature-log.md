# 文献台账

**本文件当前是空表。** 下面只有格式说明；真实文献行追加到表头下面。

一篇一行。这张表只做索引，**细读笔记单独放 `notes/papers/<citekey>.md`**，结论的权威成品
永远在 `docs/<topic>.md`。三者的分工见 `notes/papers/README.md` 的「流向」一节。

## 这张表要防的两件事

它存在的理由不是「收藏论文」，而是两个已经在本项目发生过的失败模式：

- **读了但没用上。** 一次检索的残留（仓库根那个未跟踪的 `oa1.json`，OpenAlex `/works`
  的原始响应，query 是 `Desiccation resistance water balance Circellium bacchus`）没有任何
  地方记录它查的是什么、结论是什么、有没有并进 `docs/`。有这张表，那次检索要么变成几行
  台账，要么当场被判「不相关，未入档」。
- **用了但没台账。** `docs/biology.md` 是 51k 的成品综述，引用嵌在论证里，没有一处能一眼
  扫出「这篇读过没有、当初查的什么 query、结论进了哪几份文档」。两次引用事故（大象幼崽
  死亡率的统计层级错位、虎鲸「约 8 倍」无文献支持后更正为 14×/3×）都发生在这个盲区里。

所以「已并入 `docs/`」那一列是本表最重要的一列：**它是唯一能一眼看出一篇论文有没有真正
落地的地方。**

## 列的规矩

列顺序固定，不要增删或调换前六列：

`citekey | 标题 | 年份 | 状态 | 与本项目的关系 | 笔记路径 | 已并入 docs/`

- **citekey**：`<第一作者姓小写><年份><标题第一个实词小写>`，与 `notes/papers/<citekey>.md`
  的文件名和 frontmatter 严格一致。一旦写进 `docs/` 就不要再改。
- **标题**：原文标题，不翻译。后面用括号附 DOI 或 arXiv ID——**必须来自真实检索**，
  凭记忆写等于伪造。
- **状态**：`待读` / `略读` / `精读` / `已并入` / `已弃`（写清为什么弃）。
- **与本项目的关系**：写实质关联——支持了哪条设计判断 / 是哪个参数的依据 / 结论与本项目
  某条假设冲突。**不要写「相关」**。
- **笔记路径**：`notes/papers/<citekey>.md`。还没写笔记就写「未写」，那本身是一条待办。
- **已并入 `docs/`**：写到小节级（例：`docs/biology.md` 的「亲代照料」）。**没有并入就写
  「未并入」并说明为什么**（不相关 / 只作背景 / 等某个实验做完再决定）。留空不算填。

「已并入」有值，说明那条结论已经按 `[现实]` 标签活在 `docs/` 里了——`CLAUDE.md`：
「Research lands in `docs/`, or it did not happen」。一直是「未并入」的行，要么它确实不
相关（好结论，写清楚），要么它是一笔欠账。

| citekey | 标题 | 年份 | 状态 | 与本项目的关系 | 笔记路径 | 已并入 docs/ |
| --- | --- | --- | --- | --- | --- | --- |
| colosimo2005 | Widespread Parallel Evolution in Sticklebacks by Repeated Fixation of Ectodysplasin Alleles（10.1126/science.1107239） | 2005 | 略读 | 护甲可作单基因可逆开关，支持护甲做低维可遗传性状 | notes/papers/colosimo2005.md | `docs/trait_defense_catalog.md` 护甲/棘刺节 |
| barrett2008 | Natural Selection on a Major Armor Gene in Threespine Stickleback（10.1126/science.1159978） | 2008 | 精读 | 护甲代价=生长拖累的野外实测，喂本项目代价货币原则；也是「护甲在本世界是 A 还是 B 类取决于代价记哪本账」这条张力的出处 | notes/papers/barrett2008.md | `docs/trait_defense_catalog.md` 护甲节 + `docs/trait_addition_feasibility.md` §B.2 |
| harvell1990 | The Ecology and Evolution of Inducible Defenses（10.1086/416841） | 1990 | 略读 | 诱导 vs 组成型判据；本项目昼夜波动满足「风险波动」 | notes/papers/harvell1990.md | `docs/trait_defense_catalog.md` 诱导型防御节 |
| tollrianharvell1999 | The Ecology and Evolution of Inducible Defenses（book, ISBN 9780691004945 / 10.1515/9780691228198） | 1999 | 待读 | 诱导型防御框架专著，代价证据来源 | 未写 | `docs/trait_defense_catalog.md` 诱导型防御节（作背景锚点） |
| tollrian1993 | Neckteeth formation in Daphnia pulex...（10.1093/plankt/15.11.1309） | 1993 | 略读 | 诱导强度随风险连续标定的实证 | notes/papers/tollrian1993.md | `docs/trait_defense_catalog.md` 诱导型防御节 |
| hoogland1956 | The Spines of Sticklebacks as Means of Defence against Predators（10.1163/156853956X00156，DOI 后缀未独立确认） | 1956 | 略读 | 反击型防御经典，对应「防御压低捕食者密度」 | notes/papers/hoogland1956.md | `docs/trait_defense_catalog.md` 反击/尖刺节 |
| mappes2005 | The complex business of survival by aposematism（10.1016/j.tree.2005.07.011） | 2005 | 待读 | 警戒色+化学防御综述；依赖捕食者学习 | 未写 | `docs/trait_defense_catalog.md` 警戒色节 |
| stevens2009 | Animal camouflage: current issues and new perspectives（10.1098/rstb.2008.0217） | 2009 | 待读 | 隐蔽色机制分类，作用于「被检测」阶段 | 未写 | `docs/trait_defense_catalog.md` 隐蔽色节 |
| cuthill2005 | Disruptive coloration and background pattern matching（10.1038/nature03312） | 2005 | 待读 | 破坏性色斑降低鸟类捕食的野外实验证据 | 未写 | `docs/trait_defense_catalog.md` 隐蔽色节 |
| hamilton1971 | Geometry for the selfish herd（10.1016/0022-5193(71)90189-5） | 1971 | 略读 | 稀释效应理论基石，行为性状可由现有邻居系统承载 | notes/papers/hamilton1971.md | `docs/trait_defense_catalog.md` 集群稀释节 |
| vermeij1994 | The evolutionary interaction among species: selection, escalation, and coevolution（10.1146/annurev.es.25.110194.001251） | 1994 | 待读 | escalation vs coevolution 框架，军备竞赛背景 | 未写 | `docs/trait_defense_catalog.md` 军备竞赛节 |
| brodie2005 | Parallel arms races between garter snakes and newts...（10.1007/s10886-005-1345-x） | 2005 | 略读 | 双侧协同演化真实范例，补 attack_range_redqueen 背景 | notes/papers/brodie2005.md | `docs/trait_defense_catalog.md` 军备竞赛节（交叉引用 `docs/attack_range_redqueen.md`） |
| hague2018 | Large-effect mutations generate trade-off between predatory and locomotor ability...（10.1002/evl3.76） | 2018 | 略读 | 军备升级的运动代价实测，支持给对抗性状挂税 | notes/papers/hague2018.md | `docs/trait_defense_catalog.md` 军备竞赛节 |
| kastner2024 | Gape-limited invasive predator frequently kills avian prey too large to swallow（10.1002/ece3.11598） | 2024 | 待读 | size refuge 机制佐证（非综述、强度弱） | 未写 | `docs/trait_defense_catalog.md` 体型/速度节（标 gap） |

## 检索记录

台账记「哪篇论文」，这一节记「哪次检索」——**没有命中的检索同样要记**，否则下一个 session
会把同一条 query 再跑一遍。一行一次：日期 / 实际用过的 query / 工具 / 命中几篇入档几篇 /
结论。

| 日期 | query | 工具 | 命中 → 入档 | 结论 |
| --- | --- | --- | --- | --- |
| 2026-07-24 | stickleback Eda lateral plate armor predation Colosimo | WebSearch | 8 → 2 | Colosimo2005 定位 + Barrett2008 代价，护甲节双锚点 |
| 2026-07-24 | Tollrian Harvell inducible defenses cost / Harvell 1990 QRB / Daphnia neckteeth kairomone | WebSearch×3 | 多 → 3 | Harvell1990 判据 + Tollrian1993 剂量依赖 + 1999 专著 |
| 2026-07-24 | Hamilton 1971 selfish herd JTB | WebSearch | 9 → 1 | Hamilton1971 DOI 确认 |
| 2026-07-24 | Brodie garter snake newt TTX / Hague 2018 trade-off | WebSearch×2 | 多 → 2 | Brodie2005 界面 + Hague2018 运动代价 |
| 2026-07-24 | Mappes Marples Endler aposematism TREE | WebSearch | 9 → 1 | Mappes2005 DOI 确认 |
| 2026-07-24 | Vermeij escalation / 1994 Annu Rev | WebSearch×2 | 多 → 1 | Vermeij1994 DOI 确认；1987 专著无 DOI |
| 2026-07-24 | Stevens Merilaita camouflage / Cuthill disruptive coloration Nature | WebSearch×2 | 多 → 2 | Stevens2009 综述 + Cuthill2005 实验 |
| 2026-07-24 | Hoogland Tinbergen stickleback spines defence Behaviour | WebSearch + WebFetch(Brill 403) | 9 → 1 | venue 确认，DOI 后缀未独立确认 |
| 2026-07-24 | body size refuge gape-limited escape speed antipredator review | WebSearch | 8 → 1 | 无综述锚点，仅零散实证（kastner2024），标 gap |

<!--
示例行（真实检索请照这个粒度写，不要照抄这两行的内容——它们只是格式演示）：

| 2026-07-23 | "piosphere grazing gradient distance to water" | WebSearch + 期刊官网 | 6 → 2 | 两篇进 docs/biology.md 第 1 节；其余是同一综述的转述 |
| 2026-07-23 | "dung beetle desiccation resistance water balance" | OpenAlex /works | 9 → 0 | 与本项目无实质关联（本项目没有昆虫水生理机制），未入档 |
-->
