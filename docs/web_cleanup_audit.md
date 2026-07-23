# web/ 前端清理审计

一次针对 `web/main.js`、`web/render.js`、`web/index.html` 三个文件的纯清理与
性能审计。目标是**行为与视觉零变化**——这不是重设计。范围：死代码、重复逻辑、
啰嗦的 DOM/渲染代码、过期注释、每帧可提出循环的重复计算、无必要的 GL 状态查询。

红线（`CLAUDE.md` 的 cross-file 契约，碰了会静默出错）全程未碰：`parse()` 里每个
`getFloat32` 偏移量、`HEADER_BYTES`、terrain 的 `UNTR` magic 判别、三处重复的
`herb`/`carn` 物种颜色（`check.py --contracts` 校验，改动后仍 21 项全过）。

## 1. 结论先行

代码库维护得相当好，**没有找到死代码**（见 §4 的排查清单）。真正落地的改动是两类
性能点加两处过期注释，全部行为中性：

| 改动 | 文件 | 类别 | 原因 |
| --- | --- | --- | --- |
| 缓存 GL attribute/uniform location | `render.js` | 性能 | 每帧省掉 ~14 次字符串式 `getUniformLocation`/`getAttribLocation` 驱动查询 |
| 采样器 uniform 只在 init 绑定一次 | `render.js` | 性能 | `u_plant`=0 / `u_terrain`=1 永不变，无需每帧重设 |
| 缓存渲染循环触碰的 DOM 元素 | `main.js` | 性能 | `tick()`+`drawAllSparks()` 每帧省掉 ~22 次 `getElementById` |
| 修正 `--plant` 过期注释 | `index.html` | 注释 | 原注释声称等于一个已不存在的 `render.js` 常量 |
| 修正 `C` 颜色块注释 | `main.js` | 注释 | 原注释把所有 6 个颜色都说成"锁定到 shader"，实际只有 `herb`/`carn` 是 |

## 2. 渲染性能点（`render.js`）[对应 `web/render.js`]

**每帧的 location 查询。** `Renderer.draw()` 原本每帧对两个 program 各做一批
`gl.getUniformLocation(prog, "name")` / `gl.getAttribLocation(prog, "name")`——共 14
次按字符串名的驱动侧查询。这些 location 在 program **link 之后就固定不变**，属于典型
的"每帧重复计算能提出循环的量"。

修法：`init()` 里 link 完两个 program 后，一次性把全部 location 解析进
`this.plantLoc` / `this.pointLoc` 两个对象，`draw()` 直接读缓存。

**采样器只绑一次。** `u_plant`（TEXTURE0）与 `u_terrain`（TEXTURE1）的纹理单元号是
常量，原来每帧 `gl.uniform1i` 重设两次。改为 `init()` 里 `useProgram(plantProg)` 后
绑定一次即可——采样器 uniform 是 program 状态的一部分，不随 `useProgram` 复位。

未改动 `u_texel`（随地形变，但只在 `draw` 里按 `1/terrainGrid` 算，成本可忽略）与
每帧必变的矩阵/hover/sizeRef uniform——这些本就该每帧写。

**未碰 GLSL。** 两段 shader 源码一字未动，contract 里"append never insert"的
wire 偏移、terrain magic 均无关。尽管如此仍按 `docs/conventions.md` §10 做了看的验证
（见 §5）。

## 3. DOM 热路径（`main.js`）[对应 `web/main.js`]

`tick()`（每帧刷统计数字）与 `drawAllSparks()`（每帧重绘六条 sparkline）合计每帧调
`document.getElementById` 约 22 次。DOM 是静态的——`reset` 不重建节点，只清 history 与
inspector——所以把这些元素在 IIFE 启动时一次性解析进 `ui` 对象是安全的。命名用 `ui`
而非 `el`，避免与 `drawSparkline(el, …)`、`drawRetina`/`clearInspector` 里的局部
`el`（都指 `#retina`/sparkline canvas）撞名。

未把 inspector 面板里那 ~18 个元素也缓存——它们只在点选个体（收到 JSON 帧）时更新，
不在每帧热路径上，缓存收益为零、只增噪声。

## 4. 排查过但未改（无死代码）

逐项确认均在用，故保留：

- `main.js`：`C` 的全部 6 个键（`herb`/`carn` 画点例图与 inspector，`omni` 杂食标签，
  `halo`/`plant` sparkline，`rule` 基线/刻度）；`SPEEDS`/`STRIDE`/`PICK_RADIUS`/`HIST`
  常量；`hist` 七条序列全部被 `pushHistory` 写、被 `drawAllSparks` 读；每个函数都有调用点。
- `render.js`：`Mat4` 六个方法（`create`/`perspective`/`lookAt`/`multiply`/`invert`/
  `transformPoint`）全部被相机、picking 或 `canvasFromWorld` 用到；`HEIGHT_SCALE_FRAC`/
  `AGENT_HOVER_FRAC`/`FOVY`/`NEAR` 与全部相机调参常量均被引用；`_uintIndicesExt`、
  `terrainIndexType`（>256 grid 的 32 位索引路径）是活的降级逻辑。
- `foodBuf`/`_hBuf`/`_camInitialized` 的重分配/初始化守卫都正确，未重复分配。

`drawSparkline` 里 `xOf = (i, n)` 的 `n < 2 ? 0` 分支在所有调用点其实都已被
`if (n < 2) continue/return` 挡在外面，属冗余但无害的防御，未动。

## 5. 视觉验证（`docs/conventions.md` §10）[本世界实测]

`render.js` 属渲染路径，即便改动是行为中性的 location 缓存，也按约定用眼睛验了，没有
只靠读代码。做法：一个独立 headless 夹具（scratchpad，不入库）`Renderer.init` 一块
canvas，喂合成地形（对角高斯山脊 + 蜿蜒河带 + 森林带 + 植物/果实场）与 400 个随机
agent（食性 0..1、能量 4..12），连画两帧走一遍缓存 location 的 `draw` 路径，再用
`google-chrome --headless=new --use-gl=angle --use-angle=swiftshader
--enable-unsafe-swiftshader` 截图。

结果符合预期：位移后的地形网格有真实浮雕（山脊亮、山谷暗）、蓝色河流蜿蜒、森林带更
深冷、agent 点按食性从紫（草食）到红（肉食）着色、点大小随能量变化、透视相机正确。
即渲染与改动前一致，缓存 location 未改变任何输出。

## 6. 机械检查

- `node --check web/main.js && node --check web/render.js` —— 通过。
- `check.py --contracts` —— 21 项全过（wire 偏移、terrain magic、`herb`/`carn` 三处
  颜色一致性均不受影响）。
