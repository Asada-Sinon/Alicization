"""把两份 `map_*.json` 打成 `map.html`——**在地图上**看两个生态型分别待在哪。

与 `replay.html` 的分工，以及为什么不合成一页
--------------------------------------------
`replay.html` 画的是 `forage_pref` 直方图，回答「有没有裂成两群」；本页画的是
**个体在地图上的位置**，回答「那两群各自待在哪」。

**两页的数据来自不同的 run，所以不能共用一根时间轴。** `replay.html` 的无捕食面板
取自 350k 步、跑到第 717 代的 `R38n_s0_r1`（那是「维持数百代」最有说服力的一条），
而地图数据是本轮新录的 100k 步（约 170 代）。把两者摆在同一根滑杆下，
世代跨度差四倍，看起来像同一时刻其实不是——所以分成两页，每页内部自洽。

本页的直方图**由同一帧的 800 个抽样个体现算**，不读 `traj`，
因此地图与直方图必然是同一时刻的同一批个体。

数据口径
--------
每帧 800 个体（`record_map.py --cap`，均匀抽样），坐标 uint16、性状 uint8，
base64 编码。两臂除 `diet_delta` 外配置完全相同，与 R20 的 `wn1off` / `wn1on` 一致。

底图为什么在这里现算
--------------------
`record_map.py` 只存了高程与可饮水两层，**而这一页要回答的是「吃果子的是不是聚在
林子边」——没有「果子长在哪」这一层，图上就只是一堆点**（第一次截图验收才看出来的，
读代码看不出来）。

但不必重录：**`terrain.build(cfg)` 不吃随机数**，同一份配置必然生成同一张地形，
所以 `capacity`（草的承载量）与 `fruit_capacity`（果的承载量，只长在林冠下）
在这里按 `record_map.WORLD` 现算即可，已经录好的数据照用。
代价是本脚本要导入 JAX（CPU 上跑一次 `build`，约两秒）。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260816-replay/make_map_page.py
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

HERE = os.path.dirname(__file__)
PANELS = [
    {"key": "nopred", "title": "没有捕食者",
     "note": "两种吃法都在，看它们各自待在哪"},
    {"key": "pred", "title": "有捕食者",
     "note": "同一个世界，只多了捕食者"},
]

HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>两种吃法的食草者，各自待在地图的哪里</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;
      --fruit:#e8a33d;--grass:#3fb950}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.6 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:28px 20px 40px}
h1{font-size:22px;margin:0 0 6px;font-weight:650}
.sub{color:var(--dim);margin:0 0 22px;font-size:14px}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.panels{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.ptitle{font-weight:650;margin-bottom:2px}
.pnote{color:var(--dim);font-size:13px;margin-bottom:12px;min-height:20px}
canvas.map{width:100%;aspect-ratio:1;display:block;background:#0b0f14;border-radius:6px}
canvas.hist{width:100%;height:64px;display:block;margin-top:10px;background:#0b0f14;
            border-radius:6px}
.stats{display:flex;gap:16px;margin-top:10px;font-size:13px;flex-wrap:wrap}
.stat b{font-variant-numeric:tabular-nums;font-size:16px}
.fruit{color:var(--fruit)} .grass{color:var(--grass)} .dim{color:var(--dim)}
.controls{display:flex;align-items:center;gap:14px;margin:24px 0 8px}
button{background:#21262d;color:var(--fg);border:1px solid var(--line);border-radius:6px;
       padding:8px 18px;font-size:14px;cursor:pointer}
button:hover{background:#30363d}
input[type=range]{flex:1;accent-color:var(--fruit)}
.axis{display:flex;justify-content:space-between;color:var(--dim);font-size:12px;margin-top:4px}
.legend{display:flex;gap:20px;margin-top:18px;font-size:13px;color:var(--dim);flex-wrap:wrap}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;
    vertical-align:-1px}
.foot{margin-top:26px;padding-top:16px;border-top:1px solid var(--line);
      color:var(--dim);font-size:13px}
.foot code{background:#21262d;padding:1px 5px;border-radius:4px;font-size:12px}
</style></head><body><div class="wrap">
<p style="margin:0 0 18px"><a href="index.html" style="color:#8b949e;text-decoration:none;font-size:14px">← 回到总览</a></p>
<h1>两种吃法的食草者，各自待在地图的哪里</h1>
<p class="sub">每个点是一只活着的个体：<b>绿到琥珀</b>的是食草者（颜色表示它偏爱吃草还是吃果子），
<b>红的是食肉者</b>。
底图是这个世界本身：<b>暗琥珀的地方长果子</b>（只长在林冠下）、绿的地方长草、蓝的是水。
所以「吃果子的有没有聚在长果子的地方」可以直接看出来。拖动进度条看它随世代变化。
下方的小直方图由同一帧的同一批个体现算，因此地图和直方图永远同步。</p>

<div class="panels" id="panels"></div>

<div class="controls">
  <button id="play">▶ 播放</button>
  <input type="range" id="slider" min="0" max="1000" value="0">
</div>
<div class="axis"><span>开局</span><span id="gnow"></span><span>最后</span></div>

<div class="legend">
  <span><i class="sw" style="background:var(--fruit)"></i>偏爱果子</span>
  <span><i class="sw" style="background:var(--grass)"></i>偏爱草</span>
  <span><i class="sw" style="background:#f24038"></i>食肉者</span>
  <span><i class="sw" style="background:rgb(85,65,34)"></i>果子长在这（林冠下）</span>
  <span><i class="sw" style="background:rgb(27,75,47)"></i>草长在这</span>
  <span><i class="sw" style="background:rgb(32,74,138)"></i>水</span>
  <span class="dim">两边按<b>步数比例</b>对齐（都是 10 万步）。注意两边的<b>世代数不同</b>
        ——有捕食者那边代际更替快约三倍，同样的步数它已经繁衍了三倍的世代。</span>
</div>

<div class="foot">
数据是已跑完的模拟，不是实时演算。每帧从存活个体里<b>均匀抽 800 个</b>存下来
（画布上已经很密），显示的「存活」是抽样前的真实数。两臂除 <code>diet_delta</code>
外配置完全相同，与 <code>docs/multispecies_feasibility.md</code> §27 的
<code>wn1off</code> / <code>wn1on</code> 两臂一致。直方图的分界与判决同口径：
<code>forage_pref &lt; 0.35</code> 记为果专精侧，
且<b>所有比例都只在食草者里算</b>（<code>diet &lt; 0.35</code>，与
<code>trajectory.py</code> 判决用的口径一致）——食肉者的 <code>forage_pref</code>
没有意义，混进去会把这个数算错。食性介于两者之间的个体画成灰点，两边都不计入。
</div>
</div>
<script>
const DATA = __DATA__;
const NB = 20, LO_CUT = 0.35;
// 与判决同口径：`trajectory.py` 的 `herb = alive & (diet < 0.35)`，
// 食肉者判定沿用 `metrics.py` 的 `is_carn = diet > 0.65`。
// **食草者以外的个体绝不能进直方图**——它们的 `forage_pref` 没有意义。
const HERB_MAX = 0.35, CARN_MIN = 0.65;
const CARN_COLOR = '#f24038';   // 与 web/render.js 的食肉者色一致
function unb64(s){ const b=atob(s), u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++) u[i]=b.charCodeAt(i); return u; }
function u16(s){ const u=unb64(s); return new Uint16Array(u.buffer, u.byteOffset, u.length/2); }
// 琥珀(偏果) → 绿(偏草)
function ramp(t){ const a=[232,163,61], b=[63,185,80];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`; }

const P = DATA.panels.map(p=>{
  // 地形只解一次，预渲染成离屏画布
  const g = p.terrain.grid, H = unb64(p.terrain.height), W = unb64(p.terrain.water);
  const CAP = unb64(p.terrain.capacity), FCAP = unb64(p.terrain.fruit_capacity);
  const off = document.createElement('canvas'); off.width=g; off.height=g;
  const ictx = off.getContext('2d'), img = ictx.createImageData(g,g);
  for(let i=0;i<g*g;i++){
    const h=H[i]/255, c=CAP[i]/255, fc=FCAP[i]/255;
    // 底图叠三层：草的承载量→暗绿，果的承载量→暗琥珀，高程→一点浮雕。
    // **都压得很暗**，因为个体的点要画在上面还得看得清（亮度差是主要区分手段）。
    let r = 13 + c*14 + fc*118 + h*26;
    let gg = 17 + c*58 + fc*66 + h*24;
    let bb = 21 + c*26 + fc*10 + h*20;
    if(W[i]){ r=32; gg=74; bb=138; }
    img.data[i*4]=r; img.data[i*4+1]=gg; img.data[i*4+2]=bb; img.data[i*4+3]=255;
  }
  ictx.putImageData(img,0,0);
  // 每帧的性状先解一次并缓存——拖进度条时不要反复 atob
  const frames = p.frames.map(f=>({
    t:f.t, g:f.g, n:f.n,
    x:u16(f.x), y:u16(f.y), p:unb64(f.p), d:unb64(f.d),
  }));
  return {title:p.title, note:p.note, terr:off, tg:g, frames, world:p.world, fcap:FCAP};
});

const host = document.getElementById('panels');
const maps=[], hists=[], els=[];
P.forEach((p,i)=>{
  const d = document.createElement('div'); d.className='panel';
  d.innerHTML = `<div class="ptitle">${p.title}</div><div class="pnote">${p.note}</div>
    <canvas id="m${i}" class="map"></canvas>
    <canvas id="h${i}" class="hist"></canvas>
    <div class="stats">
      <span class="stat dim">第 <b id="g${i}">–</b> 代</span>
      <span class="stat fruit">偏果的占 <b id="lo${i}">–</b></span>
      <span class="stat dim">存活 <b id="n${i}">–</b></span>
      <span class="stat" style="color:#f24038">食肉者 <b id="c${i}">–</b></span>
    </div>
    <div class="stats" style="margin-top:4px">
      <span class="stat dim">站在果子地上的比例：<b id="of${i}" class="fruit">–</b>
        <span class="dim">(偏果)</span> · <b id="og${i}" class="grass">–</b>
        <span class="dim">(偏草)</span></span>
    </div>`;
  host.appendChild(d);
  maps.push(document.getElementById('m'+i));
  hists.push(document.getElementById('h'+i));
  els.push({g:document.getElementById('g'+i), lo:document.getElementById('lo'+i),
            n:document.getElementById('n'+i), c:document.getElementById('c'+i),
            of:document.getElementById('of'+i),
            og:document.getElementById('og'+i)});
});

function fit(){ [...maps,...hists].forEach(c=>{ const r=c.getBoundingClientRect(),
  dpr=devicePixelRatio||1; c.width=r.width*dpr; c.height=r.height*dpr; }); draw(cur); }

function draw(frac){
  P.forEach((p,i)=>{
    const idx = Math.min(p.frames.length-1, Math.round(frac*(p.frames.length-1)));
    const f = p.frames[idx];
    const c = maps[i], ctx = c.getContext('2d'), Wd=c.width, Hd=c.height;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(p.terr, 0, 0, Wd, Hd);
    // 个体：点大小随画布，偏果的画在上层，免得被绿点盖住看不出聚集
    const rad = Math.max(1.6, Wd/260);
    // 画序：偏草的最先、偏果的其次、食肉者最后（画在最上层）。
    // 食肉者数量少但要看得见，食草者里偏果的少、要能看出聚集。
    const order = Array.from(f.p.keys()).sort((a,b)=>{
      const ca = f.d[a]/255 > CARN_MIN, cb = f.d[b]/255 > CARN_MIN;
      if(ca !== cb) return ca ? 1 : -1;
      return f.p[b]-f.p[a];
    });
    for(const k of order){
      const dv = f.d[k]/255;
      if(dv > CARN_MIN){                       // 食肉者：红点，稍大
        ctx.fillStyle = CARN_COLOR; ctx.globalAlpha = 0.95;
        ctx.beginPath();
        ctx.arc(f.x[k]/65535*Wd, f.y[k]/65535*Hd, rad*1.35, 0, 6.2832);
        ctx.fill();
      } else if(dv < HERB_MAX){                // 食草者：按觅食偏好上色
        ctx.fillStyle = ramp(f.p[k]/255); ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.arc(f.x[k]/65535*Wd, f.y[k]/65535*Hd, rad, 0, 6.2832);
        ctx.fill();
      } else {                                 // 中间食性：灰点，不参与任何统计
        ctx.fillStyle = '#6e7681'; ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(f.x[k]/65535*Wd, f.y[k]/65535*Hd, rad*0.8, 0, 6.2832);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    // 同一帧现算直方图
    const h = new Array(NB).fill(0); let lo=0, nherb=0, ncarn=0;
    for(let k=0;k<f.p.length;k++){
      const dv=f.d[k]/255;
      if(dv > CARN_MIN){ ncarn++; continue; }
      if(dv >= HERB_MAX) continue;              // 中间食性：两边都不算
      nherb++;
      const v=f.p[k]/255;
      h[Math.min(NB-1, Math.floor(v*NB))]++; if(v<LO_CUT) lo++;
    }
    const hc = hists[i], hx = hc.getContext('2d'), HW=hc.width, HH=hc.height;
    hx.clearRect(0,0,HW,HH);
    const pk = Math.max(...h, 1), pad=4*(devicePixelRatio||1), bw=(HW-pad*2)/NB;
    for(let k=0;k<NB;k++){
      hx.fillStyle = ramp(k/(NB-1)); hx.globalAlpha=0.92;
      const bh=(h[k]/pk)*(HH-pad*2);
      hx.fillRect(pad+k*bw+bw*0.12, HH-pad-bh, bw*0.76, bh);
    }
    hx.globalAlpha=1;
    // **肉眼分不清「偏果的聚在果子地」和「所有个体都得靠近水」**——两者在图上重叠，
    // 所以直接算：两侧各有多大比例正站在长果子的格上。差值才是证据，单看偏果那个数没意义。
    const G = p.tg, FC = p.fcap;
    let nf=0, nfOn=0, ng=0, ngOn=0;
    for(let k=0;k<f.p.length;k++){
      const dv = f.d[k]/255;
      if(dv >= HERB_MAX) continue;              // 食肉者与中间食性都不算
      const v = f.p[k]/255;
      const ix = Math.min(G-1, Math.floor(f.x[k]/65536*G));
      const iy = Math.min(G-1, Math.floor(f.y[k]/65536*G));
      const on = FC[iy*G+ix] > 25;          // >≈10% 的果层峰值，即「这格确实长果子」
      if(v < LO_CUT){ nf++; if(on) nfOn++; }
      else if(v > 1-LO_CUT){ ng++; if(on) ngOn++; }
    }
    els[i].of.textContent = nf ? (nfOn/nf*100).toFixed(0)+'%' : '—';
    els[i].og.textContent = ng ? (ngOn/ng*100).toFixed(0)+'%' : '—';
    els[i].g.textContent = f.g.toFixed(0);
    els[i].lo.textContent = nherb ? (lo/nherb*100).toFixed(1)+'%' : '—';
    els[i].n.textContent = f.n.toLocaleString();
    els[i].c.textContent = (ncarn/f.p.length*100).toFixed(1)+'%';
  });
  const f0 = P[0].frames[Math.min(P[0].frames.length-1, Math.round(frac*(P[0].frames.length-1)))];
  document.getElementById('gnow').textContent = (f0.t/1000).toFixed(0)+'k 步';
}

let cur=0, timer=null;
const slider=document.getElementById('slider'), play=document.getElementById('play');
slider.addEventListener('input', ()=>{ cur=slider.value/1000; draw(cur); });
play.addEventListener('click', ()=>{
  if(timer){ clearInterval(timer); timer=null; play.textContent='▶ 播放'; return; }
  play.textContent='❚❚ 暂停';
  timer=setInterval(()=>{
    cur += 0.004; if(cur>=1){ cur=1; clearInterval(timer); timer=null; play.textContent='▶ 播放'; }
    slider.value = cur*1000; draw(cur);
  }, 40);
});
// #0.5 直接跳到某个进度——截图验收与分享某一时刻都用得上
const hv = parseFloat(location.hash.slice(1));
if(!isNaN(hv) && hv>=0 && hv<=1){ cur=hv; slider.value=cur*1000; }
addEventListener('resize', fit); fit();
</script></body></html>"""


def terrain_planes():
    """现算 `capacity` / `fruit_capacity` 两层——见 docstring「底图为什么在这里现算」。"""
    import dataclasses

    from underworld import Config
    from underworld import terrain as T
    from record_map import WORLD

    cfg = dataclasses.replace(Config(), **WORLD)     # 地形与 seed、diet_delta 都无关
    t = T.build(cfg)
    g = cfg.grid
    out = {}
    for name in ("capacity", "fruit_capacity"):
        v = np.asarray(getattr(t, name)).reshape(g, g).astype(np.float32)
        hi = float(v.max())
        # 各自按自己的最大值归一——两层的绝对量级差一个数量级，共用标尺会把果层压没
        out[name] = (np.clip(v / hi, 0, 1) * 255).astype(np.uint8) if hi > 1e-9 \
            else np.zeros((g, g), np.uint8)
        print(f"    {name}: max={hi:.3f}  非零格 {float((v > 1e-6).mean())*100:.1f}%")
    return {k: base64.b64encode(v.ravel().tobytes()).decode("ascii") for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    # `--seed` 主要是为了能用一份几帧的小数据把整条管线（编码→解码→渲染）先验通，
    # 不必等 20 分钟的正式录制落地才发现前端读错了字节序。
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(HERE, "map.html")
    print("  现算底图的两层承载量（`terrain.build` 不吃随机数，故与录制时必然一致）：")
    extra = terrain_planes()
    panels = []
    for p in PANELS:
        path = os.path.join(HERE, f"map_{p['key']}_s{a.seed}.json")
        if not os.path.exists(path):
            raise SystemExit(f"缺 {path}——先跑 record_map.py（有捕食那臂要加 --pred）")
        d = json.load(open(path))
        panels.append({"title": p["title"], "note": p["note"], "world": d["world"],
                       "terrain": {**d["terrain"], **extra}, "frames": d["frames"]})
        f0, f1 = d["frames"][0], d["frames"][-1]
        print(f"  {p['title']:<8} {len(d['frames']):>4} 帧   "
              f"世代 {f0['g']:.0f} → {f1['g']:.0f}   存活 {f0['n']} → {f1['n']}")
    html = HTML.replace("__DATA__", json.dumps({"panels": panels}, separators=(",", ":")))
    open(out_path, "w").write(html)
    print(f"\n✓ {out_path}（{os.path.getsize(out_path)/1e6:.1f} MB，自包含，双击可开）")


if __name__ == "__main__":
    main()
