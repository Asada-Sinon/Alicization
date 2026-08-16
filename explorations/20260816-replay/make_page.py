"""把 `replay_data.json` 打成一个**自包含的** `replay.html`（数据内嵌，双击即可打开）。

为什么内嵌：`fetch` 本地 JSON 会被浏览器的同源策略挡掉，那样页面就必须起个服务器
才能看。93 KB 内嵌进去，双击就能开。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260816-replay/make_page.py
"""
import json
import os

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "replay_data.json")
OUT = os.path.join(HERE, "replay.html")

HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>两个吃法不同的食草者是怎么长出来的</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;
      --fruit:#e8a33d;--grass:#3fb950;--warn:#f24038}
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
canvas{width:100%;height:190px;display:block;background:#0b0f14;border-radius:6px}
canvas.spark{height:54px;margin-top:8px}
.stats{display:flex;gap:18px;margin-top:10px;font-size:13px;flex-wrap:wrap}
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
<h1>两个吃法不同的食草者是怎么长出来的</h1>
<p class="sub">横轴是「偏爱吃草还是吃果子」，柱子高度是有多少个体落在那儿。
拖动下面的进度条看它随世代变化。两边是同一个世界，只差有没有捕食者。</p>

<div class="panels" id="panels"></div>

<div class="controls">
  <button id="play">▶ 播放</button>
  <input type="range" id="slider" min="0" max="1000" value="0">
</div>
<div class="axis"><span>第 1 代</span><span id="gnow"></span><span>最后</span></div>

<div class="legend">
  <span><i class="sw" style="background:var(--fruit)"></i>偏爱果子</span>
  <span><i class="sw" style="background:var(--grass)"></i>偏爱草</span>
  <span class="dim">面板下方的细曲线是<b>偏果占比的全程走势</b>（虚线为 50%），竖线是当前位置。</span>
  <span class="dim">两边按<b>世代比例</b>对齐——有捕食者的世界代际更替快得多，
        同样的步数它经历的世代数是另一边的三倍。</span>
</div>

<div class="foot">
数据是已跑完的模拟，不是实时演算。左边取自 <code>__SRC0__</code>、右边 <code>__SRC1__</code>，
每边 12 个种子各跑两遍中的一个。结论与统计见 <code>docs/multispecies_feasibility.md</code> §27。
</div>
</div>
<script>
const DATA = __DATA__;
const P = DATA.panels, NB = DATA.bins;
const cvs = [], ctxs = [], els = [], sparks = [], sctxs = [];
const host = document.getElementById('panels');
P.forEach((p,i)=>{
  const d = document.createElement('div'); d.className='panel';
  d.innerHTML = `<div class="ptitle">${p.title}</div><div class="pnote">${p.note}</div>
    <canvas id="c${i}"></canvas>
    <canvas id="s${i}" class="spark"></canvas>
    <div class="stats">
      <span class="stat dim">第 <b id="g${i}" class="fg">–</b> 代</span>
      <span class="stat fruit">偏果的占 <b id="lo${i}">–</b></span>
      <span class="stat dim">存活 <b id="n${i}">–</b></span>
    </div>`;
  host.appendChild(d);
  const c = document.getElementById('c'+i);
  cvs.push(c); ctxs.push(c.getContext('2d'));
  const sp = document.getElementById('s'+i);
  sparks.push(sp); sctxs.push(sp.getContext('2d'));
  els.push({g:document.getElementById('g'+i), lo:document.getElementById('lo'+i),
            n:document.getElementById('n'+i)});
});
function fit(){ [...cvs,...sparks].forEach(c=>{ const r=c.getBoundingClientRect(), dpr=devicePixelRatio||1;
  c.width=r.width*dpr; c.height=r.height*dpr; }); draw(cur); }
// 柱子颜色：从琥珀(偏果)渐变到绿(偏草)
function barColor(k){
  const t=k/(NB-1), a=[232,163,61], b=[63,185,80];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`;
}
let peak = 0;
P.forEach(p=>p.frames.forEach(f=>f.h.forEach(v=>{ if(v>peak) peak=v; })));
function draw(frac){
  P.forEach((p,i)=>{
    const idx = Math.min(p.frames.length-1, Math.round(frac*(p.frames.length-1)));
    const f = p.frames[idx], ctx = ctxs[i], c = cvs[i];
    const dpr = devicePixelRatio||1, W=c.width, H=c.height, pad=6*dpr;
    ctx.clearRect(0,0,W,H);
    const bw = (W-pad*2)/NB;
    for(let k=0;k<NB;k++){
      const h = (f.h[k]/peak)*(H-pad*2);
      ctx.fillStyle = barColor(k);
      ctx.globalAlpha = 0.92;
      ctx.fillRect(pad+k*bw+bw*0.12, H-pad-h, bw*0.76, h);
    }
    ctx.globalAlpha=1;
    els[i].g.textContent = f.g.toFixed(0);
    els[i].lo.textContent = (f.lo*100).toFixed(1)+'%';
    els[i].n.textContent = f.n.toLocaleString();
  });
  // 全程曲线：果侧占比随世代——不拖进度条也能一眼看到走势
  P.forEach((p,i)=>{
    const ctx=sctxs[i], c=sparks[i], dpr=devicePixelRatio||1;
    const W=c.width, H=c.height, pad=4*dpr;
    ctx.clearRect(0,0,W,H);
    // 0.5 参考线
    ctx.strokeStyle='#30363d'; ctx.lineWidth=1*dpr; ctx.setLineDash([3*dpr,3*dpr]);
    ctx.beginPath(); ctx.moveTo(0,H-pad-(H-pad*2)*0.5); ctx.lineTo(W,H-pad-(H-pad*2)*0.5); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle='#e8a33d'; ctx.lineWidth=2*dpr; ctx.beginPath();
    p.frames.forEach((f,k)=>{
      const x=pad+(W-pad*2)*k/(p.frames.length-1), y=H-pad-(H-pad*2)*Math.min(f.lo,1);
      k?ctx.lineTo(x,y):ctx.moveTo(x,y);
    });
    ctx.stroke();
    const idx=Math.min(p.frames.length-1, Math.round(frac*(p.frames.length-1)));
    const cx=pad+(W-pad*2)*idx/(p.frames.length-1);
    ctx.strokeStyle='#e6edf3'; ctx.lineWidth=1*dpr; ctx.globalAlpha=.55;
    ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx,H); ctx.stroke(); ctx.globalAlpha=1;
  });
  const g0 = P[0].frames[Math.min(P[0].frames.length-1, Math.round(frac*(P[0].frames.length-1)))].g;
  document.getElementById('gnow').textContent = '第 '+g0.toFixed(0)+' 代';
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
  }, 33);
});
// 支持 URL 锚点 #0.5 直接跳到某个进度——既方便截图验收，也方便分享某一时刻
const hv = parseFloat(location.hash.slice(1));
if(!isNaN(hv) && hv>=0 && hv<=1){ cur=hv; slider.value=cur*1000; }
addEventListener('resize', fit); fit();
</script></body></html>"""


def main():
    data = json.load(open(DATA))
    html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                .replace("__SRC0__", data["panels"][0]["src"])
                .replace("__SRC1__", data["panels"][1]["src"]))
    open(OUT, "w").write(html)
    print(f"✓ {OUT}（{os.path.getsize(OUT)/1024:.0f} KB，自包含，双击可开）")


if __name__ == "__main__":
    main()
