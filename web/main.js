// Dashboard controller: connect the websocket, parse binary snapshots (and JSON
// inspector detail), drive the renderer, and wire the controls + click-select.
(function () {
  "use strict";

  const SPEEDS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  const STRIDE = 5;              // floats per agent: x, y, diet, energy, id
  const HEADER_BYTES = 68;       // protocol v6
  const PICK_RADIUS = 6.0;       // world units
  const HIST = 600;              // samples kept per series (~20s at 30fps)

  // Series colours. Locked to render.js's shader constants and validated as a
  // set (all six dataviz checks pass on the --stone surface); if you change one
  // here, change it in render.js and index.html's :root too.
  const C = {
    herb: "#9e52eb",
    carn: "#f24038",
    omni: "#9fb2d0",
    halo: "#4a97ea",
    plant: "#1a8033",
    rule: "#1b2440",
  };

  const canvas = document.getElementById("view");
  const ring = document.getElementById("ring");
  const sigil = document.getElementById("sigil");
  const $ = (id) => document.getElementById(id);
  const DPR = () => Math.min(window.devicePixelRatio || 1, 2);

  Renderer.init(canvas);

  let latest = null;
  let ws = null;
  let playing = true;
  let selectedId = null;
  let speed = 4;

  // --- history ring buffers (client-side; the server streams only "now") ---
  const hist = {
    pop: [], carn: [], std: [], plant: [], cv: [], hv: [], fst: [],
  };
  function pushHistory(s) {
    const add = (key, v) => {
      const a = hist[key];
      a.push(v);
      if (a.length > HIST) a.shift();
    };
    add("pop", s.n);
    add("carn", s.carnFrac);
    add("std", s.dietStd);
    add("plant", s.plantTotal);
    add("cv", s.carnSpeed);
    add("hv", s.herbSpeed);
    add("fst", s.forestFrac);
  }
  function clearHistory() {
    for (const k in hist) hist[k].length = 0;
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { setConn(true); if (selectedId !== null) send({ type: "select", id: selectedId }); };
    ws.onclose = () => { setConn(false); setTimeout(connect, 1000); };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") { updateInspector(JSON.parse(ev.data)); return; }
      // Terrain and snapshots share the socket; tell them apart by magic rather
      // than by length, which would break the moment the grid size changed.
      const magic = new Uint8Array(ev.data, 0, 4);
      if (magic[0] === 85 && magic[1] === 78 && magic[2] === 84 && magic[3] === 82) {
        const t = parseTerrain(ev.data);          // "UNTR"
        Renderer.setTerrain(t.grid, t.height, t.forest, t.water);
        return;
      }
      latest = parse(ev.data);
      pushHistory(latest);
    };
  }

  function setConn(on) {
    $("conn").classList.toggle("on", on);
    $("conntxt").textContent = on ? "已连接 · 时间流同步中" : "连接中断 · 正在重连";
  }

  function parse(buffer) {
    const dv = new DataView(buffer);
    const frame = dv.getUint32(4, true);
    const n = dv.getUint32(8, true);
    const grid = dv.getUint32(12, true);
    const world = dv.getFloat32(16, true);
    const meanEnergy = dv.getFloat32(20, true);
    const plantTotal = dv.getFloat32(24, true);
    const meanAge = dv.getFloat32(28, true);
    const meanDiet = dv.getFloat32(32, true);
    const carnFrac = dv.getFloat32(36, true);
    const meanWater = dv.getFloat32(40, true);
    const dietStd = dv.getFloat32(44, true);
    const carnSpeed = dv.getFloat32(48, true);
    const herbSpeed = dv.getFloat32(52, true);
    const meanElev = dv.getFloat32(56, true);
    const forestFrac = dv.getFloat32(60, true);
    const fruitTotal = dv.getFloat32(64, true);
    const agents = new Float32Array(buffer, HEADER_BYTES, n * STRIDE);
    const planes = HEADER_BYTES + n * STRIDE * 4;
    const plant = new Uint8Array(buffer, planes, grid * grid);
    const fruit = new Uint8Array(buffer, planes + grid * grid, grid * grid);
    return { frame, n, grid, world, agents, plant, fruit, meanEnergy, plantTotal,
      meanAge, meanDiet, carnFrac, meanWater, dietStd, carnSpeed, herbSpeed,
      meanElev, forestFrac, fruitTotal };
  }

  // Static terrain, sent once per world: 12-byte header then three u8 planes.
  function parseTerrain(buffer) {
    const dv = new DataView(buffer);
    const grid = dv.getUint32(4, true);
    const world = dv.getFloat32(8, true);
    const n = grid * grid;
    return {
      grid, world,
      height: new Uint8Array(buffer, 12, n),
      forest: new Uint8Array(buffer, 12 + n, n),
      water: new Uint8Array(buffer, 12 + 2 * n, n),
    };
  }

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }

  // --- click to select nearest agent ---
  canvas.addEventListener("click", (e) => {
    if (!latest) return;
    const w = Renderer.worldFromClient(e.clientX, e.clientY);
    if (!w) return;
    const a = latest.agents;
    let best = -1, bestD = PICK_RADIUS * PICK_RADIUS;
    for (let i = 0; i < latest.n; i++) {
      const dx = a[i * STRIDE] - w[0], dy = a[i * STRIDE + 1] - w[1];
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD) { bestD = d2; best = i; }
    }
    if (best >= 0) {
      selectedId = a[best * STRIDE + 4] | 0;
      send({ type: "select", id: selectedId });
    } else {
      selectedId = null;
      send({ type: "select", id: -1 });
      ring.style.display = "none";
      clearInspector();
    }
  });

  // --- controls ---
  $("play").onclick = () => {
    playing = !playing;
    $("play").textContent = playing ? "停止时间流" : "恢复时间流";
    $("play").classList.toggle("primary", playing);
    send({ type: playing ? "play" : "pause" });
  };
  $("speed").oninput = (e) => {
    speed = SPEEDS[parseInt(e.target.value, 10)];
    $("speedval").textContent = speed;
    send({ type: "speed", value: speed });
  };
  $("reset").onclick = () => {
    selectedId = null; ring.style.display = "none"; clearInspector();
    clearHistory();
    send({ type: "reset" });
  };

  // --- sparklines -------------------------------------------------------
  // Ambient trend strips, not interrogable charts: the x-axis scrolls
  // continuously, so a pinned crosshair tooltip would point at a sample that
  // has already slid away. The current value is a permanent direct label
  // instead, and the window's min/max ride along in the native title.
  function drawSparkline(el, series, opts) {
    const dpr = DPR();
    const w = el.clientWidth || 260, h = el.clientHeight || 34;
    if (el.width !== Math.round(w * dpr) || el.height !== Math.round(h * dpr)) {
      el.width = Math.round(w * dpr);
      el.height = Math.round(h * dpr);
    }
    const ctx = el.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // one pass for the domain -- this runs for every chart every frame, so no
    // intermediate arrays and no spread
    let dLo = Infinity, dHi = -Infinity, count = 0;
    for (const s of series) {
      for (const v of s.data) {
        if (!Number.isFinite(v)) continue;
        if (v < dLo) dLo = v;
        if (v > dHi) dHi = v;
        count++;
      }
    }
    if (count < 2) return;

    let lo = opts.min !== undefined ? opts.min : dLo;
    let hi = opts.max !== undefined ? opts.max : dHi;
    if (hi - lo < 1e-6) { hi = lo + 1; lo -= 0.5; }
    const pad = (hi - lo) * 0.12;
    if (opts.min === undefined) lo -= pad;
    if (opts.max === undefined) hi += pad;

    const PAD_T = 3, PAD_B = 3;
    const yOf = (v) => h - PAD_B - ((v - lo) / (hi - lo)) * (h - PAD_T - PAD_B);
    const xOf = (i, n) => (n < 2 ? 0 : (i / (n - 1)) * w);

    // recessive baseline / midline
    ctx.strokeStyle = C.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const mid = opts.mid !== undefined ? opts.mid : null;
    if (mid !== null && mid >= lo && mid <= hi) {
      ctx.moveTo(0, Math.round(yOf(mid)) + 0.5);
      ctx.lineTo(w, Math.round(yOf(mid)) + 0.5);
    }
    ctx.moveTo(0, h - 0.5);
    ctx.lineTo(w, h - 0.5);
    ctx.stroke();

    for (const s of series) {
      const d = s.data, n = d.length;
      if (n < 2) continue;
      if (s.fill) {
        const g = ctx.createLinearGradient(0, 0, 0, h);
        g.addColorStop(0, s.color + "33");
        g.addColorStop(1, s.color + "00");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let i = 0; i < n; i++) ctx.lineTo(xOf(i, n), yOf(d[i]));
        ctx.lineTo(w, h);
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = xOf(i, n), y = yOf(d[i]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
      // leading dot: where the world is right now
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(xOf(n - 1, n) - 1.5, yOf(d[n - 1]), 2.2, 0, 6.2832);
      ctx.fill();
    }

    const f = opts.fmt || ((x) => x.toFixed(2));
    el.title = `窗口内 最低 ${f(dLo)} · 最高 ${f(dHi)}`;
  }

  const pct = (x) => (x * 100).toFixed(0) + "%";

  function drawAllSparks() {
    drawSparkline($("sp_pop"), [{ data: hist.pop, color: C.halo, fill: true }],
      { fmt: (x) => x.toFixed(0) });
    // anchored at 0 (so the height still reads as a true proportion) but with a
    // free top -- on a fixed 0..1 axis a ~20% carnivore share flatlines against
    // the baseline and its oscillation, the interesting part, is invisible
    drawSparkline($("sp_carn"), [{ data: hist.carn, color: C.carn, fill: true }],
      { min: 0, fmt: pct });
    drawSparkline($("sp_std"), [{ data: hist.std, color: C.halo, fill: true }],
      { min: 0, fmt: (x) => x.toFixed(2) });
    drawSparkline($("sp_plant"), [{ data: hist.plant, color: C.plant, fill: true }],
      { fmt: (x) => x.toFixed(0) });
    // Both series are the same measure in the same units (world units/s), so
    // they legitimately share one axis -- this is not a dual-axis chart.
    drawSparkline($("sp_vel"), [
      { data: hist.cv, color: C.carn },
      { data: hist.hv, color: C.herb },
    ], { min: 0, fmt: (x) => x.toFixed(1) });
    drawSparkline($("sp_forest"), [{ data: hist.fst, color: C.plant, fill: true }],
      { min: 0, max: 1, fmt: pct });
  }

  // --- inspector panel ---
  function dietLabel(d) {
    if (d < 0.35) return ["草食动物", C.herb];
    if (d > 0.65) return ["肉食动物", C.carn];
    return ["杂食动物", C.omni];
  }
  function clearInspector() {
    $("itype").textContent = "在世界中点选一个个体，解析它的摇光";
    $("itype").className = "empty";
    $("itype").style.color = "";
    ["i_id", "i_diet", "i_energy", "i_water", "i_size", "i_age", "i_gen", "i_speed",
      "i_food", "i_meat", "i_dmg", "i_drink", "i_turn", "i_thrust"].forEach(
      (id) => ($(id).textContent = "–"));
    $("ibar").style.left = "0%";
    const el = $("retina");
    el.getContext("2d").clearRect(0, 0, el.width, el.height);
    $("neurons").innerHTML = "";
  }

  // The signature element: what the creature sees, drawn as a sigil. Four
  // concentric channel bands (food / prey / threat / water, inner -> outer),
  // additively lit, inside a ring ticked at every sector boundary so the
  // wedge divisions are actually legible. Forward is gilt and up.
  function drawRetina(d) {
    const el = $("retina");
    const dpr = DPR();
    const S = 164;
    if (el.width !== S * dpr) {
      el.width = S * dpr; el.height = S * dpr;
      el.style.width = S + "px"; el.style.height = S + "px";
    }
    const ctx = el.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, S, S);

    const cx = S / 2, cy = S / 2, R = d.sectors, w = (2 * Math.PI) / R;
    const bands = [
      [17, 27, d.retina_food, [79, 214, 106]],
      [28, 38, d.retina_prey, [79, 198, 230]],
      [39, 49, d.retina_pred, [242, 64, 56]],
      [50, 60, d.retina_water, [74, 151, 234]],
      [61, 71, d.retina_peer, [230, 190, 79]],
    ];

    ctx.globalCompositeOperation = "lighter";
    for (let s = 0; s < R; s++) {
      const a0 = -Math.PI / 2 + s * w;   // sector 0 points up (forward)
      const a1 = a0 + w * 0.93;          // a hair of surface between wedges
      for (const [ri, ro, vals, c] of bands) {
        const v = Math.max(0, Math.min(1, (vals && vals[s]) || 0));
        ctx.beginPath();
        ctx.arc(cx, cy, ro, a0, a1);
        ctx.arc(cx, cy, ri, a1, a0, true);
        ctx.closePath();
        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.07 + 0.8 * v})`;
        ctx.shadowColor = `rgba(${c[0]},${c[1]},${c[2]},${0.85 * v})`;
        ctx.shadowBlur = 9 * v;
        ctx.fill();
      }
    }
    ctx.shadowBlur = 0;
    ctx.globalCompositeOperation = "source-over";

    // outer ring + a tick at every sector boundary
    ctx.strokeStyle = C.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, 70, 0, 6.2832);
    ctx.stroke();
    for (let s = 0; s < R; s++) {
      const a = -Math.PI / 2 + s * w;
      const fwd = s === 0;
      const r0 = 70, r1 = fwd ? 79 : 74;
      ctx.strokeStyle = fwd ? "#d9b26a" : C.rule;
      ctx.lineWidth = fwd ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
      ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.stroke();
    }
    // the creature itself, at the centre
    ctx.fillStyle = dietLabel(d.diet)[1];
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, 6.2832);
    ctx.fill();
  }

  function drawNeurons(hidden) {
    const nn = $("neurons");
    nn.innerHTML = "";
    for (const h of hidden) {
      const v = Math.max(-1, Math.min(1, h));
      const m = Math.abs(v);
      const c = v >= 0 ? `242,64,56` : `74,151,234`;
      const el = document.createElement("div");
      el.style.background = `rgba(${c},${m})`;
      if (m > 0.25) el.style.boxShadow =
        `inset 0 0 0 1px ${C.rule}, 0 0 ${6 * m}px rgba(${c},${0.7 * m})`;
      el.title = v.toFixed(2);
      nn.appendChild(el);
    }
  }

  function updateInspector(d) {
    if (!d.alive) {
      $("itype").textContent = `个体 #${d.id} 已死亡`;
      $("itype").className = "empty";
      $("itype").style.color = "";
      return;
    }
    const [label, color] = dietLabel(d.diet);
    $("itype").textContent = label;
    $("itype").className = "";
    $("itype").style.color = color;
    $("ibar").style.left = (d.diet * 100).toFixed(0) + "%";
    $("i_id").textContent = d.id;
    $("i_diet").textContent = d.diet.toFixed(2);
    $("i_energy").textContent = d.energy.toFixed(1);
    $("i_water").textContent = d.water.toFixed(1);
    $("i_size").textContent = d.size.toFixed(2);
    $("i_age").textContent = d.age.toFixed(0);
    $("i_gen").textContent = d.generation.toFixed(0);
    $("i_speed").textContent = d.speed.toFixed(1);
    $("i_food").textContent = d.food.toFixed(2);
    $("i_meat").textContent = d.meat.toFixed(2);
    $("i_dmg").textContent = d.damage.toFixed(2);
    $("i_drink").textContent = d.drink.toFixed(2);
    if (d.hidden) {
      $("i_turn").textContent = d.turn.toFixed(2);
      $("i_thrust").textContent = ((d.thrust + 1) / 2).toFixed(2);
      drawRetina(d);
      drawNeurons(d.hidden);
    }
  }

  // --- render + stats loop ---
  let frames = 0, lastFps = performance.now();
  const fmt = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : "–");

  function positionRing() {
    if (selectedId === null || !latest) { ring.style.display = "none"; return; }
    const a = latest.agents;
    for (let i = 0; i < latest.n; i++) {
      if ((a[i * STRIDE + 4] | 0) === selectedId) {
        const p = Renderer.canvasFromWorld(a[i * STRIDE], a[i * STRIDE + 1]);
        ring.style.left = p.x + "px";
        ring.style.top = p.y + "px";
        ring.style.display = "block";
        return;
      }
    }
    ring.style.display = "none";   // selected agent not currently alive/visible
  }

  // Hug the letterboxed square the world actually draws into. viewportRect() is
  // canvas-relative and #sigil is positioned against #stage, so add the canvas's
  // own offset (#stage is padded).
  function positionSigil() {
    const r = Renderer.viewportRect();
    if (!r || !r.size) return;
    sigil.style.left = canvas.offsetLeft + r.x + "px";
    sigil.style.top = canvas.offsetTop + r.y + "px";
    sigil.style.width = r.size + "px";
    sigil.style.height = r.size + "px";
  }

  function tick() {
    if (latest) {
      try { Renderer.draw(latest, STRIDE); } catch (e) { console.error(e); }
      positionRing();
      positionSigil();
      $("sigilcap").textContent =
        `观 测 法 阵 · ${latest.world.toFixed(0)}²`;
      $("sv_pop").textContent = latest.n.toLocaleString();
      $("sv_carn").textContent = pct(latest.carnFrac);
      $("sv_std").textContent = fmt(latest.dietStd, 2);
      $("sv_plant").textContent = fmt(latest.plantTotal, 0);
      $("sv_cv").textContent = fmt(latest.carnSpeed, 1);
      $("sv_hv").textContent = fmt(latest.herbSpeed, 1);
      $("sv_forest").textContent = pct(latest.forestFrac);
      $("elev").textContent = fmt(latest.meanElev, 2);
      $("diet").textContent = fmt(latest.meanDiet, 2);
      $("energy").textContent = fmt(latest.meanEnergy, 2);
      $("water").textContent = fmt(latest.meanWater, 2);
      $("age").textContent = fmt(latest.meanAge, 0);
      $("winlen").textContent = (hist.pop.length * speed).toLocaleString();
      $("frame").textContent = latest.frame.toLocaleString();
      drawAllSparks();
    }
    frames++;
    const now = performance.now();
    if (now - lastFps > 500) {
      $("fps").textContent = Math.round((frames * 1000) / (now - lastFps));
      frames = 0;
      lastFps = now;
    }
    requestAnimationFrame(tick);
  }

  connect();
  requestAnimationFrame(tick);
})();
