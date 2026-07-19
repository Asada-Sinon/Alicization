// Dashboard controller: connect the websocket, parse binary snapshots (and JSON
// inspector detail), drive the renderer, and wire the controls + click-select.
(function () {
  "use strict";

  const SPEEDS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  const STRIDE = 5;              // floats per agent: x, y, diet, energy, id
  const HEADER_BYTES = 60;
  const PICK_RADIUS = 6.0;       // world units
  const canvas = document.getElementById("view");
  const ring = document.getElementById("ring");
  const $ = (id) => document.getElementById(id);

  Renderer.init(canvas);

  let latest = null;
  let ws = null;
  let playing = true;
  let selectedId = null;

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { setConn(true); if (selectedId !== null) send({ type: "select", id: selectedId }); };
    ws.onclose = () => { setConn(false); setTimeout(connect, 1000); };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") updateInspector(JSON.parse(ev.data));
      else latest = parse(ev.data);
    };
  }

  function setConn(on) {
    $("conn").classList.toggle("on", on);
    $("conntxt").textContent = on ? "在线" : "已断开";
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
    const streamAmp = dv.getFloat32(44, true);
    const streamK = dv.getFloat32(48, true);
    const streamBaseY = dv.getFloat32(52, true);
    const streamHW = dv.getFloat32(56, true);
    const agents = new Float32Array(buffer, HEADER_BYTES, n * STRIDE);
    const plant = new Uint8Array(buffer, HEADER_BYTES + n * STRIDE * 4, grid * grid);
    return { frame, n, grid, world, agents, plant, meanEnergy, plantTotal,
      meanAge, meanDiet, carnFrac, meanWater, streamAmp, streamK, streamBaseY,
      streamHW };
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
    $("play").textContent = playing ? "⏸ 暂停" : "▶ 播放";
    $("play").classList.toggle("primary", playing);
    send({ type: playing ? "play" : "pause" });
  };
  $("speed").oninput = (e) => {
    const v = SPEEDS[parseInt(e.target.value, 10)];
    $("speedval").textContent = v;
    send({ type: "speed", value: v });
  };
  $("reset").onclick = () => {
    selectedId = null; ring.style.display = "none"; clearInspector();
    send({ type: "reset" });
  };

  // --- inspector panel ---
  function dietLabel(d) {
    if (d < 0.35) return ["草食动物", "#9b5de5"];
    if (d > 0.65) return ["肉食动物", "#e8443b"];
    return ["杂食动物", "#e6d64b"];
  }
  function clearInspector() {
    $("itype").textContent = "点击一个个体";
    $("itype").className = "muted";
    ["i_id", "i_diet", "i_energy", "i_water", "i_age", "i_gen", "i_speed",
      "i_food", "i_meat", "i_dmg", "i_drink", "i_turn", "i_thrust"].forEach(
      (id) => ($(id).textContent = "–"));
    $("ibar").style.left = "0%";
    const ctx = $("retina").getContext("2d");
    ctx.clearRect(0, 0, 150, 150);
    $("neurons").innerHTML = "";
  }

  function drawRetina(d) {
    const ctx = $("retina").getContext("2d");
    ctx.clearRect(0, 0, 150, 150);
    const cx = 75, cy = 75, R = d.sectors, w = (2 * Math.PI) / R;
    // four concentric channel bands: food, prey, threat, water (inner -> outer)
    const bands = [
      [13, 25, d.retina_food, [80, 220, 90]],
      [25, 37, d.retina_prey, [80, 200, 230]],
      [37, 49, d.retina_pred, [235, 70, 60]],
      [49, 61, d.retina_water, [70, 140, 235]],
    ];
    for (let s = 0; s < R; s++) {
      const a0 = -Math.PI / 2 + s * w;   // forward (sector 0) points up
      const a1 = a0 + w * 0.94;          // small gap between wedges
      for (const [ri, ro, vals, c] of bands) {
        const v = Math.max(0, Math.min(1, vals[s] || 0));
        ctx.beginPath();
        ctx.arc(cx, cy, ro, a0, a1);
        ctx.arc(cx, cy, ri, a1, a0, true);
        ctx.closePath();
        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.1 + 0.85 * v})`;
        ctx.fill();
      }
    }
    // forward indicator
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy - 2);
    ctx.lineTo(cx, cy - 13);
    ctx.stroke();
  }

  function drawNeurons(hidden) {
    const nn = $("neurons");
    nn.innerHTML = "";
    for (const h of hidden) {
      const v = Math.max(-1, Math.min(1, h));
      const c = v >= 0
        ? `rgba(235,90,80,${Math.abs(v)})`
        : `rgba(80,150,235,${Math.abs(v)})`;
      const el = document.createElement("div");
      el.style.cssText =
        `width:13px;height:13px;border-radius:2px;background:${c};` +
        `box-shadow:inset 0 0 0 1px #1c2836`;
      el.title = v.toFixed(2);
      nn.appendChild(el);
    }
  }
  function updateInspector(d) {
    if (!d.alive) {
      $("itype").textContent = `#${d.id} 已死亡`;
      $("itype").className = "muted";
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

  function tick() {
    if (latest) {
      try { Renderer.draw(latest, STRIDE); } catch (e) { console.error(e); }
      positionRing();
      $("pop").textContent = latest.n.toLocaleString();
      $("carn").textContent = (latest.carnFrac * 100).toFixed(0) + "%";
      $("diet").textContent = fmt(latest.meanDiet, 2);
      $("energy").textContent = fmt(latest.meanEnergy, 2);
      $("water").textContent = fmt(latest.meanWater, 2);
      $("age").textContent = fmt(latest.meanAge, 0);
      $("plant").textContent = fmt(latest.plantTotal, 0);
      $("frame").textContent = latest.frame.toLocaleString();
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
