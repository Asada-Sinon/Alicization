// Minimal WebGL renderer: a plant-field background (grid texture, with a
// meandering stream blended in) plus one gl.POINTS draw call for the whole
// living population, coloured by diet (purple = herbivore, red = carnivore).
// Scales to ~1e5 agents in one draw.
// Exposes window.Renderer, plus world<->pixel helpers for click-selection.
(function () {
  "use strict";

  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error("shader: " + gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  function program(gl, vsrc, fsrc) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsrc));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error("link: " + gl.getProgramInfoLog(p));
    }
    return p;
  }

  const PLANT_VS = `
    attribute vec2 a_quad;
    varying vec2 v_uv;
    void main() {
      v_uv = a_quad * 0.5 + 0.5;
      gl_Position = vec4(a_quad, 0.0, 1.0);
    }`;

  const PLANT_FS = `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_plant;
    uniform float u_world;
    uniform float u_streamAmp;
    uniform float u_streamK;
    uniform float u_streamBaseY;
    uniform float u_streamHW;
    void main() {
      float l = texture2D(u_plant, v_uv).r;
      vec3 bg = vec3(0.03, 0.05, 0.08);
      vec3 food = vec3(0.10, 0.50, 0.20);
      vec3 col = bg + food * l;

      // meandering stream: a sine centerline in y as a function of x, matching
      // ecology.stream_dist -- blended in as a soft-edged blue band.
      float x = v_uv.x * u_world;
      float y = v_uv.y * u_world;
      float centerY = u_streamBaseY +
        u_streamAmp * sin(6.28318530718 * u_streamK * x / u_world);
      float dy = abs(y - centerY);
      dy = min(dy, u_world - dy);
      float t = clamp(1.0 - dy / u_streamHW, 0.0, 1.0);
      vec3 water = vec3(0.10, 0.38, 0.68);
      col = mix(col, water, t * 0.9);

      gl_FragColor = vec4(col, 1.0);
    }`;

  const POINT_VS = `
    attribute vec2 a_pos;
    attribute float a_diet;
    attribute float a_energy;
    uniform float u_world;
    uniform float u_scale;
    varying float v_diet;
    varying float v_energy;
    void main() {
      vec2 clip = a_pos / u_world * 2.0 - 1.0;
      gl_Position = vec4(clip, 0.0, 1.0);
      float s = clamp(a_energy / 12.0, 0.2, 1.6);
      // carnivores render a touch larger
      gl_PointSize = u_scale * (2.5 + 3.0 * s + 2.0 * a_diet);
      v_diet = a_diet;
      v_energy = a_energy;
    }`;

  const POINT_FS = `
    precision mediump float;
    varying float v_diet;
    varying float v_energy;
    void main() {
      vec2 d = gl_PointCoord - vec2(0.5);
      if (dot(d, d) > 0.25) discard;
      // purple (herbivore) -> red (carnivore)
      vec3 herb = vec3(0.62, 0.32, 0.92);
      vec3 carn = vec3(0.95, 0.25, 0.22);
      vec3 c = mix(herb, carn, clamp(v_diet, 0.0, 1.0));
      float b = clamp(0.5 + v_energy * 0.05, 0.5, 1.0);
      gl_FragColor = vec4(c * b, 1.0);
    }`;

  const Renderer = {
    init(canvas) {
      const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
      if (!gl) throw new Error("WebGL not available");
      this.gl = gl;
      this.canvas = canvas;
      this._world = 256;
      this._vp = [0, 0, 1];

      this.plantProg = program(gl, PLANT_VS, PLANT_FS);
      this.pointProg = program(gl, POINT_VS, POINT_FS);

      this.quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      gl.bufferData(gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

      this.tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

      this.agentBuf = gl.createBuffer();
    },

    resize() {
      const c = this.canvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.floor(c.clientWidth * dpr);
      const h = Math.floor(c.clientHeight * dpr);
      if (c.width !== w || c.height !== h) {
        c.width = w;
        c.height = h;
      }
      const side = Math.min(w, h);
      this._vp = [Math.floor((w - side) / 2), Math.floor((h - side) / 2), side];
    },

    draw(snap, STRIDE_FLOATS) {
      const gl = this.gl;
      this.resize();
      this._world = snap.world;
      const [vx, vy, vs] = this._vp;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0.0, 0.0, 0.0, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.viewport(vx, vy, vs, vs);

      // --- plant background (+ stream overlay) ---
      gl.useProgram(this.plantProg);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, snap.grid, snap.grid, 0,
        gl.LUMINANCE, gl.UNSIGNED_BYTE, snap.plant);
      gl.uniform1i(gl.getUniformLocation(this.plantProg, "u_plant"), 0);
      gl.uniform1f(gl.getUniformLocation(this.plantProg, "u_world"), snap.world);
      gl.uniform1f(gl.getUniformLocation(this.plantProg, "u_streamAmp"), snap.streamAmp);
      gl.uniform1f(gl.getUniformLocation(this.plantProg, "u_streamK"), snap.streamK);
      gl.uniform1f(gl.getUniformLocation(this.plantProg, "u_streamBaseY"), snap.streamBaseY);
      gl.uniform1f(gl.getUniformLocation(this.plantProg, "u_streamHW"), snap.streamHW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      const qloc = gl.getAttribLocation(this.plantProg, "a_quad");
      gl.enableVertexAttribArray(qloc);
      gl.vertexAttribPointer(qloc, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // --- agents (interleaved x,y,diet,energy,id straight from the wire) ---
      if (snap.n > 0) {
        const stride = STRIDE_FLOATS * 4;  // bytes per agent
        gl.useProgram(this.pointProg);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.agentBuf);
        gl.bufferData(gl.ARRAY_BUFFER, snap.agents, gl.DYNAMIC_DRAW);
        const posLoc = gl.getAttribLocation(this.pointProg, "a_pos");
        const dietLoc = gl.getAttribLocation(this.pointProg, "a_diet");
        const enLoc = gl.getAttribLocation(this.pointProg, "a_energy");
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(dietLoc);
        gl.vertexAttribPointer(dietLoc, 1, gl.FLOAT, false, stride, 8);
        gl.enableVertexAttribArray(enLoc);
        gl.vertexAttribPointer(enLoc, 1, gl.FLOAT, false, stride, 12);
        gl.uniform1f(gl.getUniformLocation(this.pointProg, "u_world"), snap.world);
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        gl.uniform1f(gl.getUniformLocation(this.pointProg, "u_scale"), dpr);
        gl.drawArrays(gl.POINTS, 0, snap.n);
      }
    },

    // --- world <-> canvas-local-pixel mapping (matches the square viewport) ---
    worldFromClient(clientX, clientY) {
      const c = this.canvas;
      const rect = c.getBoundingClientRect();
      const bufX = (clientX - rect.left) * (c.width / rect.width);
      const bufYtop = (clientY - rect.top) * (c.height / rect.height);
      const bufYbot = c.height - bufYtop;               // GL origin is bottom-left
      const [vx, vy, vs] = this._vp;
      const u = (bufX - vx) / vs;
      const v = (bufYbot - vy) / vs;
      if (u < 0 || u > 1 || v < 0 || v > 1) return null;
      return [u * this._world, v * this._world];
    },

    // Returns pixel coords relative to the canvas top-left (for overlays).
    canvasFromWorld(wx, wy) {
      const c = this.canvas;
      const rect = c.getBoundingClientRect();
      const [vx, vy, vs] = this._vp;
      const bufX = vx + (wx / this._world) * vs;
      const bufYbot = vy + (wy / this._world) * vs;
      const bufYtop = c.height - bufYbot;
      return {
        x: bufX * (rect.width / c.width),
        y: bufYtop * (rect.height / c.height),
      };
    },
  };

  window.Renderer = Renderer;
})();
