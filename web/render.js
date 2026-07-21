// WebGL renderer: a heightfield terrain mesh (built once from the UNTR
// message) plus one gl.POINTS draw call for the whole living population,
// coloured by diet (purple = herbivore, red = carnivore). Both are seen
// through a perspective orbit camera -- world<->screen mapping for
// click-selection goes through the camera matrices too.
//
// The simulation itself is still a flat 2D plane (agents carry only x, y);
// this file is the only place "3D" exists. See docs/three_d.md 2.1 for the
// design this follows.
//
// WebGL1, no build step, no gl-matrix -- Mat4 below is the whole matrix
// library this needs, not a trimmed copy of a bigger one.
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

  // --------------------------------------------------------------------
  // Mat4: the minimal 4x4 matrix library this file needs (perspective,
  // lookAt, multiply, invert). Column-major, flat length-16 arrays -- the
  // same layout WebGL expects, so these upload with uniformMatrix4fv(...,
  // false, m) with no transpose.
  // --------------------------------------------------------------------
  const Mat4 = {
    create() {
      return new Float32Array(16);
    },
    perspective(out, fovy, aspect, near, far) {
      const f = 1.0 / Math.tan(fovy / 2);
      const nf = 1 / (near - far);
      out.set([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
      ]);
      return out;
    },
    // eye/center/up are plain [x,y,z] arrays. up is world +Z (the terrain
    // is a Z-up heightfield: x,y are the simulation plane, z is elevation).
    lookAt(out, eye, center, up) {
      let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
      let len = Math.hypot(z0, z1, z2) || 1;
      z0 /= len; z1 /= len; z2 /= len;
      let x0 = up[1] * z2 - up[2] * z1;
      let x1 = up[2] * z0 - up[0] * z2;
      let x2 = up[0] * z1 - up[1] * z0;
      len = Math.hypot(x0, x1, x2) || 1;
      x0 /= len; x1 /= len; x2 /= len;
      const y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
      out.set([
        x0, y0, z0, 0,
        x1, y1, z1, 0,
        x2, y2, z2, 0,
        -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]),
        -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]),
        -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]),
        1,
      ]);
      return out;
    },
    // out = a * b
    multiply(out, a, b) {
      const r = new Float32Array(16);
      for (let c = 0; c < 4; c++) {
        for (let row = 0; row < 4; row++) {
          r[c * 4 + row] =
            a[0 * 4 + row] * b[c * 4 + 0] +
            a[1 * 4 + row] * b[c * 4 + 1] +
            a[2 * 4 + row] * b[c * 4 + 2] +
            a[3 * 4 + row] * b[c * 4 + 3];
        }
      }
      out.set(r);
      return out;
    },
    // Standard cofactor-expansion inverse. Returns null if singular (out
    // left untouched in that case).
    invert(out, a) {
      const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
      const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
      const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
      const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
      const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
      const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
      const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
      const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
      const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
      const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
      let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
      if (!det) return null;
      det = 1.0 / det;
      out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
      out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
      out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
      out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
      out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
      out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
      out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
      out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
      out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
      out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
      out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
      out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
      out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
      out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
      out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
      out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
      return out;
    },
    // m * [x,y,z,1], full 4-component result (caller divides by w).
    transformPoint(out, m, x, y, z) {
      out[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
      out[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
      out[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
      out[3] = m[3] * x + m[7] * y + m[11] * z + m[15];
      return out;
    },
  };

  // Terrain mesh vertices carry world-space position (already displaced by
  // height -- baked once on the CPU when the UNTR message lands, see
  // _buildTerrainMesh) plus a UV for sampling the plant/terrain textures.
  const PLANT_VS = `
    attribute vec3 a_pos;
    attribute vec2 a_uv;
    uniform mat4 u_viewProj;
    varying vec2 v_uv;
    void main() {
      v_uv = a_uv;
      gl_Position = u_viewProj * vec4(a_pos, 1.0);
    }`;

  // Terrain arrives once as a static texture (height / forest / water in r,g,b),
  // so the client no longer duplicates any world-generation formula -- there is
  // nothing here that has to be kept in sync with the Python side.
  //
  // This fragment shader is unchanged from the flat-quad version. The relief-
  // shading block below was written when the terrain was still a textured quad
  // with no real depth; now that it is an actual displaced mesh (real
  // silhouettes, real self-occlusion, real perspective foreshortening), that
  // block might look redundant -- but it is exactly a slope-based hillshade (a
  // real cartographic technique, not just a flat-quad trick), so it still adds
  // legitimate directional shading on top of the geometry rather than fighting
  // it. Screenshot-verified it still reads as terrain, not a flat lawn, so it
  // was kept rather than removed -- see the commit message for the reasoning.
  const PLANT_FS = `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_plant;
    uniform sampler2D u_terrain;
    uniform float u_texel;
    void main() {
      vec3 t = texture2D(u_terrain, v_uv).rgb;
      float h = t.r, forest = t.g, water = t.b;

      vec2 food = texture2D(u_plant, v_uv).ra;
      float l = food.r, fr = food.g;
      // Most vegetated ground sits AT its carrying capacity, so the ceiling sets
      // the mood, not the curve -- a bright peak paints one flat lawn and drowns
      // the agents. Keep the peak a deep night forest.
      l = pow(l, 1.8);
      vec3 bg = vec3(0.027, 0.039, 0.071);   // --void
      vec3 col = bg + vec3(0.045, 0.24, 0.105) * l;

      // canopy reads darker and colder than open grass, so the forest belt is
      // legible as terrain rather than just "more food"
      col = mix(col, vec3(0.014, 0.112, 0.072), forest * 0.7);

      // Fruit: a warm amber against all that green, because it is the one food
      // worth crossing the map for and has to be findable by eye. Kept as an
      // additive bloom rather than a mix so a stripped patch fades out instead
      // of punching a hole in the canopy colour.
      col += vec3(0.42, 0.20, 0.03) * pow(fr, 1.3);

      // Bare rock takes over as the ground climbs. Deliberately a dark slate:
      // a light rock colour plus the relief highlight below turns every peak
      // into a glossy pale blob instead of a mountain.
      float rock = smoothstep(0.62, 0.88, h);
      col = mix(col, vec3(0.115, 0.112, 0.135), rock);

      // Relief shading: slope against a fixed light, so the range reads as 3D.
      // Sampling the height plane itself means no formula is duplicated. Kept
      // asymmetric -- shadows may go deep, highlights stay restrained, which is
      // what stops the peaks blowing out.
      float hL = texture2D(u_terrain, v_uv - vec2(u_texel, 0.0)).r;
      float hR = texture2D(u_terrain, v_uv + vec2(u_texel, 0.0)).r;
      float hD = texture2D(u_terrain, v_uv - vec2(0.0, u_texel)).r;
      float hU = texture2D(u_terrain, v_uv + vec2(0.0, u_texel)).r;
      vec2 slope = vec2(hR - hL, hU - hD);
      float shade = clamp(dot(normalize(vec2(-0.7, 0.7)), slope) * 7.0, -0.55, 0.30);
      col *= (1.0 + shade);

      // rivers and sea last, so water always wins over whatever it covers
      col = mix(col, vec3(0.10, 0.38, 0.68), water * 0.92);

      gl_FragColor = vec4(col, 1.0);
    }`;

  // a_pos is world (x,y); a_h is the agent's ground elevation, sampled on the
  // CPU once per frame from the same raw height field the terrain mesh and
  // the picking ray-march use (Renderer._heightAt) -- one source of truth for
  // "how tall is the ground here" rather than a second copy in a shader.
  const POINT_VS = `
    attribute vec2 a_pos;
    attribute float a_h;
    attribute float a_diet;
    attribute float a_energy;
    uniform mat4 u_viewProj;
    uniform float u_scale;
    uniform float u_hover;
    uniform float u_sizeRef;
    varying float v_diet;
    varying float v_energy;
    void main() {
      gl_Position = u_viewProj * vec4(a_pos, a_h + u_hover, 1.0);
      float s = clamp(a_energy / 12.0, 0.2, 1.6);
      // carnivores render a touch larger. The perspective divide already
      // makes gl_Position read as near/far, but gl_PointSize is a raw pixel
      // count the rasterizer does NOT divide by w -- so without the explicit
      // u_sizeRef/w term every agent would be the same size regardless of
      // camera distance, which would defeat the point of a perspective camera.
      float base = u_scale * (2.5 + 3.0 * s + 2.0 * a_diet);
      gl_PointSize = clamp(base * (u_sizeRef / max(gl_Position.w, 0.001)), 1.0, 64.0);
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

  // Vertical exaggeration for the terrain mesh. The wire format only sends
  // height remapped to [0,255] over [min,max] (protocol.encode_terrain), so
  // the client never learns the true elevation range in world units -- this
  // is a rendering choice, not a measurement. Picked for a visually legible
  // relief at the default camera distance, not derived from anything.
  const HEIGHT_SCALE_FRAC = 0.16;
  // How far above the ground an agent's point sprite floats, so it reads as
  // "standing on the terrain" instead of being half-buried in a slope.
  const AGENT_HOVER_FRAC = 0.004;

  const FOVY = 45 * Math.PI / 180;
  const NEAR = 1.0;

  // Orbit camera tuning. Angles are radians; 0 pitch is eye-level with the
  // ground plane, PI/2 is straight down.
  const ROT_SPEED = 0.006;          // radians per CSS pixel dragged
  const PAN_SPEED = 0.0016;         // fraction of camera distance per pixel
  const ZOOM_SPEED = 0.0018;        // exponential factor per wheel unit
  const PITCH_MIN = 0.10, PITCH_MAX = 1.50;
  const DIST_MIN_FRAC = 0.10, DIST_MAX_FRAC = 3.5;
  const DRAG_CLICK_THRESHOLD = 5;   // px; drags past this suppress click-select

  const Renderer = {
    init(canvas) {
      const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
      if (!gl) throw new Error("WebGL not available");
      this.gl = gl;
      this.canvas = canvas;
      this._world = 256;
      this._vp = [0, 0, 1];

      gl.enable(gl.DEPTH_TEST);

      this.plantProg = program(gl, PLANT_VS, PLANT_FS);
      this.pointProg = program(gl, POINT_VS, POINT_FS);

      const mkTex = () => {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return t;
      };
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      this.tex = mkTex();            // plant field, re-uploaded every frame
      this.terrainTex = mkTex();     // static terrain, uploaded once per world
      this.terrainGrid = 0;

      // Terrain mesh buffers (static: rebuilt only when setTerrain is called).
      this.terrainVBuf = gl.createBuffer();
      this.terrainIBuf = gl.createBuffer();
      this.terrainIndexCount = 0;
      this.terrainIndexType = gl.UNSIGNED_SHORT;
      this._uintIndicesExt = gl.getExtension("OES_element_index_uint");

      // Per-frame agent buffers: position/diet/energy come straight off the
      // wire (a_pos etc.), but a_h (ground height under each agent) is
      // recomputed on the CPU every frame from the raw height field, so it
      // needs its own small buffer separate from the raw wire bytes.
      this.agentBuf = gl.createBuffer();
      this.agentHBuf = gl.createBuffer();

      this._camInitialized = false;
      this._cam = {
        yaw: -2.356, pitch: 0.85, dist: 256, target: [0, 0, 0],
      };
      this._suppressClick = false;
      this._drag = null;

      this._viewM = Mat4.create();
      this._projM = Mat4.create();
      this._viewProjM = Mat4.create();
      this._invViewProjM = Mat4.create();

      this._attachControls(canvas);
    },

    // --- orbit camera interaction: left-drag rotates, right/middle/shift-
    // drag pans, wheel zooms. Lives here (not main.js) because it only
    // touches camera state this file owns. ---
    _attachControls(canvas) {
      const cam = this._cam;
      const self = this;

      const onDown = (e) => {
        const mode = (e.button === 0 && !e.shiftKey) ? "rotate" : "pan";
        self._drag = { mode, lastX: e.clientX, lastY: e.clientY, total: 0 };
        e.preventDefault();
      };
      const onMove = (e) => {
        const d = self._drag;
        if (!d) return;
        const dx = e.clientX - d.lastX, dy = e.clientY - d.lastY;
        d.lastX = e.clientX; d.lastY = e.clientY;
        d.total += Math.abs(dx) + Math.abs(dy);
        if (d.mode === "rotate") {
          cam.yaw -= dx * ROT_SPEED;
          cam.pitch = Math.min(PITCH_MAX, Math.max(PITCH_MIN, cam.pitch - dy * ROT_SPEED));
        } else {
          const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
          // ground-plane forward/right, ignoring pitch, so panning stays
          // level regardless of camera tilt
          const fwd = [cy, sy], right = [sy, -cy];
          const k = cam.dist * PAN_SPEED;
          cam.target[0] += (-dx * right[0] + dy * fwd[0]) * k;
          cam.target[1] += (-dx * right[1] + dy * fwd[1]) * k;
          const w = self._world;
          cam.target[0] = Math.min(2 * w, Math.max(-w, cam.target[0]));
          cam.target[1] = Math.min(2 * w, Math.max(-w, cam.target[1]));
        }
      };
      const onUp = () => {
        if (self._drag && self._drag.total > DRAG_CLICK_THRESHOLD) {
          self._suppressClick = true;
        }
        self._drag = null;
      };
      const onWheel = (e) => {
        e.preventDefault();
        const w = self._world;
        cam.dist *= Math.pow(1 + ZOOM_SPEED, e.deltaY);
        cam.dist = Math.min(w * DIST_MAX_FRAC, Math.max(w * DIST_MIN_FRAC, cam.dist));
      };

      canvas.addEventListener("mousedown", onDown);
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
      canvas.addEventListener("wheel", onWheel, { passive: false });
      canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    },

    // Consumed by main.js's click handler so an orbit drag doesn't also
    // fire a click-to-select.
    consumeClickSuppression() {
      const v = this._suppressClick;
      this._suppressClick = false;
      return v;
    },

    // Bilinear height sample (world units) at world (x, y). Shared by the
    // terrain mesh build, per-frame agent elevation, and click picking, so
    // there is exactly one place that knows how texel index maps to world
    // position (cell centres, matching underworld/terrain.py's iy*g+ix
    // layout).
    _heightAt(x, y) {
      const g = this.terrainGrid;
      if (!g) return 0;
      const cell = this._world / g;
      const gx = x / cell - 0.5, gy = y / cell - 0.5;
      const ix0 = Math.floor(gx), iy0 = Math.floor(gy);
      const fx = gx - ix0, fy = gy - iy0;
      const wrap = (i) => ((i % g) + g) % g;
      const ix0w = wrap(ix0), ix1w = wrap(ix0 + 1);
      const iy0w = wrap(iy0), iy1w = wrap(iy0 + 1);
      const h = this._heightData;
      const h00 = h[iy0w * g + ix0w], h10 = h[iy0w * g + ix1w];
      const h01 = h[iy1w * g + ix0w], h11 = h[iy1w * g + ix1w];
      const hx0 = h00 + (h10 - h00) * fx;
      const hx1 = h01 + (h11 - h01) * fx;
      const hv = hx0 + (hx1 - hx0) * fy;
      return (hv / 255.0) * this._heightScale;
    },

    // Static terrain: three grid*grid u8 planes packed into one RGB texture,
    // plus a heightfield mesh built once from the same height plane. Called
    // once when the terrain message arrives (and again after a reset -- the
    // terrain itself never changes, terrain.build has no RNG, so this is
    // idempotent and only the camera default is skipped on repeat calls, so
    // a reset doesn't yank the user's view back to the default angle).
    setTerrain(grid, world, height, forest, water) {
      const gl = this.gl;
      const n = grid * grid;
      const rgb = new Uint8Array(n * 3);
      for (let i = 0; i < n; i++) {
        rgb[i * 3] = height[i];
        rgb[i * 3 + 1] = forest[i];
        rgb[i * 3 + 2] = water[i];
      }
      gl.bindTexture(gl.TEXTURE_2D, this.terrainTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, grid, grid, 0,
        gl.RGB, gl.UNSIGNED_BYTE, rgb);
      this.terrainGrid = grid;
      this._world = world;
      this._heightData = height;
      this._heightScale = world * HEIGHT_SCALE_FRAC;
      this._agentHover = world * AGENT_HOVER_FRAC;

      this._buildTerrainMesh(grid, world, height);

      if (!this._camInitialized) {
        this._camInitialized = true;
        const cam = this._cam;
        cam.target = [world / 2, world / 2, this._heightScale * 0.25];
        cam.dist = world * 0.9;
      }
    },

    // grid*grid vertices at plant-cell centres (matching docs/three_d.md
    // 2.1's count: (grid-1)^2 quads, 2*(grid-1)^2 triangles), baked with
    // world position + height once here rather than every frame -- the
    // terrain is static so there is nothing to recompute per draw.
    _buildTerrainMesh(grid, world, height) {
      const gl = this.gl;
      const cell = world / grid;
      const verts = new Float32Array(grid * grid * 5); // x,y,z,u,v
      for (let iy = 0; iy < grid; iy++) {
        for (let ix = 0; ix < grid; ix++) {
          const idx = iy * grid + ix;
          const o = idx * 5;
          verts[o] = (ix + 0.5) * cell;
          verts[o + 1] = (iy + 0.5) * cell;
          verts[o + 2] = (height[idx] / 255.0) * this._heightScale;
          verts[o + 3] = (ix + 0.5) / grid;
          verts[o + 4] = (iy + 0.5) / grid;
        }
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.terrainVBuf);
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

      const quads = (grid - 1) * (grid - 1);
      const indexCount = quads * 6;
      const vertCount = grid * grid;
      // 16-bit indices address up to 65536 distinct vertices (0..65535
      // inclusive); beyond that (grid > 256) OES_element_index_uint is
      // required. grid=128 (the default) needs 16384 vertices, comfortably
      // 16-bit.
      const need32 = vertCount > 65536;
      if (need32 && !this._uintIndicesExt) {
        console.warn("terrain grid too large for 16-bit indices and " +
          "OES_element_index_uint is unavailable; mesh will be truncated");
      }
      const use32 = need32 && this._uintIndicesExt;
      const IndexArray = use32 ? Uint32Array : Uint16Array;
      this.terrainIndexType = use32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
      const indices = new IndexArray(indexCount);
      let k = 0;
      for (let iy = 0; iy < grid - 1; iy++) {
        for (let ix = 0; ix < grid - 1; ix++) {
          const a = iy * grid + ix;
          const b = iy * grid + ix + 1;
          const c = (iy + 1) * grid + ix;
          const d = (iy + 1) * grid + ix + 1;
          indices[k++] = a; indices[k++] = c; indices[k++] = b;
          indices[k++] = b; indices[k++] = c; indices[k++] = d;
        }
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.terrainIBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
      this.terrainIndexCount = indexCount;
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

    // Recompute view/projection from the current orbit camera. The drawing
    // viewport is always a centred square (see resize()), so aspect is
    // always 1 -- no need to track canvas aspect ratio here.
    _updateCamera() {
      const cam = this._cam;
      const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
      const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
      const eye = [
        cam.target[0] + cam.dist * cp * cy,
        cam.target[1] + cam.dist * cp * sy,
        cam.target[2] + cam.dist * sp,
      ];
      const far = Math.max(this._world * 6, cam.dist * 3);
      Mat4.perspective(this._projM, FOVY, 1.0, NEAR, far);
      Mat4.lookAt(this._viewM, eye, cam.target, [0, 0, 1]);
      Mat4.multiply(this._viewProjM, this._projM, this._viewM);
      this._far = far;
    },

    draw(snap, STRIDE_FLOATS) {
      const gl = this.gl;
      this.resize();
      this._world = snap.world;
      const [vx, vy, vs] = this._vp;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0.0, 0.0, 0.0, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.viewport(vx, vy, vs, vs);

      if (!this.terrainGrid) return;   // nothing sensible to draw until terrain lands
      this._updateCamera();

      // --- terrain mesh ---
      gl.useProgram(this.plantProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      // Grass and fruit share one LUMINANCE_ALPHA texture -- grass in .r, fruit
      // in .a -- so the two food layers cost one upload and one sample instead
      // of two of each.
      const cells = snap.grid * snap.grid;
      if (!this.foodBuf || this.foodBuf.length !== cells * 2) {
        this.foodBuf = new Uint8Array(cells * 2);
      }
      for (let i = 0; i < cells; i++) {
        this.foodBuf[i * 2] = snap.plant[i];
        this.foodBuf[i * 2 + 1] = snap.fruit[i];
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE_ALPHA, snap.grid, snap.grid, 0,
        gl.LUMINANCE_ALPHA, gl.UNSIGNED_BYTE, this.foodBuf);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.terrainTex);
      gl.uniform1i(gl.getUniformLocation(this.plantProg, "u_plant"), 0);
      gl.uniform1i(gl.getUniformLocation(this.plantProg, "u_terrain"), 1);
      gl.uniform1f(gl.getUniformLocation(this.plantProg, "u_texel"),
        1.0 / this.terrainGrid);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.plantProg, "u_viewProj"),
        false, this._viewProjM);
      gl.activeTexture(gl.TEXTURE0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.terrainVBuf);
      const aLoc = gl.getAttribLocation(this.plantProg, "a_pos");
      const uvLoc = gl.getAttribLocation(this.plantProg, "a_uv");
      gl.enableVertexAttribArray(aLoc);
      gl.vertexAttribPointer(aLoc, 3, gl.FLOAT, false, 20, 0);
      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 20, 12);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.terrainIBuf);
      gl.drawElements(gl.TRIANGLES, this.terrainIndexCount, this.terrainIndexType, 0);

      // --- agents (x,y,diet,energy,id straight from the wire; height
      // sampled fresh each frame from the same field the mesh was built
      // from) ---
      if (snap.n > 0) {
        const stride = STRIDE_FLOATS * 4;  // bytes per agent
        const agents = snap.agents;
        if (!this._hBuf || this._hBuf.length < snap.n) {
          this._hBuf = new Float32Array(Math.max(snap.n, 1024));
        }
        for (let i = 0; i < snap.n; i++) {
          this._hBuf[i] = this._heightAt(agents[i * STRIDE_FLOATS], agents[i * STRIDE_FLOATS + 1]);
        }

        gl.useProgram(this.pointProg);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.agentBuf);
        gl.bufferData(gl.ARRAY_BUFFER, agents, gl.DYNAMIC_DRAW);
        const posLoc = gl.getAttribLocation(this.pointProg, "a_pos");
        const dietLoc = gl.getAttribLocation(this.pointProg, "a_diet");
        const enLoc = gl.getAttribLocation(this.pointProg, "a_energy");
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(dietLoc);
        gl.vertexAttribPointer(dietLoc, 1, gl.FLOAT, false, stride, 8);
        gl.enableVertexAttribArray(enLoc);
        gl.vertexAttribPointer(enLoc, 1, gl.FLOAT, false, stride, 12);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.agentHBuf);
        gl.bufferData(gl.ARRAY_BUFFER, this._hBuf.subarray(0, snap.n), gl.DYNAMIC_DRAW);
        const hLoc = gl.getAttribLocation(this.pointProg, "a_h");
        gl.enableVertexAttribArray(hLoc);
        gl.vertexAttribPointer(hLoc, 1, gl.FLOAT, false, 0, 0);

        gl.uniformMatrix4fv(gl.getUniformLocation(this.pointProg, "u_viewProj"),
          false, this._viewProjM);
        gl.uniform1f(gl.getUniformLocation(this.pointProg, "u_hover"), this._agentHover);
        gl.uniform1f(gl.getUniformLocation(this.pointProg, "u_sizeRef"), this._cam.dist);
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        gl.uniform1f(gl.getUniformLocation(this.pointProg, "u_scale"), dpr);
        gl.drawArrays(gl.POINTS, 0, snap.n);
      }
    },

    // --- world <-> canvas-local-pixel mapping (matches the square viewport),
    // now going through the camera instead of a linear world<->clip map. ---

    // Ray-march the JS-side height field from the clicked pixel's camera ray
    // until it crosses the terrain surface, then bisect for precision. No
    // GPU readback -- the CPU already has the raw height bytes from
    // parseTerrain. Returns [worldX, worldY] or null (off-mesh / no terrain
    // in that direction, e.g. a click above the horizon).
    worldFromClient(clientX, clientY) {
      if (!this.terrainGrid) return null;
      const c = this.canvas;
      const rect = c.getBoundingClientRect();
      const bufX = (clientX - rect.left) * (c.width / rect.width);
      const bufYtop = (clientY - rect.top) * (c.height / rect.height);
      const bufYbot = c.height - bufYtop;               // GL origin is bottom-left
      const [vx, vy, vs] = this._vp;
      const u = (bufX - vx) / vs;
      const v = (bufYbot - vy) / vs;
      if (u < 0 || u > 1 || v < 0 || v > 1) return null;
      const ndcX = u * 2 - 1, ndcY = v * 2 - 1;

      this._updateCamera();
      const inv = Mat4.invert(this._invViewProjM, this._viewProjM);
      if (!inv) return null;
      const pNear = Mat4.transformPoint(new Float32Array(4), inv, ndcX, ndcY, -1);
      const pFar = Mat4.transformPoint(new Float32Array(4), inv, ndcX, ndcY, 1);
      if (Math.abs(pNear[3]) < 1e-8 || Math.abs(pFar[3]) < 1e-8) return null;
      const near = [pNear[0] / pNear[3], pNear[1] / pNear[3], pNear[2] / pNear[3]];
      const far = [pFar[0] / pFar[3], pFar[1] / pFar[3], pFar[2] / pFar[3]];
      let dir = [far[0] - near[0], far[1] - near[1], far[2] - near[2]];
      const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
      dir = [dir[0] / len, dir[1] / len, dir[2] / len];

      const world = this._world;
      const cell = world / this.terrainGrid;
      const step = Math.max(cell, world / 4096);
      const maxDist = (this._far || world * 6) + this._cam.dist;
      const maxSteps = Math.min(4096, Math.ceil(maxDist / step) + 1);

      const sample = (t) => {
        const x = near[0] + dir[0] * t, y = near[1] + dir[1] * t, z = near[2] + dir[2] * t;
        if (x < 0 || x >= world || y < 0 || y >= world) return null;
        return { x, y, z, delta: z - this._heightAt(x, y) };
      };

      let prevT = 0, prev = sample(0);
      if (!prev) return null;
      if (prev.delta < 0) return null;   // camera ray starts under the mesh; bail out
      for (let s = 1; s <= maxSteps; s++) {
        const t = s * step;
        const cur = sample(t);
        if (!cur) return null;           // marched off the single rendered tile
        if (cur.delta <= 0) {
          // bisect between prevT and t for sub-cell precision
          let lo = prevT, hi = t;
          for (let it = 0; it < 12; it++) {
            const mid = (lo + hi) / 2;
            const m = sample(mid);
            if (!m) { hi = mid; continue; }
            if (m.delta > 0) lo = mid; else hi = mid;
          }
          const hit = sample((lo + hi) / 2);
          return hit ? [hit.x, hit.y] : null;
        }
        prevT = t; prev = cur;
      }
      return null;
    },

    // The square drawing area in CSS pixels relative to the canvas top-left.
    // The world is letterboxed inside a non-square canvas, so overlays that need
    // to hug the world (the sigil frame) can't just wrap #stage -- they align here.
    viewportRect() {
      const c = this.canvas;
      const rect = c.getBoundingClientRect();
      const [vx, vy, vs] = this._vp;
      const sx = rect.width / (c.width || 1);
      const sy = rect.height / (c.height || 1);
      return {
        x: vx * sx,
        y: (c.height - vy - vs) * sy,   // GL origin is bottom-left
        size: vs * sx,
      };
    },

    // Returns pixel coords relative to the canvas top-left (for overlays), or
    // null if the point is behind the camera / off the letterboxed square.
    canvasFromWorld(wx, wy) {
      if (!this.terrainGrid) return null;
      this._updateCamera();
      const wz = this._heightAt(wx, wy) + this._agentHover;
      const clip = Mat4.transformPoint(new Float32Array(4), this._viewProjM, wx, wy, wz);
      if (clip[3] <= 1e-6) return null;          // behind the eye
      const ndcX = clip[0] / clip[3], ndcY = clip[1] / clip[3];
      const c = this.canvas;
      const rect = c.getBoundingClientRect();
      const [vx, vy, vs] = this._vp;
      const bufX = vx + (ndcX * 0.5 + 0.5) * vs;
      const bufYbot = vy + (ndcY * 0.5 + 0.5) * vs;
      const bufYtop = c.height - bufYbot;
      return {
        x: bufX * (rect.width / c.width),
        y: bufYtop * (rect.height / c.height),
      };
    },
  };

  window.Renderer = Renderer;
})();
