import { vertex, bloom, composite } from './shader-library.js?v=20260913g';
import { scenes, gridVertex, gridFragment } from './shader-scenes.js?v=20260913g';

function makeProgram(gl, vertexSource, fragmentSource) {
  const shaders = [gl.VERTEX_SHADER, gl.FRAGMENT_SHADER].map((type, index) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, index ? fragmentSource : vertexSource);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const error = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(error);
    }
    return shader;
  });
  const program = gl.createProgram();
  shaders.forEach(shader => gl.attachShader(program, shader));
  gl.linkProgram(program);
  shaders.forEach(shader => gl.deleteShader(shader));
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const error = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(error);
  }
  const uniforms = {};
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let index = 0; index < count; index++) {
    const info = gl.getActiveUniform(program, index);
    uniforms[info.name] = gl.getUniformLocation(program, info.name);
  }
  return { program, uniforms };
}

function makeTarget(gl, width, height, floating) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, floating ? gl.RGBA16F : gl.RGBA8,
    width, height, 0, gl.RGBA, floating ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null);
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteTexture(texture); gl.deleteFramebuffer(framebuffer);
    throw new Error('Unable to allocate a complete render target.');
  }
  return { texture, framebuffer, width, height };
}

export class ShaderRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false,
      powerPreference: 'high-performance', preserveDrawingBuffer: false });
    if (!this.gl) throw new Error('WebGL 2 unavailable.');
    const gl = this.gl;
    this.floating = Boolean(gl.getExtension('EXT_color_buffer_float'));
    this.encoding = this.floating ? 1 : 8;
    this.programs = {
      scene: makeProgram(gl, vertex, scenes[canvas.dataset.scene]),
      bloom: makeProgram(gl, vertex, bloom),
      composite: makeProgram(gl, vertex, composite)
    };
    if (canvas.dataset.scene === 'universe') {
      this.programs.grid = makeProgram(gl, gridVertex, gridFragment);
    }
    this.targets = [];
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.scale = 1;
    this.frames = 0;
    canvas.dataset.renderer = this.floating ? 'webgl2-hdr' : 'webgl2';
  }

  resize() {
    const { canvas, gl } = this;
    const rect = canvas.getBoundingClientRect();
    const maxPixels = this.programs.grid ? 1400000
      : (rect.width > 900 && rect.height > 500 ? 900000 : 260000);
    const ratio = Math.min(devicePixelRatio || 1, 1.5,
      Math.sqrt(maxPixels / Math.max(1, rect.width * rect.height))) * this.scale;
    const width = Math.max(2, Math.round(rect.width * ratio));
    const height = Math.max(2, Math.round(rect.height * ratio));
    if (canvas.width === width && canvas.height === height && this.targets.length) return;
    this.targets.forEach(target => {
      gl.deleteTexture(target.texture); gl.deleteFramebuffer(target.framebuffer);
    });
    canvas.width = width; canvas.height = height;
    this.targets = [1, 4, 4, 12, 12].map(divisor =>
      makeTarget(gl, Math.max(2, Math.round(width / divisor)),
        Math.max(2, Math.round(height / divisor)), this.floating));
    canvas.dataset.resolution = width + '×' + height;
  }

  use(name, target) {
    const gl = this.gl;
    const { program, uniforms } = this.programs[name];
    gl.useProgram(program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target?.framebuffer || null);
    gl.viewport(0, 0, target?.width || this.canvas.width, target?.height || this.canvas.height);
    if (uniforms.uEncoding) gl.uniform1f(uniforms.uEncoding, this.encoding);
    return uniforms;
  }

  texture(location, texture, unit = 0) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(location, unit);
  }

  blur(source, target, direction, threshold) {
    const gl = this.gl;
    const u = this.use('bloom', target);
    this.texture(u.uTexture, source.texture);
    gl.uniform2fv(u.uDirection, direction);
    gl.uniform1f(u.uThreshold, threshold);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  render(time, pointer) {
    this.resize();
    const gl = this.gl;
    const [scene, narrowX, narrowY, wideX, wideY] = this.targets;
    let u = this.use('scene', scene);
    gl.uniform2f(u.uResolution, scene.width, scene.height);
    gl.uniform1f(u.uTime, time);
    gl.uniform2fv(u.uPointer, pointer);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (this.programs.grid) {
      u = this.use('grid', scene);
      gl.uniform2f(u.uResolution, scene.width, scene.height);
      gl.uniform1f(u.uTime, time);
      gl.uniform2fv(u.uPointer, pointer);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.drawArrays(gl.TRIANGLES, 0, 160 * 160 * 6);
      gl.disable(gl.BLEND);
    }
    this.blur(scene, narrowX, [2 / scene.width, 0], .35);
    this.blur(narrowX, narrowY, [0, 1 / narrowX.height], 0);
    this.blur(narrowY, wideX, [3 / narrowY.width, 0], 0);
    this.blur(wideX, wideY, [0, 1.5 / wideX.height], 0);
    u = this.use('composite', null);
    this.texture(u.uTexture, scene.texture);
    this.texture(u.uBloom, narrowY.texture, 1);
    this.texture(u.uWideBloom, wideY.texture, 2);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.canvas.dataset.frames = String(++this.frames);
  }

  dispose() {
    const gl = this.gl;
    this.targets.forEach(target => {
      gl.deleteTexture(target.texture); gl.deleteFramebuffer(target.framebuffer);
    });
    Object.values(this.programs).forEach(({ program }) => gl.deleteProgram(program));
    gl.deleteVertexArray(this.vao);
  }
}
