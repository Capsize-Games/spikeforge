import { ShaderRenderer } from './shader-renderer.js?v=20260913g';

// Lifecycle-owned adapter for React or other client-side routers. No global
// DOM selectors; every resource and listener is released by dispose().
function createClock(canvas, state) {
  const request = () => {
    if (!state.frame && !state.disposed && !document.hidden && state.visible)
      state.frame = requestAnimationFrame(draw);
  };
  function draw(now) {
    state.frame = 0;
    if (state.disposed || document.hidden || !state.visible || !state.renderer) return;
    const elapsed = state.previous ? now - state.previous : 34;
    if (!state.paused && elapsed < 32) { request(); return; }
    if (!state.paused) state.time += Math.min(elapsed, 100) / 1000;
    state.previous = now;
    try { state.renderer.render(state.time, [0, 0]); }
    catch { state.renderer.dispose(); state.renderer = null; canvas.dataset.renderer = 'unavailable'; return; }
    state.slow = elapsed > 65 ? state.slow + 1 : Math.max(0, state.slow - 1);
    if (state.slow > 35) {
      state.renderer.scale = Math.max(.6, state.renderer.scale * .85);
      state.slow = 0;
    }
    if (!state.paused) request();
  }
  return () => { state.previous = 0; request(); };
}

export function mountScene(canvas) {
  const media = matchMedia('(prefers-reduced-motion: reduce)');
  const state = { renderer: null, paused: media.matches, visible: true,
    frame: 0, previous: 0, time: 0, disposed: false, slow: 0 };
  const reset = createClock(canvas, state);
  const motion = () => { state.paused = media.matches; reset(); };
  const initialize = () => {
    try { state.renderer = new ShaderRenderer(canvas); reset(); }
    catch { canvas.dataset.renderer = 'unavailable'; }
  };
  const lost = event => {
    event.preventDefault(); state.renderer = null;
    canvas.dataset.renderer = 'unavailable';
  };
  const visibility = new IntersectionObserver(entries => {
    state.visible = entries[0].isIntersecting; reset();
  });
  const sizing = new ResizeObserver(reset);
  visibility.observe(canvas); sizing.observe(canvas);
  media.addEventListener('change', motion);
  document.addEventListener('visibilitychange', reset);
  canvas.addEventListener('webglcontextlost', lost);
  canvas.addEventListener('webglcontextrestored', initialize);
  initialize();
  return {
    setPaused(value) { state.paused = value; reset(); },
    dispose() {
      state.disposed = true; cancelAnimationFrame(state.frame);
      visibility.disconnect(); sizing.disconnect();
      media.removeEventListener('change', motion);
      document.removeEventListener('visibilitychange', reset);
      canvas.removeEventListener('webglcontextlost', lost);
      canvas.removeEventListener('webglcontextrestored', initialize);
      state.renderer?.dispose(); state.renderer = null;
    }
  };
}
