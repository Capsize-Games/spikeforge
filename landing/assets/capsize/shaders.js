import { ShaderRenderer } from './shader-renderer.js?v=20260913g';

// One clock controls all surfaces. Paused, hidden, and off-screen scenes do
// not submit GPU work. Reduced-motion is respected on load and when changed.
const media = matchMedia('(prefers-reduced-motion: reduce)');
const button = document.querySelector('.motion-control');
const surfaces = new Map();
const masthead = document.querySelector('.masthead');
let paused = media.matches;
let raf = 0;
let lastFrame = 0;
let sceneTime = 0;
let pointer = [0, 0];
let smoothedPointer = [0, 0];
let slowFrames = 0;

function updateHeader() {
  masthead?.classList.toggle('is-compact', window.scrollY > 56);
}
document.addEventListener('scroll', updateHeader, { passive: true });
updateHeader();

function fail(canvas, error) {
  canvas.dataset.renderer = 'unavailable';
  console.warn('Capsize visual renderer:', error.message);
}

function initialize(canvas) {
  try {
    const renderer = new ShaderRenderer(canvas);
    surfaces.set(canvas, { renderer, visible: true, dirty: true });
    requestFrame();
  } catch (error) { fail(canvas, error); }
}

function requestFrame() {
  if (!raf && !document.hidden) raf = requestAnimationFrame(draw);
}

function draw(now) {
  raf = 0;
  if (document.hidden) return;
  const elapsed = lastFrame ? now - lastFrame : 34;
  if (!paused && elapsed < 32) { requestFrame(); return; }
  if (!paused) sceneTime += Math.min(elapsed, 100) / 1000;
  lastFrame = now;
  smoothedPointer = smoothedPointer.map((value, index) => value + (pointer[index] - value) * .035);
  let active = false;
  for (const [canvas, surface] of surfaces) {
    if (!surface.visible) continue;
    active = true;
    if (paused && !surface.dirty) continue;
    try {
      surface.renderer.render(sceneTime, smoothedPointer);
      surface.dirty = false;
    } catch (error) {
      surface.renderer.dispose();
      surfaces.delete(canvas);
      fail(canvas, error);
    }
  }
  slowFrames = elapsed > 65 ? slowFrames + 1 : Math.max(0, slowFrames - 1);
  if (slowFrames > 35) {
    for (const surface of surfaces.values()) {
      surface.renderer.scale = Math.max(.6, surface.renderer.scale * .85);
    }
    slowFrames = 0;
  }
  if (!paused && active) requestFrame();
}

function updateMotion() {
  if (button) {
    button.textContent = paused ? 'Resume motion' : 'Pause motion';
    button.setAttribute('aria-label', paused ? 'Resume visual animation' : 'Pause visual animation');
    button.setAttribute('aria-pressed', String(paused));
  }
  document.dispatchEvent(new CustomEvent('capsize-motion', { detail: { paused } }));
  lastFrame = 0;
  requestFrame();
}
button?.addEventListener('click', () => { paused = !paused; updateMotion(); });
media.addEventListener('change', () => { paused = media.matches; updateMotion(); });
document.addEventListener('visibilitychange', () => { lastFrame = 0; requestFrame(); });

const visibility = new IntersectionObserver(entries => {
  for (const entry of entries) {
    const surface = surfaces.get(entry.target);
    if (surface) { surface.visible = entry.isIntersecting; surface.dirty = true; }
  }
  requestFrame();
}, { rootMargin: '40px' });
const sizing = new ResizeObserver(entries => {
  for (const entry of entries) {
    const surface = surfaces.get(entry.target);
    if (surface) surface.dirty = true;
  }
  requestFrame();
});
document.querySelectorAll('[data-scene]').forEach(canvas => {
  initialize(canvas);
  visibility.observe(canvas);
  sizing.observe(canvas);
  canvas.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    surfaces.delete(canvas);
    canvas.dataset.renderer = 'unavailable';
  });
  canvas.addEventListener('webglcontextrestored', () => initialize(canvas));
});
document.querySelector('.universe')?.addEventListener('pointermove', event => {
  if (paused || event.pointerType === 'touch') return;
  const r = event.currentTarget.getBoundingClientRect();
  pointer = [(event.clientX - r.left) / r.width - .5, .5 - (event.clientY - r.top) / r.height];
});
document.querySelector('.universe')?.addEventListener('pointerleave', () => { pointer = [0, 0]; });
window.addEventListener('pagehide', () => {
  cancelAnimationFrame(raf);
  raf = 0;
});
window.addEventListener('pageshow', () => { lastFrame = 0; requestFrame(); });
updateMotion();
