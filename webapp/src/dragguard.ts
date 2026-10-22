/* Drags that start in a picture never select the page. Every viewer (the field's ZoomViewer, DeepZoom of the Cores
   view, Mark bad areas and grey / white matter, the zoom-level panels, the minimap) is marked `VIEWERS`; a press of the
   left or middle button there:
   - has no default action (`mousedown` prevented: no text or image selection starts, no native drag of a picture, no
     autoscroll), and whatever had the keyboard focus (a slider, a field) lets it go, so the page's keys work;
   - marks the page as dragging (`html.viewer-dragging`: nothing on it can be selected meanwhile; `selectstart` and
     `dragstart` are refused) until the button is let go anywhere, the pointer is cancelled or lost, or the window
     loses focus; any selection made by accident is cleared at both ends.
   Each viewer also holds the pointer (`setPointerCapture`) while a button is down, so the moves and the release reach
   it even beyond the window, and ends its drag on every one of those ends (viewer.tsx, deepzoom.tsx, levelmap.tsx). */

export const VIEWERS = '.viewer, .dz, .level-map, .minimap'
const DRAGGING = 'viewer-dragging'
let installed = false
let active = false

export function clearSelection() {
  try { window.getSelection()?.removeAllRanges() } catch { /* no selection API */ }
}

function inViewer(target: EventTarget | null): boolean {
  return target instanceof Element && !!target.closest(VIEWERS) && !target.closest('input, textarea, select, button, a, [contenteditable="true"]')
}

function begin() {
  active = true
  document.documentElement.classList.add(DRAGGING)
  clearSelection()
}

function end() {
  if (!active) return
  active = false
  document.documentElement.classList.remove(DRAGGING)
  clearSelection()
}

/** Once, at start (main.tsx): the page-wide part. */
export function installDragGuard() {
  if (installed || typeof window === 'undefined') return
  installed = true
  window.addEventListener('pointerdown', (e) => {
    if ((e.button === 0 || e.button === 1) && inViewer(e.target)) begin()
  }, true)
  window.addEventListener('mousedown', (e) => {
    if ((e.button !== 0 && e.button !== 1) || !inViewer(e.target)) return
    e.preventDefault()
    const focused = document.activeElement
    if (focused instanceof HTMLElement && focused !== document.body && !focused.closest(VIEWERS)) focused.blur()
  }, true)
  const refuse = (e: Event) => { if (active || inViewer(e.target)) e.preventDefault() }
  window.addEventListener('selectstart', refuse, true)
  window.addEventListener('dragstart', refuse, true)
  for (const name of ['pointerup', 'pointercancel', 'mouseup', 'lostpointercapture']) window.addEventListener(name, end, true)
  window.addEventListener('blur', end)
}
