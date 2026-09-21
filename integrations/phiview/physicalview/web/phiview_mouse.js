// Mouse gestures send controls only; the server owns all camera and scene math.
export function installMouseControls({$, command, getState, getFrameId}) {
  const viewport = $('viewport'), frame = $('frame');
  let mode = 'view', selection = null, drag = null, objectGesture = null, epoch = 0;
  let orbit = [0, 0], pan = [0, 0], zoom = 0;
  let keys = new Set(), boost = false, inputBusy = false, wasActive = false;
  try { const saved = localStorage.getItem('phiview.mouseMode'); if (['select','move'].includes(saved)) mode = saved; } catch {}

  function refresh() {
    $('mouseView').setAttribute('aria-pressed', String(mode === 'view'));
    $('mouseSelect').setAttribute('aria-pressed', String(mode === 'select'));
    $('mouseMove').setAttribute('aria-pressed', String(mode === 'move'));
    viewport.dataset.mouseMode = mode;
    const fixed = getState()?.camera_view && getState().camera_view !== 'free';
    $('mouseHint').textContent = mode === 'view'
      ? fixed ? 'Fixed robot camera · choose Free view to navigate'
        : 'Left-drag orbit · right-drag pan · wheel zoom'
      : mode === 'move' ? 'Drag the selected simulatable object across its horizontal plane'
      : $('shooting').checked ? 'Shooting enabled · click to fire'
        : 'Click an object · left-drag a box · Esc deselect';
  }
  function clearInput() {
    epoch++; selection = null; drag = null; objectGesture = null;
    $('selectionBox').style.display = 'none';
    viewport.classList.remove('dragging');
    keys.clear(); boost = false; orbit = [0, 0]; pan = [0, 0]; zoom = 0;
    command('selection_end');
    command('move_end');
    command('input', {keys: [], look: [0, 0]});
  }
  function setMode(next) {
    clearInput(); mode = next;
    if (mode !== 'select') $('shooting').checked = false;
    if (mode === 'move') command('view', {mode: 'simulation'});
    $('crosshair').style.display = $('shooting').checked ? 'block' : 'none';
    try { localStorage.setItem('phiview.mouseMode', mode); } catch {}
    refresh();
  }
  function deselect() { clearInput(); command('deselect'); }
  $('mouseView').onclick = () => setMode('view');
  $('mouseSelect').onclick = () => setMode('select');
  $('mouseMove').onclick = () => setMode('move');
  $('deselect').onclick = deselect;
  $('shooting').onchange = () => {
    if ($('shooting').checked) setMode('select');
    $('crosshair').style.display = $('shooting').checked ? 'block' : 'none';
    refresh();
  };
  const typing = e => ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
  window.addEventListener('keydown', e => {
    if (typing(e)) return;
    const k = e.key.toLowerCase();
    if ('wasdqe'.includes(k) && k.length === 1) { keys.add(k); e.preventDefault(); }
    if (k === 'shift') boost = true;
    if (k === 'f') command('focus');
    if (k === 'escape') { e.preventDefault(); deselect(); document.exitPointerLock?.(); }
  });
  window.addEventListener('keyup', e => {
    keys.delete(e.key.toLowerCase()); if (e.key === 'Shift') boost = false;
  });
  window.addEventListener('blur',clearInput);
  document.addEventListener('visibilitychange', () => { if (document.hidden) clearInput(); });
  viewport.addEventListener('contextmenu', e => e.preventDefault());
  const point = (e, r) => [Math.max(0, Math.min(1, (e.clientX-r.left)/r.width)),
                            Math.max(0, Math.min(1, (e.clientY-r.top)/r.height))];
  viewport.addEventListener('pointerdown', e => {
    if (![0, 1, 2].includes(e.button) || selection || drag || objectGesture) return;
    if (mode !== 'view' && (e.button !== 0 || e.target !== frame)) return;
    e.preventDefault(); viewport.setPointerCapture(e.pointerId); viewport.focus();
    if (mode === 'view') {
      if (getState()?.camera_view !== 'free') return;
      drag = {type: e.button === 2 || e.shiftKey ? 'pan' : e.button === 1 ? 'zoom' : 'orbit',
              x: e.clientX, y: e.clientY};
      viewport.classList.add('dragging');
    } else if (mode === 'move') {
      const rect = frame.getBoundingClientRect(), start = point(e, rect);
      keys.clear();
      objectGesture = {epoch, rect, point: start, pending: null,
        begin: command('move_begin', {frame: getFrameId(), x: start[0], y: start[1]})};
    } else {
      const rect = frame.getBoundingClientRect(), start = point(e, rect), fid = getFrameId();
      keys.clear();
      selection = {epoch, rect, start, end: start, frame: fid, shoot: $('shooting').checked,
                   pin: command('selection_begin', {frame: fid})};
    }
  });
  viewport.addEventListener('pointermove', e => {
    if (objectGesture) objectGesture.pending = point(e, objectGesture.rect);
    if (drag) {
      const height = Math.max(frame.getBoundingClientRect().height, 1);
      const dx = (e.clientX-drag.x)/height, dy = (e.clientY-drag.y)/height;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.type === 'orbit') { orbit[0] += dx; orbit[1] += dy; }
      else if (drag.type === 'pan') { pan[0] += dx; pan[1] += dy; }
      else zoom += dy*4;
    }
    if (!selection) return;
    const g = selection; g.end = point(e, g.rect);
    if (g.shoot) return;
    const r = viewport.getBoundingClientRect();
    Object.assign($('selectionBox').style, {display: 'block',
      left: (g.rect.left-r.left+Math.min(g.start[0],g.end[0])*g.rect.width)+'px',
      top: (g.rect.top-r.top+Math.min(g.start[1],g.end[1])*g.rect.height)+'px',
      width: (Math.abs(g.start[0]-g.end[0])*g.rect.width)+'px',
      height: (Math.abs(g.start[1]-g.end[1])*g.rect.height)+'px'});
  });
  viewport.addEventListener('pointerup', async e => {
    drag = null; viewport.classList.remove('dragging');
    if (e.button === 0 && objectGesture) {
      const g = objectGesture; objectGesture = null;
      const last = point(e, g.rect), result = await g.begin;
      if (!result || g.epoch !== epoch) return;
      await command('move', {drag: result.drag, x: last[0], y: last[1]});
      await command('move_end', {drag: result.drag});
      return;
    }
    if (e.button !== 0 || !selection) return;
    const g = selection; g.end = point(e, g.rect); selection = null;
    $('selectionBox').style.display = 'none';
    if (!await g.pin || g.epoch !== epoch || mode !== 'select') return;
    const distance = Math.hypot((g.end[0]-g.start[0])*g.rect.width, (g.end[1]-g.start[1])*g.rect.height);
    try {
      if (!g.shoot && distance >= 6) await command('box_select', {frame: g.frame, box: [...g.start, ...g.end]});
      else await command(g.shoot ? 'shoot' : 'pick', {frame: g.frame,
        x: Math.min(g.end[0], .999999), y: Math.min(g.end[1], .999999)});
    } finally { if (g.epoch === epoch) await command('selection_end'); }
  });
  viewport.addEventListener('pointercancel', clearInput);
  viewport.addEventListener('lostpointercapture', () => { if (selection || drag || objectGesture) clearInput(); });
  viewport.addEventListener('wheel', e => {
    e.preventDefault();
    if (mode !== 'view' || getState()?.camera_view !== 'free') return;
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? viewport.clientHeight : 1;
    zoom += Math.max(-1, Math.min(1, e.deltaY*unit/500));
  }, {passive: false});
  const timer = setInterval(async () => {
    if (inputBusy || selection) return;
    if (objectGesture) {
      const g = objectGesture, point = g.pending; g.pending = null;
      if (!point) return;
      inputBusy = true;
      try {
        const result = await g.begin;
        if (result && g.epoch === epoch && objectGesture === g)
          await command('move', {drag: result.drag, x: point[0], y: point[1]});
      } finally { inputBusy = false; }
      return;
    }
    const navigating = orbit.some(Boolean) || pan.some(Boolean) || zoom;
    const active = keys.size;
    if (!navigating && !active && !wasActive) return;
    inputBusy = true;
    const gesture = {orbit, pan, zoom}; orbit = [0, 0]; pan = [0, 0]; zoom = 0;
    try {
      if (navigating && mode === 'view') await command('navigate', gesture);
      await command('input', {keys: [...keys], look: [0, 0], boost});
      wasActive = active;
    } finally { inputBusy = false; }
  }, 33);
  window.addEventListener('pagehide', () => { clearInterval(timer); });
  refresh();
  return {get selecting() { return selection !== null; }, update: refresh};
}
