() => {
  document.documentElement.classList.remove('dark');
  document.body.classList.remove('dark');
  if (window.hdrGSStudioBound) return;
  window.hdrGSStudioBound = true;
  const send = (panel, command) => panel.querySelector('iframe')?.contentWindow?.postMessage(
    {kind:'hdr-gs-viewer-command', ...command}, location.origin);
  document.addEventListener('click', async event => {
    const button = event.target.closest('[data-viewer-action]');
    if (!button) return;
    const panel = button.closest('.gaussian-panel');
    if (!panel) return;
    if (button.dataset.viewerAction === 'reset') send(panel, {action:'reset'});
    if (button.dataset.viewerAction === 'fullscreen') {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await panel.requestFullscreen();
      } catch (error) { console.warn('Fullscreen is unavailable.', error); }
    }
  });
  document.addEventListener('input', event => {
    if (!event.target.matches('.viewer-exposure input')) return;
    const value = Number(event.target.value), panel = event.target.closest('.gaussian-panel');
    panel.querySelector('.viewer-exposure output').textContent = value.toFixed(1) + ' EV';
    send(panel, {action:'exposure', value});
  });
  document.addEventListener('fullscreenchange', () => {
    document.querySelectorAll('[data-viewer-action="fullscreen"]').forEach(button => {
      button.textContent = document.fullscreenElement === button.closest('.gaussian-panel') ? 'Exit fullscreen' : 'Fullscreen';
    });
  });
  window.addEventListener('message', event => {
    if (event.origin !== location.origin || event.data?.kind !== 'hdr-gs-viewer-ready') return;
    const panel = [...document.querySelectorAll('.gaussian-panel')].find(
      candidate => candidate.querySelector('iframe')?.contentWindow === event.source);
    if (!panel || panel.dataset.mode !== event.data.mode) return;
    const link = panel.querySelector('.viewer-download');
    if (typeof event.data.download === 'string' && event.data.download.startsWith('blob:' + location.origin + '/')) {
      link.href = event.data.download;
      link.download = event.data.filename || panel.dataset.mode + '.splat';
    }
    panel.dataset.loaded = 'true';
  });
}
