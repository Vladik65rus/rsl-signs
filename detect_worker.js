// © 2026 Савченко Владислав Эдуардович (НГТУ). Все права защищены. Копирование без разрешения автора запрещено.
// Фоновый поток для нейросети MediaPipe: руки — в одном потоке, тело и лицо — в другом.
// Так они считаются одновременно, а страница рисует камеру на полной скорости экрана.
let H = null, P = null, F = null, kind = '', n = 0;
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      kind = m.kind;
      const V = await import(m.bundle);
      const fs = await V.FilesetResolver.forVisionTasks(m.wasm);
      if (kind === 'hand') {
        H = await V.HandLandmarker.createFromOptions(fs, { baseOptions: { modelAssetBuffer: m.bins.hand, delegate: 'CPU' },
          runningMode: 'VIDEO', numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 });
      } else {
        P = await V.PoseLandmarker.createFromOptions(fs, { baseOptions: { modelAssetBuffer: m.bins.pose, delegate: 'CPU' }, runningMode: 'VIDEO', numPoses: 1 });
        if (m.bins.face) try {
          F = await V.FaceLandmarker.createFromOptions(fs, { baseOptions: { modelAssetBuffer: m.bins.face, delegate: 'CPU' }, runningMode: 'VIDEO', numFaces: 1 });
        } catch (e2) { F = null; }
      }
      self.postMessage({ type: 'ready', kind });
      return;
    }
    if (m.type === 'frame') {
      const t0 = performance.now(), bmp = m.bmp, out = { type: 'res', kind, id: m.id, ts: m.ts };
      if (kind === 'hand') {
        const r = H.detectForVideo(bmp, m.ts);
        out.hands = (r.landmarks || []).map(l => l.map(p => [p.x, p.y, p.z]));
        out.labels = (r.landmarks || []).map((_, k) => ((r.handednesses?.[k]?.[0]?.categoryName || r.handedness?.[k]?.[0]?.categoryName) === 'Right') ? 0 : 1);
      } else {
        const r = P.detectForVideo(bmp, m.ts);
        const l = r.landmarks?.[0];
        out.pose = l ? l.map(p => [p.x, p.y, p.z, p.visibility ?? 1]) : null;
        if (F && m.face) {                                  // лицо — каждый 2-й раз: оно только для экрана
          const fr = F.detectForVideo(bmp, m.ts);
          out.face = fr.faceLandmarks?.[0] ? fr.faceLandmarks[0].map(p => [p.x, p.y]) : null;
        }
      }
      bmp.close();
      out.dt = performance.now() - t0;
      self.postMessage(out);
    }
  } catch (err) {
    try { m.bmp?.close?.(); } catch (e3) {}
    self.postMessage({ type: 'error', kind, msg: String(err?.message || err) });
  }
};
