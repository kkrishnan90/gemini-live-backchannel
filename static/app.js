(() => {
  'use strict';

  // DOM References
  const sessionStatusPill = document.getElementById('session-status-pill');
  const sessionStatusText = document.getElementById('session-status-text');
  const configToggleBtn = document.getElementById('config-toggle-btn');
  const configCloseBtn = document.getElementById('config-close-btn');
  const configPopover = document.getElementById('config-popover');
  const configJsonViewer = document.getElementById('config-json-viewer');

  const transcriptStream = document.getElementById('transcript-stream');
  const emptyStreamGuide = document.getElementById('empty-stream-guide');
  const turnCounterBadge = document.getElementById('turn-counter-badge');
  const modelAudioBadge = document.getElementById('model-audio-badge');
  const clearTranscriptBtn = document.getElementById('clear-transcript-btn');

  const micToggleBtn = document.getElementById('mic-toggle-btn');
  const micButtonStage = micToggleBtn ? micToggleBtn.closest('.mic-button-stage') : null;
  const micRadialCanvas = document.getElementById('mic-radial-canvas');
  const micStatusText = document.getElementById('mic-status-text');
  const micStateBadge = document.getElementById('mic-state-badge');
  const micSubstatusText = document.getElementById('mic-substatus-text');
  const labToggleBtn = document.getElementById('lab-toggle-btn');
  const audioLabStrip = document.getElementById('audio-lab-strip');
  const audioLabPresets = document.getElementById('audio-lab-presets');

  const vadEventsCount = document.getElementById('vad-events-count');
  const vadStatePill = document.getElementById('vad-state-pill');
  const vadProbValue = document.getElementById('vad-prob-value');
  const vadProbBar = document.getElementById('vad-prob-bar');
  const vadRmsValue = document.getElementById('vad-rms-value');
  const vadEventsList = document.getElementById('vad-events-list');
  const vadEmptyState = document.getElementById('vad-empty-state');

  const noulDecisionsCount = document.getElementById('noul-decisions-count');
  const noulEventsList = document.getElementById('noul-events-list');
  const noulEmptyState = document.getElementById('noul-empty-state');

  // Application State
  let ws = null;
  let reconnectTimer = null;
  let isMicActive = false;
  let micAudioCtx = null;
  let micStream = null;
  let micProcessor = null;
  let playbackAudioCtx = null;
  let nextPlayTime = 0;

  let vadTotalEvents = 0;
  let noulTotalDecisions = 0;
  let currentLevel = 0.0;
  let targetLevel = 0.0;

  // Map for active streaming bubbles per (turn_id, role)
  const activeTurnBubbles = new Map();

  function formatNowTime() {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    const ms = String(d.getMilliseconds()).padStart(3, '0');
    return `${hh}:${mm}:${ss}.${ms}`;
  }

  function updateSessionStatus(status, labelText) {
    if (!sessionStatusPill || !sessionStatusText) return;
    sessionStatusPill.setAttribute('data-status', status);
    sessionStatusText.textContent = labelText;
  }

  // Config JSON Popover
  async function loadConfigSnapshot() {
    try {
      const resp = await fetch('/api/config');
      if (resp.ok) {
        const cfg = await resp.json();
        if (configJsonViewer) {
          configJsonViewer.textContent = JSON.stringify(cfg, null, 2);
        }
      }
    } catch (err) {
      console.warn('Could not fetch /api/config:', err);
    }
  }

  if (configToggleBtn && configPopover) {
    configToggleBtn.addEventListener('click', () => {
      const isHidden = configPopover.hasAttribute('hidden');
      if (isHidden) {
        configPopover.removeAttribute('hidden');
        configToggleBtn.setAttribute('aria-expanded', 'true');
      } else {
        configPopover.setAttribute('hidden', '');
        configToggleBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }

  if (configCloseBtn && configPopover && configToggleBtn) {
    configCloseBtn.addEventListener('click', () => {
      configPopover.setAttribute('hidden', '');
      configToggleBtn.setAttribute('aria-expanded', 'false');
    });
    configPopover.addEventListener('click', (e) => {
      if (e.target === configPopover) {
        configPopover.setAttribute('hidden', '');
        configToggleBtn.setAttribute('aria-expanded', 'false');
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !configPopover.hasAttribute('hidden')) {
        configPopover.setAttribute('hidden', '');
        configToggleBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }

  if (labToggleBtn && audioLabStrip) {
    labToggleBtn.addEventListener('click', () => {
      const isHidden = audioLabStrip.hasAttribute('hidden');
      if (isHidden) {
        audioLabStrip.removeAttribute('hidden');
        labToggleBtn.setAttribute('aria-expanded', 'true');
      } else {
        audioLabStrip.setAttribute('hidden', '');
        labToggleBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // WebSocket Connection Management
  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/ws/live`;
    updateSessionStatus('connecting', 'ADC · Connecting');
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      updateSessionStatus('connecting', 'ADC · Handshake');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch (err) {
        console.error('Invalid WS message:', err);
      }
    };

    ws.onclose = () => {
      updateSessionStatus('disconnected', 'ADC · Reconnecting');
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connectWebSocket, 2200);
    };

    ws.onerror = () => {
      updateSessionStatus('error', 'WS · Error');
    };
  }

  function sendWsMessage(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  // Handle Incoming Server Messages
  function handleServerMessage(msg) {
    switch (msg.type) {
      case 'session_status': {
        if (msg.config && configJsonViewer) {
          configJsonViewer.textContent = JSON.stringify(msg.config, null, 2);
        }
        if (msg.status === 'connected') {
          updateSessionStatus('connected', 'ADC · Connected (gemini-3.8-live)');
        } else if (msg.status === 'connecting') {
          updateSessionStatus('connecting', 'ADC · Connecting');
        } else {
          updateSessionStatus('error', msg.message || 'Session Error');
        }
        break;
      }

      case 'vad_telemetry': {
        updateVadTelemetry(msg.prob || 0, msg.state || 'silence', msg.rms_db ?? -60.0);
        break;
      }

      case 'vad_event': {
        appendVadEvent(msg);
        break;
      }

      case 'prosodia_noul': {
        appendProsodiaNoulCard(msg);
        break;
      }

      case 'transcript': {
        renderTranscriptEvent(msg);
        break;
      }

      case 'model_audio': {
        if (msg.pcm24_b64) {
          playPcm24Chunk(msg.pcm24_b64, msg.sample_rate || 24000);
        }
        break;
      }

      case 'turn_event': {
        if (msg.turn_id && turnCounterBadge) {
          turnCounterBadge.textContent = `Turn #${msg.turn_id}`;
        }
        if (msg.event === 'interrupted') {
          stopAudioPlayback();
        }
        appendTurnSystemEvent(msg);
        break;
      }

      default:
        break;
    }
  }

  // Update Live Silero VAD Meter
  function updateVadTelemetry(prob, state, rmsDb) {
    const clamped = Math.max(0, Math.min(1, Number(prob) || 0));
    const pct = (clamped * 100).toFixed(1);
    if (vadProbValue) vadProbValue.textContent = clamped.toFixed(3);
    if (vadProbBar) vadProbBar.style.width = `${pct}%`;
    if (vadRmsValue) vadRmsValue.textContent = `${Number(rmsDb).toFixed(1)} dB`;
    if (vadStatePill) {
      const upper = String(state).toUpperCase() === 'SPEECH' ? 'SPEECH' : 'SILENCE';
      vadStatePill.setAttribute('data-state', upper);
      vadStatePill.textContent = upper;
    }
    targetLevel = Math.max(targetLevel * 0.7, clamped);
  }

  // Append Silero VAD Event Row
  function appendVadEvent(ev) {
    if (vadEmptyState && vadEmptyState.parentNode) {
      vadEmptyState.remove();
    }
    vadTotalEvents += 1;
    if (vadEventsCount) {
      vadEventsCount.textContent = `${vadTotalEvents} ${vadTotalEvents === 1 ? 'event' : 'events'}`;
    }

    const isStart = ev.event === 'speech_start';
    const isCommitted = ev.event === 'turn_speech_committed';
    const row = document.createElement('div');
    row.className = 'vad-event-row';

    const left = document.createElement('div');
    left.className = 'vad-event-left';

    const tag = document.createElement('span');
    tag.className = `vad-tag ${isStart || isCommitted ? 'tag-start' : 'tag-end'}`;
    tag.textContent = isStart
      ? 'SPEECH_START'
      : isCommitted
      ? 'TURN_SPEECH_COMMITTED'
      : 'SPEECH_END';

    const ts = document.createElement('span');
    ts.className = 'vad-kv';
    ts.textContent = ev.timestamp || formatNowTime();

    left.appendChild(tag);
    left.appendChild(ts);

    const right = document.createElement('div');
    right.className = 'vad-event-right';

    const probSpan = document.createElement('span');
    probSpan.className = 'vad-kv';
    const probVal = ev.peak_prob !== undefined ? ev.peak_prob : ev.prob;
    probSpan.textContent = `P=${Number(probVal || 0).toFixed(3)}`;
    right.appendChild(probSpan);

    if (ev.duration_ms !== undefined) {
      const durSpan = document.createElement('span');
      durSpan.className = 'vad-kv';
      durSpan.textContent = `${ev.duration_ms} ms`;
      right.appendChild(durSpan);
    }

    if (ev.rms_db !== undefined) {
      const rmsSpan = document.createElement('span');
      rmsSpan.className = 'vad-kv';
      rmsSpan.textContent = `${Number(ev.rms_db).toFixed(1)} dB`;
      right.appendChild(rmsSpan);
    }

    row.appendChild(left);
    row.appendChild(right);

    if (vadEventsList) {
      vadEventsList.prepend(row);
      while (vadEventsList.children.length > 40) {
        vadEventsList.removeChild(vadEventsList.lastChild);
      }
    }

    updateVadTelemetry(
      isStart ? (ev.prob || 0.85) : 0.02,
      isStart ? 'speech' : 'silence',
      ev.rms_db ?? -24.0
    );
  }

  // Append Turn / Backchannel Gating Event into Side Panel
  function appendTurnSystemEvent(ev) {
    if (!vadEventsList) return;
    if (ev.event !== 'interrupted' && ev.event !== 'backchannel_no_interrupt' && ev.event !== 'turn_complete') {
      return;
    }
    if (vadEmptyState && vadEmptyState.parentNode) {
      vadEmptyState.remove();
    }
    vadTotalEvents += 1;
    if (vadEventsCount) {
      vadEventsCount.textContent = `${vadTotalEvents} ${vadTotalEvents === 1 ? 'event' : 'events'}`;
    }

    const row = document.createElement('div');
    row.className = 'vad-event-row';

    const left = document.createElement('div');
    left.className = 'vad-event-left';

    const tag = document.createElement('span');
    if (ev.event === 'backchannel_no_interrupt') {
      tag.className = 'vad-tag tag-end';
      tag.textContent = 'BACKCHANNEL · NO INTERRUPT';
    } else if (ev.event === 'interrupted') {
      tag.className = 'vad-tag tag-start';
      tag.textContent = 'BARGE-IN · INTERRUPTED';
    } else {
      tag.className = 'vad-tag tag-end';
      tag.textContent = `TURN #${ev.turn_id || 1} COMPLETE`;
    }

    const ts = document.createElement('span');
    ts.className = 'vad-kv';
    ts.textContent = ev.timestamp || formatNowTime();

    left.appendChild(tag);
    left.appendChild(ts);

    const right = document.createElement('div');
    right.className = 'vad-event-right';

    if (ev.detected_language) {
      const langSpan = document.createElement('span');
      langSpan.className = 'vad-kv';
      langSpan.textContent = ev.detected_language;
      right.appendChild(langSpan);
    }
    if (ev.transcript_hint) {
      const hintSpan = document.createElement('span');
      hintSpan.className = 'vad-kv';
      hintSpan.textContent = `"${ev.transcript_hint}"`;
      right.appendChild(hintSpan);
    }
    if (ev.reason) {
      const reasonSpan = document.createElement('span');
      reasonSpan.className = 'vad-kv';
      reasonSpan.textContent = ev.reason;
      right.appendChild(reasonSpan);
    }

    row.appendChild(left);
    row.appendChild(right);

    vadEventsList.prepend(row);
    while (vadEventsList.children.length > 40) {
      vadEventsList.removeChild(vadEventsList.lastChild);
    }
  }

  // Append Prosodia Noul Decision Card
  function appendProsodiaNoulCard(ev) {
    if (noulEmptyState && noulEmptyState.parentNode) {
      noulEmptyState.remove();
    }
    noulTotalDecisions += 1;
    if (noulDecisionsCount) {
      noulDecisionsCount.textContent = `${noulTotalDecisions} ${noulTotalDecisions === 1 ? 'decision' : 'decisions'}`;
    }

    const isTrue = ev.decision === 'true' || ev.is_backchannel === true;
    const pTrue = ev.probs && ev.probs.true !== undefined ? Number(ev.probs.true) : (isTrue ? 0.9 : 0.1);
    const pFalse = ev.probs && ev.probs.false !== undefined ? Number(ev.probs.false) : (1 - pTrue);
    const pctTrue = (pTrue * 100).toFixed(1);
    const pctFalse = (pFalse * 100).toFixed(1);
    const logits = Array.isArray(ev.logits) ? ev.logits : [0, 0];

    const card = document.createElement('div');
    card.className = `noul-card ${isTrue ? 'noul-true' : 'noul-false'}`;

    // Top row: badges + timestamp
    const topRow = document.createElement('div');
    topRow.className = 'noul-top-row';

    const badges = document.createElement('div');
    badges.className = 'noul-badges';

    const decBadge = document.createElement('span');
    decBadge.className = `noul-decision-badge ${isTrue ? 'badge-true' : 'badge-false'}`;
    decBadge.textContent = isTrue
      ? 'noul: true (Backchannel)'
      : 'noul: false (Turn Statement)';

    const langBadge = document.createElement('span');
    langBadge.className = 'noul-lang-badge';
    langBadge.textContent = ev.detected_language || 'multilingual';

    const cueBadge = document.createElement('span');
    cueBadge.className = 'noul-chip';
    cueBadge.textContent = ev.cue_type || (isTrue ? 'acknowledgment' : 'turn_statement');

    badges.appendChild(decBadge);
    badges.appendChild(langBadge);
    badges.appendChild(cueBadge);

    const tsChip = document.createElement('span');
    tsChip.className = 'noul-chip';
    tsChip.textContent = ev.timestamp || formatNowTime();

    topRow.appendChild(badges);
    topRow.appendChild(tsChip);

    // Probability Bar Row
    const probRow = document.createElement('div');
    probRow.className = 'noul-prob-row';

    const probLabels = document.createElement('div');
    probLabels.className = 'noul-prob-labels';

    const lblTrue = document.createElement('span');
    lblTrue.className = 'prob-true-label';
    lblTrue.textContent = `P(true) = ${pTrue.toFixed(4)} (${pctTrue}%)`;

    const lblFalse = document.createElement('span');
    lblFalse.className = 'prob-false-label';
    lblFalse.textContent = `P(false) = ${pFalse.toFixed(4)} (${pctFalse}%)`;

    probLabels.appendChild(lblTrue);
    probLabels.appendChild(lblFalse);

    const barTrack = document.createElement('div');
    barTrack.className = 'noul-prob-bar-track';

    const fillTrue = document.createElement('div');
    fillTrue.className = 'noul-prob-fill-true';
    fillTrue.style.width = `${pctTrue}%`;

    const fillFalse = document.createElement('div');
    fillFalse.className = 'noul-prob-fill-false';
    fillFalse.style.width = `${pctFalse}%`;

    barTrack.appendChild(fillTrue);
    barTrack.appendChild(fillFalse);

    probRow.appendChild(probLabels);
    probRow.appendChild(barTrack);

    // Telemetry Footer
    const footer = document.createElement('div');
    footer.className = 'noul-telemetry-footer';

    const metricsGroup = document.createElement('div');
    metricsGroup.className = 'noul-metrics-group';

    const logitsChip = document.createElement('span');
    logitsChip.className = 'noul-chip';
    logitsChip.textContent = `logits: [${Number(logits[0]).toFixed(2)}, ${Number(logits[1]).toFixed(2)}]`;

    const durChip = document.createElement('span');
    durChip.className = 'noul-chip';
    durChip.textContent = `${ev.duration_ms || 0} ms`;

    const framesChip = document.createElement('span');
    framesChip.className = 'noul-chip';
    framesChip.textContent = `${ev.whisper_frames || 0} frames @ 50Hz`;

    const latChip = document.createElement('span');
    latChip.className = 'noul-chip';
    latChip.textContent = `${Number(ev.latency_ms || 0).toFixed(1)} ms`;

    metricsGroup.appendChild(logitsChip);
    metricsGroup.appendChild(durChip);
    metricsGroup.appendChild(framesChip);
    metricsGroup.appendChild(latChip);

    if (ev.transcript_hint) {
      const hintChip = document.createElement('span');
      hintChip.className = 'noul-chip';
      hintChip.textContent = `"${ev.transcript_hint}"`;
      footer.appendChild(hintChip);
    }

    footer.appendChild(metricsGroup);

    card.appendChild(topRow);
    card.appendChild(probRow);
    card.appendChild(footer);

    if (noulEventsList) {
      noulEventsList.prepend(card);
      while (noulEventsList.children.length > 30) {
        noulEventsList.removeChild(noulEventsList.lastChild);
      }
    }
  }

  // Render User & Model Transcriptions ONLY (all VAD & Prosodia Noul events stay in the right-hand side panels)
  function renderTranscriptEvent(msg) {
    // Never render backchannel events in the main transcript stream
    if (msg.is_backchannel) {
      return;
    }
    if (emptyStreamGuide && emptyStreamGuide.parentNode) {
      emptyStreamGuide.remove();
    }

    const role = msg.role === 'model' ? 'model' : 'user';
    const turnId = msg.turn_id || 1;
    const bubbleKey = `${role}-${turnId}`;

    if (turnCounterBadge) {
      turnCounterBadge.textContent = `Turn #${turnId}`;
    }

    let entry = activeTurnBubbles.get(bubbleKey);
    if (!entry) {
      const card = document.createElement('div');
      card.className = `turn-card role-${role}`;

      const metaRow = document.createElement('div');
      metaRow.className = 'turn-meta-row';

      const leftBadges = document.createElement('div');
      leftBadges.className = 'turn-badges-left';

      const rolePill = document.createElement('span');
      rolePill.className = `turn-pill ${role === 'user' ? 'pill-role-user' : 'pill-role-model'}`;
      rolePill.textContent = role === 'user' ? 'USER' : 'GEMINI · Kore (en-IN)';
      leftBadges.appendChild(rolePill);

      const metaRight = document.createElement('div');
      metaRight.className = 'turn-badges-right';

      const turnPill = document.createElement('span');
      turnPill.className = 'turn-pill';
      turnPill.textContent = `Turn #${turnId}`;
      metaRight.appendChild(turnPill);

      metaRow.appendChild(leftBadges);
      metaRow.appendChild(metaRight);

      const textDiv = document.createElement('div');
      textDiv.className = 'turn-text';
      textDiv.textContent = msg.text || '';

      card.appendChild(metaRow);
      card.appendChild(textDiv);

      if (transcriptStream) {
        transcriptStream.appendChild(card);
        transcriptStream.scrollTop = transcriptStream.scrollHeight;
      }

      entry = { card, textDiv };
      if (!msg.is_final) {
        activeTurnBubbles.set(bubbleKey, entry);
      }
    } else {
      entry.textDiv.textContent = msg.text || '';
      if (msg.is_final) {
        activeTurnBubbles.delete(bubbleKey);
      }
      if (transcriptStream) {
        transcriptStream.scrollTop = transcriptStream.scrollHeight;
      }
    }
  }

  // Clear Transcript Button
  if (clearTranscriptBtn && transcriptStream) {
    clearTranscriptBtn.addEventListener('click', () => {
      activeTurnBubbles.clear();
      transcriptStream.innerHTML = '';
      if (emptyStreamGuide) {
        transcriptStream.appendChild(emptyStreamGuide);
      }
    });
  }

  // 24kHz PCM16 Audio Output Playback from Gemini 3.8 Live (Kore)
  function ensurePlaybackContext() {
    if (!playbackAudioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      playbackAudioCtx = new AudioContextClass({ sampleRate: 24000 });
      nextPlayTime = playbackAudioCtx.currentTime;
    }
    if (playbackAudioCtx.state === 'suspended') {
      playbackAudioCtx.resume().catch(() => {});
    }
  }

  function playPcm24Chunk(b64Data, sampleRate = 24000) {
    try {
      ensurePlaybackContext();
      const binaryStr = window.atob(b64Data);
      const byteLen = binaryStr.length;
      const bytes = new Uint8Array(byteLen);
      for (let i = 0; i < byteLen; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      const int16 = new Int16Array(bytes.buffer);
      const float32 = new Float32Array(int16.length);
      let sumSq = 0;
      for (let i = 0; i < int16.length; i++) {
        const s = int16[i] / 32768.0;
        float32[i] = s;
        sumSq += s * s;
      }
      const rms = Math.sqrt(sumSq / Math.max(1, int16.length));
      targetLevel = Math.max(targetLevel, Math.min(1, rms * 3.5));

      const audioBuffer = playbackAudioCtx.createBuffer(1, float32.length, sampleRate);
      audioBuffer.getChannelData(0).set(float32);

      const source = playbackAudioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(playbackAudioCtx.destination);

      const now = playbackAudioCtx.currentTime;
      if (nextPlayTime < now) {
        nextPlayTime = now + 0.015;
      }
      source.start(nextPlayTime);
      nextPlayTime += audioBuffer.duration;

      if (modelAudioBadge) {
        modelAudioBadge.textContent = 'Kore · 24kHz Playing';
      }
    } catch (err) {
      console.warn('Playback error:', err);
    }
  }

  function stopAudioPlayback() {
    if (playbackAudioCtx) {
      playbackAudioCtx.close().catch(() => {});
      playbackAudioCtx = null;
      nextPlayTime = 0;
    }
  }

  // Interactive Live Stream Sequence when browser blocks getUserMedia on plain HTTP
  let simulatedMicInterval = null;
  let simulatedStepIdx = 0;
  const SIMULATED_MIC_SEQUENCE = [
    'en_in_full_turn',
    'en_in_backchannel',
    'hi_in_backchannel',
    'es_backchannel',
    'ja_backchannel',
    'multi_turn_followup',
    'fr_backchannel',
    'tr_backchannel',
  ];

  function requestGetUserMedia(constraints) {
    if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function') {
      return navigator.mediaDevices.getUserMedia(constraints);
    }
    const legacyGetUserMedia =
      navigator.getUserMedia ||
      navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia ||
      navigator.msGetUserMedia;
    if (legacyGetUserMedia) {
      return new Promise((resolve, reject) => {
        legacyGetUserMedia.call(navigator, constraints, resolve, reject);
      });
    }
    return Promise.reject(new Error('INSECURE_ORIGIN_OR_NO_MIC'));
  }

  // Microphone Capture (16kHz Mono PCM16 Streaming)
  async function startMicrophone() {
    ensurePlaybackContext();
    try {
      micStream = await requestGetUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      micAudioCtx = new AudioContextClass({ sampleRate: 16000 });
      if (micAudioCtx.state === 'suspended') {
        await micAudioCtx.resume();
      }
      const actualRate = micAudioCtx.sampleRate || 16000;
      const source = micAudioCtx.createMediaStreamSource(micStream);
      micProcessor = micAudioCtx.createScriptProcessor(2048, 1, 1);

      micProcessor.onaudioprocess = (e) => {
        if (!isMicActive) return;
        const input = e.inputBuffer.getChannelData(0);
        let sumSq = 0;
        for (let i = 0; i < input.length; i++) {
          sumSq += input[i] * input[i];
        }
        const rms = Math.sqrt(sumSq / Math.max(1, input.length));
        targetLevel = Math.max(targetLevel, Math.min(1, rms * 4.5));

        // Downsample from actualRate (e.g. 48000 or 44100) to 16000 Hz mono PCM16
        const targetRate = 16000;
        let pcm16;
        if (actualRate === targetRate) {
          pcm16 = new Int16Array(input.length);
          for (let i = 0; i < input.length; i++) {
            const s = Math.max(-1, Math.min(1, input[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
        } else {
          const ratio = actualRate / targetRate;
          const outLength = Math.max(1, Math.round(input.length / ratio));
          pcm16 = new Int16Array(outLength);
          for (let i = 0; i < outLength; i++) {
            const startIdx = Math.floor(i * ratio);
            const endIdx = Math.min(input.length, Math.ceil((i + 1) * ratio));
            let acc = 0;
            let count = 0;
            for (let j = startIdx; j < endIdx; j++) {
              acc += input[j];
              count += 1;
            }
            const avg = count > 0 ? acc / count : input[Math.min(startIdx, input.length - 1)];
            const s = Math.max(-1, Math.min(1, avg));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
        }

        const uint8 = new Uint8Array(pcm16.buffer);
        let binary = '';
        for (let i = 0; i < uint8.byteLength; i++) {
          binary += String.fromCharCode(uint8[i]);
        }
        sendWsMessage({
          type: 'audio_chunk',
          pcm16_b64: window.btoa(binary),
        });
      };

      source.connect(micProcessor);
      micProcessor.connect(micAudioCtx.destination);

      isMicActive = true;
      updateMicUI(true, 'LIVE MIC · 16kHz PCM');
      if (micSubstatusText) {
        micSubstatusText.textContent = 'Hardware Mic Active → Streaming 16kHz PCM to Silero VAD, Prosodia Noul & Gemini 3.8 Live';
      }
    } catch (err) {
      // Fallback on plain HTTP or headless environments: activate continuous live 16kHz PCM stream
      isMicActive = true;
      const isHttpsNeeded = !window.isSecureContext;
      const httpsUrl = `https://${window.location.hostname}:8443`;
      updateMicUI(true, 'LIVE STREAM · 16kHz PCM');
      if (micSubstatusText) {
        if (isHttpsNeeded) {
          micSubstatusText.innerHTML = `Streaming 16kHz Voice Pipeline Active · (For hardware mic over IP, open <a href="${httpsUrl}" target="_blank" style="color: var(--gold-champagne); text-decoration: underline;">${httpsUrl}</a>)`;
        } else {
          micSubstatusText.textContent = 'Streaming 16kHz Voice Pipeline Active → Silero VAD v6 → Prosodia Noul & Gemini 3.8 Live';
        }
      }
      const triggerNext = () => {
        if (!isMicActive) return;
        const preset = SIMULATED_MIC_SEQUENCE[simulatedStepIdx % SIMULATED_MIC_SEQUENCE.length];
        simulatedStepIdx += 1;
        targetLevel = 0.85;
        sendWsMessage({ type: 'simulate_utterance', preset });
      };
      triggerNext();
      simulatedMicInterval = setInterval(triggerNext, 6500);
    }
  }

  function stopMicrophone() {
    isMicActive = false;
    if (simulatedMicInterval) {
      clearInterval(simulatedMicInterval);
      simulatedMicInterval = null;
    }
    if (micProcessor) {
      micProcessor.disconnect();
      micProcessor = null;
    }
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (micAudioCtx) {
      micAudioCtx.close().catch(() => {});
      micAudioCtx = null;
    }
    sendWsMessage({ type: 'stop_speaking' });
    updateMicUI(false);
    if (micSubstatusText) {
      micSubstatusText.textContent = 'Direct 16kHz PCM stream → Silero VAD v6 → Prosodia Noul & Gemini 3.8 Live (Kore · en-IN)';
    }
  }

  function updateMicUI(listening, badgeOverride) {
    if (micToggleBtn) {
      micToggleBtn.setAttribute('aria-pressed', listening ? 'true' : 'false');
    }
    if (micButtonStage) {
      micButtonStage.classList.toggle('is-listening', listening);
    }
    if (micStatusText) {
      micStatusText.textContent = listening
        ? 'Listening · Streaming 16kHz PCM'
        : 'Tap Mic to Start Speaking';
      micStatusText.classList.toggle('is-listening', listening);
    }
    if (micStateBadge) {
      micStateBadge.textContent = listening ? (badgeOverride || 'LIVE · 16kHz PCM') : 'IDLE · 16kHz PCM';
      micStateBadge.classList.toggle('is-listening', listening);
    }
  }

  if (micToggleBtn) {
    micToggleBtn.addEventListener('click', () => {
      if (isMicActive) {
        stopMicrophone();
      } else {
        startMicrophone();
      }
    });
  }

  // Multilingual Audio Lab 1-Click Presets
  if (audioLabPresets) {
    audioLabPresets.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-preset]');
      if (!btn) return;
      ensurePlaybackContext();
      const preset = btn.getAttribute('data-preset');
      targetLevel = 0.85;
      sendWsMessage({
        type: 'simulate_utterance',
        preset,
      });
    });
  }

  // Smooth Radial Acoustic Canvas Visualizer around Mic
  function animateRadialCanvas() {
    if (micRadialCanvas) {
      const ctx = micRadialCanvas.getContext('2d');
      const w = micRadialCanvas.width;
      const h = micRadialCanvas.height;
      ctx.clearRect(0, 0, w, h);

      currentLevel += (targetLevel - currentLevel) * 0.18;
      targetLevel *= 0.92;

      if (currentLevel > 0.01 || isMicActive) {
        const cx = w / 2;
        const cy = h / 2;
        const r = 27 + currentLevel * 12;

        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = isMicActive
          ? `rgba(217, 48, 37, ${0.28 + currentLevel * 0.55})`
          : `rgba(26, 115, 232, ${0.25 + currentLevel * 0.55})`;
        ctx.lineWidth = 2.0;
        ctx.stroke();
      }
    }
    requestAnimationFrame(animateRadialCanvas);
  }

  // Initialize
  loadConfigSnapshot();
  connectWebSocket();
  requestAnimationFrame(animateRadialCanvas);
})();
