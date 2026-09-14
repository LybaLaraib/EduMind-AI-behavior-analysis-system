(() => {
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("captureCanvas");
  const preview = document.getElementById("previewOverlay");
  const placeholder = document.getElementById("cameraPlaceholder");
  const liveLabels = document.getElementById("liveLabels");
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const badge = document.getElementById("sessionBadge");
  const scoreEl = document.getElementById("attentionScore");
  const scoreCircle = document.getElementById("scoreCircle");
  const frameCounter = document.getElementById("frameCounter");
  const factsList = document.getElementById("factsList");
  const recEl = document.getElementById("outRecommendation");
  const metricsLine = document.getElementById("metricsLine");
  const postureStatus = document.getElementById("postureStatus");
  const eyesHoldStatus = document.getElementById("eyesHoldStatus");
  const inferenceSteps = document.getElementById("inferenceSteps");
  const livePulse = document.getElementById("livePulse");
  const sessEyesClosed = document.getElementById("sessEyesClosed");
  const sessLookingAway = document.getElementById("sessLookingAway");
  const sessHeadDown = document.getElementById("sessHeadDown");
  const sessPhone = document.getElementById("sessPhone");

  let stream = null;
  let analyzeTimer = null;
  let monitoringActive = false;
  let sessionEpoch = 0;
  let frames = 0;
  let displayScore = 0;
  let pendingPreview = null;
  let previewRaf = null;
  const CIRCUMFERENCE = 2 * Math.PI * 52;
  const ANALYZE_MS = 1000;
  const ANALYZE_TIMEOUT_MS = 45000;

  function formatFactTotal(entry) {
    if (!entry) return "0 sec · 0 frames";
    const sec = entry.seconds ?? 0;
    const fr = entry.frames ?? 0;
    const secLabel =
      sec < 60
        ? `${Number(sec).toFixed(1)} sec`
        : `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} sec`;
    return `${secLabel} · ${fr} frames`;
  }

  function updateSessionFacts(totals) {
    if (!totals) return;
    if (sessEyesClosed) sessEyesClosed.textContent = formatFactTotal(totals.eyes_closed);
    if (sessLookingAway) sessLookingAway.textContent = formatFactTotal(totals.looking_away);
    if (sessHeadDown) sessHeadDown.textContent = formatFactTotal(totals.head_down);
    if (sessPhone) sessPhone.textContent = formatFactTotal(totals.phone_suspected);
  }

  function setScore(score) {
    const target = Math.round(score);
    const step = () => {
      if (Math.abs(displayScore - target) < 1) {
        displayScore = target;
      } else {
        displayScore += (target - displayScore) * 0.25;
      }
      const val = Math.round(displayScore);
      scoreEl.textContent = val;
      scoreCircle.style.strokeDashoffset =
        CIRCUMFERENCE - (val / 100) * CIRCUMFERENCE;
      if (val !== target) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function pillClass(id, value) {
    const v = (value || "").toLowerCase();
    if (id === "outRisk") {
      if (v === "low") return "pill-ok";
      if (v === "high") return "pill-danger";
      return "pill-warn";
    }
    if (id === "outAttention") {
      if (v === "high") return "pill-ok";
      if (v === "low") return "pill-danger";
      return "pill-warn";
    }
    if (["attentive"].includes(v)) return "pill-ok";
    if (["disengaged"].includes(v)) return "pill-danger";
    if (["distracted"].includes(v)) return "pill-warn";
    return "";
  }

  function updatePills(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value || "—";
    el.className = "pill " + pillClass(id, value);
  }

  function renderLiveLabels(labels) {
    if (!liveLabels) return;
    if (!labels?.length) {
      liveLabels.innerHTML = "";
      return;
    }
    liveLabels.innerHTML = labels
      .map(
        (lb) =>
          `<span class="live-label live-label-${lb.tone || "warn"}">${lb.text}</span>`
      )
      .join("");
  }

  function updateInferenceSteps(steps) {
    if (!inferenceSteps) return;
    if (!steps?.length) {
      inferenceSteps.innerHTML = '<li class="muted">No inference steps yet.</li>';
      return;
    }
    inferenceSteps.innerHTML = steps
      .slice(-6)
      .map(
        (s) =>
          `<li><strong>${s.technique || "Inference"}</strong> — <code>${s.rule || ""}</code><br><span class="muted small">${s.result || ""}</span></li>`
      )
      .join("");
  }

  function flushPreview() {
    if (!pendingPreview || !monitoringActive) {
      pendingPreview = null;
      return;
    }
    const src = pendingPreview;
    pendingPreview = null;
    preview.hidden = false;
    if (preview.src !== src) {
      preview.onload = () => {
        preview.style.opacity = "1";
        if (pendingPreview) schedulePreviewFlush();
      };
      preview.onerror = () => {
        preview.hidden = true;
        preview.style.opacity = "0";
      };
      preview.style.opacity = "0.35";
      preview.src = src;
    } else {
      preview.style.opacity = "1";
    }
    if (pendingPreview) schedulePreviewFlush();
  }

  function schedulePreviewFlush() {
    if (previewRaf) return;
    previewRaf = requestAnimationFrame(() => {
      previewRaf = null;
      flushPreview();
    });
  }

  function smoothShowPreview(src) {
    if (!src || !monitoringActive) return;
    pendingPreview = src;
    schedulePreviewFlush();
  }

  function updateUI(data) {
    if (!monitoringActive) return;

    const inf = data.inference || {};
    const obs = data.observation || {};

    updatePills("outAttention", inf.attention);
    updatePills("outBehavior", inf.behavior);
    updatePills("outRisk", inf.risk);

    const rec = inf.recommendation || "—";
    if (recEl && recEl.textContent !== rec) {
      recEl.classList.remove("advice-pulse");
      void recEl.offsetWidth;
      recEl.textContent = rec;
      recEl.classList.add("advice-pulse");
    }

    const reasonEl = document.getElementById("outReason");
    if (reasonEl) reasonEl.textContent = inf.reason || "—";

    const facts = obs.facts || {};
    const active = Object.keys(facts).filter((k) => facts[k]);
    factsList.innerHTML =
      active.length === 0
        ? '<li class="muted">No active facts</li>'
        : active.map((f) => `<li>${f}</li>`).join("");

    const m = obs.metrics || {};
    const earDetail =
      m.ear_left != null
        ? `EAR L${m.ear_left} R${m.ear_right} (avg ${m.ear})`
        : `EAR ${m.ear ?? "—"}`;
    metricsLine.textContent = `Yaw ${m.head_yaw ?? "—"}° · Pitch ${m.head_pitch ?? "—"}° · ${earDetail}`;

    if (active.length) {
      postureStatus.textContent = active.slice(0, 4).join(" · ");
    } else if (facts.eyes_closing || m.eyes_closed_raw > 0) {
      postureStatus.textContent = "eyes_closing";
    } else if (facts.no_face || obs.message?.toLowerCase().includes("no face")) {
      postureStatus.textContent = "No face — center yourself in frame";
    } else {
      postureStatus.textContent = "Tracking…";
    }

    if (eyesHoldStatus) {
      const progress = m.eyes_closed_progress ?? 0;
      if (facts.eyes_closed) {
        eyesHoldStatus.textContent = "Eyes: closed (confirmed 2s+)";
      } else if (m.eyes_closed_raw > 0) {
        eyesHoldStatus.textContent = `Eyes: holding ${Math.round(progress * 100)}% to 2s`;
      } else {
        eyesHoldStatus.textContent = "Eyes: open";
      }
    }

    renderLiveLabels(data.labels || obs.labels);
    updateInferenceSteps(inf.inference_steps);
    updateSessionFacts(data.session_facts);

    setScore(data.attention_score || 0);
    frames = data.frame || frames;
    frameCounter.textContent = `Frames analyzed: ${frames}`;

    if (data.preview) smoothShowPreview(data.preview);
    if (livePulse) livePulse.classList.add("active");
  }

  function captureFrameDataUrl() {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return null;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    canvas.width = w;
    canvas.height = h;
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();
    return canvas.toDataURL("image/jpeg", 0.72);
  }

  async function fetchJson(url, options = {}, timeoutMs = ANALYZE_TIMEOUT_MS) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || `Request failed (${res.status})`);
      }
      return data;
    } finally {
      clearTimeout(timer);
    }
  }

  async function captureAndAnalyze(epoch) {
    if (!monitoringActive || epoch !== sessionEpoch) return;

    const image = captureFrameDataUrl();
    if (!image) {
      if (postureStatus) postureStatus.textContent = "Waiting for camera frames…";
      return;
    }

    try {
      const data = await fetchJson(
        "/api/analyze",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image, annotate: true }),
        },
        ANALYZE_TIMEOUT_MS
      );
      if (!monitoringActive || epoch !== sessionEpoch) return;
      if (data.ok) {
        updateUI(data);
      } else if (postureStatus) {
        postureStatus.textContent = data.error || "Analysis failed";
      }
    } catch (err) {
      if (!monitoringActive || epoch !== sessionEpoch) return;
      console.error("Analyze failed", err);
      if (postureStatus) {
        postureStatus.textContent =
          err.name === "AbortError"
            ? "Analysis timed out — retrying…"
            : "Analysis error — retrying…";
      }
    }
  }

  async function runAnalyzeLoop(epoch) {
    if (!monitoringActive || epoch !== sessionEpoch) return;

    const started = performance.now();
    await captureAndAnalyze(epoch);
    if (!monitoringActive || epoch !== sessionEpoch) return;

    const elapsed = performance.now() - started;
    const delay = Math.max(80, ANALYZE_MS - elapsed);
    analyzeTimer = setTimeout(() => runAnalyzeLoop(epoch), delay);
  }

  function waitForVideoReady() {
    return new Promise((resolve, reject) => {
      if (video.videoWidth > 0 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        resolve();
        return;
      }

      const timeout = setTimeout(() => {
        cleanup();
        reject(new Error("Camera stream did not become ready"));
      }, 12000);

      const cleanup = () => {
        clearTimeout(timeout);
        video.removeEventListener("loadedmetadata", onReady);
        video.removeEventListener("canplay", onReady);
        video.removeEventListener("playing", onReady);
      };

      const onReady = () => {
        if (video.videoWidth > 0) {
          cleanup();
          resolve();
        }
      };

      video.addEventListener("loadedmetadata", onReady);
      video.addEventListener("canplay", onReady);
      video.addEventListener("playing", onReady);
      onReady();
    });
  }

  async function requestCameraStream() {
    const constraints = {
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    };
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      if (err.name === "OverconstrainedError" || err.name === "NotFoundError") {
        return await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      throw err;
    }
  }

  function stopCameraTracks() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.srcObject = null;
  }

  async function startMonitoring() {
    const isLocalhost = ["localhost", "127.0.0.1", "::1"].includes(location.hostname);
    const isSecureOrigin = window.isSecureContext || isLocalhost;

    if (!navigator.mediaDevices?.getUserMedia) {
      alert(
        isSecureOrigin
          ? "Camera is unavailable in this browser."
          : "Webcam requires HTTPS or localhost."
      );
      return;
    }

    if (!isSecureOrigin) {
      alert("Open the app at http://localhost:5000 or use HTTPS.");
      return;
    }

    btnStart.disabled = true;

    try {
      stream = await requestCameraStream();
      video.srcObject = stream;
      video.style.opacity = "1";
      placeholder.style.display = "none";
      preview.hidden = true;
      preview.removeAttribute("src");
      pendingPreview = null;

      await waitForVideoReady();
      try {
        await video.play();
      } catch (_) {
        /* autoplay with muted is usually allowed */
      }

      const startData = await fetchJson(
        "/api/session/start",
        { method: "POST" },
        15000
      );
      if (startData.active === false) {
        throw new Error("Could not start session on server");
      }

      monitoringActive = true;
      sessionEpoch += 1;
      const epoch = sessionEpoch;

      badge.textContent = "Session active";
      badge.classList.add("active");
      btnStop.disabled = false;
      if (postureStatus) postureStatus.textContent = "Analyzing…";

      updateSessionFacts({
        eyes_closed: { seconds: 0, frames: 0 },
        looking_away: { seconds: 0, frames: 0 },
        head_down: { seconds: 0, frames: 0 },
        phone_suspected: { seconds: 0, frames: 0 },
      });

      runAnalyzeLoop(epoch);
    } catch (err) {
      monitoringActive = false;
      stopCameraTracks();
      placeholder.style.display = "flex";
      preview.hidden = true;
      btnStart.disabled = false;
      btnStop.disabled = true;
      alert(
        err.message === "Camera stream did not become ready"
          ? "Camera opened but video is not ready. Try another browser or camera."
          : "Camera access denied or unavailable."
      );
      console.error(err);
    }
  }

  async function stopMonitoring() {
    monitoringActive = false;
    sessionEpoch += 1;

    if (analyzeTimer) {
      clearTimeout(analyzeTimer);
      analyzeTimer = null;
    }

    stopCameraTracks();

    preview.hidden = true;
    preview.removeAttribute("src");
    pendingPreview = null;
    video.style.opacity = "1";
    placeholder.style.display = "flex";
    if (liveLabels) liveLabels.innerHTML = "";
    badge.textContent = "Session idle";
    badge.classList.remove("active");
    btnStart.disabled = false;
    btnStop.disabled = true;
    if (livePulse) livePulse.classList.remove("active");

    try {
      await fetchJson("/api/session/stop", { method: "POST" }, 15000);
    } catch (err) {
      console.warn("Session stop failed", err);
    }
  }

  btnStart?.addEventListener("click", startMonitoring);
  btnStop?.addEventListener("click", stopMonitoring);

  function endSessionBeacon() {
    try {
      navigator.sendBeacon?.("/api/session/stop", new Blob([], { type: "application/json" }));
    } catch (_) {
      /* best-effort */
    }
  }

  window.addEventListener("pagehide", () => {
    if (monitoringActive) {
      monitoringActive = false;
      sessionEpoch += 1;
      if (analyzeTimer) clearTimeout(analyzeTimer);
      stopCameraTracks();
      endSessionBeacon();
    }
  });

  scoreCircle.style.strokeDasharray = CIRCUMFERENCE;
  preview.style.transition = "opacity 0.25s ease";
  setScore(0);
})();
