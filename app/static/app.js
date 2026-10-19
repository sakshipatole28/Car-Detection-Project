// Detect page: upload -> /api/v1/detect -> draw boxes on a canvas in the centre stage.
// The API returns every box above 0.05; the slider filters locally (exact, since
// greedy NMS commutes with thresholding), so no extra request is needed.
(() => {
  const $ = (id) => document.getElementById(id);
  const fileInput = $("file-input"), stage = $("stage"), slider = $("threshold");
  const canvas = $("canvas"), ctx = canvas.getContext("2d");
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let state = null;        // { data, img }
  let previewUrl = null;
  let animFrame = 0;
  let last = null;         // { blob, name, previewSrc } so a model switch can re-run
  let model = (window.CARDET && window.CARDET.defaultModel) || "cnn";
  const HINTS = {
    cnn: "Fine-tuned SSDlite CNN — works for front, side, rear and three-quarter views.",
    hog: "Classic HOG + logistic regression — side views only, scans 9 window sizes.",
  };
  const SCAN = { cnn: "Running CNN…", hog: "Scanning 9 scales…" };

  const show = (which) => {
    for (const id of ["empty", "loading", "error", "output"]) $(id).hidden = id !== which;
  };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  async function detect(blob, name, previewSrc) {
    last = { blob, name, previewSrc };
    $("scan-text").textContent = SCAN[model] || "Scanning…";
    // Preview + laser scan while the server works. PGM can't be shown by
    // browsers, so the scan runs over an empty stage in that case.
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = previewSrc ? null : URL.createObjectURL(blob);
    const preview = $("preview");
    preview.hidden = false;
    preview.onerror = () => { preview.hidden = true; };
    preview.src = previewSrc || previewUrl;
    $("r-status").textContent = "Scanning…";
    show("loading");

    const form = new FormData();
    form.append("file", blob, name);
    try {
      const url = `/api/v1/detect?return_image=true&source=ui&model=${encodeURIComponent(model)}`;
      const res = await fetch(url, { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
      const img = new Image();
      img.onload = () => {
        state = { data, img };
        show("output");
        render(true);
      };
      img.src = "data:image/png;base64," + data.image_png_base64;
    } catch (err) {
      $("error-msg").textContent = err.message;
      $("r-status").textContent = "Error";
      show("error");
    }
  }

  // Fit the image inside the stage, then draw at device resolution.
  function layout() {
    const { img } = state;
    const maxW = stage.clientWidth - 24;
    const maxH = Math.max(220, window.innerHeight * 0.68);
    const k = Math.min(maxW / img.width, maxH / img.height, 4);
    const w = Math.round(img.width * k), h = Math.round(img.height * k);
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    return canvas.width / img.width;
  }

  function drawFrame(boxes, s, t) {
    const { img } = state;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const hud = css("--hud") || "#c8f43a";
    const lw = Math.max(2, Math.round(canvas.width / 360));
    const font = Math.max(12, lw * 6);
    ctx.font = `700 ${font}px "Chakra Petch", system-ui, sans-serif`;
    ctx.textBaseline = "middle";

    // Two passes so every label sits on top of every box outline.
    const geo = boxes.map((b, i) => {
      // Staggered pop-in: each box grows from its centre with a slight overshoot.
      const p = Math.min(1, Math.max(0, (t - i * 110) / 380));
      const e = 1 + 1.7 * Math.pow(p - 1, 3) + 0.7 * Math.pow(p - 1, 2); // easeOutBack
      const cx = (b.x + b.w / 2) * s, cy = (b.y + b.h / 2) * s;
      const w = b.w * s * e, h = b.h * s * e;
      return { b, i, p, x: cx - w / 2, y: cy - h / 2, w, h };
    }).filter((g) => g.p > 0);

    for (const { p, x, y, w, h } of geo) {
      ctx.globalAlpha = Math.min(1, p * 1.6);
      ctx.fillStyle = "rgba(200, 244, 58, 0.10)";
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = hud;
      ctx.lineWidth = lw;
      ctx.strokeRect(x, y, w, h);
      // Corner brackets for a HUD look.
      const c = Math.min(w, h) * 0.28;
      ctx.lineWidth = lw * 2;
      ctx.beginPath();
      for (const [px, py, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
        ctx.moveTo(px, py + dy * c); ctx.lineTo(px, py); ctx.lineTo(px + dx * c, py);
      }
      ctx.stroke();
    }

    for (const { b, i, p, x, y } of geo) {
      // Label tag: "P1 98%"
      ctx.globalAlpha = Math.min(1, p * 1.6);
      const label = `P${i + 1}  ${(b.score * 100).toFixed(0)}%`;
      const tw = ctx.measureText(label).width + font;
      const th = font * 1.6;
      const ty = y - th - lw >= 0 ? y - th - lw : y + lw;
      ctx.fillStyle = hud;
      ctx.beginPath();
      ctx.moveTo(x, ty); ctx.lineTo(x + tw + th * 0.4, ty); ctx.lineTo(x + tw, ty + th); ctx.lineTo(x, ty + th);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = "#0b0d10";
      ctx.fillText(label, x + font * 0.5, ty + th / 2 + 1);
    }
    ctx.globalAlpha = 1;
  }

  function countUp(el, to) {
    const from = parseInt(el.textContent, 10) || 0;
    if (reduceMotion || from === to) { el.textContent = to; return; }
    const t0 = performance.now(), dur = 450;
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      el.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function render(animate = false) {
    if (!state) return;
    const { data } = state;
    const thr = parseFloat(slider.value);
    const boxes = data.detections.filter((b) => b.score >= thr);
    const s = layout();

    cancelAnimationFrame(animFrame);
    if (animate && !reduceMotion && boxes.length) {
      const t0 = performance.now();
      const total = boxes.length * 110 + 400;
      const tick = (now) => {
        drawFrame(boxes, s, now - t0);
        if (now - t0 < total) animFrame = requestAnimationFrame(tick);
      };
      animFrame = requestAnimationFrame(tick);
    } else {
      drawFrame(boxes, s, Infinity);
    }

    const n = boxes.length;
    countUp($("count"), n);
    $("chip-count").innerHTML = `<span class="dot"></span> <b>${n}</b> ${n === 1 ? "car" : "cars"}`;
    $("chip-time").textContent = (data.latency_ms / 1000).toFixed(2) + " s";
    $("count-label").textContent = n === 1 ? "car found" : "cars found";
    $("r-status").textContent = "Done";
    $("r-file").textContent = data.filename;
    $("r-size").textContent = `${data.image_width}×${data.image_height}` +
      (data.resize_factor !== 1 ? ` (×${data.resize_factor})` : "");
    $("r-latency").textContent = `${Math.round(data.latency_ms)} ms`;
    $("r-model").textContent = data.model === "cnn" ? "CNN (any angle)" : "HOG (side view)";

    const list = $("det-list");
    if (!n) {
      list.innerHTML = `<li class="empty-row">No cars above ${thr.toFixed(2)}</li>`;
      return;
    }
    list.replaceChildren(...boxes.map((b, i) => {
      const li = document.createElement("li");
      li.style.setProperty("--i", animate ? i : 0);
      if (!animate) li.style.animation = "none";
      const pct = (b.score * 100).toFixed(1);
      li.innerHTML = `<span class="pos">P${i + 1}</span>
        <span class="meta">x ${b.x}, y ${b.y} · ${b.w}×${b.h}</span>
        <span class="pct">${pct}%</span>
        <span class="bar"><span style="width:${pct}%"></span></span>`;
      if (!animate) li.querySelector(".bar span").style.animation = "none";
      return li;
    }));
  }

  // ── Inputs ────────────────────────────────────────────────────────────────
  fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    if (f) detect(f, f.name);
    fileInput.value = "";
  });
  ["dragenter", "dragover"].forEach((ev) => stage.addEventListener(ev, (e) => {
    e.preventDefault(); stage.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((ev) => stage.addEventListener(ev, (e) => {
    e.preventDefault(); stage.classList.remove("dragover");
  }));
  stage.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) detect(f, f.name);
  });
  document.querySelectorAll(".sample").forEach((btn) => btn.addEventListener("click", async () => {
    const blob = await (await fetch(btn.dataset.src)).blob();
    detect(blob, btn.dataset.name, btn.dataset.src);
    if (window.innerWidth <= 1100) stage.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
  }));
  slider.addEventListener("input", () => {
    $("threshold-out").textContent = parseFloat(slider.value).toFixed(2);
    render(false);
  });

  // Model switch: reset the slider to that model's tuned threshold and re-run.
  const segs = document.querySelectorAll(".seg");
  const selectModel = (btn, rerun) => {
    segs.forEach((b) => b.setAttribute("aria-checked", String(b === btn)));
    model = btn.dataset.model;
    slider.value = btn.dataset.threshold;
    $("threshold-out").textContent = parseFloat(slider.value).toFixed(2);
    $("model-hint").textContent = HINTS[model] || "";
    if (rerun && last) detect(last.blob, last.name, last.previewSrc);
  };
  segs.forEach((btn) => btn.addEventListener("click", () => {
    if (btn.dataset.model !== model) selectModel(btn, true);
  }));
  const initial = [...segs].find((b) => b.dataset.model === model);
  if (initial) selectModel(initial, false);
  const fromUrl = new URLSearchParams(location.search).get("model");
  const urlBtn = [...segs].find((b) => b.dataset.model === fromUrl);
  if (urlBtn) selectModel(urlBtn, false);

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => render(false), 120);
  });
  window.addEventListener("themechange", () => render(false));

  // Deep link: /?sample=scale-80.png runs that sample on load.
  const wanted = new URLSearchParams(location.search).get("sample");
  if (wanted) document.querySelector(`.sample[data-name="${CSS.escape(wanted)}"]`)?.click();
})();
