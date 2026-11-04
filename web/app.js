/* Podcaster — front-end controller */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const state = {
    voices: [],
    styles: [],
    currentJobId: null,
    eventSource: null,
    activeJob: null,
  };

  /* ------------------------------------------------------------------ */
  /* helpers                                                             */
  /* ------------------------------------------------------------------ */

  const toast = (message, kind = "") => {
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = message;
    $("#toasts").appendChild(el);
    setTimeout(() => {
      el.style.transition = "opacity .3s, transform .3s";
      el.style.opacity = "0";
      el.style.transform = "translateX(20px)";
      setTimeout(() => el.remove(), 320);
    }, kind === "err" ? 6500 : 3600);
  };

  const api = async (path, options = {}) => {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
    if (!res.ok) {
      const detail = (data && (data.detail || data.message)) || `HTTP ${res.status}`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  };

  const fmtTime = (ts) => {
    if (!ts) return "—";
    return new Date(ts * 1000).toLocaleTimeString();
  };

  const fmtDuration = (seconds) => {
    if (!seconds && seconds !== 0) return "—";
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  /* ------------------------------------------------------------------ */
  /* tabs                                                                */
  /* ------------------------------------------------------------------ */

  const showView = (name) => {
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
    if (name === "jobs") loadJobs();
    if (name === "settings") loadConfig();
  };

  $("#tabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (tab) showView(tab.dataset.view);
  });

  /* ------------------------------------------------------------------ */
  /* compose: source type                                                */
  /* ------------------------------------------------------------------ */

  const sourceType = () => {
    const active = $("#sourceType .seg.active");
    return active ? active.dataset.type : "url";
  };

  $("#sourceType").addEventListener("click", (e) => {
    const btn = e.target.closest(".seg");
    if (!btn) return;
    $$("#sourceType .seg").forEach((b) => b.classList.toggle("active", b === btn));
    $$(".src-pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === btn.dataset.type));
  });

  $("#exportFormat").addEventListener("click", (e) => {
    const btn = e.target.closest(".seg");
    if (!btn) return;
    $$("#exportFormat .seg").forEach((b) => b.classList.toggle("active", b === btn));
    updateExportHint();
  });

  /* upload ---------------------------------------------------------------- */

  const dropzone = $("#dropzone");
  const fileInput = $("#sourceFile");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
  });

  async function uploadFile(file) {
    $("#dzLabel").innerHTML = `Uploading <strong>${escapeHtml(file.name)}</strong>…`;
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      window.__uploadedPath = data.path;
      $("#sourcePath").value = data.path;
      $("#dzLabel").innerHTML = `✅ <strong>${escapeHtml(file.name)}</strong> (${formatBytes(data.bytes)})`;
      toast("File uploaded", "ok");
    } catch (err) {
      $("#dzLabel").innerHTML = `Drop a file here or <strong>browse</strong>`;
      toast(err.message, "err");
    }
  }

  const formatBytes = (n) => (n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`);
  const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ------------------------------------------------------------------ */
  /* hosts                                                               */
  /* ------------------------------------------------------------------ */

  function renderHosts(count = 2, genders = null, names = null) {
    const wrap = $("#hosts");
    wrap.innerHTML = "";
    for (let i = 0; i < count; i++) {
      const gender = genders?.[i] || (i % 2 === 0 ? "male" : "female");
      const name = names?.[i] || "";
      const row = document.createElement("div");
      row.className = "host-row";
      row.innerHTML = `
        <input type="text" class="h-name" placeholder="Host ${i + 1} name" value="${escapeHtml(name)}" />
        <select class="h-gender">
          <option value="male"${gender === "male" ? " selected" : ""}>Male</option>
          <option value="female"${gender === "female" ? " selected" : ""}>Female</option>
        </select>
        <div style="display:flex;align-items:center;gap:6px">
          <select class="h-voice"><option value="">auto voice</option></select>
          <button class="host-remove" title="Remove host">✕</button>
        </div>`;
      wrap.appendChild(row);
      row.querySelector(".host-remove").addEventListener("click", () => {
        if ($("#hosts").children.length <= 1) return;
        row.remove();
      });
      row.querySelector(".h-voice").addEventListener("change", () => refreshVoiceHints());
    }
    refreshVoiceHints();
  }

  function refreshVoiceHints() {
    const language = $("#language").value;
    const pool = state.voices.filter((v) => !v.gender || true);
    $$("#hosts .host-row").forEach((row) => {
      const gender = row.querySelector(".h-gender").value;
      const select = row.querySelector(".h-voice");
      const current = select.value;
      const matching = pool.filter((v) => v.gender === gender);
      const options = [`<option value="">auto voice</option>`].concat(
        matching.map((v) => `<option value="${escapeHtml(v.id)}">${escapeHtml(v.label)}</option>`)
      );
      select.innerHTML = options.join("");
      if (matching.some((v) => v.id === current)) select.value = current;
    });
  }

  $("#addHost").addEventListener("click", () => {
    renderHosts($("#hosts").children.length + 1);
  });
  $("#resetHosts").addEventListener("click", () => renderHosts(2));
  $("#language").addEventListener("change", refreshVoiceHints);

  /* ------------------------------------------------------------------ */
  /* style + length                                                      */
  /* ------------------------------------------------------------------ */

  async function loadStyles() {
    const data = await api("/api/styles");
    state.styles = data.styles;
    const select = $("#style");
    select.innerHTML = data.styles
      .map((s) => `<option value="${s.id}">${escapeHtml(s.label)}</option>`)
      .join("");
    updateStyleDesc();
  }

  const updateStyleDesc = () => {
    const style = state.styles.find((s) => s.id === $("#style").value);
    $("#styleDesc").textContent = style ? style.description : "";
  };
  $("#style").addEventListener("change", updateStyleDesc);

  const updateLength = () => {
    const v = Number($("#lengthRange").value);
    $("#lengthOut").textContent = `${v} min`;
  };
  $("#lengthRange").addEventListener("input", updateLength);

  const updateExportHint = () => {
    const fmt = $("#exportFormat .seg.active").dataset.fmt;
    const slides = $("#showSlides").checked;
    $("#exportHint").textContent = fmt === "video"
      ? (slides
        ? "Video export writes an .mp4 of illustrated slides — one AI image per point, " +
          "opening with the episode title and the hosts."
        : "Video export writes an .mp4 with an animated waveform and per-speaker colour.")
      : "Audio export writes a single .mp3 with a short pause between turns.";
    // Slides and subtitles are burned into the video only.
    $("#subtitleOption").hidden = fmt !== "video";
    $("#slideOption").hidden = fmt !== "video";
  };

  $("#showSlides").addEventListener("change", updateExportHint);

  /* ------------------------------------------------------------------ */
  /* file browser                                                        */
  /* ------------------------------------------------------------------ */

  let browserTarget = null;

  const openBrowser = (targetInputId, title) => {
    browserTarget = targetInputId;
    $("#browserTitle").textContent = title;
    $("#browserModal").hidden = false;
    loadBrowse("");
  };

  async function loadBrowse(path) {
    const data = await api(`/api/browse?path=${encodeURIComponent(path)}`);
    $("#browserPath").textContent = data.path;
    $("#browserInput").value = data.path;
    const list = $("#browserList");
    list.innerHTML = "";
    if (data.parent && data.parent !== data.path) {
      const up = document.createElement("li");
      up.className = "dir";
      up.innerHTML = `<span>📁</span><span>..</span>`;
      up.addEventListener("click", () => loadBrowse(data.parent));
      list.appendChild(up);
    }
    data.entries.forEach((entry) => {
      const li = document.createElement("li");
      li.className = entry.is_dir ? "dir" : "file";
      li.innerHTML = `<span>${entry.is_dir ? "📁" : "📄"}</span><span>${escapeHtml(entry.name)}</span>`;
      li.addEventListener("click", () => {
        if (entry.is_dir) loadBrowse(entry.path);
        else { $("#browserInput").value = entry.path; }
      });
      list.appendChild(li);
    });
    if (!data.entries.length) {
      list.innerHTML = `<li class="file">Empty folder</li>`;
    }
  }

  $("#browseExport").addEventListener("click", () => openBrowser("exportPath", "Choose the output file location"));
  $("#browseSource").addEventListener("click", () => openBrowser("sourcePath", "Choose a source document"));
  $("#browserClose").addEventListener("click", () => { $("#browserModal").hidden = true; });
  $("#browserCancel").addEventListener("click", () => { $("#browserModal").hidden = true; });
  $("#browserModal").addEventListener("click", (e) => {
    if (e.target === $("#browserModal")) $("#browserModal").hidden = true;
  });
  $("#browserPick").addEventListener("click", () => {
    const value = $("#browserInput").value.trim();
    if (browserTarget && value) $(`#${browserTarget}`).value = value;
    $("#browserModal").hidden = true;
  });

  /* ------------------------------------------------------------------ */
  /* generate                                                            */
  /* ------------------------------------------------------------------ */

  const collectHosts = () => {
    const names = [], genders = [], personas = [], overrides = {};
    $$("#hosts .host-row").forEach((row) => {
      names.push(row.querySelector(".h-name").value.trim());
      genders.push(row.querySelector(".h-gender").value);
      const voice = row.querySelector(".h-voice").value;
      if (voice) overrides[row.querySelector(".h-name").value.trim() || names[names.length - 1]] = voice;
    });
    return { names, genders, personas, overrides };
  };

  $("#generateBtn").addEventListener("click", async () => {
    const type = sourceType();
    const hosts = collectHosts();
    const payload = {
      source_type: type,
      source_url: $("#sourceUrl").value.trim(),
      source_text: $("#sourceText").value,
      source_path: $("#sourcePath").value.trim() || window.__uploadedPath || "",
      language: $("#language").value,
      style: $("#style").value,
      length_minutes: Number($("#lengthRange").value),
      host_count: hosts.names.length,
      host_genders: hosts.genders,
      host_names: hosts.names,
      host_personas: hosts.personas,
      voice_overrides: hosts.overrides,
      export_format: $("#exportFormat .seg.active").dataset.fmt,
      export_path: $("#exportPath").value.trim(),
      show_subtitles: $("#showSubtitles").checked,
      show_slides: $("#showSlides").checked,
      enable_review: $("#enableReview").checked,
      title: $("#podTitle").value.trim(),
      extra_instructions: $("#extraInstructions").value.trim(),
      tts_speed: Number($("#ttsSpeed").value),
      gap_seconds: Number($("#gapSeconds").value),
      chunk_chars: 6000,
      keep_workdir: true,
    };

    if (type === "url" && !payload.source_url) { toast("Add a URL first", "err"); return; }
    if (type === "upload" && !payload.source_path) { toast("Upload a document first", "err"); return; }
    if (type === "text" && !payload.source_text.trim()) { toast("Paste some text first", "err"); return; }

    const btn = $("#generateBtn");
    btn.disabled = true;
    $("#cancelBtn").hidden = false;
    $("#launchNote").textContent = "Running…";
    try {
      const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
      state.currentJobId = job.id;
      toast("Job started", "ok");
      showView("jobs");
      selectJob(job.id);
      openStream(job.id);
    } catch (err) {
      toast(err.message, "err");
    } finally {
      btn.disabled = false;
      $("#cancelBtn").hidden = true;
      $("#launchNote").textContent = "Nothing runs until you press the button.";
    }
  });

  $("#cancelBtn").addEventListener("click", async () => {
    if (!state.currentJobId) return;
    await api(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
    toast("Cancellation requested");
  });

  /* ------------------------------------------------------------------ */
  /* jobs                                                                */
  /* ------------------------------------------------------------------ */

  async function loadJobs() {
    const data = await api("/api/jobs");
    const list = $("#jobList");
    const running = data.jobs.filter((j) => j.status === "running" || j.status === "queued").length;
    const badge = $("#jobBadge");
    badge.hidden = running === 0;
    badge.textContent = running;
    list.innerHTML = "";
    if (!data.jobs.length) {
      list.innerHTML = `<li class="job-item"><div class="job-item-sub">No jobs yet.</div></li>`;
      return;
    }
    data.jobs.forEach((job) => {
      const li = document.createElement("li");
      li.className = `job-item${job.id === state.currentJobId ? " active" : ""}`;
      li.dataset.id = job.id;
      const title = job.request.title || job.request.source_url || job.request.source_type;
      li.innerHTML = `
        <div class="job-item-title">
          <span>${escapeHtml(String(title).slice(0, 42) || "Untitled")}</span>
          <span class="status-pill ${job.status}">${job.status}</span>
        </div>
        <div class="job-item-sub">${escapeHtml(job.step)} · ${Math.round((job.progress || 0) * 100)}% · ${fmtDuration(job.elapsed)}</div>`;
      li.addEventListener("click", () => selectJob(job.id));
      list.appendChild(li);
    });
  }

  async function selectJob(jobId) {
    state.currentJobId = jobId;
    $$(".job-item").forEach((el) => el.classList.toggle("active", el.dataset.id === jobId));
    const job = await api(`/api/jobs/${jobId}`);
    renderJob(job);
  }

  const REVIEW_STATUSES = ["awaiting_script_review", "awaiting_audio_review", "awaiting_slides_review"];
  const RESUMABLE_STATUSES = ["interrupted", "running", ...REVIEW_STATUSES];
  const DELETABLE_STATUSES = ["done", "error", "cancelled", "interrupted"];

  function renderJob(job) {
    state.activeJob = job;
    $("#jobEmpty").hidden = true;
    $("#jobDetailBody").hidden = false;

    const title = job.request.title || job.request.source_url ||
      (job.request.source_type === "text" ? "Pasted text" : "Podcast");
    $("#jdTitle").textContent = title;
    $("#jdMeta").textContent = [
      job.request.language === "zh" ? "中文" : "English",
      job.request.style,
      `${job.request.length_minutes} min target`,
      job.request.export_format,
      job.request.enable_review ? "review enabled" : "",
      `started ${fmtTime(job.started_at)}`,
    ].filter(Boolean).join(" · ");
    const pill = $("#jdStatus");
    pill.textContent = job.status.replace(/_/g, " ");
    pill.className = `status-pill ${job.status}`;
    $("#jdProgress").style.width = `${Math.round((job.progress || 0) * 100)}%`;
    $("#jdStep").textContent = job.step || "—";

    const logs = $("#jdLogs");
    logs.innerHTML = "";
    (job.logs || []).forEach((entry) => {
      const div = document.createElement("div");
      div.className = `log-line ${entry.level}`;
      div.innerHTML = `<span class="ts">${fmtTime(entry.t)}</span>${escapeHtml(entry.message)}`;
      logs.appendChild(div);
    });
    logs.scrollTop = logs.scrollHeight;

    const script = $("#jdScript");
    if (job.result && job.result.script_text) {
      script.textContent = job.result.script_text;
    } else if (job.review_data && job.review_data.script_turns && job.review_data.script_turns.length) {
      script.textContent = job.review_data.script_turns.map(t => `${t.speaker}: ${t.text}`).join("\n");
    } else {
      script.innerHTML = `<p class="muted">Not available yet.</p>`;
    }

    const actions = $("#jdActions");
    actions.innerHTML = "";

    if (RESUMABLE_STATUSES.includes(job.status)) {
      const resumeBtn = document.createElement("button");
      resumeBtn.className = "btn resume";
      resumeBtn.textContent = "▶ Resume";
      resumeBtn.onclick = () => resumeJob(job.id);
      actions.appendChild(resumeBtn);
    }

    if (job.result && job.result.export_path) {
      const link = document.createElement("a");
      link.className = "btn primary";
      link.href = `/api/download?path=${encodeURIComponent(job.result.export_path)}`;
      link.textContent = `⬇ Download ${job.result.export_path.split(/[\\/]/).pop()}`;
      link.setAttribute("download", "");
      actions.appendChild(link);

      if (job.result.audio_path && job.result.audio_path !== job.result.export_path) {
        const a = document.createElement("a");
        a.className = "btn ghost";
        a.href = `/api/download?path=${encodeURIComponent(job.result.audio_path)}`;
        a.textContent = "⬇ Audio only";
        actions.appendChild(a);
      }
    }
    if (job.result && job.result.duration_seconds) {
      const info = document.createElement("span");
      info.className = "muted small";
      info.textContent = `Length ${fmtDuration(job.result.duration_seconds)} · ${job.result.turns} turns`;
      actions.appendChild(info);
    }
    if (job.error) {
      const err = document.createElement("span");
      err.className = "muted small";
      err.style.color = "#fca5a5";
      err.textContent = job.error;
      actions.appendChild(err);
    }

    if (DELETABLE_STATUSES.includes(job.status)) {
      const deleteBtn = document.createElement("button");
      deleteBtn.className = "btn danger sm";
      deleteBtn.textContent = "🗑 Delete";
      deleteBtn.style.marginLeft = "auto";
      deleteBtn.onclick = () => deleteJob(job.id);
      actions.appendChild(deleteBtn);
    }

    renderReviewPanel(job);
  }

  function renderReviewPanel(job) {
    const panel = $("#jdReview");
    const content = $("#reviewContent");
    const titleEl = $("#reviewTitle");

    if (!REVIEW_STATUSES.includes(job.status)) {
      panel.hidden = true;
      return;
    }

    panel.hidden = false;
    content.innerHTML = "";

    if (job.status === "awaiting_script_review") {
      titleEl.textContent = "Script Review";
      const data = job.review_data || {};
      const turns = data.script_turns || [];
      const hosts = (data.hosts || []).map(h => h.name);

      if (data.script_title) {
        const titleField = document.createElement("div");
        titleField.className = "review-turn";
        titleField.innerHTML = `
          <span class="review-turn-speaker">Title</span>
          <input type="text" class="review-title-input" value="${escapeHtml(data.script_title)}" />
          <span></span>`;
        content.appendChild(titleField);
      }

      turns.forEach((turn, i) => {
        const row = document.createElement("div");
        row.className = "review-turn";
        row.dataset.index = turn.index;
        row.innerHTML = `
          <select class="review-turn-speaker-select">
            ${hosts.map(h => `<option value="${escapeHtml(h)}"${h === turn.speaker ? " selected" : ""}>${escapeHtml(h)}</option>`).join("")}
          </select>
          <textarea class="review-turn-text">${escapeHtml(turn.text)}</textarea>
          <button class="btn ghost sm review-turn-delete" title="Remove turn">✕</button>`;
        content.appendChild(row);
      });

      content.querySelectorAll(".review-turn-delete").forEach(btn => {
        btn.onclick = () => {
          const row = btn.closest(".review-turn");
          if (row) row.remove();
        };
      });

      const notes = document.createElement("div");
      notes.className = "review-notes";
      notes.innerHTML = `
        <label class="field">
          <span>Notes for regeneration <em class="hint">optional</em></span>
          <textarea id="reviewNotes" rows="2" placeholder="Add instructions for regenerating the script..."></textarea>
        </label>`;
      content.appendChild(notes);

      $("#reviewRegenerate").onclick = () => regenerateScript(job.id);
      $("#reviewRegenerate").hidden = false;

    } else if (job.status === "awaiting_audio_review") {
      titleEl.textContent = "Audio Review";
      const data = job.review_data || {};
      const clips = data.audio_clips || [];

      clips.forEach((clip) => {
        const row = document.createElement("div");
        row.className = "review-clip";
        row.dataset.index = clip.index;
        const audioSrc = clip.files && clip.files[0]
          ? `/api/jobs/${job.id}/clips/${clip.files[0]}`
          : "";
        row.innerHTML = `
          <span class="review-turn-speaker">${escapeHtml(clip.speaker)}</span>
          <div>
            <div class="review-turn-text" style="margin-bottom:6px">${escapeHtml(clip.text)}</div>
            ${audioSrc ? `<audio controls src="${audioSrc}"></audio>` : "<span class='muted'>No audio</span>"}
          </div>
          <button class="btn ghost sm review-clip-regen" data-index="${clip.index}">Regenerate</button>`;
        content.appendChild(row);
      });

      content.querySelectorAll(".review-clip-regen").forEach(btn => {
        btn.onclick = () => regenerateClip(job.id, parseInt(btn.dataset.index, 10));
      });
      $("#reviewRegenerate").hidden = true;

    } else if (job.status === "awaiting_slides_review") {
      titleEl.textContent = "Slides Review";
      const data = job.review_data || {};
      const slides = data.slides || [];

      slides.forEach((slide) => {
        const row = document.createElement("div");
        row.className = "review-slide";
        row.dataset.index = slide.index;
        const imgPath = slide.image ? slide.image.split(/[\\/]/).pop() : "";
        const imgSrc = imgPath ? `/api/jobs/${job.id}/slides/${imgPath}` : "";
        row.innerHTML = `
          ${imgSrc ? `<img src="${imgSrc}" alt="Slide ${slide.index + 1}" />` : "<div style='width:160px;height:90px;background:#0a1220;border-radius:6px'></div>"}
          <div>
            <strong>Slide ${slide.index + 1}: ${escapeHtml(slide.title || "")}</strong>
            <div class="review-slide-prompt">${escapeHtml(slide.prompt || "")}</div>
            <input type="text" class="slide-prompt-input" placeholder="New prompt..." style="margin-top:8px" />
          </div>
          <button class="btn ghost sm review-slide-regen" data-index="${slide.index}">Regenerate</button>`;
        content.appendChild(row);
      });

      content.querySelectorAll(".review-slide-regen").forEach(btn => {
        btn.onclick = () => {
          const row = btn.closest(".review-slide");
          const prompt = row.querySelector(".slide-prompt-input").value.trim();
          if (!prompt) { toast("Enter a new prompt", "err"); return; }
          regenerateSlide(job.id, parseInt(btn.dataset.index, 10), prompt);
        };
      });
      $("#reviewRegenerate").hidden = true;
    }

    $("#reviewApprove").onclick = () => approveReview(job.id);
  }

  async function approveReview(jobId) {
    try {
      if (state.activeJob && state.activeJob.status === "awaiting_script_review") {
        await saveScriptEdits(jobId);
      }
      await api(`/api/jobs/${jobId}/approve`, { method: "POST" });
      toast("Review approved", "ok");
    } catch (err) {
      toast(err.message, "err");
    }
  }

  async function saveScriptEdits(jobId) {
    const titleInput = $(".review-title-input");
    const turns = [];
    $$("#reviewContent .review-turn").forEach(row => {
      if (row.querySelector(".review-title-input")) return;
      const speaker = row.querySelector(".review-turn-speaker-select")?.value || "";
      const text = row.querySelector(".review-turn-text")?.value || "";
      turns.push({ index: parseInt(row.dataset.index, 10), speaker, text });
    });
    const payload = { turns };
    if (titleInput) payload.title = titleInput.value.trim();
    await api(`/api/jobs/${jobId}/script`, { method: "PATCH", body: JSON.stringify(payload) });
  }

  async function regenerateScript(jobId) {
    const notes = $("#reviewNotes")?.value || "";
    try {
      toast("Regenerating script...");
      const data = await api(`/api/jobs/${jobId}/script/regenerate`, {
        method: "POST", body: JSON.stringify({ notes })
      });
      if (state.activeJob) {
        state.activeJob.review_data = data.script;
        renderReviewPanel(state.activeJob);
      }
      toast("Script regenerated", "ok");
    } catch (err) {
      toast(err.message, "err");
    }
  }

  async function regenerateClip(jobId, turnIndex) {
    try {
      toast(`Regenerating turn ${turnIndex + 1}...`);
      await api(`/api/jobs/${jobId}/clips/${turnIndex}/regenerate`, {
        method: "POST", body: JSON.stringify({})
      });
      const job = await api(`/api/jobs/${jobId}`);
      renderJob(job);
      toast(`Turn ${turnIndex + 1} regenerated`, "ok");
    } catch (err) {
      toast(err.message, "err");
    }
  }

  async function regenerateSlide(jobId, slideIndex, prompt) {
    try {
      toast(`Regenerating slide ${slideIndex + 1}...`);
      await api(`/api/jobs/${jobId}/slides/${slideIndex}/regenerate`, {
        method: "POST", body: JSON.stringify({ prompt })
      });
      const job = await api(`/api/jobs/${jobId}`);
      renderJob(job);
      toast(`Slide ${slideIndex + 1} regenerated`, "ok");
    } catch (err) {
      toast(err.message, "err");
    }
  }

  function openStream(jobId) {
    if (state.eventSource) state.eventSource.close();
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    state.eventSource = es;
    let lastSeq = 0;

    const TERMINAL = ["done", "error", "cancelled"];

    const setStatusPill = (status) => {
      const pill = $("#jdStatus");
      pill.textContent = status.replace(/_/g, " ");
      pill.className = `status-pill ${status}`;
    };

    es.onmessage = async (msg) => {
      let payload;
      try { payload = JSON.parse(msg.data); } catch { return; }
      if (payload.type === "snapshot") {
        lastSeq = payload.job.seq || 0;
        renderJob(payload.job);
        if (TERMINAL.includes(payload.job.status)) {
          es.close();
        }
        return;
      }
      if (payload.seq && payload.seq <= lastSeq) return;
      lastSeq = payload.seq || lastSeq;

      if (payload.type === "step") {
        const job = state.activeJob;
        if (job) {
          job.step = payload.step;
          job.progress = payload.progress;
          $("#jdProgress").style.width = `${Math.round(payload.progress * 100)}%`;
          $("#jdStep").textContent = payload.step;
        }
        return;
      }
      if (payload.type === "log") {
        const job = state.activeJob;
        if (!job) return;
        job.logs = job.logs || [];
        job.logs.push(payload.entry);
        const div = document.createElement("div");
        div.className = `log-line ${payload.entry.level}`;
        div.innerHTML = `<span class="ts">${fmtTime(payload.entry.t)}</span>${escapeHtml(payload.entry.message)}`;
        const box = $("#jdLogs");
        box.appendChild(div);
        box.scrollTop = box.scrollHeight;
        return;
      }
      if (payload.type === "review_data") {
        const job = state.activeJob;
        if (job) {
          job.review_data = payload.data;
          renderReviewPanel(job);
        }
        return;
      }
      if (payload.type === "status") {
        const status = payload.status;
        setStatusPill(status);
        if (state.activeJob) state.activeJob.status = status;
        loadJobs();

        if (REVIEW_STATUSES.includes(status)) {
          const job = await api(`/api/jobs/${jobId}`);
          renderJob(job);
          toast(`Review required: ${status.replace(/_/g, " ")}`, "ok");
          return;
        }
        if (!TERMINAL.includes(status)) {
          return;
        }
        es.close();
        const job = await api(`/api/jobs/${jobId}`);
        renderJob(job);
        toast(status === "done" ? "Podcast ready" :
              status === "error" ? "Job failed" : "Job cancelled",
              status === "done" ? "ok" : "err");
      }
    };
    es.onerror = () => { /* the browser retries automatically */ };
  }

  async function resumeJob(jobId) {
    try {
      toast("Resuming job...");
      const data = await api(`/api/jobs/${jobId}/resume`, { method: "POST" });
      if (data.ok) {
        toast("Job resumed", "ok");
        selectJob(jobId);
        openStream(jobId);
        loadJobs();
      }
    } catch (err) {
      toast(err.message, "err");
    }
  }

  async function deleteJob(jobId) {
    if (!confirm("Delete this job and its temporary files?\n\nFinal exported audio/video will be kept.")) {
      return;
    }
    try {
      const data = await api(`/api/jobs/${jobId}`, { method: "DELETE" });
      if (data.ok) {
        toast(`Deleted job (${data.deleted_files?.length || 0} temp files removed)`, "ok");
        loadJobs();
        $("#jobEmpty").hidden = false;
        $("#jobDetailBody").hidden = true;
        state.currentJobId = null;
      }
    } catch (err) {
      toast(err.message, "err");
    }
  }

  $("#refreshJobs").addEventListener("click", loadJobs);

  /* ------------------------------------------------------------------ */
  /* settings                                                            */
  /* ------------------------------------------------------------------ */

  const CONFIG_FIELDS = {
    llm_base_url: "#llmBaseUrl", llm_model: "#llmModel", llm_api_key: "#llmApiKey",
    llm_temperature: "#llmTemperature", llm_max_tokens: "#llmMaxTokens",
    tts_provider: "#ttsProvider", tts_base_url: "#ttsBaseUrl", tts_model: "#ttsModel",
    tts_api_key: "#ttsApiKey", output_dir: "#outputDir",
    tts_parallel: "#ttsParallel", tts_parallel_workers: "#ttsParallelWorkers",
    image_base_url: "#imageBaseUrl", image_model: "#imageModel", image_api_key: "#imageApiKey",
  };

  async function loadConfig() {
    const cfg = await api("/api/config");
    Object.entries(CONFIG_FIELDS).forEach(([key, selector]) => {
      const el = $(selector);
      if (!el) return;
      if (key.endsWith("_api_key")) {
        el.value = "";
        el.placeholder = cfg[`${key}_set`] ? `saved (${cfg[`${key}_hint`]}) — type to replace` : "sk-…";
      } else if (el.type === "checkbox") {
        el.checked = !!cfg[key];
      } else {
        el.value = cfg[key] ?? "";
      }
    });
    $("#saveStatus").textContent = "Settings loaded.";
  }

  $("#saveSettings").addEventListener("click", async () => {
    const payload = {};
    Object.entries(CONFIG_FIELDS).forEach(([key, selector]) => {
      const el = $(selector);
      if (!el) return;
      if (key.endsWith("_api_key")) {
        const value = el.value.trim();
        if (!value) return;  // keep existing
        payload[key] = value;
      } else if (el.type === "checkbox") {
        payload[key] = el.checked;
      } else if (key === "llm_temperature" || key === "llm_max_tokens" || key === "tts_parallel_workers") {
        payload[key] = Number(el.value);
      } else {
        payload[key] = el.value.trim();
      }
    });
    try {
      await api("/api/config", { method: "PUT", body: JSON.stringify(payload) });
      $("#saveStatus").textContent = "Saved just now.";
      toast("Settings saved", "ok");
      refreshHealth();
    } catch (err) {
      toast(err.message, "err");
    }
  });

  $("#testBtn").addEventListener("click", async () => {
    const box = $("#testResults");
    box.innerHTML = `<div class="test-row">Testing…</div>`;
    const payload = {};
    Object.entries(CONFIG_FIELDS).forEach(([key, selector]) => {
      const el = $(selector);
      if (!el) return;
      const value = el.value.trim();
      if (key.endsWith("_api_key") && !value) return;
      payload[key] = key === "llm_temperature" || key === "llm_max_tokens" ? Number(value) : value;
    });
    try {
      const data = await api("/api/config/test", { method: "POST", body: JSON.stringify(payload) });
      box.innerHTML = "";
      Object.entries(data.results).forEach(([name, res]) => {
        const div = document.createElement("div");
        div.className = `test-row ${res.ok ? "ok" : "bad"}`;
        div.innerHTML = `<b>${name.toUpperCase()}</b> — ${escapeHtml(res.message || "")}`;
        box.appendChild(div);
      });
    } catch (err) {
      box.innerHTML = `<div class="test-row bad">${escapeHtml(err.message)}</div>`;
    }
  });

  $("#verifyVoiceBtn").addEventListener("click", async () => {
    const box = $("#voiceVerifyResult");
    const voiceId = $("#verifyVoiceId").value.trim();
    if (!voiceId) { toast("Enter a voice id", "err"); return; }
    box.innerHTML = `<div class="test-row">Testing ${escapeHtml(voiceId)}…</div>`;
    try {
      const data = await api("/api/voices/verify", {
        method: "POST",
        body: JSON.stringify({ voice_id: voiceId, language: $("#language").value }),
      });
      box.innerHTML = `<div class="test-row ${data.ok ? "ok" : "bad"}">${escapeHtml(data.message)}</div>`;
    } catch (err) {
      box.innerHTML = `<div class="test-row bad">${escapeHtml(err.message)}</div>`;
    }
  });

  /* ------------------------------------------------------------------ */
  /* boot                                                                */
  /* ------------------------------------------------------------------ */

  async function loadVoices() {
    const data = await api("/api/voices");
    state.voices = data.voices;
    $("#voiceCount").textContent = `${data.voices.length} voices available in the catalogue.`;
  }

  async function refreshHealth() {
    const pill = $("#healthPill");
    const text = $("#healthText");
    try {
      const data = await api("/api/health");
      const ok = data.llm_key && data.tts_key;
      pill.className = `health ${ok ? "ok" : "bad"}`;
      text.textContent = ok ? "LLM + TTS ready" : "Missing API key — open Settings";
    } catch {
      pill.className = "health bad";
      text.textContent = "offline";
    }
  }

  (async function init() {
    try {
      await Promise.all([loadStyles(), loadVoices(), loadConfig()]);
      renderHosts(2);
      updateLength();
      updateExportHint();
      refreshHealth();
      loadJobs();
      const ff = await api("/api/ffmpeg");
      if (!ff.available) toast("ffmpeg was not found on PATH — video export will fail", "err");
    } catch (err) {
      toast(`Startup problem: ${err.message}`, "err");
    }
  })();
})();
