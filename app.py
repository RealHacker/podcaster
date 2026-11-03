"""Podcaster — FastAPI application.

A local tool that turns documents or web pages into a podcast episode:
LLM writes the script, TTS speaks it, ffmpeg assembles the result.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from podcaster import config as config_mod
from podcaster import media as media_mod
from podcaster import voices as voices_mod
from podcaster.jobs import JobStore
from podcaster.llm import LLMClient
from podcaster.pipeline import PodcastRequest
from podcaster.script import STYLE_ORDER, STYLES

APP_DIR = config_mod.APP_DIR
WEB_DIR = APP_DIR / "web"

app = FastAPI(title="Podcaster", version="1.0.0",
              description="Turn documents and web pages into podcasts.")
store = JobStore(max_workers=2)


# --------------------------------------------------------------------------- #
# Static UI
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return config_mod.public_view(config_mod.load())


@app.put("/api/config")
async def put_config(request: Request) -> dict[str, Any]:
    payload = await request.json()
    return config_mod.public_view(config_mod.merge_patch(payload))


@app.post("/api/config/test")
async def test_config(request: Request) -> dict[str, Any]:
    payload = await request.json()
    cfg = config_mod.load()
    for key in ("llm_base_url", "llm_model", "llm_api_key", "tts_base_url", "tts_model",
                "tts_api_key", "tts_provider"):
        if payload.get(key):
            cfg[key] = payload[key]

    results: dict[str, Any] = {}

    llm = LLMClient(cfg.get("llm_base_url", ""), cfg.get("llm_api_key", ""),
                     cfg.get("llm_model", ""))
    if not llm.api_key:
        results["llm"] = {"ok": False, "message": "No API key set"}
    else:
        try:
            reply = llm.chat([{"role": "user", "content": "Reply with the single word: ready"}],
                             temperature=0.1, max_tokens=32, retries=2)
            models = llm.list_models()
            results["llm"] = {"ok": True, "message": f"Connected — model replied “{reply[:60]}”",
                              "models": models}
        except Exception as exc:
            results["llm"] = {"ok": False, "message": str(exc)[:400]}

    tts = {"provider": cfg.get("tts_provider", "minimax")}
    try:
        from podcaster.tts import build_tts

        engine = build_tts(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "probe.mp3"
            text = "语音测试。" if cfg.get("tts_provider") != "openai" else "Voice test."
            engine.synthesize(text, voices_mod.load_catalog()[0]["id"], out, language="zh")
            size = out.stat().st_size
        tts.update({"ok": size > 1000, "message": f"Connected — generated {size:,} bytes of audio"})
    except Exception as exc:
        tts.update({"ok": False, "message": str(exc)[:400]})
    results["tts"] = tts

    return {"ok": all(v.get("ok") for v in results.values() if isinstance(v, dict)),
            "results": results}


@app.get("/api/voices")
async def get_voices(language: str | None = None, gender: str | None = None) -> dict[str, Any]:
    return {"voices": voices_mod.list_voices(language, gender)}


@app.post("/api/voices/verify")
async def verify_voice(request: Request) -> dict[str, Any]:
    payload = await request.json()
    voice_id = (payload.get("voice_id") or "").strip()
    if not voice_id:
        raise HTTPException(400, "voice_id is required")
    cfg = config_mod.load()
    from podcaster.tts import build_tts

    engine = build_tts(cfg)
    text = "你好，这是一次音色测试。" if payload.get("language", "zh") == "zh" else "Hello, this is a voice test."
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "v.mp3"
        try:
            engine.synthesize(text, voice_id, out, language=payload.get("language", "zh"))
        except Exception as exc:
            return {"ok": False, "voice_id": voice_id, "message": str(exc)[:400]}
        size = out.stat().st_size
    return {"ok": True, "voice_id": voice_id, "message": f"Voice works — {size:,} bytes generated"}


@app.get("/api/styles")
async def get_styles() -> dict[str, Any]:
    return {"styles": [{"id": k, "label": v["label"], "description": v["en"]}
                       for k, v in STYLES.items()]}


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #

@app.get("/api/browse")
async def browse(path: str = "") -> dict[str, Any]:
    target = Path(path).expanduser() if path else Path.home()
    if not target.exists():
        target = Path.home()
    if target.is_file():
        target = target.parent
    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith("."):
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "is_dir": child.is_dir(),
            })
    except PermissionError:
        pass
    return {"path": str(target), "parent": str(target.parent), "entries": entries[:500],
            "home": str(Path.home())}


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

@app.post("/api/jobs")
async def create_job(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        req = PodcastRequest(
            source_type=payload.get("source_type", "url"),
            source_url=payload.get("source_url", ""),
            source_text=payload.get("source_text", ""),
            source_path=payload.get("source_path", ""),
            language=payload.get("language", "en"),
            length_minutes=float(payload.get("length_minutes", 5)),
            style=payload.get("style", "narrative"),
            host_count=int(payload.get("host_count", 2)),
            host_genders=payload.get("host_genders", []) or [],
            host_names=payload.get("host_names", []) or [],
            host_personas=payload.get("host_personas", []) or [],
            voice_overrides=payload.get("voice_overrides", {}) or {},
            export_format=payload.get("export_format", "audio"),
            export_path=payload.get("export_path", ""),
            title=payload.get("title", ""),
            extra_instructions=payload.get("extra_instructions", ""),
            tts_speed=float(payload.get("tts_speed", 1.0)),
            gap_seconds=float(payload.get("gap_seconds", 0.28)),
            chunk_chars=int(payload.get("chunk_chars", 6000)),
            keep_workdir=bool(payload.get("keep_workdir", True)),
            show_subtitles=bool(payload.get("show_subtitles", False)),
            show_slides=bool(payload.get("show_slides", False)),
            enable_review=bool(payload.get("enable_review", False)),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid request: {exc}") from exc

    if req.language not in ("en", "zh"):
        raise HTTPException(400, "language must be 'en' or 'zh'")
    if req.export_format not in ("audio", "video"):
        raise HTTPException(400, "export_format must be 'audio' or 'video'")
    if req.style not in STYLE_ORDER:
        raise HTTPException(400, f"style must be one of {STYLE_ORDER}")
    if req.length_minutes <= 0 or req.length_minutes > 240:
        raise HTTPException(400, "length_minutes must be between 0 and 240")

    job = store.create(req)
    return job.to_dict()


@app.get("/api/jobs")
async def list_jobs() -> dict[str, Any]:
    return {"jobs": [job.to_dict(include_logs=False) for job in store.list()]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job.cancel()
    return {"ok": True, "status": job.status}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    from podcaster.jobs import TERMINAL_STATUSES

    async def stream():
        events = job._events
        snapshot = job.to_dict()
        yield f"data: {json.dumps({'type': 'snapshot', 'job': snapshot})}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.to_thread(events.get, True, 15)
            except Exception:
                yield ": keep-alive\n\n"
                continue
            if event.get("type") == "status" and event.get("status") in TERMINAL_STATUSES:
                yield f"data: {json.dumps(event)}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- #
# Review workflow API
# --------------------------------------------------------------------------- #

@app.get("/api/jobs/{job_id}/script")
async def get_job_script(job_id: str) -> dict[str, Any]:
    """Get the current script for review."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "title": job.review_data.script_title,
        "outline": job.review_data.script_outline,
        "turns": job.review_data.script_turns,
        "hosts": job.review_data.hosts,
    }


@app.patch("/api/jobs/{job_id}/script")
async def patch_job_script(job_id: str, request: Request) -> dict[str, Any]:
    """Update script turns or title during review."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "awaiting_script_review":
        raise HTTPException(400, "Job is not awaiting script review")

    payload = await request.json()
    if "title" in payload:
        job.review_data.script_title = str(payload["title"]).strip()
    if "turns" in payload:
        turns = payload["turns"]
        if not isinstance(turns, list):
            raise HTTPException(400, "turns must be a list")
        job.review_data.script_turns = [
            {"index": t.get("index", i), "speaker": t.get("speaker", ""), "text": t.get("text", "")}
            for i, t in enumerate(turns)
        ]
    job._emit({"type": "review_data", "data": job.review_data.to_dict()})
    return {"ok": True, "script": job.review_data.to_dict()}


@app.post("/api/jobs/{job_id}/script/regenerate")
async def regenerate_job_script(job_id: str, request: Request) -> dict[str, Any]:
    """Regenerate the script with optional notes (extra instructions)."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "awaiting_script_review":
        raise HTTPException(400, "Job is not awaiting script review")

    payload = await request.json()
    notes = str(payload.get("notes", "") or "").strip()

    cfg = config_mod.load()
    from podcaster.llm import LLMClient
    from podcaster.script import generate_script, Host
    from podcaster.chunking import split_into_chunks
    from podcaster.pipeline import resolve_source, build_hosts

    try:
        doc = resolve_source(job.request)
        chunks = split_into_chunks(doc.text, job.request.chunk_chars or 6000)
        hosts = build_hosts(job.request)

        client = LLMClient(
            base_url=cfg.get("llm_base_url", ""),
            api_key=cfg.get("llm_api_key", ""),
            model=cfg.get("llm_model", ""),
            temperature=float(cfg.get("llm_temperature", 0.7)),
            max_tokens=int(cfg.get("llm_max_tokens", 4096)),
        )

        extra = job.request.extra_instructions
        if notes:
            extra = f"{extra}\n\n{notes}" if extra else notes

        script = generate_script(
            client,
            source_text=doc.text,
            chunks=chunks,
            hosts=hosts,
            language=job.request.language,
            style=job.request.style,
            target_minutes=job.request.length_minutes,
            title_hint=job.request.title or doc.title,
            extra_instructions=extra,
        )

        job.review_data.script_title = script.title
        job.review_data.script_outline = list(script.outline)
        job.review_data.script_turns = [
            {"index": t.index, "speaker": t.speaker, "text": t.text}
            for t in script.turns
        ]
        job._emit({"type": "review_data", "data": job.review_data.to_dict()})
        job.log("info", f"Script regenerated: {len(script.turns)} turns")

    except Exception as exc:
        raise HTTPException(500, f"Script regeneration failed: {exc}") from exc

    return {"ok": True, "script": job.review_data.to_dict()}


@app.post("/api/jobs/{job_id}/approve")
async def approve_job(job_id: str) -> dict[str, Any]:
    """Approve the current review stage and continue the pipeline."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status not in ("awaiting_script_review", "awaiting_audio_review", "awaiting_slides_review"):
        raise HTTPException(400, f"Job is not awaiting review (status: {job.status})")

    job.approve_and_resume()
    return {"ok": True, "status": job.status}


@app.get("/api/jobs/{job_id}/clips")
async def get_job_clips(job_id: str) -> dict[str, Any]:
    """Get the audio clips for review."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "clips": job.review_data.audio_clips,
        "hosts": job.review_data.hosts,
    }


@app.get("/api/jobs/{job_id}/clips/{filename:path}")
async def get_clip_file(job_id: str, filename: str) -> FileResponse:
    """Serve an audio clip file from the job workdir."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    clips_dir = Path(job.workdir) / "clips"
    clip_path = clips_dir / filename
    if not clip_path.exists() or not clip_path.is_file():
        raise HTTPException(404, "Clip not found")
    if not clip_path.resolve().is_relative_to(clips_dir.resolve()):
        raise HTTPException(403, "Access denied")

    return FileResponse(str(clip_path), media_type="audio/mpeg", filename=filename)


@app.post("/api/jobs/{job_id}/clips/{turn_index}/regenerate")
async def regenerate_clip(job_id: str, turn_index: int, request: Request) -> dict[str, Any]:
    """Regenerate a specific audio clip with optional updated text or voice."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "awaiting_audio_review":
        raise HTTPException(400, "Job is not awaiting audio review")

    payload = await request.json()
    new_text = payload.get("text")
    new_voice_id = payload.get("voice_id")

    clips = job.review_data.audio_clips
    clip_data = next((c for c in clips if c["index"] == turn_index), None)
    if not clip_data:
        raise HTTPException(404, f"Turn {turn_index} not found")

    cfg = config_mod.load()
    from podcaster.tts import build_tts, split_for_tts

    try:
        tts_cfg = dict(cfg)
        tts_cfg["tts_speed"] = job.request.tts_speed or cfg.get("tts_speed", 1.0)
        tts = build_tts(tts_cfg)

        text = new_text if new_text else clip_data["text"]
        voice = new_voice_id if new_voice_id else clip_data["voice_id"]
        max_chars = 4800 if "2.8" in (cfg.get("tts_model") or "") else 4000
        want_subtitles = job.request.export_format == "video" and job.request.show_subtitles

        clips_dir = Path(job.workdir) / "clips"
        pieces = split_for_tts(text, max_chars)
        new_files: list[str] = []
        for pi, piece in enumerate(pieces):
            suffix = "" if len(pieces) == 1 else f"_p{pi}"
            out = clips_dir / f"clip_{turn_index + 1:04d}{suffix}.mp3"
            tts.synthesize(piece, voice, out, language=job.request.language,
                           speed=job.request.tts_speed or None,
                           subtitle_enable=want_subtitles)
            new_files.append(out.name)

        clip_data["text"] = text
        clip_data["voice_id"] = voice
        clip_data["files"] = new_files
        job._emit({"type": "review_data", "data": job.review_data.to_dict()})
        job.log("info", f"Regenerated audio for turn {turn_index + 1}")

    except Exception as exc:
        raise HTTPException(500, f"Clip regeneration failed: {exc}") from exc

    return {"ok": True, "clip": clip_data}


@app.get("/api/jobs/{job_id}/slides")
async def get_job_slides(job_id: str) -> dict[str, Any]:
    """Get the slides for review."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"slides": job.review_data.slides}


@app.get("/api/jobs/{job_id}/slides/{filename:path}")
async def get_slide_file(job_id: str, filename: str) -> FileResponse:
    """Serve a slide image file from the job workdir."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    slides_dir = Path(job.workdir) / "slides"
    slide_path = slides_dir / filename
    if not slide_path.exists() or not slide_path.is_file():
        raise HTTPException(404, "Slide not found")
    if not slide_path.resolve().is_relative_to(slides_dir.resolve()):
        raise HTTPException(403, "Access denied")

    media_type = "image/jpeg"
    if slide_path.suffix.lower() == ".png":
        media_type = "image/png"
    elif slide_path.suffix.lower() == ".webp":
        media_type = "image/webp"

    return FileResponse(str(slide_path), media_type=media_type, filename=filename)


@app.post("/api/jobs/{job_id}/slides/{slide_index}/regenerate")
async def regenerate_slide(job_id: str, slide_index: int, request: Request) -> dict[str, Any]:
    """Regenerate a specific slide with an updated prompt."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "awaiting_slides_review":
        raise HTTPException(400, "Job is not awaiting slides review")

    payload = await request.json()
    new_prompt = payload.get("prompt")
    if not new_prompt:
        raise HTTPException(400, "prompt is required")

    slides = job.review_data.slides
    slide_data = next((s for s in slides if s["index"] == slide_index), None)
    if not slide_data:
        raise HTTPException(404, f"Slide {slide_index} not found")

    cfg = config_mod.load()
    from podcaster import images as images_mod
    from podcaster.slides import build_image_prompt

    try:
        image_client = images_mod.build_image_client(cfg)
        slides_dir = Path(job.workdir) / "slides"
        target = slides_dir / f"slide_{slide_index + 1:02d}.jpg"

        result = image_client.generate(build_image_prompt(new_prompt), target)
        slide_data["prompt"] = new_prompt
        slide_data["image"] = str(result.path)
        job._emit({"type": "review_data", "data": job.review_data.to_dict()})
        job.log("info", f"Regenerated slide {slide_index + 1}")

    except Exception as exc:
        raise HTTPException(500, f"Slide regeneration failed: {exc}") from exc

    return {"ok": True, "slide": slide_data}


# --------------------------------------------------------------------------- #
# Uploads and downloads
# --------------------------------------------------------------------------- #

UPLOAD_DIR = config_mod.DATA_DIR / "uploads"


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in {".pdf", ".md", ".markdown", ".txt", ".text", ".html", ".htm", ".docx", ".rtf"}:
        raise HTTPException(400, f"Unsupported file type '{suffix}'")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / f"{os.urandom(6).hex()}_{Path(file.filename or 'upload').name}"
    data = await file.read()
    if len(data) > 60 * 1024 * 1024:
        raise HTTPException(413, "File is larger than 60 MB")
    target.write_bytes(data)
    return {"path": str(target), "name": file.filename, "bytes": len(data)}


@app.get("/api/download")
async def download(path: str) -> FileResponse:
    target = Path(path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(target), filename=target.name)


@app.get("/api/ffmpeg")
async def ffmpeg_status() -> dict[str, Any]:
    path = shutil.which("ffmpeg")
    return {"available": bool(path), "path": path,
            "version": (media_mod._run([path, "-version"], timeout=30).splitlines() or [""])[0]
            if path else ""}


@app.get("/api/health")
async def health() -> dict[str, Any]:
    cfg = config_mod.load()
    return {"ok": True, "llm_key": bool(cfg.get("llm_api_key")), "tts_key": bool(cfg.get("tts_api_key"))}
