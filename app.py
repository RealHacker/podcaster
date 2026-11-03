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
        from podcaster.tts import build_tts, is_qwen_dialect

        engine = build_tts(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "probe.mp3"
            is_qwen = is_qwen_dialect(cfg.get("tts_model", ""), cfg.get("tts_base_url", ""))
            if is_qwen:
                text = "Voice test." if cfg.get("tts_provider") == "openai" else "语音测试。"
                voice_id = voices_mod.get_default_qwen_voice()
            elif cfg.get("tts_provider") == "openai":
                text = "Voice test."
                voice_id = voices_mod.load_catalog()[0]["id"]
            else:
                text = "语音测试。"
                voice_id = voices_mod.load_catalog()[0]["id"]
            engine.synthesize(text, voice_id, out, language="en" if is_qwen else "zh")
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
            if event.get("type") == "status" and event.get("status") in ("done", "error", "cancelled"):
                yield f"data: {json.dumps(event)}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
