"""The end-to-end podcast pipeline.

source → chunks → script → TTS clips → merged audio → (optional) video
"""
from __future__ import annotations

import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from . import config as config_mod
from . import images as images_mod
from . import media as media_mod
from . import slides as slides_mod
from . import subtitles as subtitles_mod
from . import voices as voices_mod
from .chunking import split_into_chunks
from .llm import LLMClient
from .script import Host, Script, Turn, generate_script, parse_turns
from .sources import SourceDoc, fetch_url, load_local_file
from .tts import build_tts, is_qwen_dialect, split_for_tts

if TYPE_CHECKING:
    from .jobs import Job

Logger = Callable[[str, str], None]  # (level, message)


class PipelineError(RuntimeError):
    pass


@dataclass
class PodcastRequest:
    source_type: str = "url"
    source_url: str = ""
    source_text: str = ""
    source_path: str = ""
    language: str = "en"
    length_minutes: float = 5.0
    style: str = "narrative"
    host_count: int = 2
    host_genders: list[str] = field(default_factory=list)
    host_names: list[str] = field(default_factory=list)
    host_personas: list[str] = field(default_factory=list)
    voice_overrides: dict[str, str] = field(default_factory=dict)
    export_format: str = "audio"
    export_path: str = ""
    title: str = ""
    extra_instructions: str = ""
    tts_speed: float = 1.0
    gap_seconds: float = 0.28
    chunk_chars: int = 6000
    keep_workdir: bool = True
    show_subtitles: bool = False
    show_slides: bool = False
    enable_review: bool = False


DEFAULT_NAMES_MALE = ["Marcus", "Daniel", "Alex", "Kenji", "Victor", "Sam"]
DEFAULT_NAMES_FEMALE = ["Sofia", "Maya", "Elena", "Aiko", "Nina", "Clara"]
DEFAULT_PERSONAS = {
    "narrative": ("Narrator", "Co-host"),
    "interview": ("Host", "Guest"),
    "chitchat": ("Friend", "Friend"),
    "deep_dive": ("Analyst", "Analyst"),
    "news": ("Anchor", "Co-anchor"),
    "educational": ("Teacher", "Listener"),
    "storytelling": ("Storyteller", "Listener"),
    "comedy": ("Comedian", "Sidekick"),
}


def _slugify(text: str, fallback: str = "podcast") -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff\- ]+", "", (text or "").strip().lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:80] or fallback


def build_hosts(req: PodcastRequest, cfg: dict[str, Any] | None = None) -> list[Host]:
    """Build host list with assigned voices.

    When ``cfg`` is provided and indicates Qwen TTS dialect, Qwen-compatible
    voices (Cherry, Ethan, etc.) are assigned instead of MiniMax voices.
    """
    count = max(1, min(6, int(req.host_count or 2)))
    genders: list[str] = []
    for i in range(count):
        if i < len(req.host_genders) and req.host_genders[i]:
            genders.append(req.host_genders[i].lower())
        else:
            genders.append("male" if i % 2 == 0 else "female")

    used: set[str] = set()
    male_pool = list(DEFAULT_NAMES_MALE)
    female_pool = list(DEFAULT_NAMES_FEMALE)
    personas = DEFAULT_PERSONAS.get(req.style, ("Host", "Co-host"))

    hosts: list[Host] = []
    for i in range(count):
        gender = genders[i]
        custom = req.host_names[i].strip() if i < len(req.host_names) else ""
        if custom:
            name = custom
        else:
            pool = male_pool if gender == "male" else female_pool
            name = pool[0] if pool else f"Host {i + 1}"
            if pool:
                pool.remove(name)
        while name in used:
            name = f"{name} {i + 1}"
        used.add(name)

        persona = ""
        if i < len(req.host_personas) and req.host_personas[i].strip():
            persona = req.host_personas[i].strip()
        else:
            persona = personas[i] if i < len(personas) else "Host"

        hosts.append(Host(name=name, gender=gender, persona=persona))

    is_qwen = False
    if cfg:
        is_qwen = is_qwen_dialect(
            cfg.get("tts_model", ""),
            cfg.get("tts_base_url", "")
        )
    voices_mod.assign_voices(hosts, req.language, req.voice_overrides, is_qwen=is_qwen)
    return hosts


def resolve_source(req: PodcastRequest) -> SourceDoc:
    kind = (req.source_type or "url").lower()
    if kind == "url":
        if not req.source_url.strip():
            raise PipelineError("A source URL is required.")
        return fetch_url(req.source_url)
    if kind in ("upload", "file"):
        if not req.source_path:
            raise PipelineError("No uploaded file was provided.")
        return load_local_file(req.source_path)
    if kind == "text":
        text = (req.source_text or "").strip()
        if not text:
            raise PipelineError("The pasted source text is empty.")
        return SourceDoc(text=text, title=req.title or "Pasted text",
                         origin="pasted text", kind="text", meta={"chars": len(text)})
    raise PipelineError(f"Unknown source type '{kind}'.")


def resolve_export_path(req: PodcastRequest, title: str) -> Path:
    ext = ".mp4" if req.export_format == "video" else ".mp3"
    raw = (req.export_path or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if path.is_dir() or raw.endswith(("/", "\\")):
            path = path / f"{_slugify(title)}{ext}"
        else:
            if not path.suffix:
                path = path.with_suffix(ext)
            elif path.suffix.lower() not in (".mp3", ".mp4", ".m4a", ".aac", ".wav"):
                path = path.with_suffix(ext)
    else:
        cfg = config_mod.load()
        outdir = Path(cfg.get("output_dir") or (config_mod.APP_DIR / "output"))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = outdir / f"{_slugify(title)}-{stamp}{ext}"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _clip_units(clip_records: list[dict[str, Any]], clips_dir: Path) -> list[dict[str, Any]]:
    """One entry per rendered audio clip, in episode order."""
    units: list[dict[str, Any]] = []
    for record in clip_records:
        cue_sets = record.get("subtitles") or []
        source_sets = record.get("sources") or []
        for index, filename in enumerate(record["files"]):
            path = clips_dir / filename
            units.append({
                "path": path,
                "speaker": record["speaker"],
                "turn": record["index"],
                "cues": cue_sets[index] if index < len(cue_sets) else [],
                "source": source_sets[index] if index < len(source_sets) else "",
                "duration": media_mod.probe_duration(path),
            })
    return units


def _build_video_segments(
    req: PodcastRequest,
    script: Script,
    hosts: list[Host],
    clip_records: list[dict[str, Any]],
    clips_dir: Path,
    workdir: Path,
    client: LLMClient,
    cfg: dict[str, Any],
    log: Logger,
    on_step: Callable[[str, float], None],
    job: "Job | None" = None,
    should_run_phase: Callable[[str], bool] = lambda _: True,
) -> tuple[list[media_mod.VideoSegment], str]:
    """Build the video segments and pick the layout.

    Returns the segments plus the render mode: ``waveform`` for the usual
    animated bars, ``slides`` for the illustrated layout.
    """
    gender_by_name = {h.name: h.gender for h in hosts}
    host_names = [h.name for h in hosts]

    def make_segment(audio: Path, speaker: str, cues: list[Any], source: str, *,
                     slide: Path | None = None, title_card: bool = False):
        return media_mod.VideoSegment(
            audio=audio,
            speaker=speaker,
            color=voices_mod.color_for(gender_by_name.get(speaker, "mixed")),
            label=speaker,
            subtitles=subtitles_mod.build_lines(cues, source),
            slide=slide,
            title_card=title_card,
            hosts=list(host_names) if title_card else [],
        )

    units = _clip_units(clip_records, clips_dir)
    if not units:
        raise PipelineError("No audio was produced for the video.")

    plan: list[slides_mod.Slide] = []
    if req.show_slides:
        total_seconds = sum(u["duration"] for u in units)
        turn_infos = []
        for record in clip_records:
            seconds = sum(u["duration"] for u in units if u["turn"] == record["index"])
            turn_infos.append(slides_mod.TurnInfo(
                index=record["index"], speaker=record["speaker"],
                text=record["text"], seconds=seconds))
        on_step("Planning the slides", 0.87)
        plan = slides_mod.plan_slides(client, turn_infos, total_seconds,
                                       language=req.language, title=script.title)
        if plan:
            on_step(f"Drawing {len(plan)} slide images", 0.88)
            image_client = images_mod.build_image_client(cfg)
            slides_mod.generate_images(
                image_client, plan, workdir / "slides", log=log,
                progress=lambda stage, pct: on_step(stage, 0.88 + pct * 0.04))
            if not any(slide.image for slide in plan):
                log("warn", "No slide images could be generated — using the waveform layout.")
                plan = []
            # ---- Slides review gate --------------------------------------- #
            elif should_run_phase("slides_review") and req.enable_review and job is not None:
                if job:
                    job.set_phase("slides_review")
                job.set_review_data(
                    slides=[{
                        "index": s.index,
                        "title": s.title,
                        "prompt": s.prompt,
                        "image": str(s.image) if s.image else None,
                        "turn_indexes": s.turn_indexes,
                    } for s in plan]
                )
                if not job.wait_for_review("awaiting_slides_review"):
                    raise PipelineError("Cancelled by the user.")
                job.set_status("running")
                on_step("Resuming after slides review", 0.92)
                updated_slides = job.review_data.slides
                if updated_slides:
                    for i, slide_data in enumerate(updated_slides):
                        if i < len(plan) and slide_data.get("image"):
                            plan[i].image = Path(slide_data["image"])
        else:
            log("warn", "The slide planner returned nothing — using the waveform layout.")

    if not plan:
        return ([make_segment(u["path"], u["speaker"], u["cues"], u["source"]) for u in units],
                "waveform")

    # ---- illustrated layout ------------------------------------------------ #
    slide_for_turn: dict[int, int] = {}
    for slide in plan:
        for turn in slide.turn_indexes:
            slide_for_turn[turn] = slide.index
    images = [slide.image for slide in plan]

    segments: list[media_mod.VideoSegment] = []

    # The opening title card: the episode title and the hosts, spoken over the
    # first few seconds of the intro.
    first = units[0]
    card_seconds = min(first["duration"], slides_mod.TITLE_CARD_SECONDS)
    if card_seconds > 0.05:
        card_audio = media_mod.trim_audio(first["path"], 0.0, card_seconds,
                                          workdir / "title_card.mp3")
        split_ms = card_seconds * 1000.0
        card_cues = [c for c in first["cues"] if c.begin_ms < split_ms]
        rest_cues = [replace(c, begin_ms=c.begin_ms - split_ms, end_ms=c.end_ms - split_ms)
                     for c in first["cues"] if c.begin_ms >= split_ms]
        segments.append(make_segment(card_audio, first["speaker"], card_cues,
                                     first["source"], title_card=True))
        if first["duration"] > card_seconds + 0.2:
            rest_audio = media_mod.trim_audio(first["path"], card_seconds,
                                              first["duration"] - card_seconds,
                                              workdir / "intro_rest.mp3")
            segments.append(make_segment(rest_audio, first["speaker"], rest_cues,
                                         first["source"], slide=images[0]))

    for unit in units[1:]:
        index = slide_for_turn.get(unit["turn"], 0)
        segments.append(make_segment(unit["path"], unit["speaker"], unit["cues"],
                                     unit["source"], slide=images[index]))
    return segments, "slides"


def _synthesize_turn(
    turn_index: int,
    turn: Turn,
    voice: str,
    tts: Any,
    clips_dir: Path,
    req: PodcastRequest,
    max_chars: int,
    want_subtitles: bool,
) -> dict[str, Any]:
    """Synthesize a single turn (may be called in parallel)."""
    pieces = split_for_tts(turn.text, max_chars)
    turn_clips: list[Path] = []
    turn_subtitles: list[list[subtitles_mod.SubtitleCue]] = []
    turn_sources: list[str] = []
    for pi, piece in enumerate(pieces):
        suffix = "" if len(pieces) == 1 else f"_p{pi}"
        out = clips_dir / f"clip_{turn_index + 1:04d}{suffix}.mp3"
        result = tts.synthesize(piece, voice, out, language=req.language,
                                speed=req.tts_speed or None,
                                subtitle_enable=want_subtitles)
        turn_clips.append(out)
        turn_subtitles.append(result.subtitles)
        turn_sources.append(piece)
    return {
        "index": turn_index,
        "speaker": turn.speaker,
        "text": turn.text,
        "voice_id": voice,
        "files": [c.name for c in turn_clips],
        "subtitles": turn_subtitles,
        "sources": turn_sources,
    }


def run_pipeline(
    req: PodcastRequest,
    workdir: Path,
    *,
    on_step: Callable[[str, float], None],
    log: Logger,
    should_cancel: Callable[[], bool] = lambda: False,
    job: "Job | None" = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    cfg = config_mod.load()
    workdir.mkdir(parents=True, exist_ok=True)

    # Phase ordering for resume support
    phases = ["source", "chunks", "script", "script_review", "tts", "audio_review", "merge", "video", "slides_review", "done"]
    skip_until = phases.index(resume_from) if resume_from and resume_from in phases else 0

    def should_run_phase(phase: str) -> bool:
        return phases.index(phase) >= skip_until

    def set_phase(phase: str) -> None:
        if job:
            job.set_phase(phase)

    # ---- 1. Source -------------------------------------------------------- #
    set_phase("source")
    if should_run_phase("source"):
        on_step("Reading the source material", 0.03)
        doc = resolve_source(req)
        log("info", f"Loaded {doc.kind} source: {len(doc.text):,} characters from {doc.origin}")
        if len(doc.text.strip()) < 120:
            raise PipelineError("The source material is too short to build a podcast from.")
        if job:
            job.save_pipeline_state("source", {"text": doc.text, "title": doc.title, "origin": doc.origin, "kind": doc.kind})
    else:
        source_state = job.load_pipeline_state("source") if job else None
        if not source_state:
            raise PipelineError("Cannot resume: source state not found")
        doc = SourceDoc(text=source_state["text"], title=source_state["title"],
                        origin=source_state["origin"], kind=source_state["kind"], meta={})
        log("info", f"Resumed with source: {len(doc.text):,} characters")

    # ---- 2. Chunks -------------------------------------------------------- #
    set_phase("chunks")
    chunk_size = max(1200, min(20000, int(req.chunk_chars or 6000)))
    if should_run_phase("chunks"):
        on_step("Splitting the source into chunks", 0.07)
        chunks = split_into_chunks(doc.text, chunk_size)
        log("info", f"Split into {len(chunks)} chunk(s) of ~{chunk_size} characters")
        if job:
            job.save_pipeline_state("chunks", chunks)
    else:
        chunks = job.load_pipeline_state("chunks") if job else None
        if not chunks:
            chunks = split_into_chunks(doc.text, chunk_size)

    # ---- 3. Script -------------------------------------------------------- #
    set_phase("script")
    hosts = build_hosts(req, cfg)
    log("info", "Casting: " + ", ".join(f"{h.name} ({h.gender}) → {h.voice_id}" for h in hosts))

    client = LLMClient(
        base_url=cfg.get("llm_base_url", ""),
        api_key=cfg.get("llm_api_key", ""),
        model=cfg.get("llm_model", ""),
        temperature=float(cfg.get("llm_temperature", 0.7)),
        max_tokens=int(cfg.get("llm_max_tokens", 4096)),
    )

    script = None
    if should_run_phase("script"):
        on_step("Writing the podcast script", 0.1)
        if not client.api_key:
            raise PipelineError("The LLM API key is not configured. Add it in Settings.")
        script = generate_script(
            client,
            source_text=doc.text,
            chunks=chunks,
            hosts=hosts,
            language=req.language,
            style=req.style,
            target_minutes=req.length_minutes,
            title_hint=req.title or doc.title,
            extra_instructions=req.extra_instructions,
            progress=lambda stage, pct: on_step(stage, 0.1 + pct * 0.5),
        )
        if not script.turns:
            raise PipelineError("The script generator produced no usable dialogue lines.")
        script.title = req.title or script.title or doc.title
        log("info", f"Script written: {len(script.turns)} turns, "
                    f"~{len(script.text):,} characters of dialogue")
        if job:
            job.save_pipeline_state("script", {
                "title": script.title,
                "outline": script.outline,
                "turns": [{"index": t.index, "speaker": t.speaker, "text": t.text, "segment": t.segment}
                          for t in script.turns],
            })
    else:
        script_state = job.load_pipeline_state("script") if job else None
        if not script_state:
            raise PipelineError("Cannot resume: script state not found")
        script = Script(
            title=script_state["title"],
            outline=script_state.get("outline", []),
            turns=[Turn(speaker=t["speaker"], text=t["text"], index=t["index"], segment=t.get("segment", ""))
                   for t in script_state["turns"]],
        )
        log("info", f"Resumed with script: {len(script.turns)} turns")

    # ---- Script review gate ----------------------------------------------- #
    set_phase("script_review")
    if should_run_phase("script_review") and req.enable_review and job is not None:
        job.set_review_data(
            script_title=script.title,
            script_outline=list(script.outline),
            script_turns=[{"index": t.index, "speaker": t.speaker, "text": t.text}
                          for t in script.turns],
            hosts=[h.to_dict() for h in hosts],
        )
        job.save_pipeline_state("script", {
            "title": script.title,
            "outline": script.outline,
            "turns": [{"index": t.index, "speaker": t.speaker, "text": t.text, "segment": t.segment}
                      for t in script.turns],
        })
        if not job.wait_for_review("awaiting_script_review"):
            raise PipelineError("Cancelled by the user.")
        job.set_status("running")
        on_step("Resuming after script review", 0.60)
        updated_turns = job.review_data.script_turns
        if updated_turns:
            script.turns = [Turn(speaker=t["speaker"], text=t["text"], index=t["index"])
                            for t in updated_turns]
            script.title = job.review_data.script_title or script.title
            log("info", f"Script updated: {len(script.turns)} turns after review")
            if job:
                job.save_pipeline_state("script", {
                    "title": script.title,
                    "outline": script.outline,
                    "turns": [{"index": t.index, "speaker": t.speaker, "text": t.text, "segment": t.segment}
                              for t in script.turns],
                })

    # ---- 4. TTS ----------------------------------------------------------- #
    set_phase("tts")
    tts_cfg = dict(cfg)
    tts_cfg["tts_speed"] = req.tts_speed or cfg.get("tts_speed", 1.0)
    tts = build_tts(tts_cfg)
    clips_dir = workdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clips: list[Path] = []
    clip_records: list[dict[str, Any]] = []
    voice_by_name = {h.name: h for h in hosts}
    max_chars = getattr(tts, "max_chars", lambda: 4000)()
    total = len(script.turns)
    want_subtitles = req.export_format == "video" and bool(req.show_subtitles)

    if should_run_phase("tts"):
        use_parallel = cfg.get("tts_parallel", False)
        parallel_workers = max(1, min(6, int(cfg.get("tts_parallel_workers", 2))))

        if use_parallel and total > 1:
            log("info", f"Using parallel TTS with {parallel_workers} workers")
            on_step(f"Generating audio for {total} turns (parallel)", 0.60)
            futures_map: dict[Any, int] = {}
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                for i, turn in enumerate(script.turns):
                    if should_cancel():
                        raise PipelineError("Cancelled by the user.")
                    host = voice_by_name.get(turn.speaker) or hosts[0]
                    voice = host.voice_id or hosts[0].voice_id
                    future = executor.submit(
                        _synthesize_turn, i, turn, voice, tts, clips_dir, req,
                        max_chars, want_subtitles)
                    futures_map[future] = i

                results_by_index: dict[int, dict[str, Any]] = {}
                completed = 0
                for future in as_completed(futures_map):
                    if should_cancel():
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise PipelineError("Cancelled by the user.")
                    idx = futures_map[future]
                    try:
                        result = future.result()
                        results_by_index[idx] = result
                    except Exception as exc:
                        raise PipelineError(f"TTS failed for turn {idx + 1}: {exc}") from exc
                    completed += 1
                    on_step(f"Generating audio ({completed}/{total})", 0.60 + 0.25 * (completed / total))

            for i in range(total):
                record = results_by_index[i]
                clip_records.append(record)
                for fname in record["files"]:
                    clips.append(clips_dir / fname)
        else:
            for i, turn in enumerate(script.turns):
                if should_cancel():
                    raise PipelineError("Cancelled by the user.")
                on_step(f"Generating audio for turn {i + 1}/{total}", 0.6 + 0.25 * (i / max(1, total)))
                host = voice_by_name.get(turn.speaker) or hosts[0]
                voice = host.voice_id or hosts[0].voice_id
                record = _synthesize_turn(i, turn, voice, tts, clips_dir, req, max_chars, want_subtitles)
                clips.extend(clips_dir / fname for fname in record["files"])
                clip_records.append(record)

        if job:
            job.save_pipeline_state("clip_records", clip_records)
    else:
        clip_records = job.load_pipeline_state("clip_records") if job else None
        if not clip_records:
            raise PipelineError("Cannot resume: audio clips state not found")
        clips = [clips_dir / fname for r in clip_records for fname in r["files"]]
        # Verify clips exist
        missing = [str(c) for c in clips if not c.exists()]
        if missing:
            raise PipelineError(f"Cannot resume: missing audio files: {missing[:3]}")
        log("info", f"Resumed with {len(clips)} audio clips")

    if should_cancel():
        raise PipelineError("Cancelled by the user.")

    # ---- Audio review gate ------------------------------------------------ #
    set_phase("audio_review")
    if should_run_phase("audio_review") and req.enable_review and job is not None:
        job.set_review_data(
            audio_clips=[{
                "index": r["index"],
                "speaker": r["speaker"],
                "text": r["text"],
                "voice_id": r["voice_id"],
                "files": r["files"],
            } for r in clip_records]
        )
        if not job.wait_for_review("awaiting_audio_review"):
            raise PipelineError("Cancelled by the user.")
        job.set_status("running")
        on_step("Resuming after audio review", 0.86)
        clips = [clips_dir / fname for r in clip_records for fname in r["files"]]

    # ---- 5. Merge audio --------------------------------------------------- #
    set_phase("merge")
    export_path = resolve_export_path(req, script.title)

    # For a video export the final artefact is the MP4, so the merged audio is
    # written next to it as an .mp3 rather than onto the export path.
    if req.export_format == "video":
        audio_target = export_path.with_suffix(".mp3")
    else:
        audio_target = export_path

    merged = None
    if should_run_phase("merge"):
        on_step("Merging the audio clips", 0.86)
        merged = media_mod.merge_audio(clips, audio_target, gap_seconds=req.gap_seconds,
                                       workdir=workdir)
        log("info", f"Merged {len(clips)} clips into {audio_target.name} "
                    f"({merged.duration / 60:.1f} min)")
        if job:
            job.save_pipeline_state("merged", {"path": str(merged.path), "duration": merged.duration})
    else:
        merged_state = job.load_pipeline_state("merged") if job else None
        if merged_state and Path(merged_state["path"]).exists():
            merged = media_mod.MergedAudio(path=Path(merged_state["path"]), duration=merged_state["duration"])
            log("info", f"Resumed with merged audio: {merged.path.name}")
        else:
            on_step("Re-merging audio clips", 0.86)
            merged = media_mod.merge_audio(clips, audio_target, gap_seconds=req.gap_seconds,
                                           workdir=workdir)
            log("info", f"Re-merged {len(clips)} clips")

    # ---- 6. Video (optional) ---------------------------------------------- #
    set_phase("video")
    video_path: Path | None = None
    slide_count = 0
    if req.export_format == "video" and should_run_phase("video"):
        video_segments, mode = _build_video_segments(
            req, script, hosts, clip_records, clips_dir, workdir, client, cfg, log, on_step, job, should_run_phase)
        slide_count = len({segment.slide for segment in video_segments if segment.slide})
        on_step("Rendering the slide video" if mode == "slides"
                else "Rendering the waveform video", 0.93)
        font = media_mod.find_font(req.language)
        video_export = export_path.with_suffix(".mp4")
        video_path = media_mod.render_video(
            video_segments, video_export,
            title=script.title,
            font_path=font,
            workdir=workdir,
            progress=lambda stage, pct: on_step(stage, 0.93 + pct * 0.06),
            mode=mode,
        )
        log("info", f"Rendered video: {video_path.name}")
        export_path = video_path
    elif req.export_format == "video":
        video_export = export_path.with_suffix(".mp4")
        if video_export.exists():
            video_path = video_export
            export_path = video_path
            log("info", f"Resumed with existing video: {video_path.name}")

    # ---- 7. Artefacts ------------------------------------------------------ #
    script_path = workdir / "script.txt"
    script_path.write_text(
        f"# {script.title}\n"
        f"Style: {req.style} | Language: {req.language} | "
        f"Target: {req.length_minutes} min | Actual: {merged.duration / 60:.1f} min\n\n"
        + script.text + "\n", encoding="utf-8")

    if not req.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    return {
        "script": script,
        "script_path": str(script_path),
        "audio_path": str(merged.path),
        "video_path": str(video_path) if video_path else None,
        "export_path": str(export_path),
        "duration_seconds": merged.duration,
        "turns": len(script.turns),
        "clips": len(clips),
        "hosts": [h.to_dict() for h in hosts],
        "outline": script.outline,
        "source_chars": len(doc.text),
        "chunks": len(chunks),
        "slides": slide_count,
    }
