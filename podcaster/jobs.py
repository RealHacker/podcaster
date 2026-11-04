"""Job registry: creation, progress tracking and a live event feed."""
from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as config_mod
from .pipeline import PodcastRequest, PipelineError, run_pipeline

REVIEW_STATUSES = ("awaiting_script_review", "awaiting_audio_review", "awaiting_slides_review")
TERMINAL_STATUSES = ("done", "error", "cancelled")
INTERRUPTED_STATUSES = ("running",) + REVIEW_STATUSES
RESUMABLE_STATUSES = INTERRUPTED_STATUSES + ("interrupted",)

PHASES = (
    "source",
    "chunks",
    "script",
    "script_review",
    "tts",
    "audio_review",
    "merge",
    "video",
    "slides_review",
    "done",
)

logger = logging.getLogger(__name__)


@dataclass
class ReviewData:
    """Holds reviewable artifacts at a particular stage."""
    script_title: str = ""
    script_outline: list[str] = field(default_factory=list)
    script_turns: list[dict[str, Any]] = field(default_factory=list)
    audio_clips: list[dict[str, Any]] = field(default_factory=list)
    slides: list[dict[str, Any]] = field(default_factory=list)
    hosts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "script_title": self.script_title,
            "script_outline": list(self.script_outline),
            "script_turns": list(self.script_turns),
            "audio_clips": list(self.audio_clips),
            "slides": list(self.slides),
            "hosts": list(self.hosts),
        }


@dataclass
class Job:
    id: str
    request: PodcastRequest
    status: str = "queued"          # queued | running | done | error | cancelled | awaiting_*_review | interrupted
    step: str = "Queued"
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    workdir: str = ""
    review_data: ReviewData = field(default_factory=ReviewData)
    current_phase: str = "source"   # tracks pipeline phase for resume
    _events: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _resume: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _seq: int = field(default=0, repr=False)
    _pipeline_state: dict[str, Any] = field(default_factory=dict, repr=False)
    _persist_callback: Any = field(default=None, repr=False)  # callback to trigger persistence

    # -- updates ---------------------------------------------------------- #
    def _emit(self, payload: dict[str, Any]) -> None:
        # Every event carries a monotonic sequence number so that a client
        # which received a snapshot can drop the events the snapshot already
        # contains instead of rendering them twice.
        with self._lock:
            self._seq += 1
            payload["seq"] = self._seq
        try:
            self._events.put_nowait(payload)
        except queue.Full:
            pass

    def _trigger_persist(self) -> None:
        """Notify the store to persist job state."""
        if self._persist_callback:
            try:
                self._persist_callback()
            except Exception:
                pass

    def set_phase(self, phase: str) -> None:
        """Track which pipeline phase the job is in for resume support."""
        with self._lock:
            self.current_phase = phase
        self._trigger_persist()

    def log(self, level: str, message: str) -> None:
        entry = {"t": time.time(), "level": level, "message": message}
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > 600:
                del self.logs[:200]
        self._emit({"type": "log", "entry": entry})

    def set_step(self, step: str, progress: float | None = None) -> None:
        with self._lock:
            self.step = step
            if progress is not None:
                self.progress = max(0.0, min(1.0, progress))
            self.message = step
        self._emit({"type": "step", "step": step, "progress": self.progress})

    def set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            if error:
                self.error = error
            if status in TERMINAL_STATUSES:
                self.finished_at = time.time()
                self.progress = 1.0 if status == "done" else self.progress
        self._emit({"type": "status", "status": status, "error": error})
        self._trigger_persist()

    def cancel(self) -> None:
        self._cancel.set()
        self._resume.set()
        self.log("warn", "Cancellation requested")

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def wait_for_review(self, status: str, timeout: float | None = None) -> bool:
        """Block until the user approves or cancels. Returns True if approved."""
        self._resume.clear()
        self.set_status(status)
        self.log("info", f"Waiting for review ({status})")
        approved = self._resume.wait(timeout=timeout)
        return approved and not self.cancelled

    def approve_and_resume(self) -> None:
        """Signal the pipeline to continue after a review gate."""
        self._resume.set()
        self.log("info", "Review approved, resuming")

    def set_review_data(self, **kwargs: Any) -> None:
        """Update review data fields."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.review_data, key):
                    setattr(self.review_data, key, value)
        self._emit({"type": "review_data", "data": self.review_data.to_dict()})

    def save_pipeline_state(self, key: str, value: Any) -> None:
        """Save intermediate pipeline state for resumption."""
        with self._lock:
            self._pipeline_state[key] = value
        state_file = Path(self.workdir) / "pipeline_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            state_file.write_text(json.dumps(self._pipeline_state, default=str), encoding="utf-8")
        except OSError:
            pass

    def load_pipeline_state(self, key: str, default: Any = None) -> Any:
        """Load saved pipeline state."""
        with self._lock:
            if key in self._pipeline_state:
                return self._pipeline_state[key]
        state_file = Path(self.workdir) / "pipeline_state.json"
        if state_file.exists():
            try:
                self._pipeline_state = json.loads(state_file.read_text(encoding="utf-8"))
                return self._pipeline_state.get(key, default)
            except (OSError, json.JSONDecodeError):
                pass
        return default

    # -- serialisation ---------------------------------------------------- #
    def to_dict(self, include_logs: bool = True, include_review: bool = True) -> dict[str, Any]:
        with self._lock:
            data = {
                "id": self.id,
                "status": self.status,
                "step": self.step,
                "progress": self.progress,
                "message": self.message,
                "error": self.error,
                "seq": self._seq,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed": ((self.finished_at or time.time()) - (self.started_at or self.created_at)),
                "result": self._serialize_result(self.result),
                "current_phase": self.current_phase,
                "workdir": self.workdir,
                "request": {
                    "language": self.request.language,
                    "style": self.request.style,
                    "length_minutes": self.request.length_minutes,
                    "host_count": self.request.host_count,
                    "export_format": self.request.export_format,
                    "export_path": self.request.export_path,
                    "title": self.request.title,
                    "source_type": self.request.source_type,
                    "source_url": self.request.source_url,
                    "enable_review": getattr(self.request, "enable_review", False),
                },
            }
            if include_logs:
                data["logs"] = list(self.logs)
            if include_review and self.status in REVIEW_STATUSES:
                data["review_data"] = self.review_data.to_dict()
        return data

    @staticmethod
    def _serialize_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
        if not result:
            return None
        out = dict(result)
        script = out.pop("script", None)
        if script is not None:
            out["script_text"] = script.text
            out["script_title"] = script.title
            out["script_outline"] = script.outline
        out.pop("clips", None)
        return out

    def to_persist_dict(self) -> dict[str, Any]:
        """Full serialization for disk persistence, includes all fields needed for resume."""
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "step": self.step,
                "progress": self.progress,
                "message": self.message,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "logs": list(self.logs[-100:]),  # keep last 100 logs
                "result": self._serialize_result(self.result),
                "workdir": self.workdir,
                "current_phase": self.current_phase,
                "review_data": self.review_data.to_dict(),
                "request": self._serialize_request(),
                "pipeline_state": dict(self._pipeline_state),
            }

    def _serialize_request(self) -> dict[str, Any]:
        """Serialize the full PodcastRequest for persistence."""
        req = self.request
        return {
            "source_type": req.source_type,
            "source_url": req.source_url,
            "source_text": req.source_text,
            "source_path": req.source_path,
            "language": req.language,
            "length_minutes": req.length_minutes,
            "style": req.style,
            "host_count": req.host_count,
            "host_genders": list(req.host_genders),
            "host_names": list(req.host_names),
            "host_personas": list(req.host_personas),
            "voice_overrides": dict(req.voice_overrides),
            "export_format": req.export_format,
            "export_path": req.export_path,
            "title": req.title,
            "extra_instructions": req.extra_instructions,
            "tts_speed": req.tts_speed,
            "gap_seconds": req.gap_seconds,
            "chunk_chars": req.chunk_chars,
            "keep_workdir": req.keep_workdir,
            "show_subtitles": req.show_subtitles,
            "show_slides": req.show_slides,
            "enable_review": req.enable_review,
        }

    @classmethod
    def from_persist_dict(cls, data: dict[str, Any]) -> "Job":
        """Restore a Job from persisted data."""
        req_data = data.get("request", {})
        request = PodcastRequest(
            source_type=req_data.get("source_type", "url"),
            source_url=req_data.get("source_url", ""),
            source_text=req_data.get("source_text", ""),
            source_path=req_data.get("source_path", ""),
            language=req_data.get("language", "en"),
            length_minutes=float(req_data.get("length_minutes", 5.0)),
            style=req_data.get("style", "narrative"),
            host_count=int(req_data.get("host_count", 2)),
            host_genders=list(req_data.get("host_genders", [])),
            host_names=list(req_data.get("host_names", [])),
            host_personas=list(req_data.get("host_personas", [])),
            voice_overrides=dict(req_data.get("voice_overrides", {})),
            export_format=req_data.get("export_format", "audio"),
            export_path=req_data.get("export_path", ""),
            title=req_data.get("title", ""),
            extra_instructions=req_data.get("extra_instructions", ""),
            tts_speed=float(req_data.get("tts_speed", 1.0)),
            gap_seconds=float(req_data.get("gap_seconds", 0.28)),
            chunk_chars=int(req_data.get("chunk_chars", 6000)),
            keep_workdir=bool(req_data.get("keep_workdir", True)),
            show_subtitles=bool(req_data.get("show_subtitles", False)),
            show_slides=bool(req_data.get("show_slides", False)),
            enable_review=bool(req_data.get("enable_review", False)),
        )

        review_dict = data.get("review_data", {})
        review_data = ReviewData(
            script_title=review_dict.get("script_title", ""),
            script_outline=list(review_dict.get("script_outline", [])),
            script_turns=list(review_dict.get("script_turns", [])),
            audio_clips=list(review_dict.get("audio_clips", [])),
            slides=list(review_dict.get("slides", [])),
            hosts=list(review_dict.get("hosts", [])),
        )

        job = cls(
            id=data["id"],
            request=request,
            status=data.get("status", "queued"),
            step=data.get("step", ""),
            progress=float(data.get("progress", 0.0)),
            message=data.get("message", ""),
            error=data.get("error"),
            created_at=float(data.get("created_at", time.time())),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            logs=list(data.get("logs", [])),
            result=data.get("result"),
            workdir=data.get("workdir", ""),
            review_data=review_data,
            current_phase=data.get("current_phase", "source"),
        )
        job._pipeline_state = dict(data.get("pipeline_state", {}))
        return job


class JobStore:
    JOBS_FILE = "jobs.json"
    MAX_PERSISTED_JOBS = 100

    def __init__(self, max_workers: int = 2, persist_dir: Path | None = None):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._persist_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                             thread_name_prefix="podcaster")
        self._persist_dir = persist_dir or config_mod.DATA_DIR
        self._jobs_file = self._persist_dir / self.JOBS_FILE
        self._load_persisted_jobs()

    def _load_persisted_jobs(self) -> None:
        """Load jobs from disk on startup."""
        if not self._jobs_file.exists():
            return
        try:
            data = json.loads(self._jobs_file.read_text(encoding="utf-8"))
            jobs_data = data.get("jobs", [])
            for job_data in jobs_data:
                try:
                    job = Job.from_persist_dict(job_data)
                    # Mark running/review jobs as interrupted
                    if job.status in INTERRUPTED_STATUSES:
                        job.status = "interrupted"
                        job.log("warn", "Job was interrupted by app shutdown")
                    job._persist_callback = self._persist
                    self._jobs[job.id] = job
                    self._order.append(job.id)
                except Exception as e:
                    logger.warning(f"Failed to restore job {job_data.get('id')}: {e}")
            logger.info(f"Loaded {len(self._jobs)} persisted jobs")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load persisted jobs: {e}")

    def _persist(self) -> None:
        """Persist all jobs to disk."""
        with self._persist_lock:
            try:
                self._persist_dir.mkdir(parents=True, exist_ok=True)
                with self._lock:
                    jobs_data = []
                    for job_id in self._order[-self.MAX_PERSISTED_JOBS:]:
                        if job_id in self._jobs:
                            jobs_data.append(self._jobs[job_id].to_persist_dict())
                data = {
                    "version": 1,
                    "updated_at": time.time(),
                    "jobs": jobs_data,
                }
                temp_file = self._jobs_file.with_suffix(".tmp")
                temp_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                temp_file.replace(self._jobs_file)
            except OSError as e:
                logger.error(f"Failed to persist jobs: {e}")

    def create(self, request: PodcastRequest) -> Job:
        job_id = uuid.uuid4().hex[:12]
        workdir = config_mod.DATA_DIR / "jobs" / job_id
        job = Job(id=job_id, request=request, workdir=str(workdir))
        job._persist_callback = self._persist
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            if len(self._order) > self.MAX_PERSISTED_JOBS:
                oldest = self._order.pop(0)
                self._jobs.pop(oldest, None)
        self._persist()
        self._executor.submit(self._run, job)
        return job

    def _run(self, job: Job, resume_from: str | None = None) -> None:
        job.started_at = time.time()
        job.set_status("running")
        job.log("info", "Job started" if not resume_from else f"Job resumed from {resume_from}")
        try:
            result = run_pipeline(
                job.request,
                Path(job.workdir),
                on_step=job.set_step,
                log=job.log,
                should_cancel=lambda: job.cancelled,
                job=job,
                resume_from=resume_from,
            )
            job.result = result
            job.set_phase("done")
            job.set_status("done")
            job.set_step("Finished", 1.0)
            job.log("info", f"Done — output at {result['export_path']}")
        except PipelineError as exc:
            message = str(exc)
            status = "cancelled" if "Cancel" in message else "error"
            job.set_status(status, message if status == "error" else None)
            job.log("error" if status == "error" else "warn", message)
        except Exception as exc:  # noqa: BLE001
            import traceback

            job.set_status("error", f"{type(exc).__name__}: {exc}")
            job.log("error", f"{type(exc).__name__}: {exc}")
            for line in traceback.format_exc().splitlines()[-8:]:
                job.log("error", line)

    def resume(self, job_id: str) -> Job | None:
        """Resume an interrupted job from its last phase."""
        job = self.get(job_id)
        if not job:
            return None
        if job.status not in RESUMABLE_STATUSES:
            return None

        # Reset runtime state for resumed execution
        job._cancel = threading.Event()
        job._resume = threading.Event()
        job._events = queue.Queue()
        job.error = None

        resume_from = job.current_phase
        job.log("info", f"Resuming job from phase: {resume_from}")
        self._executor.submit(self._run, job, resume_from)
        return job

    def delete(self, job_id: str, keep_exports: bool = True) -> dict[str, Any]:
        """Delete a job and its temporary files, optionally keeping final exports."""
        job = self.get(job_id)
        if not job:
            return {"ok": False, "error": "Job not found"}

        if job.status in ("running", "queued") + REVIEW_STATUSES:
            return {"ok": False, "error": "Cannot delete a running job"}

        deleted_files: list[str] = []
        kept_exports: list[str] = []

        # Identify final exports to keep
        exports_to_keep: set[Path] = set()
        if keep_exports and job.result:
            for key in ("export_path", "audio_path", "video_path"):
                path_str = job.result.get(key)
                if path_str:
                    export_path = Path(path_str)
                    if export_path.exists():
                        exports_to_keep.add(export_path.resolve())
                        kept_exports.append(str(export_path))

        # Delete workdir (temp files)
        workdir = Path(job.workdir)
        if workdir.exists():
            # Don't delete exports that happen to be in workdir (unlikely but safe)
            for item in workdir.rglob("*"):
                if item.is_file() and item.resolve() not in exports_to_keep:
                    try:
                        item.unlink()
                        deleted_files.append(str(item))
                    except OSError:
                        pass
            # Remove empty directories
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except OSError:
                pass

        # Remove from store
        with self._lock:
            self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)
        self._persist()

        return {
            "ok": True,
            "deleted_files": deleted_files,
            "kept_exports": kept_exports,
        }

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            ids = list(reversed(self._order))
            return [self._jobs[i] for i in ids if i in self._jobs]

    def events(self, job_id: str) -> "queue.Queue[dict[str, Any]]":
        job = self.get(job_id)
        return job._events if job else queue.Queue()
