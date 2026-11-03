"""Job registry: creation, progress tracking and a live event feed."""
from __future__ import annotations

import json
import queue
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
    status: str = "queued"          # queued | running | done | error | cancelled | awaiting_*_review
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
    _events: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _resume: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _seq: int = field(default=0, repr=False)
    _pipeline_state: dict[str, Any] = field(default_factory=dict, repr=False)

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


class JobStore:
    def __init__(self, max_workers: int = 2):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                             thread_name_prefix="podcaster")

    def create(self, request: PodcastRequest) -> Job:
        job_id = uuid.uuid4().hex[:12]
        workdir = config_mod.DATA_DIR / "jobs" / job_id
        job = Job(id=job_id, request=request, workdir=str(workdir))
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            if len(self._order) > 60:
                oldest = self._order.pop(0)
                self._jobs.pop(oldest, None)
        self._executor.submit(self._run, job)
        return job

    def _run(self, job: Job) -> None:
        job.started_at = time.time()
        job.set_status("running")
        job.log("info", "Job started")
        try:
            result = run_pipeline(
                job.request,
                Path(job.workdir),
                on_step=job.set_step,
                log=job.log,
                should_cancel=lambda: job.cancelled,
                job=job,
            )
            job.result = result
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
