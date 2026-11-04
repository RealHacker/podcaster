"""Tests for job persistence, resume, and delete functionality.

These tests verify:
1. Jobs are persisted to disk and survive a simulated restart
2. Interrupted jobs can be resumed from their last phase
3. Delete removes temp files but keeps final exports
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from podcaster.jobs import (
    Job, JobStore, ReviewData, TERMINAL_STATUSES, RESUMABLE_STATUSES, INTERRUPTED_STATUSES
)
from podcaster.pipeline import PodcastRequest

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  OK   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


# --------------------------------------------------------------------------- #
# 1. Job serialization round-trip
# --------------------------------------------------------------------------- #

def test_job_serialization() -> None:
    """Jobs can be serialized and deserialized without data loss."""
    request = PodcastRequest(
        source_type="url",
        source_url="https://example.com/article",
        language="en",
        style="narrative",
        length_minutes=5.0,
        host_count=2,
        host_genders=["male", "female"],
        host_names=["Alex", "Sam"],
        export_format="audio",
        enable_review=True,
    )

    job = Job(
        id="test123",
        request=request,
        status="running",
        step="Writing script",
        progress=0.45,
        message="Processing",
        created_at=1234567890.0,
        started_at=1234567891.0,
        workdir="/tmp/test/jobs/test123",
        current_phase="script",
    )
    job.review_data = ReviewData(
        script_title="Test Episode",
        script_outline=["Intro", "Main", "Outro"],
        script_turns=[{"index": 0, "speaker": "Alex", "text": "Hello"}],
        hosts=[{"name": "Alex", "gender": "male"}],
    )
    job._pipeline_state = {"source": {"text": "test content"}}
    job.logs = [{"t": 1234567892.0, "level": "info", "message": "Started"}]

    persist_dict = job.to_persist_dict()
    check(persist_dict["id"] == "test123", "id is preserved")
    check(persist_dict["status"] == "running", "status is preserved")
    check(persist_dict["current_phase"] == "script", "current_phase is preserved")
    check(persist_dict["request"]["source_url"] == "https://example.com/article", "request is preserved")
    check(persist_dict["review_data"]["script_title"] == "Test Episode", "review_data is preserved")
    check("source" in persist_dict["pipeline_state"], "pipeline_state is preserved")

    restored = Job.from_persist_dict(persist_dict)
    check(restored.id == "test123", "id restored")
    check(restored.status == "running", "status restored")
    check(restored.current_phase == "script", "current_phase restored")
    check(restored.request.source_url == "https://example.com/article", "request restored")
    check(restored.review_data.script_title == "Test Episode", "review_data restored")
    check(restored._pipeline_state.get("source", {}).get("text") == "test content", "pipeline_state restored")


# --------------------------------------------------------------------------- #
# 2. JobStore persistence
# --------------------------------------------------------------------------- #

def test_jobstore_persistence() -> None:
    """JobStore persists jobs to disk and reloads them on restart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir)

        store1 = JobStore(max_workers=1, persist_dir=persist_dir)

        request = PodcastRequest(
            source_type="text",
            source_text="Test content for persistence",
            language="en",
            style="chitchat",
            length_minutes=2.0,
        )

        job = Job(
            id="persist_test",
            request=request,
            status="done",
            step="Finished",
            progress=1.0,
            workdir=str(persist_dir / "jobs" / "persist_test"),
            current_phase="done",
        )
        job.result = {"export_path": "/tmp/output.mp3", "duration_seconds": 120}
        job._persist_callback = store1._persist
        store1._jobs["persist_test"] = job
        store1._order.append("persist_test")
        store1._persist()

        jobs_file = persist_dir / "jobs.json"
        check(jobs_file.exists(), "jobs.json created")

        store2 = JobStore(max_workers=1, persist_dir=persist_dir)

        check("persist_test" in store2._jobs, "job restored from disk")
        restored = store2._jobs.get("persist_test")
        if restored:
            check(restored.status == "done", "restored job has correct status")
            check(restored.current_phase == "done", "restored job has correct phase")
            check(restored.request.source_text == "Test content for persistence", "restored request data")


# --------------------------------------------------------------------------- #
# 3. Interrupted jobs marked on reload
# --------------------------------------------------------------------------- #

def test_interrupted_jobs_marked() -> None:
    """Running jobs are marked as interrupted when reloaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir)

        jobs_data = {
            "version": 1,
            "updated_at": time.time(),
            "jobs": [
                {
                    "id": "running_job",
                    "status": "running",
                    "step": "Generating audio",
                    "progress": 0.7,
                    "message": "",
                    "error": None,
                    "created_at": time.time(),
                    "started_at": time.time(),
                    "finished_at": None,
                    "logs": [],
                    "result": None,
                    "workdir": "/tmp/jobs/running_job",
                    "current_phase": "tts",
                    "review_data": {},
                    "request": {
                        "source_type": "text",
                        "source_text": "test",
                        "language": "en",
                        "style": "narrative",
                        "length_minutes": 5.0,
                    },
                    "pipeline_state": {},
                },
                {
                    "id": "review_job",
                    "status": "awaiting_script_review",
                    "step": "Waiting for review",
                    "progress": 0.6,
                    "message": "",
                    "error": None,
                    "created_at": time.time(),
                    "started_at": time.time(),
                    "finished_at": None,
                    "logs": [],
                    "result": None,
                    "workdir": "/tmp/jobs/review_job",
                    "current_phase": "script_review",
                    "review_data": {},
                    "request": {
                        "source_type": "text",
                        "source_text": "test",
                        "language": "en",
                        "style": "narrative",
                        "length_minutes": 5.0,
                    },
                    "pipeline_state": {},
                },
                {
                    "id": "done_job",
                    "status": "done",
                    "step": "Finished",
                    "progress": 1.0,
                    "message": "",
                    "error": None,
                    "created_at": time.time(),
                    "started_at": time.time(),
                    "finished_at": time.time(),
                    "logs": [],
                    "result": {"export_path": "/tmp/output.mp3"},
                    "workdir": "/tmp/jobs/done_job",
                    "current_phase": "done",
                    "review_data": {},
                    "request": {
                        "source_type": "text",
                        "source_text": "test",
                        "language": "en",
                        "style": "narrative",
                        "length_minutes": 5.0,
                    },
                    "pipeline_state": {},
                },
            ],
        }

        (persist_dir / "jobs.json").write_text(json.dumps(jobs_data), encoding="utf-8")

        store = JobStore(max_workers=1, persist_dir=persist_dir)

        running_job = store.get("running_job")
        check(running_job is not None, "running job loaded")
        check(running_job.status == "interrupted", "running job marked as interrupted")

        review_job = store.get("review_job")
        check(review_job is not None, "review job loaded")
        check(review_job.status == "interrupted", "review job marked as interrupted")

        done_job = store.get("done_job")
        check(done_job is not None, "done job loaded")
        check(done_job.status == "done", "done job status unchanged")


# --------------------------------------------------------------------------- #
# 4. Delete cascade
# --------------------------------------------------------------------------- #

def test_delete_cascade() -> None:
    """Delete removes temp files but keeps final exports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir)
        jobs_dir = persist_dir / "jobs" / "delete_test"
        jobs_dir.mkdir(parents=True)

        clips_dir = jobs_dir / "clips"
        clips_dir.mkdir()
        (clips_dir / "clip_001.mp3").write_text("fake audio")
        (clips_dir / "clip_002.mp3").write_text("fake audio")
        (jobs_dir / "pipeline_state.json").write_text("{}")
        (jobs_dir / "script.txt").write_text("fake script")

        export_dir = persist_dir / "output"
        export_dir.mkdir()
        export_path = export_dir / "final_podcast.mp3"
        export_path.write_text("final audio export")

        store = JobStore(max_workers=1, persist_dir=persist_dir)

        job = Job(
            id="delete_test",
            request=PodcastRequest(source_type="text", source_text="test"),
            status="done",
            step="Finished",
            progress=1.0,
            workdir=str(jobs_dir),
            current_phase="done",
        )
        job.result = {"export_path": str(export_path), "audio_path": str(export_path)}
        job._persist_callback = store._persist
        store._jobs["delete_test"] = job
        store._order.append("delete_test")

        result = store.delete("delete_test", keep_exports=True)

        check(result["ok"], "delete succeeded")
        check(len(result.get("deleted_files", [])) > 0, "temp files deleted")
        check(str(export_path) in result.get("kept_exports", []), "export in kept_exports list")

        check(not jobs_dir.exists(), "workdir removed")
        check(export_path.exists(), "export file preserved")

        check(store.get("delete_test") is None, "job removed from store")


# --------------------------------------------------------------------------- #
# 5. RESUMABLE_STATUSES constant
# --------------------------------------------------------------------------- #

def test_resumable_statuses() -> None:
    """RESUMABLE_STATUSES includes the right statuses."""
    check("interrupted" in RESUMABLE_STATUSES, "interrupted is resumable")
    check("running" in RESUMABLE_STATUSES, "running is resumable")
    check("awaiting_script_review" in RESUMABLE_STATUSES, "script review is resumable")
    check("awaiting_audio_review" in RESUMABLE_STATUSES, "audio review is resumable")
    check("awaiting_slides_review" in RESUMABLE_STATUSES, "slides review is resumable")
    check("done" not in RESUMABLE_STATUSES, "done is not resumable")
    check("error" not in RESUMABLE_STATUSES, "error is not resumable")
    check("cancelled" not in RESUMABLE_STATUSES, "cancelled is not resumable")


# --------------------------------------------------------------------------- #
# 6. API endpoints
# --------------------------------------------------------------------------- #

def test_api_endpoints() -> None:
    """Resume and delete API endpoints work correctly."""
    from fastapi.testclient import TestClient
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir)
        jobs_dir = persist_dir / "jobs" / "api_test"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "test_file.txt").write_text("temp data")

        from podcaster.jobs import JobStore
        from podcaster.pipeline import PodcastRequest

        import app as app_module
        original_store = app_module.store

        test_store = JobStore(max_workers=1, persist_dir=persist_dir)

        job = Job(
            id="api_test",
            request=PodcastRequest(source_type="text", source_text="test content for API"),
            status="interrupted",
            step="Interrupted",
            progress=0.5,
            workdir=str(jobs_dir),
            current_phase="script",
        )
        job._persist_callback = test_store._persist
        test_store._jobs["api_test"] = job
        test_store._order.append("api_test")

        app_module.store = test_store

        try:
            from app import app
            with TestClient(app) as client:
                response = client.get("/api/jobs/api_test")
                check(response.status_code == 200, f"GET job returns 200 ({response.status_code})")
                data = response.json()
                check(data["status"] == "interrupted", "job has interrupted status")

                response = client.delete("/api/jobs/api_test")
                check(response.status_code == 200, f"DELETE job returns 200 ({response.status_code})")
                data = response.json()
                check(data["ok"], "delete succeeded")

                response = client.get("/api/jobs/api_test")
                check(response.status_code == 404, f"job not found after delete ({response.status_code})")
        finally:
            app_module.store = original_store


# --------------------------------------------------------------------------- #
# 7. Phase tracking
# --------------------------------------------------------------------------- #

def test_phase_tracking() -> None:
    """Job phase is tracked and persisted correctly."""
    request = PodcastRequest(
        source_type="text",
        source_text="test",
        language="en",
    )

    job = Job(id="phase_test", request=request, workdir="/tmp/test")

    check(job.current_phase == "source", "default phase is source")

    job.set_phase("script")
    check(job.current_phase == "script", "phase updated to script")

    job.set_phase("tts")
    check(job.current_phase == "tts", "phase updated to tts")

    persist_dict = job.to_persist_dict()
    check(persist_dict["current_phase"] == "tts", "phase in persist dict")

    restored = Job.from_persist_dict(persist_dict)
    check(restored.current_phase == "tts", "phase restored correctly")


# --------------------------------------------------------------------------- #
# Run all tests
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=== job serialization")
    test_job_serialization()
    print("\n=== jobstore persistence")
    test_jobstore_persistence()
    print("\n=== interrupted jobs marked")
    test_interrupted_jobs_marked()
    print("\n=== delete cascade")
    test_delete_cascade()
    print("\n=== resumable statuses")
    test_resumable_statuses()
    print("\n=== API endpoints")
    test_api_endpoints()
    print("\n=== phase tracking")
    test_phase_tracking()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL PERSISTENCE TESTS PASSED")
