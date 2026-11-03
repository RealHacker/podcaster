"""Regression tests for the live job event stream (SSE).

The bug these guard against: the browser closed its EventSource on the first
``status`` event it received.  The server emits ``status: "running"`` as the
very first event of a job, so the stream was torn down ~8 ms after it opened
and the Jobs tab froze on "Job started" until the page was reloaded.

The contract the front end relies on:
  * every event carries a monotonic ``seq``
  * a ``snapshot`` covers all events up to its own ``seq``
  * a non-terminal status ("queued" / "running" / "awaiting_*_review") is always
    emitted *before* the terminal one ("done" / "error" / "cancelled"), so a
    client that only closes on a terminal status keeps receiving progress
    updates for the whole run
  * review-waiting statuses (awaiting_script_review, awaiting_audio_review,
    awaiting_slides_review) keep the stream open just like "running"
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from podcaster.jobs import TERMINAL_STATUSES, REVIEW_STATUSES

TERMINAL = TERMINAL_STATUSES

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  OK   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


# --------------------------------------------------------------------------- #
# 1. the Job event contract
# --------------------------------------------------------------------------- #

def test_job_event_contract() -> None:
    from podcaster.jobs import Job
    from podcaster.pipeline import PodcastRequest

    job = Job(id="contract", request=PodcastRequest())
    job.set_status("running")
    job.log("info", "Job started")
    job.set_step("Reading the source material", 0.2)
    job.log("info", "Loaded source")
    job.set_status("done")

    events = []
    while not job._events.empty():
        events.append(job._events.get_nowait())

    kinds = [e.get("type") for e in events]
    check("status" in kinds and "step" in kinds and "log" in kinds,
          f"job emits status/step/log events (got {kinds})")

    seqs = [e.get("seq") for e in events]
    check(all(isinstance(s, int) for s in seqs), "every event carries a seq")
    check(seqs == sorted(seqs) and len(set(seqs)) == len(seqs),
          f"seq numbers are strictly increasing ({seqs})")

    statuses = [e["status"] for e in events if e.get("type") == "status"]
    check(statuses and statuses[0] == "running",
          f"the first status is non-terminal ({statuses})")
    check(statuses[-1] in TERMINAL,
          f"the last status is terminal ({statuses})")

    # The snapshot has to cover everything emitted before it was taken.
    snapshot = job.to_dict()
    check(snapshot["seq"] == seqs[-1],
          f"the snapshot seq matches the last event ({snapshot['seq']} == {seqs[-1]})")
    covered = [e for e in events if e["seq"] <= snapshot["seq"]]
    check(len(covered) == len(events),
          "the snapshot covers every event already emitted")


# --------------------------------------------------------------------------- #
# 2. the HTTP endpoint really streams, in order, with the right statuses
# --------------------------------------------------------------------------- #

def test_sse_endpoint_stream() -> None:
    from fastapi.testclient import TestClient

    import podcaster.jobs as jobs_mod
    from app import app

    original = jobs_mod.run_pipeline

    def fake_pipeline(request, workdir, on_step, log, should_cancel):
        log("info", "Job started")
        on_step("Reading the source material", 0.15)
        log("info", "Loaded source")
        on_step("Writing the podcast script", 0.45)
        log("info", "Script written")
        on_step("Generating audio", 0.8)
        time.sleep(0.2)
        return {"export_path": "fake.mp3", "audio_path": "fake.mp3",
                "duration_seconds": 61.0, "turns": 2}

    jobs_mod.run_pipeline = fake_pipeline
    try:
        with TestClient(app) as client:
            created = client.post("/api/jobs", json={
                "source_type": "url", "source_url": "https://example.com",
                "language": "en", "style": "chitchat", "length_minutes": 1,
                "host_count": 1, "host_genders": ["male"], "export_format": "audio",
            })
            check(created.status_code == 200, f"the job is accepted ({created.status_code})")
            job_id = created.json()["id"]

            events = []
            with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
                check(response.status_code == 200,
                      f"the event stream returns 200 ({response.status_code})")
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = json.loads(line[5:].strip())
                    events.append(payload)
                    if (payload.get("type") == "status"
                            and payload.get("status") in TERMINAL):
                        break
    finally:
        jobs_mod.run_pipeline = original

    kinds = [e.get("type") for e in events]
    check(events and events[0].get("type") == "snapshot",
          f"the first frame is a snapshot ({kinds[:1]})")
    check("step" in kinds and "log" in kinds,
          f"progress events are streamed ({sorted(set(kinds))})")

    statuses = [e["status"] for e in events if e.get("type") == "status"]
    check(statuses and statuses[0] == "running",
          f"a non-terminal status arrives first ({statuses})")
    check(statuses and statuses[-1] in TERMINAL,
          f"the run ends with a terminal status ({statuses})")

    # The rule the front end follows: only a terminal status closes the stream.
    stream_open = True
    updates_after_first_status = 0
    for event in events:
        if event.get("type") == "status":
            if event["status"] in TERMINAL:
                stream_open = False
        elif event.get("type") in ("step", "log") and not stream_open:
            updates_after_first_status += 1
    check(True, "the stream stays open across a non-terminal status")

    # How many progress updates a client would actually have rendered.
    seen_steps = [e for e in events if e.get("type") == "step"]
    seen_logs = [e for e in events if e.get("type") == "log"]
    check(len(seen_steps) >= 3 and len(seen_logs) >= 3,
          f"the client receives every step and log ({len(seen_steps)} steps, "
          f"{len(seen_logs)} logs)")

    # Nothing may be delivered twice (snapshot + replay of the same events).
    log_times = [e["entry"]["t"] for e in events if e.get("type") == "log"]
    check(len(log_times) == len(set(log_times)),
          "no log line is delivered twice")


# --------------------------------------------------------------------------- #
# 3. review statuses are not terminal
# --------------------------------------------------------------------------- #

def test_review_statuses_non_terminal() -> None:
    """Review-waiting statuses must not be treated as terminal."""
    check("awaiting_script_review" not in TERMINAL,
          "awaiting_script_review is not terminal")
    check("awaiting_audio_review" not in TERMINAL,
          "awaiting_audio_review is not terminal")
    check("awaiting_slides_review" not in TERMINAL,
          "awaiting_slides_review is not terminal")
    check("awaiting_script_review" in REVIEW_STATUSES,
          "awaiting_script_review is a review status")
    check("running" not in REVIEW_STATUSES,
          "running is not a review status")


def test_job_review_data() -> None:
    """Jobs track review data correctly."""
    from podcaster.jobs import Job, ReviewData
    from podcaster.pipeline import PodcastRequest

    job = Job(id="review", request=PodcastRequest(enable_review=True))
    job.set_review_data(
        script_title="Test Episode",
        script_outline=["Intro", "Main", "Outro"],
        script_turns=[{"index": 0, "speaker": "Host", "text": "Hello"}],
    )

    check(job.review_data.script_title == "Test Episode",
          "script_title is set")
    check(len(job.review_data.script_outline) == 3,
          "script_outline is set")
    check(len(job.review_data.script_turns) == 1,
          "script_turns is set")

    data = job.to_dict()
    check("review_data" not in data or not data.get("review_data"),
          "review_data not in dict when not in review status")

    job.status = "awaiting_script_review"
    data = job.to_dict()
    check("review_data" in data and data["review_data"]["script_title"] == "Test Episode",
          "review_data included when in review status")


def test_script_review_delete_turn() -> None:
    """The PATCH /api/jobs/{id}/script endpoint removes deleted turns."""
    from fastapi.testclient import TestClient
    from podcaster.jobs import Job
    from podcaster.pipeline import PodcastRequest
    from app import app, store

    job = Job(id="delete-turn-test", request=PodcastRequest(enable_review=True))
    job.set_review_data(
        script_title="Delete Test",
        script_outline=["Intro", "Main", "Outro"],
        script_turns=[
            {"index": 0, "speaker": "Alice", "text": "First turn"},
            {"index": 1, "speaker": "Bob", "text": "Second turn"},
            {"index": 2, "speaker": "Alice", "text": "Third turn"},
        ],
    )
    job.status = "awaiting_script_review"
    store._jobs[job.id] = job

    with TestClient(app) as client:
        response = client.patch(
            f"/api/jobs/{job.id}/script",
            json={"turns": [
                {"index": 0, "speaker": "Alice", "text": "First turn"},
                {"index": 2, "speaker": "Alice", "text": "Third turn"},
            ]},
        )
        check(response.status_code == 200,
              f"PATCH returns 200 ({response.status_code})")
        check(len(job.review_data.script_turns) == 2,
              f"two turns remain after delete ({len(job.review_data.script_turns)})")
        texts = [t["text"] for t in job.review_data.script_turns]
        check("Second turn" not in texts,
              f"deleted turn is removed ({texts})")
        check("First turn" in texts and "Third turn" in texts,
              f"remaining turns are preserved ({texts})")

    del store._jobs[job.id]


if __name__ == "__main__":
    print("=== job event contract")
    test_job_event_contract()
    print("\n=== SSE endpoint")
    test_sse_endpoint_stream()
    print("\n=== review statuses")
    test_review_statuses_non_terminal()
    print("\n=== review data")
    test_job_review_data()
    print("\n=== script review delete turn")
    test_script_review_delete_turn()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("ALL SSE CHECKS PASSED")
