from __future__ import annotations

from pathlib import Path

import orjson

from app.core.models import JobRecord, PipelineResult
from app.core.paths import app_data_dir


def recent_jobs_path() -> Path:
    return app_data_dir() / "recent_jobs.json"


def load_recent_jobs() -> list[JobRecord]:
    path = recent_jobs_path()
    if not path.exists():
        return []
    return [JobRecord.model_validate(item) for item in orjson.loads(path.read_bytes())]


def save_recent_jobs(records: list[JobRecord]) -> None:
    path = recent_jobs_path()
    path.write_bytes(orjson.dumps([r.model_dump(mode="json") for r in records], option=orjson.OPT_INDENT_2))


def append_recent_job(result: PipelineResult) -> None:
    records = load_recent_jobs()
    records.insert(
        0,
        JobRecord(
            input_path=result.input_media.source_path,
            media_type=result.input_media.media_type.value,
            duration_seconds=result.input_media.duration_seconds,
            backend=result.backend.mode,
            output_dir=result.output_dir,
            status="success",
        ),
    )
    save_recent_jobs(records[:20])
