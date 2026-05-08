from __future__ import annotations

from pathlib import Path

from app.core.errors import PreprocessError
from app.core.config import AppConfig
from app.core.models import InputMedia
from app.services.ffmpeg_service import ffmpeg_extract_wav


def preprocess_media(media: InputMedia, job_dir: Path, config: AppConfig) -> Path:
    output = job_dir / "normalized.wav"
    try:
        ffmpeg_extract_wav(media.source_path, output)
    except Exception as exc:
        raise PreprocessError(str(exc)) from exc
    return output
