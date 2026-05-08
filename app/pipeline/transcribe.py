from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig
from app.core.models import BackendSelection, TranscriptSegment
from app.services.whisper_service import transcribe_audio


def transcribe(audio_path: Path, config: AppConfig, backend: BackendSelection) -> list[TranscriptSegment]:
    return transcribe_audio(audio_path, config, backend)
