from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig
from app.core.models import BackendSelection, SpeakerTurn
from app.services.pyannote_service import diarize_audio


def diarize(audio_path: Path, config: AppConfig, backend: BackendSelection) -> list[SpeakerTurn]:
    return diarize_audio(audio_path, config, backend)
