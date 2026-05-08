from __future__ import annotations

import os
from pathlib import Path

from app.core.config import AppConfig
from app.core.errors import DependencyMissingError, DiarizationError
from app.core.models import BackendSelection, SpeakerTurn


def pyannote_readiness() -> str:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return "token missing"
    try:
        import pyannote.audio  # noqa: F401
    except Exception as exc:
        return f"missing: {exc}"
    return "ready (token present)"


def diarize_audio(audio_path: Path, config: AppConfig, backend: BackendSelection) -> list[SpeakerTurn]:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise DependencyMissingError(
            "Diarization requires HUGGINGFACE_HUB_TOKEN and accepted access terms."
        )

    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise DependencyMissingError(f"pyannote.audio is unavailable: {exc}") from exc

    try:
        pipeline = Pipeline.from_pretrained(config.diarization.model, token=token)
        if backend.diarization_device == "cuda":
            pipeline.to(torch.device("cuda"))

        kwargs = {}
        mode = config.diarization.speaker_count_mode
        if mode == "fixed" and config.diarization.fixed_speakers > 0:
            kwargs["num_speakers"] = config.diarization.fixed_speakers
        elif mode == "minmax":
            if config.diarization.min_speakers > 0:
                kwargs["min_speakers"] = config.diarization.min_speakers
            if config.diarization.max_speakers > 0:
                kwargs["max_speakers"] = config.diarization.max_speakers

        output = pipeline(str(audio_path), **kwargs)
        diarization = getattr(output, "exclusive_speaker_diarization", None) or output.speaker_diarization
        return [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=str(speaker))
            for turn, speaker in diarization
        ]
    except Exception as exc:
        raise DiarizationError(str(exc)) from exc
