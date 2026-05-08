from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig
from app.core.errors import DependencyMissingError, TranscriptionError
from app.core.models import BackendSelection, TranscriptSegment


def whisper_readiness() -> str:
    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:
        return f"missing: {exc}"
    return "ready"


def _default_model(config: AppConfig, backend: BackendSelection) -> str:
    if config.transcription.model != "auto":
        return config.transcription.model
    if backend.mode == "cuda":
        return "medium"
    if backend.mode == "mps":
        return "small"
    return "small"


def transcribe_audio(
    audio_path: Path,
    config: AppConfig,
    backend: BackendSelection,
) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise DependencyMissingError(f"faster-whisper is unavailable: {exc}") from exc

    model_name = _default_model(config, backend)
    language = None if config.transcription.language == "auto" else config.transcription.language

    try:
        model = WhisperModel(
            model_name,
            device=backend.transcription_device,
            compute_type=backend.compute_type,
        )
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=config.transcription.beam_size,
            language=language,
            vad_filter=config.transcription.vad_filter,
        )
        detected_language = getattr(info, "language", None)
        output = [
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                avg_logprob=getattr(segment, "avg_logprob", None),
                language=detected_language,
            )
            for segment in segments
        ]
    except Exception as exc:
        raise TranscriptionError(str(exc)) from exc

    return output
