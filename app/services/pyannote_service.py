from __future__ import annotations

import os
import wave
from pathlib import Path

from app.core.config import AppConfig
from app.core.errors import DependencyMissingError, DiarizationError
from app.core.models import BackendSelection, SpeakerTurn


def pyannote_readiness(model_id: str = "pyannote/speaker-diarization-community-1") -> str:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return "token missing"
    try:
        import pyannote.audio  # noqa: F401
    except Exception as exc:
        return f"missing: {exc}"
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except Exception:
        return "ready (token present; access not verified)"

    try:
        HfApi().model_info(model_id, token=token)
    except GatedRepoError:
        return f"gated: request access to {model_id}"
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 401:
            return "token invalid or unauthorized"
        if status_code == 403:
            return f"gated: request access to {model_id}"
        return f"hub error: {exc}"
    except Exception as exc:
        return f"unable to verify access: {exc}"

    return "ready (access verified)"


def _load_wav_for_pyannote(audio_path: Path, torch):
    with wave.open(str(audio_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw_audio = wav.readframes(frames)

    if sample_width != 2:
        raise DiarizationError(
            f"Expected 16-bit PCM WAV for diarization, got {sample_width * 8}-bit audio."
        )

    waveform = torch.frombuffer(bytearray(raw_audio), dtype=torch.int16).to(torch.float32) / 32768.0
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(dim=1)

    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def _speaker_turns_from_annotation(annotation) -> list[SpeakerTurn]:
    if hasattr(annotation, "itertracks"):
        return [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]

    return [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=str(speaker))
        for turn, speaker in annotation
    ]


def diarize_audio(
    audio_path: Path,
    config: AppConfig,
    backend: BackendSelection,
) -> list[SpeakerTurn]:
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

        pyannote_audio = _load_wav_for_pyannote(audio_path, torch)
        output = pipeline(pyannote_audio, **kwargs)
        diarization = (
            getattr(output, "exclusive_speaker_diarization", None) or output.speaker_diarization
        )
        return _speaker_turns_from_annotation(diarization)
    except Exception as exc:
        raise DiarizationError(str(exc)) from exc
