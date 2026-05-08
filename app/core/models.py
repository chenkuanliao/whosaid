from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    INSPECT = "inspect"
    PREPROCESS = "preprocess"
    HARDWARE = "hardware"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    ALIGN = "align"
    EXPORT = "export"
    DONE = "done"


class MediaType(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"


class InputMedia(BaseModel):
    source_path: Path
    media_type: MediaType
    duration_seconds: float
    sample_rate: int | None = None
    channels: int | None = None
    codec: str | None = None
    needs_ffmpeg: bool = True
    is_long: bool = False


class HardwareInfo(BaseModel):
    cuda_available: bool = False
    cuda_device_name: str | None = None
    mps_available: bool = False
    cpu_only: bool = True
    platform: str

    def summary(self) -> str:
        if self.cuda_available:
            return f"CUDA ({self.cuda_device_name or 'GPU'})"
        if self.mps_available:
            return "MPS"
        return f"CPU ({self.platform})"


class BackendSelection(BaseModel):
    mode: Literal["cuda", "mps", "cpu"]
    compute_type: str
    transcription_device: str
    diarization_device: str
    reason: str


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    language: str | None = None


class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str


class AlignedSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class ExportArtifact(BaseModel):
    format: str
    path: Path


class PipelineProgress(BaseModel):
    stage: PipelineStage
    message: str
    percent: float | None = None


class PipelineResult(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_media: InputMedia
    backend: BackendSelection
    transcript_segments: list[TranscriptSegment]
    speaker_turns: list[SpeakerTurn]
    aligned_segments: list[AlignedSegment]
    artifacts: list[ExportArtifact]
    output_dir: Path


class GeneralConfig(BaseModel):
    output_dir: str = ""
    temp_dir: str = ""
    logging_level: str = "INFO"
    keep_temp_files: bool = False
    recent_jobs_limit: int = 20


class HardwareConfig(BaseModel):
    mode: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    compute_type: str = "auto"


class TranscriptionConfig(BaseModel):
    model: str = "auto"
    language: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    chunking_enabled: bool = True
    chunk_length_minutes: int = 30


class DiarizationConfig(BaseModel):
    enabled: bool = True
    model: str = "pyannote/speaker-diarization-community-1"
    speaker_count_mode: Literal["auto", "fixed", "minmax"] = "auto"
    fixed_speakers: int = 0
    min_speakers: int = 0
    max_speakers: int = 0


class ExportConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["txt", "json", "srt"])


class AppConfigModel(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


class JobRecord(BaseModel):
    input_path: Path
    media_type: str
    duration_seconds: float
    backend: str
    output_dir: Path
    status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
