from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import DependencyMissingError


@dataclass
class FFMpegStatus:
    available: bool
    message: str


def check_ffmpeg() -> FFMpegStatus:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return FFMpegStatus(True, "ok")
    return FFMpegStatus(False, "ffmpeg and ffprobe are required on PATH")


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise DependencyMissingError(f"Required binary '{name}' was not found on PATH.")
    return path


def ffprobe_json(media_path: Path) -> dict:
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,sample_rate,channels",
        "-of",
        "json",
        str(media_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def ffmpeg_extract_wav(input_path: Path, output_path: Path) -> None:
    ffmpeg = _require_binary("ffmpeg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
