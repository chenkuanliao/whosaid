from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig
from app.core.errors import UnsupportedMediaError
from app.core.models import InputMedia, MediaType
from app.services.ffmpeg_service import ffprobe_json


def inspect_media(media_path: Path, config: AppConfig | None = None) -> InputMedia:
    if not media_path.exists():
        raise FileNotFoundError(media_path)

    probe = ffprobe_json(media_path)
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    media_type = MediaType.AUDIO
    sample_rate = None
    channels = None
    codec = None

    for stream in streams:
        if stream.get("codec_type") == "video":
            media_type = MediaType.VIDEO
        if stream.get("codec_type") == "audio" and sample_rate is None:
            sample_rate = int(stream.get("sample_rate", 0) or 0) or None
            channels = int(stream.get("channels", 0) or 0) or None
            codec = stream.get("codec_name")

    try:
        duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError) as exc:
        raise UnsupportedMediaError("Could not determine media duration.") from exc

    if duration <= 0:
        raise UnsupportedMediaError("Unsupported media or empty duration.")

    return InputMedia(
        source_path=media_path,
        media_type=media_type,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        codec=codec,
        is_long=duration >= 90 * 60,
    )
