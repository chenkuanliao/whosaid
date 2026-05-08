from __future__ import annotations

from pathlib import Path

import orjson

from app.core.errors import ExportError
from app.core.models import AlignedSegment, ExportArtifact, PipelineResult


def _fmt_timestamp(seconds: float, srt: bool = False) -> str:
    total_ms = int(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    sep = "," if srt else "."
    return f"{hours:02}:{minutes:02}:{secs:02}{sep}{ms:03}"


def export_txt(segments: list[AlignedSegment], output_path: Path) -> None:
    lines = [f"[{_fmt_timestamp(s.start)} - {_fmt_timestamp(s.end)}] {s.speaker}: {s.text}" for s in segments]
    output_path.write_text("\n".join(lines) + "\n")


def export_srt(segments: list[AlignedSegment], output_path: Path) -> None:
    chunks = []
    for idx, seg in enumerate(segments, start=1):
        chunks.append(str(idx))
        chunks.append(f"{_fmt_timestamp(seg.start, srt=True)} --> {_fmt_timestamp(seg.end, srt=True)}")
        chunks.append(f"{seg.speaker}: {seg.text}")
        chunks.append("")
    output_path.write_text("\n".join(chunks))


def export_json(result: PipelineResult, output_path: Path) -> None:
    output_path.write_bytes(orjson.dumps(result.model_dump(mode="json"), option=orjson.OPT_INDENT_2))


def export_all(result: PipelineResult, formats: list[str]) -> list[ExportArtifact]:
    artifacts: list[ExportArtifact] = []
    stem = result.input_media.source_path.stem
    for fmt in formats:
        path = result.output_dir / f"{stem}.transcript.{fmt}"
        try:
            if fmt == "txt":
                export_txt(result.aligned_segments, path)
            elif fmt == "srt":
                export_srt(result.aligned_segments, path)
            elif fmt == "json":
                export_json(result, path)
            else:
                continue
            artifacts.append(ExportArtifact(format=fmt, path=path))
        except Exception as exc:
            raise ExportError(f"Failed to export {fmt}: {exc}") from exc
    return artifacts
