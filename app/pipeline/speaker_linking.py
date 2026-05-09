from __future__ import annotations

from pathlib import Path

import orjson

from app.core.models import PipelineResult
from app.pipeline.export import export_json, export_srt, export_txt


def find_result_json(output_path: Path) -> Path | None:
    path = output_path.expanduser()
    if path.is_file():
        return path if path.name.endswith(".transcript.json") else None
    if not path.is_dir():
        return None

    candidates = sorted(
        path.glob("*.transcript.json"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_result(result_path: Path) -> PipelineResult:
    return PipelineResult.model_validate(orjson.loads(result_path.read_bytes()))


def speaker_ids(result: PipelineResult) -> list[str]:
    speakers = {segment.speaker for segment in result.aligned_segments}
    if not speakers:
        speakers.update(turn.speaker for turn in result.speaker_turns)
    return sorted(speakers)


def apply_speaker_names(result: PipelineResult, names: dict[str, str]) -> None:
    replacements = {speaker: name.strip() for speaker, name in names.items() if name.strip()}
    if not replacements:
        return

    for turn in result.speaker_turns:
        turn.speaker = replacements.get(turn.speaker, turn.speaker)
    for segment in result.aligned_segments:
        segment.speaker = replacements.get(segment.speaker, segment.speaker)


def exported_formats(result: PipelineResult, result_path: Path) -> list[str]:
    formats = []
    for artifact in result.artifacts:
        if artifact.format in {"txt", "json", "srt"} and artifact.path.exists():
            formats.append(artifact.format)

    if not formats:
        stem = result_path.name.removesuffix(".transcript.json")
        for fmt in ("txt", "json", "srt"):
            if (result_path.parent / f"{stem}.transcript.{fmt}").exists():
                formats.append(fmt)

    return list(dict.fromkeys(formats or ["json"]))


def rewrite_exports(result: PipelineResult, result_path: Path, formats: list[str]) -> list[Path]:
    stem = result_path.name.removesuffix(".transcript.json")
    written: list[Path] = []
    for fmt in formats:
        path = result_path.parent / f"{stem}.transcript.{fmt}"
        if fmt == "txt":
            export_txt(result.aligned_segments, path)
        elif fmt == "srt":
            export_srt(result.aligned_segments, path)
        elif fmt == "json":
            export_json(result, path)
        else:
            continue
        written.append(path)
    return written
