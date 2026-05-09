from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from app.cli import cli
from app.core.models import (
    AlignedSegment,
    BackendSelection,
    ExportArtifact,
    InputMedia,
    MediaType,
    PipelineResult,
    SpeakerTurn,
    TranscriptSegment,
)
from app.pipeline.export import export_json, export_srt, export_txt

runner = CliRunner()


def _sample_result(media_path: Path, output_dir: Path, transcript_text: str) -> PipelineResult:
    return PipelineResult(
        created_at=datetime.now(UTC),
        input_media=InputMedia(
            source_path=media_path,
            media_type=MediaType.AUDIO,
            duration_seconds=12.34,
            sample_rate=16_000,
            channels=1,
            codec="pcm_s16le",
            needs_ffmpeg=False,
            is_long=False,
        ),
        backend=BackendSelection(
            mode="cpu",
            compute_type="int8",
            transcription_device="cpu",
            diarization_device="cpu",
            reason="test",
        ),
        transcript_segments=[
            TranscriptSegment(start=0.0, end=1.0, text=transcript_text, language="en"),
        ],
        speaker_turns=[
            SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
        ],
        aligned_segments=[
            AlignedSegment(start=0.0, end=1.0, speaker="SPEAKER_00", text=transcript_text),
        ],
        artifacts=[
            ExportArtifact(format="txt", path=output_dir / "sample.transcript.txt"),
            ExportArtifact(format="json", path=output_dir / "sample.transcript.json"),
            ExportArtifact(format="srt", path=output_dir / "sample.transcript.srt"),
        ],
        output_dir=output_dir,
    )


def test_run_prints_summary_without_transcript_content(monkeypatch, tmp_path: Path) -> None:
    media_path = tmp_path / "sample.wav"
    media_path.write_bytes(b"")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    transcript_text = "secret transcript line"

    result = _sample_result(media_path, output_dir, transcript_text)

    config = SimpleNamespace(
        transcription=SimpleNamespace(model="base", language="en"),
        diarization=SimpleNamespace(enabled=True),
        export=SimpleNamespace(formats=["txt", "json"]),
    )

    monkeypatch.setattr("app.cli.load_config", lambda: config)
    monkeypatch.setattr("app.cli.run_job", lambda *args, **kwargs: result)

    completed = runner.invoke(cli, ["run", str(media_path)])

    assert completed.exit_code == 0
    assert "WhoSaid Run Summary" in completed.stdout
    assert str(output_dir) in completed.stdout
    assert "sample.transcript.txt" in completed.stdout
    assert "sample.transcript.json" in completed.stdout
    assert transcript_text not in completed.stdout


def test_link_speakers_rewrites_existing_exports(tmp_path: Path) -> None:
    media_path = tmp_path / "sample.wav"
    media_path.write_bytes(b"")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result = _sample_result(media_path, output_dir, "hello there")

    export_txt(result.aligned_segments, output_dir / "sample.transcript.txt")
    export_srt(result.aligned_segments, output_dir / "sample.transcript.srt")
    export_json(result, output_dir / "sample.transcript.json")

    completed = runner.invoke(cli, ["link-speakers", str(output_dir)], input="Alice\n")

    assert completed.exit_code == 0
    assert "Updated speaker names" in completed.stdout
    assert "Alice: hello there" in (output_dir / "sample.transcript.txt").read_text()
    assert "Alice: hello there" in (output_dir / "sample.transcript.srt").read_text()

    updated = PipelineResult.model_validate_json(
        (output_dir / "sample.transcript.json").read_text()
    )
    assert updated.speaker_turns[0].speaker == "Alice"
    assert updated.aligned_segments[0].speaker == "Alice"


def test_link_speakers_skip_leaves_outputs_unchanged(tmp_path: Path) -> None:
    media_path = tmp_path / "sample.wav"
    media_path.write_bytes(b"")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result = _sample_result(media_path, output_dir, "hello there")
    export_json(result, output_dir / "sample.transcript.json")

    completed = runner.invoke(cli, ["link-speakers", str(output_dir)], input="\n")

    assert completed.exit_code == 0
    assert "Outputs were left unchanged" in completed.stdout
    updated = PipelineResult.model_validate_json(
        (output_dir / "sample.transcript.json").read_text()
    )
    assert updated.aligned_segments[0].speaker == "SPEAKER_00"


def test_link_speakers_fails_safely_without_previous_output(monkeypatch) -> None:
    monkeypatch.setattr("app.cli.load_recent_jobs", lambda: [])

    completed = runner.invoke(cli, ["link-speakers"])

    assert completed.exit_code == 1
    assert "No previous run output found" in completed.stdout
