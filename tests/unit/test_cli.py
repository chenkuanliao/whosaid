from __future__ import annotations

from datetime import datetime, timezone
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

runner = CliRunner()


def test_run_prints_summary_without_transcript_content(monkeypatch, tmp_path: Path) -> None:
    media_path = tmp_path / "sample.wav"
    media_path.write_bytes(b"")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    transcript_text = "secret transcript line"

    result = PipelineResult(
        created_at=datetime.now(timezone.utc),
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
        ],
        output_dir=output_dir,
    )

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
