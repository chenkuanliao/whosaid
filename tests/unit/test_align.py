from app.core.models import SpeakerTurn, TranscriptSegment
from app.pipeline.align import align_segments


def test_align_segments_uses_max_overlap() -> None:
    transcript = [
        TranscriptSegment(start=0.0, end=2.0, text="hello"),
        TranscriptSegment(start=2.0, end=4.0, text="world"),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=1.5, speaker="Speaker 1"),
        SpeakerTurn(start=1.5, end=4.0, speaker="Speaker 2"),
    ]

    aligned = align_segments(transcript, turns)

    assert aligned[0].speaker == "Speaker 1"
    assert aligned[1].speaker == "Speaker 2"
