from __future__ import annotations

from app.core.models import AlignedSegment, SpeakerTurn, TranscriptSegment


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def align_segments(
    transcript_segments: list[TranscriptSegment],
    speaker_turns: list[SpeakerTurn],
) -> list[AlignedSegment]:
    if not transcript_segments:
        return []
    if not speaker_turns:
        return [
            AlignedSegment(start=s.start, end=s.end, speaker="Speaker 1", text=s.text.strip())
            for s in transcript_segments
        ]

    aligned: list[AlignedSegment] = []
    for segment in transcript_segments:
        best_speaker = "Speaker 1"
        best_overlap = -1.0
        for turn in speaker_turns:
            overlap = _overlap(segment.start, segment.end, turn.start, turn.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        text = segment.text.strip()
        if not text:
            continue
        if (
            aligned
            and aligned[-1].speaker == best_speaker
            and segment.start - aligned[-1].end <= 0.75
        ):
            aligned[-1].end = segment.end
            aligned[-1].text = f"{aligned[-1].text} {text}".strip()
        else:
            aligned.append(
                AlignedSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker=best_speaker,
                    text=text,
                )
            )
    return aligned
