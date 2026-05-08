from app.pipeline.export import _fmt_timestamp


def test_fmt_timestamp_plain() -> None:
    assert _fmt_timestamp(65.432) == "00:01:05.432"


def test_fmt_timestamp_srt() -> None:
    assert _fmt_timestamp(65.432, srt=True) == "00:01:05,432"
