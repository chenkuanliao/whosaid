class WhoSaidError(Exception):
    """Base application error."""


class DependencyMissingError(WhoSaidError):
    """A required binary or Python dependency is missing."""


class UnsupportedMediaError(WhoSaidError):
    """The provided media file could not be handled."""


class HardwareDetectionError(WhoSaidError):
    """Hardware detection failed."""


class PreprocessError(WhoSaidError):
    """Media preprocessing failed."""


class TranscriptionError(WhoSaidError):
    """Transcription failed."""


class DiarizationError(WhoSaidError):
    """Diarization failed."""


class AlignmentError(WhoSaidError):
    """Segment alignment failed."""


class ExportError(WhoSaidError):
    """Export generation failed."""
