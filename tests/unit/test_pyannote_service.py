import wave

import torch

from app.services.pyannote_service import _load_wav_for_pyannote


def test_load_wav_for_pyannote_returns_mono_waveform_dict(tmp_path) -> None:
    audio_path = tmp_path / "audio.wav"
    samples = [0, 16384, -16384, 32767]

    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))

    audio = _load_wav_for_pyannote(audio_path, torch)

    assert audio["sample_rate"] == 16000
    assert audio["waveform"].shape == (1, 4)
    assert torch.allclose(
        audio["waveform"],
        torch.tensor([[0.0, 0.5, -0.5, 32767 / 32768]], dtype=torch.float32),
    )
