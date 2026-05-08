from __future__ import annotations

import platform

from app.core.config import AppConfig
from app.core.models import BackendSelection, HardwareInfo


def detect_hardware() -> HardwareInfo:
    cuda_available = False
    cuda_device_name = None
    mps_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass

    return HardwareInfo(
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        mps_available=mps_available,
        cpu_only=not cuda_available and not mps_available,
        platform=platform.platform(),
    )


def resolve_backend(config: AppConfig, hardware: HardwareInfo) -> BackendSelection:
    mode = config.hardware.mode
    compute = config.hardware.compute_type
    if mode == "cuda":
        if not hardware.cuda_available:
            raise RuntimeError("CUDA was forced but no CUDA backend is available.")
        return BackendSelection(
            mode="cuda",
            compute_type="float16" if compute == "auto" else compute,
            transcription_device="cuda",
            diarization_device="cuda",
            reason="User forced CUDA mode.",
        )
    if mode == "mps":
        if not hardware.mps_available:
            raise RuntimeError("MPS was forced but no Apple MPS backend is available.")
        return BackendSelection(
            mode="mps",
            compute_type="float32" if compute == "auto" else compute,
            transcription_device="cpu",
            diarization_device="cpu",
            reason="User forced MPS mode; conservative CPU fallback retained for services.",
        )
    if mode == "cpu":
        return BackendSelection(
            mode="cpu",
            compute_type="int8" if compute == "auto" else compute,
            transcription_device="cpu",
            diarization_device="cpu",
            reason="User forced CPU mode.",
        )
    if hardware.cuda_available:
        return BackendSelection(
            mode="cuda",
            compute_type="float16" if compute == "auto" else compute,
            transcription_device="cuda",
            diarization_device="cuda",
            reason="CUDA detected; using NVIDIA-first path.",
        )
    if hardware.mps_available:
        return BackendSelection(
            mode="mps",
            compute_type="float32" if compute == "auto" else compute,
            transcription_device="cpu",
            diarization_device="cpu",
            reason="Apple MPS detected; using conservative transcription path.",
        )
    return BackendSelection(
        mode="cpu",
        compute_type="int8" if compute == "auto" else compute,
        transcription_device="cpu",
        diarization_device="cpu",
        reason="No accelerator detected; using CPU fallback.",
    )
