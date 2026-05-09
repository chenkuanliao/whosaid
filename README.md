# WhoSaid

Local-first CLI for turning audio or video into speaker-labeled transcripts using `faster-whisper` and `pyannote.audio`.

## Status

This repo contains an MVP codebase with:

- a CLI-first pipeline
- media inspection and FFmpeg preprocessing
- backend selection for CUDA, MPS, and CPU
- speaker-labeled alignment and export

The intended primary target is Linux with NVIDIA GPUs. Apple Silicon support is included as a graceful-degradation path.

## Environment

Use your `mainenv` shell alias locally before installing or running:

```bash
a
pip install -e .[dev]
whosaid doctor
whosaid run /path/to/media.mp4
```

## External requirements

- `ffmpeg`
- Hugging Face access token in `HUGGINGFACE_HUB_TOKEN` for diarization
- a compatible `torch` install for your hardware target

## CLI

```bash
whosaid doctor
whosaid inspect /path/to/file.mp4
whosaid run /path/to/file.mp4 --exports txt --exports json --exports srt
whosaid link-speakers
whosaid link-speakers /path/to/outputs
whosaid config init
whosaid config show
```

After `run`, use `link-speakers` to replace diarization ids like `SPEAKER_00` with names.
Press Enter at a prompt to skip that speaker and keep the original id.
