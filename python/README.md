# SP Python prototype

Read-only audio analysis and visualizer for **MP4, WebM, WAV, and MP3**.

## Setup

Python 3.10-3.12 is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py "path/to/music.mp4"
python main.py "path/to/music.webm"
python main.py "path/to/music.wav"
python main.py "path/to/music.mp3"
```

## Read-only guarantee

The program opens the selected media for reading. For MP4/WebM, FFmpeg decodes the audio stream to memory through stdout. It does not create a converted media file, modify metadata, overwrite, or delete the source.

## Current analysis status

- **Bass**: low-frequency energy estimate.
- **Perc**: onset / spectral-flux estimate; not a drum-only classifier.
- **Piano*** and **Vocal***: rough acoustic heuristics, not reliable instrument or voice recognition.

`*` marks estimates that need a dedicated ML model or source-separation stage for dependable results. Video is currently audio-analyzed only; video display is a later feature.
