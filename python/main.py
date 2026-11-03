"""
SP - audio part visualizer (Python prototype)

Read-only input support:
  - Audio: WAV / MP3
  - Video: MP4 / WebM (audio is decoded to memory for analysis; source is untouched)

This first prototype estimates Bass and Percussion using signal features.
Piano and Vocal are heuristic estimates, not reliable source separation.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

AUDIO_EXTENSIONS = {".wav", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".webm"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def load_audio_read_only(path: Path, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    """Decode audio into memory only. Never writes to or modifies the input file."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix in VIDEO_EXTENSIONS:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "Video support requires imageio-ffmpeg. Install requirements.txt."
            ) from exc

        # ffmpeg reads the video and emits decoded PCM to stdout. No output file is made.
        import subprocess

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v", "error",
            "-i", str(path),
            "-vn",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", str(sample_rate),
            "pipe:1",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0 or not result.stdout:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not decode video audio: {message or 'no audio stream'}")
        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
        return audio, sample_rate

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return np.asarray(audio, dtype=np.float32), sr


def normalize(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(values)) if values.size else 0.0
    if peak <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / peak, 0.0, 1.0).astype(np.float32)


def analyze(audio: np.ndarray, sr: int, hop_length: int = 512) -> dict[str, np.ndarray]:
    if audio.size == 0:
        raise ValueError("The selected file contains no audio.")

    n_fft = 2048
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def band_energy(low: float, high: float) -> np.ndarray:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            return np.zeros(magnitude.shape[1], dtype=np.float32)
        return np.sqrt(np.mean(magnitude[mask] ** 2, axis=0))

    bass = normalize(band_energy(35, 180))
    low_mid = band_energy(180, 2000)
    high_mid = band_energy(2000, 8000)

    # Percussion proxy: positive spectral-flux changes, weighted toward upper frequencies.
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(magnitude + 1e-8), sr=sr, hop_length=hop_length)
    flux = librosa.util.fix_length(flux, size=magnitude.shape[1])
    perc = normalize(flux * (0.35 + normalize(high_mid)))

    # Lightweight heuristics only. These are not instrument classifiers.
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    harmonic = librosa.effects.harmonic(audio)
    harmonic_rms = librosa.feature.rms(y=harmonic, frame_length=n_fft, hop_length=hop_length)[0]
    harmonic_rms = librosa.util.fix_length(harmonic_rms, size=magnitude.shape[1])

    piano = normalize(harmonic_rms * (1.0 - np.clip(flatness, 0, 1)))
    vocal_likeness = np.exp(-((centroid - 2200.0) / 1800.0) ** 2) * (1.0 - np.clip(flatness, 0, 1))
    vocal = normalize(vocal_likeness * np.sqrt(low_mid + high_mid + 1e-8))

    return {
        "Bass": bass,
        "Perc": perc,
        "Piano*": piano,
        "Vocal*": vocal,
        "times": librosa.frames_to_time(
            np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length
        ),
    }


def play_visualizer(features: dict[str, np.ndarray], title: str) -> None:
    names = ["Bass", "Perc", "Piano*", "Vocal*"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.canvas.manager.set_window_title("SP - Audio Part Visualizer")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Estimated intensity")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.2)
    lines = {
        name: ax.plot([], [], label=name, linewidth=1.5)[0]
        for name in names
    }
    ax.legend(loc="upper right")
    times = features["times"]

    def update(_frame: int):
        for name, line in lines.items():
            line.set_data(times, features[name])
        ax.set_xlim(0, max(float(times[-1]), 1.0))
        return tuple(lines.values())

    FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only audio part visualizer prototype")
    parser.add_argument("file", type=Path, help="Input .mp4, .webm, .wav, or .mp3 file")
    args = parser.parse_args()

    path = args.file.expanduser().resolve()
    if not path.is_file():
        parser.error(f"File does not exist: {path}")

    try:
        audio, sr = load_audio_read_only(path)
        features = analyze(audio, sr)
        play_visualizer(features, path.name)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
