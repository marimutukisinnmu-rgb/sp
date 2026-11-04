import argparse
import subprocess
import sys

import numpy as np

SAMPLE_RATE = 11025
MAX_OFFSET_SECONDS = 2.0


def load_audio(path):
    """Decode an audio file to mono float PCM using FFmpeg."""
    command = [
        "ffmpeg",
        "-v", "error",
        "-i", path,
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "-"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg could not read the file:\n"
            f"{result.stderr.decode(errors='replace')}"
        )

    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()

    if len(audio) == 0:
        raise RuntimeError(f"Audio is empty: {path}")

    return audio


def find_best_alignment(reference, compared):
    """Search for the best alignment within +/- 2 seconds."""
    max_offset = int(SAMPLE_RATE * MAX_OFFSET_SECONDS)

    # Use a bounded segment for efficient alignment search.
    length = min(len(reference), len(compared))
    reference = reference[:length]
    compared = compared[:length]

    # Cross-correlate a downsampled preview to estimate the offset.
    stride = 16
    ref_preview = reference[::stride]
    cmp_preview = compared[::stride]
    preview_offset = max_offset // stride

    correlation = np.correlate(
        ref_preview,
        cmp_preview,
        mode="full"
    )
    center = len(cmp_preview) - 1
    low = max(0, center - preview_offset)
    high = min(len(correlation), center + preview_offset + 1)

    best_lag = int(np.argmax(correlation[low:high]) + low - center)
    best_lag *= stride

    if best_lag > 0:
        reference = reference[best_lag:]
        compared = compared[:len(reference)]
    elif best_lag < 0:
        compared = compared[-best_lag:]
        reference = reference[:len(compared)]

    length = min(len(reference), len(compared))
    return reference[:length], compared[:length], best_lag


def calculate_similarity(reference, compared):
    """Calculate normalized waveform correlation as a percentage."""
    reference = reference.astype(np.float64)
    compared = compared.astype(np.float64)

    reference -= np.mean(reference)
    compared -= np.mean(compared)

    ref_energy = np.sum(reference ** 2)
    cmp_energy = np.sum(compared ** 2)

    if ref_energy == 0 or cmp_energy == 0:
        return 0.0

    correlation = np.sum(reference * compared) / np.sqrt(
        ref_energy * cmp_energy
    )

    # Negative correlation is not counted as positive similarity.
    return float(np.clip(correlation, 0.0, 1.0) * 100.0)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare an original audio file with a reconstructed mix "
            "and print a normalized waveform similarity percentage."
        )
    )
    parser.add_argument("original", help="Original audio file")
    parser.add_argument("combined", help="Reconstructed audio file")

    args = parser.parse_args()

    try:
        print("Loading original audio...")
        original = load_audio(args.original)

        print("Loading reconstructed audio...")
        combined = load_audio(args.combined)

        print("Aligning audio...")
        original, combined, lag = find_best_alignment(
            original,
            combined
        )

        print("Calculating similarity...")
        similarity = calculate_similarity(original, combined)

        print()
        print("=" * 40)
        print(f"Similarity: {similarity:.2f}%")
        print(f"Alignment offset: {lag / SAMPLE_RATE:.4f} seconds")
        print(f"Compared duration: {len(original) / SAMPLE_RATE:.2f} seconds")
        print("=" * 40)

    except FileNotFoundError as error:
        print(f"File not found: {error}")
        sys.exit(1)
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
