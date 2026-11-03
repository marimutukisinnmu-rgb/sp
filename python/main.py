"""SP MP3 stem splitter.

The input MP3 is read only. Demucs writes its intermediate files into a
temporary directory inside the selected output directory; only the four
finished MP3 stems are kept.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


STEMS = {
    "bass": "Bass.mp3",
    "drums": "Perc.mp3",
    "other": "MIEP.mp3",
    "vocals": "Vocals.mp3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate an MP3 into Bass, Perc, MIEP, and Vocals files."
    )
    parser.add_argument("input", type=Path, help="Input MP3 file (read only)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the four separated MP3 files will be saved",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Processing device (default: cpu; use cuda only if supported)",
    )
    parser.add_argument(
        "--mp3-bitrate",
        type=int,
        choices=(128, 192, 256, 320),
        default=320,
        help="Output MP3 bitrate in kbps (default: 320)",
    )
    return parser.parse_args()


def separate_mp3(
    input_path: Path,
    output_dir: Path,
    device: str,
    mp3_bitrate: int,
) -> int:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_path.is_file():
        print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
        return 2
    if input_path.suffix.lower() != ".mp3":
        print("Error: input must be an MP3 file.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Never overwrite existing results.
    conflicts = [name for name in STEMS.values() if (output_dir / name).exists()]
    if conflicts:
        print(
            "Error: these output files already exist; nothing was overwritten:\n"
            + "\n".join(f"  {name}" for name in conflicts),
            file=sys.stderr,
        )
        return 2

    print(f"Input (read only): {input_path}")
    print(f"Output directory: {output_dir}")
    print("Starting Demucs. The first run may download its model.")

    try:
        with tempfile.TemporaryDirectory(
            prefix=".sp-demucs-", dir=str(output_dir)
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            demucs_output = temporary_path / "demucs"

            command = [
                sys.executable,
                "-m",
                "demucs",
                "-n",
                "htdemucs",
                "--mp3",
                "--mp3-bitrate",
                str(mp3_bitrate),
                "-d",
                device,
                "-o",
                str(demucs_output),
                str(input_path),
            ]

            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                print(
                    f"Error: Demucs exited with code {result.returncode}.",
                    file=sys.stderr,
                )
                print(
                    "Install dependencies with: "
                    "python -m pip install -r requirements.txt",
                    file=sys.stderr,
                )
                return result.returncode or 1

            model_directory = demucs_output / "htdemucs"
            song_directories = (
                [path for path in model_directory.iterdir() if path.is_dir()]
                if model_directory.is_dir()
                else []
            )
            if len(song_directories) != 1:
                print(
                    "Error: could not find the separated Demucs output folder.",
                    file=sys.stderr,
                )
                return 1

            separated_directory = song_directories[0]
            staged_directory = temporary_path / "ready"
            staged_directory.mkdir()

            for stem, output_name in STEMS.items():
                source_file = separated_directory / f"{stem}.mp3"
                if not source_file.is_file():
                    print(
                        f"Error: Demucs did not create {source_file.name}.",
                        file=sys.stderr,
                    )
                    return 1
                shutil.copyfile(source_file, staged_directory / output_name)

            # Publish the complete set only after all four files were prepared.
            for output_name in STEMS.values():
                destination = output_dir / output_name
                if destination.exists():
                    print(
                        f"Error: output appeared during processing: {destination}",
                        file=sys.stderr,
                    )
                    return 2
                os.replace(staged_directory / output_name, destination)

    except FileNotFoundError as exc:
        print(f"Error: could not start Demucs: {exc}", file=sys.stderr)
        print(
            "Install dependencies with: "
            "python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 1

    print("Completed:")
    for output_name in STEMS.values():
        print(f"  {output_dir / output_name}")
    print("The input MP3 was not modified.")
    return 0


def main() -> int:
    args = parse_args()
    return separate_mp3(
        args.input,
        args.output_dir,
        args.device,
        args.mp3_bitrate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
