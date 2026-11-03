# SP Python MP3 stem splitter

Split one MP3 file into four separate MP3 files with Demucs.

## Setup

Python 3.11-3.14 is recommended.

```powershell
python -m pip install -r requirements.txt
```

The first separation run may download the Demucs model.

## Run

```powershell
python main.py "F:\music\song.mp3" -o "F:\separated"
```

The output directory is specified with **`-o`** (required). The command creates it if it does not exist.

By default, processing uses the CPU. To request CUDA on a compatible NVIDIA setup:

```powershell
python main.py "F:\music\song.mp3" -o "F:\separated" --device cuda
```

You can also set the output MP3 bitrate to 128, 192, 256, or 320 kbps. The default is 320 kbps:

```powershell
python main.py "F:\music\song.mp3" -o "F:\separated" --mp3-bitrate 320
```

## Output files

The program creates these files directly inside the directory supplied with `-o`:

| File | Demucs stem | Meaning |
|---|---|---|
| `Bass.mp3` | `bass` | Bass |
| `Perc.mp3` | `drums` | Percussion / drums |
| `MIEP.mp3` | `other` | Remaining accompaniment |
| `Vocals.mp3` | `vocals` | Vocals |

**MIEP** is the project label for “Musical Instruments Excluding Percussion.” Technically, this file is Demucs' `other` stem: the accompaniment left after Demucs separates bass, drums, and vocals. It is not a piano-only track, and it may contain several kinds of instruments and sounds.

## Read-only input and output safety

- The source MP3 is only read; it is never edited, overwritten, or deleted.
- Demucs' temporary working files are created in a temporary subfolder of the output directory and removed after processing.
- If any of the four target filenames already exists, the program stops rather than overwriting it.
- The completed files are written to the directory specified by `-o`.
