# Audio Profanity Filter

A Python tool that generates "clean" audio tracks for video files by detecting and muting profanity using Whisper speech-to-text and ffmpeg.

## Features
- **High Performance**: Optimized matching algorithm (~10x faster)
- **Automatic Speech-to-Text**: Transcription using [faster-whisper](https://github.com/guillaumekln/faster-whisper) — local or remote
- **Remote Transcription**: Offload Whisper to a separate server via [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)
- **Smart Audio Selection**: Automatically detects and processes English audio tracks
- **Interactive Mode**: Prompts to overwrite, skip, or duplicate if a clean track already exists
- **Track Cleanup**: Utility to remove duplicate/unwanted audio tracks
- **Persistent Logging**: Keeps a history of all runs in `profanity_filter.log`
- **GPU Acceleration**: CUDA support for fast transcription
- **Batch Processing**: Recursively process videos with `--recursive`
- **Live TUI Progress**: Dashboard with overall progress bar (with ETA), per-file status, and transcription sub-progress for long runs

## Requirements

- Python 3.10+
- ffmpeg (must be in PATH)
- NVIDIA GPU with CUDA support (optional, for faster local processing)
  - **Note:** AMD GPUs are not supported for acceleration - AMD users should use CPU mode or remote transcription

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wpbryant/profanity.git
   cd profanity
   ```

2. Create a virtual environment and install dependencies:

   **Linux/macOS:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   **Windows:**

   **Important:** Windows users should use Python 3.11 or 3.12. Python 3.14+ may have compatibility issues with PyAV.

   ```cmd
   # Use Python 3.11 or 3.12
   py -3.11 -m venv venv
   # or: py -3.12 -m venv venv

   venv\Scripts\activate
   pip install -r requirements-windows.txt
   ```

3. **For local GPU transcription only**, install cuDNN:

   **Linux/macOS:**
   ```bash
   pip install nvidia-cudnn-cu12
   ```

   **Windows:**
   ```cmd
   pip install nvidia-cudnn-cu12
   ```

   **AMD GPU users / remote transcription users:** Skip this step. See [Remote Transcription](#remote-transcription) to offload Whisper to another machine.

## Usage

### CPU Mode (Local)

**Linux/macOS:**
```bash
source venv/bin/activate
python profanity_filter.py /path/to/video.mkv
```

**Windows:**
```cmd
venv\Scripts\activate
python profanity_filter.py C:\path\to\video.mkv
```

### GPU Mode (Local, NVIDIA CUDA only)

**NVIDIA GPU users only** - Use the wrapper script which sets up cuDNN library paths:

**Linux/macOS:**
```bash
./run.sh /path/to/video.mkv
```

**Windows:**
```cmd
run.bat C:\path\to\video.mkv
```

### Remote Mode

Offload transcription to a remote Whisper server — no GPU or `faster-whisper` package needed on this machine:

```bash
# Via CLI flag (uses default URL from config.yaml):
python profanity_filter.py /path/to/video.mkv --remote

# Or set it in config.yaml and run normally:
python profanity_filter.py /path/to/video.mkv
```

See [Remote Transcription](#remote-transcription) for full setup details.

### Command Line Options
| Flag | Shorthand | Description |
|:---|:---|:---|
| `--recursive` | `-r` | Process directories recursively |
| `--dry-run` | `-d` | Preview detections without modifying files (fast) |
| `--remote` | | Use remote Whisper server for transcription (overrides config) |
| `--skip-clean` | `-s` | Skip files that already have a clean track |
| `--replace-clean` | `-o` | Overwrite existing clean tracks (force) |
| `--cleanup` | | Interactive mode to remove duplicate/spam tracks |
| `--no-tui` | | Disable the live progress display (plain log output) |
| `--config` | `-c` | Specify custom config file |
| `--verbose` | `-v` | Enable verbose logging |

### Examples

**Standard Run (local transcription):**
```bash
python profanity_filter.py /path/to/video.mkv
```

**Remote Transcription:**
```bash
python profanity_filter.py /path/to/video.mkv --remote
```

**Recursively Process Directory (Skip Existing):**
```bash
python profanity_filter.py /path/to/videos/ -r -s
```

**Cleanup Duplicate Tracks:**
```bash
python profanity_filter.py /path/to/videos/ -r --cleanup
```

**Dry Run with Verbose Logging:**
```bash
python profanity_filter.py /path/to/video.mkv -d -v
```

**Remote with custom server URL (via config):**
```bash
python profanity_filter.py /path/to/video.mkv -c /path/to/remote-config.yaml
```

### Progress Display (TUI)

When running in an interactive terminal (and [rich](https://github.com/Textualize/rich) is installed), the tool renders a live dashboard instead of streaming log lines:

```
 Profanity Filter — 2/8 files
 Overall     ━━━━━━━━━━━━━━━━━╸──────  34%  ETA 12:41  1.2/3.5 GB
 ─────────────────────────────────────────────────────────────────
 › movie_two.mkv                                    elapsed 04:12
   Transcribing (medium/cuda)      last: Transcribed 4102 words
   Transcription ━━━━━━━━━╸──────  61%          32:10 / 52:40
 ─────────────────────────────────────────────────────────────────
 ✓  movie_one.mkv                 12 mutes              3:41
 ✓  short_ep.mkv                   0 mutes (clean)      0:52
 ✗  broken.mp4          failed: ffprobe failed...      0:03
 »  skipped_one.mkv       skipped: already clean         -
 ─────────────────────────────────────────────────────────────────
 Mutes 12 | Done 2 | Clean 1 | Skipped 1 | Failed 0
```

- The overall bar is weighted by file size, so the ETA stays honest across mixed libraries
- During local transcription a sub-bar tracks position in the audio; remote transcription shows a pulsing bar while elapsed time ticks
- If a file already has a clean track (and neither `-s` nor `-o` is set), the Overwrite/Skip/Duplicate menu appears inside the display — press `1`, `2`, or `3`
- When the run ends, a permanent per-file summary table is printed
- Everything is still written to `profanity_filter.log` as usual

The TUI is disabled automatically when output is piped (`nohup`, cron, `| tee`) or with `--verbose`, and can be turned off explicitly with `--no-tui` or in config (`tui: false`). For unattended runs, pair it with `-s` or `-o` so nothing waits for input.

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
# Whisper settings
whisper_model: "base"    # tiny, base, small, medium, large
whisper_device: "cuda"   # cuda or cpu

# Remote transcription (used when whisper_mode is "remote" or --remote flag is set)
whisper_mode: "local"                # "local" or "remote"
whisper_api_url: "http://192.168.1.135:9000"  # whisper-asr-webservice URL
whisper_api_timeout: 600             # request timeout in seconds

# Filtering settings
padding_before_ms: 100   # silence padding before profanity
padding_after_ms: 150    # silence padding after profanity
min_severity: 3          # 1=all, 2=moderate+, 3=strong+, 4=severe only

# Audio settings
clean_track_title: "English (Clean)"
clean_track_mode: "default"  # add, default, or replace
volume_boost: 2.0        # boost clean track volume (1.0 = no change)

# Output settings
tui: true                # live progress dashboard (auto-disabled when piped)
```

### Clean Track Modes

| Mode | Behavior |
|------|----------|
| `add` | Adds clean track as an extra audio track |
| `default` | Adds clean track and sets it as the default selection |
| `replace` | Removes original audio, only keeps clean version |

### Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39M | Fastest | Lower |
| base | 74M | Fast | Good |
| small | 244M | Medium | Better |
| medium | 769M | Slow | High |
| large | 1550M | Slowest | Highest |

**Note:** When using remote transcription, the model is configured on the server side — `whisper_model` in config.yaml only affects local mode.

## Remote Transcription

Offload the transcription step to a separate machine running [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) (Docker image: `onerahmet/openai-whisper-asr-webservice`). This is useful when:

- Your machine doesn't have a GPU
- You want to process videos while keeping your machine's resources free
- You have a dedicated server with a GPU for transcription

### Setup

1. **Start the Whisper server** on your remote machine (e.g., 192.168.1.135):
   ```bash
   # CPU:
   docker run -p 9000:9000 onerahmet/openai-whisper-asr-webservice:latest

   # GPU (NVIDIA):
   docker run --gpus all -p 9000:9000 onerahmet/openai-whisper-asr-webservice:latest-gpu
   ```

   The model is configured via the `ASR_MODEL` environment variable on the server (default: `base`). See the [whisper-asr-webservice docs](https://github.com/ahmetoner/whisper-asr-webservice) for all options.

2. **Configure the filter** — either method works:

   **Option A: CLI flag** (one-off, overrides config):
   ```bash
   python profanity_filter.py /path/to/video.mkv --remote
   ```

   **Option B: config.yaml** (persistent):
   ```yaml
   whisper_mode: "remote"
   whisper_api_url: "http://192.168.1.135:9000"
   ```

3. **Run** — the rest of the pipeline (profanity detection, audio filtering, remuxing) runs locally as normal.

### Standalone Testing

The `transcribe_remote.py` script can be used independently to test your Whisper server or debug transcription output:

```bash
# Print word-level timestamps to console:
python transcribe_remote.py /path/to/audio.wav

# Save as JSON:
python transcribe_remote.py /path/to/audio.wav -o words.json

# Custom server URL:
python transcribe_remote.py /path/to/audio.wav --url http://my-server:9000

# Verbose output:
python transcribe_remote.py /path/to/audio.wav -v
```

### Dependencies

When using remote mode, `faster-whisper` is **not required** on the local machine. The only dependencies needed are `requests` and `pyyaml`:

```bash
pip install requests pyyaml
```

Adding `rich` enables the live progress display (without it the tool falls back to plain log output):

```bash
pip install requests pyyaml rich
```

## Windows Troubleshooting

### Python Version Compatibility

**Issue:** Installation fails with build errors for `av` (PyAV) or `onnxruntime`

**Symptoms:**
```
ERROR: Failed to build 'av' when getting requirements to build wheel
ERROR: No matching distribution found for onnxruntime
```

**Solution:** Use Python 3.11 or 3.12 (not 3.14+)

Python 3.14 and newer versions don't have pre-built wheels for all dependencies on Windows. Download Python 3.11 or 3.12 from [python.org/downloads](https://www.python.org/downloads/) and recreate your virtual environment:

```cmd
# Remove old venv
rmdir /s venv

# Create new venv with Python 3.11 or 3.12
py -3.12 -m venv venv

# Activate and install
venv\Scripts\activate
pip install -r requirements-windows.txt
```

### ffmpeg Requirement

**Issue:** `ffmpeg` not found or video processing fails

**Solution:** Install ffmpeg and add it to your PATH

1. Download ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract the archive and add the `bin` folder to your system PATH
3. Verify installation: `ffmpeg -version`

### Using CPU Mode

If you have an AMD GPU or encounter CUDA issues, use CPU mode:

1. Edit `config.yaml` and set `whisper_device: "cpu"`
2. Skip the cuDNN installation step
3. Use the regular Python command instead of `run.bat`:
   ```cmd
   venv\Scripts\activate
   python profanity_filter.py C:\path\to\video.mkv
   ```

Or use [remote transcription](#remote-transcription) to offload to a GPU server.

## Profanity List

The included `en.json` contains 400+ English profanity entries with:
- Severity levels (1-4)
- Pattern matching for variations
- Categories (general, sexual, racial, etc.)

You can customize severities or add entries as needed.

## Key Behaviors

### Smart Audio Selection
The script automatically detects and selects the English audio track (tagged `eng`, `en`, `english`).
- If multiple English tracks exist, it selects the first one.
- If no English track is found, it falls back to the configured `audio_track_index` (default: 0).

### Interactive Safety Checks
If a file already has a "Clean" audio track:
- **Interactive**: Pauses and asks you to [Overwrite], [Skip], or [Create New].
- **Batch (`-s`)**: Automatically skips the file.
- **Force (`-o`)**: Automatically overwrites the existing track.

### Persistent Logging
Logs are saved to `profanity_filter.log` in the same directory as your video files. This file is **append-only**, keeping a history of all runs for debugging.

## How It Works
1. Extracts audio from video file
2. Transcribes speech using Whisper with word-level timestamps (local or remote)
3. Matches words against profanity list using regex patterns
4. Generates ffmpeg filter to mute detected segments
5. Creates clean audio track with muted profanity
6. Adds clean track to original video file

## Output

- The original video file is modified in-place with the new audio track
- A `.profanity.log` file is created for each video with detection details
- A `profanity_filter.log` is appended to in the video directory (global run history)

## Credits

- Profanity word list from [dsojevic/profanity-list](https://github.com/dsojevic/profanity-list)
- Speech-to-text powered by [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- Remote API via [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)

## License

MIT
