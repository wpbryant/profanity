# Audio Profanity Filter

A Python tool that generates "clean" audio tracks for video files by detecting and muting profanity using Whisper speech-to-text and ffmpeg.

## Features

- Automatic speech-to-text transcription using [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- Configurable profanity detection with severity levels
- Adds a separate "Clean" audio track to your video files
- GPU acceleration support (CUDA)
- Batch processing for multiple files

## Requirements

- Python 3.10+
- ffmpeg (must be in PATH)
- NVIDIA GPU with CUDA support (optional, for faster processing)
  - **Note:** AMD GPUs are not supported for acceleration - AMD users should use CPU mode

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

3. **For NVIDIA GPU support only**, install cuDNN:

   **Linux/macOS:**
   ```bash
   pip install nvidia-cudnn-cu12
   ```

   **Windows:**
   ```cmd
   pip install nvidia-cudnn-cu12
   ```

   **AMD GPU users:** Skip this step and use CPU mode (see Configuration section to set `whisper_device: "cpu"`)

## Usage

### CPU Mode

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

### GPU Mode (NVIDIA CUDA only)

**NVIDIA GPU users only** - Use the wrapper script which sets up cuDNN library paths:

**Linux/macOS:**
```bash
./run.sh /path/to/video.mkv
```

**Windows:**
```cmd
run.bat C:\path\to\video.mkv
```

### Other Options

```bash
# Dry run - preview detections without modifying files
python profanity_filter.py /path/to/video.mkv --dry-run

# Replace previously added clean tracks (re-run with different config)
python profanity_filter.py /path/to/video.mkv --replace-clean

# Batch process a directory
python profanity_filter.py /path/to/videos/ --recursive

# Save transcript for debugging
python profanity_filter.py /path/to/video.mkv --dry-run --transcript transcript.txt

# Verbose output
python profanity_filter.py /path/to/video.mkv -v
```

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
# Whisper settings
whisper_model: "base"    # tiny, base, small, medium, large
whisper_device: "cuda"   # cuda or cpu

# Filtering settings
padding_before_ms: 100   # silence padding before profanity
padding_after_ms: 150    # silence padding after profanity
min_severity: 3          # 1=all, 2=moderate+, 3=strong+, 4=severe only

# Audio settings
clean_track_title: "English (Clean)"
clean_track_mode: "default"  # add, default, or replace
volume_boost: 2.0        # boost clean track volume (1.0 = no change)
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

## Profanity List

The included `en.json` contains 400+ English profanity entries with:
- Severity levels (1-4)
- Pattern matching for variations
- Categories (general, sexual, racial, etc.)

You can customize severities or add entries as needed.

## How It Works

1. Extracts audio from video file
2. Transcribes speech using Whisper with word-level timestamps
3. Matches words against profanity list using regex patterns
4. Generates ffmpeg filter to mute detected segments
5. Creates clean audio track with muted profanity
6. Adds clean track to original video file

## Output

- The original video file is modified in-place with the new audio track
- A `.profanity.log` file is created with detection details

## Credits

- Profanity word list from [dsojevic/profanity-list](https://github.com/dsojevic/profanity-list)
- Speech-to-text powered by [faster-whisper](https://github.com/guillaumekln/faster-whisper)

## License

MIT
