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
- NVIDIA GPU (optional, for faster processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/profanity-filter.git
   cd profanity-filter
   ```

2. Create a virtual environment and install dependencies:

   **Linux/macOS:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   **Windows:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. For GPU support, install cuDNN:
   ```bash
   pip install nvidia-cudnn-cu12
   ```

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

### GPU Mode (CUDA)

Use the wrapper script which sets up cuDNN library paths:

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
volume_boost: 2.0        # boost clean track volume (1.0 = no change)
```

### Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39M | Fastest | Lower |
| base | 74M | Fast | Good |
| small | 244M | Medium | Better |
| medium | 769M | Slow | High |
| large | 1550M | Slowest | Highest |

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
