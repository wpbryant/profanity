# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the filter (use wrapper script for CUDA/cuDNN support)
./run.sh /path/to/video.mkv

# Dry run - preview detections without modifying files
./run.sh /path/to/video.mkv --dry-run

# Batch process directory
./run.sh /path/to/videos/ --recursive

# Save transcript for debugging
./run.sh /path/to/video.mkv --dry-run --transcript transcript.txt

# Verbose output
./run.sh /path/to/video.mkv -v
```

**Important:** Always use `./run.sh` instead of calling Python directly - it sets `LD_LIBRARY_PATH` for CUDA/cuDNN libraries installed via pip.

## Architecture

Single-file Python script (`profanity_filter.py`) with linear processing pipeline:

1. **Audio extraction** - ffmpeg extracts audio track to temp WAV (16kHz mono)
2. **Transcription** - faster-whisper generates word-level timestamps
3. **Profanity matching** - Regex patterns from `en.json` matched against words
4. **Filter generation** - Builds ffmpeg volume filter string with mute ranges
5. **Remuxing** - Adds clean audio track to original video file (in-place modification)

### Key Functions

- `pattern_to_regex()` - Converts profanity list wildcards (`fu*ck`) to regex (`fu+ck`)
- `load_profanity_list()` - Loads `en.json`, filters by severity, compiles regex patterns
- `match_profanity()` - Matches transcribed words against compiled patterns
- `build_ffmpeg_filter()` - Generates ffmpeg `-af` filter string with volume muting
- `process_video()` - Main orchestration function for single video

### Profanity List Format (en.json)

```json
{
  "id": "fuck",
  "match": "fu*c*k|fucks|fu*c*king",  // pipe-separated, * = repeat prev char
  "severity": 4,                       // 1-4 scale
  "tags": ["general"]
}
```

The `*` wildcard means "one or more of previous character" (e.g., `fu*ck` matches `fuck`, `fuuck`, `fuuuck`).

## Configuration (config.yaml)

Key settings:
- `whisper_device: "cuda"` - Use GPU (requires cuDNN via `pip install nvidia-cudnn-cu12`)
- `min_severity: 3` - Only filter severity 3+ (strong profanity)
- `volume_boost: 2.0` - Boost clean track volume (compensates for quieter output)
- `padding_before_ms/padding_after_ms` - Silence padding around detected words
