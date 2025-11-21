## Project: Audio Profanity Filter (Python)

### Overview
Python script that generates "clean" audio tracks for video files by detecting and muting profanity using Whisper speech-to-text and ffmpeg.

### Core Requirements

**Primary Script: `profanity_filter.py`**

**Inputs:**
- Video file path (support .mkv, .mp4, .avi)
- Profanity word list file (`profanity_words.txt`)
- Configuration options (padding, Whisper model, etc.)

**Outputs:**
- Modified video file with additional "Clean" audio track
- Log file with detected profanity and timestamps

### Implementation Steps

1. **Setup & Configuration**
   - Parse command-line arguments
   - Load config file (YAML or JSON)
   - Load profanity word list
   - Validate dependencies (ffmpeg, whisper)

2. **Audio Extraction**
   - Use subprocess to call ffmpeg
   - Extract audio to temporary file (.wav or .mp3)
   - Handle multiple audio tracks (select primary)

3. **Speech Recognition**
   - Run Whisper on extracted audio
   - Get word-level timestamps
   - Use `faster-whisper` library for efficiency
   - Progress indicator during processing

4. **Profanity Detection**
   - Parse Whisper output (JSON format)
   - Match words against profanity list (case-insensitive)
   - Build list of timestamp ranges: `[(start, end), ...]`
   - Apply padding to timestamps

5. **Generate FFmpeg Filter**
   - Convert timestamp list to ffmpeg volume filter string
   - Format: `"volume=enable='between(t,START,END)':volume=0"`
   - Chain multiple filters with commas

6. **Audio Processing**
   - Apply filter to extracted audio
   - Create new clean audio file

7. **Remux Video**
   - Add clean audio track back to original video
   - Preserve all original streams
   - Set metadata: `title="English (Clean)"`
   - Output to new file or replace original

8. **Cleanup**
   - Remove temporary audio files
   - Write log file with results

### File Structure
```
profanity-filter/
├── profanity_filter.py
├── config.yaml
├── profanity_words.txt
├── requirements.txt
├── README.md
└── .gitignore
```

### Configuration File (config.yaml)
```yaml
# Whisper settings
whisper_model: "base"  # tiny, base, small, medium, large
whisper_device: "cpu"  # or "cuda" for GPU

# Filtering settings
padding_before_ms: 100
padding_after_ms: 150
profanity_file: "profanity_words.txt"

# Output settings
output_suffix: "_clean"
keep_original: true
log_detections: true

# Audio settings
audio_track_index: 0  # which audio track to process
clean_track_title: "English (Clean)"
```

### Dependencies (requirements.txt)
```
faster-whisper>=0.10.0
pyyaml>=6.0
```

### Command-Line Usage
```bash
# Basic usage
python profanity_filter.py movie.mkv

# Custom config
python profanity_filter.py movie.mkv --config custom_config.yaml

# Dry run (show detections without processing)
python profanity_filter.py movie.mkv --dry-run

# Batch process directory
python profanity_filter.py /path/to/movies/ --recursive

# Verbose output
python profanity_filter.py movie.mkv -v
```

### Error Handling
- Check ffmpeg installed and in PATH
- Verify input file exists and is valid video
- Handle Whisper model download on first run
- Check disk space before processing
- Graceful failure with clear error messages

### Example Profanity Words File
```
# profanity_words.txt (one word per line)
damn
hell
crap
# Add more as needed
```

### Success Criteria
- Successfully processes a test video file
- Accurately detects and mutes profanity
- Creates working audio track in Emby
- Clean, readable code with comments
- Basic error handling implemented

### Optional Enhancements (mention but don't require initially)
- Progress bars (tqdm)
- Multiple language support
- GPU acceleration
- Resume interrupted processing
- Integration with Emby API