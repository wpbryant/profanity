#!/usr/bin/env python3
"""
Audio Profanity Filter

Generates "clean" audio tracks for video files by detecting and muting profanity
using Whisper speech-to-text and ffmpeg.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from faster_whisper import WhisperModel


def setup_logging(verbose: bool = False, log_dir: Path = None) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = []
    
    # Console handler (time only)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    handlers.append(console_handler)

    # File handler (append mode, full date) - only if log_dir provided
    if log_dir:
        try:
            log_file = log_dir / "profanity_filter.log"
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Failed to create log file in {log_dir}: {e}")

    # Configure root logger with both handlers
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True  # Ensure we override any previous config
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    default_config = {
        "whisper_model": "base",
        "whisper_device": "cpu",
        "padding_before_ms": 100,
        "padding_after_ms": 150,
        "profanity_file": "en.json",
        "min_severity": 1,
        "output_suffix": "_clean",
        "keep_original": True,
        "log_detections": True,
        "audio_track_index": 0,
        "clean_track_title": "English (Clean)",
        "clean_track_mode": "add",  # add, default, or replace
    }

    if config_path.exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        default_config.update(user_config)

    return default_config


def pattern_to_regex(pattern: str) -> str:
    """Convert profanity list pattern to regex.

    The pattern format uses * to mean 'zero or more of previous char'.
    E.g., 'fu*ck' matches 'fuck', 'fuuck', 'fuuuck', etc.
    """
    result = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if i + 1 < len(pattern) and pattern[i + 1] == '*':
            # Character followed by * means "one or more of this char"
            result.append(re.escape(char) + '+')
            i += 2
        else:
            result.append(re.escape(char))
            i += 1
    return ''.join(result)


def load_profanity_list(profanity_path: Path, min_severity: int = 1) -> tuple[list[dict], re.Pattern]:
    """Load profanity list from JSON file."""
    with open(profanity_path, "r") as f:
        profanity_data = json.load(f)

    # Filter by severity and build match patterns
    filtered = []
    for entry in profanity_data:
        if entry.get("severity", 1) >= min_severity:
            # Split match patterns by pipe and convert to regex
            raw_patterns = [p.strip().lower() for p in entry["match"].split("|")]
            regex_patterns = []
            for p in raw_patterns:
                regex_p = pattern_to_regex(p)
                regex_patterns.append(regex_p)

            # Combine into single regex with word boundaries
            combined_regex = re.compile(
                r'\b(' + '|'.join(regex_patterns) + r')\b',
                re.IGNORECASE
            )

            filtered.append({
                "id": entry["id"],
                "patterns": raw_patterns,
                "regex": combined_regex,
                "severity": entry.get("severity", 1),
                "tags": entry.get("tags", []),
            })

    # Create combined regex for pre-filtering (optimization)
    all_regex_patterns = []
    for entry in filtered:
        # entry["regex"] is already compiled, get the pattern string
        # pattern is wrapped in \b(  )\b due to line 98
        # We need the inner parts.
        # But actually, line 98 does: r'\b(' + '|'.join(regex_patterns) + r')\b'
        # So we can just re-use the raw regex patterns from earlier?
        # We didn't save them.
        # Let's rebuild the simple list of all patterns
        # effectively we want: \b(pattern1|pattern2|...)\b
        # We can just join all patterns from all entries.
        
        # entry['patterns'] has the raw patterns. 
        # We need to run pattern_to_regex on all of them again? 
        # Or store them.
        
        # Let's iterate through filtered again
        pass

    all_patterns_regex = []
    for entry in filtered:
        for p in entry["patterns"]:
            all_patterns_regex.append(pattern_to_regex(p))

    combined_regex = re.compile(
        r'\b(' + '|'.join(all_patterns_regex) + r')\b',
        re.IGNORECASE
    )

    return filtered, combined_regex


def check_dependencies() -> bool:
    """Verify ffmpeg is installed."""
    if not shutil.which("ffmpeg"):
        logging.error("ffmpeg not found in PATH. Please install ffmpeg.")
        return False
    if not shutil.which("ffprobe"):
        logging.error("ffprobe not found in PATH. Please install ffmpeg.")
        return False
    return True


def get_audio_info(video_path: Path) -> dict:
    """Get audio stream information from video file."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    return data.get("streams", [])


def get_english_track_index(streams: list[dict]) -> int:
    """Find index of first English audio track. Returns -1 if not found."""
    for i, stream in enumerate(streams):
        tags = stream.get("tags", {})
        language = tags.get("language", "").lower()
        if language in ("eng", "en", "english"):
            return i
    return -1


def extract_audio(
    video_path: Path,
    output_path: Path,
    track_index: int = 0,
    for_transcription: bool = True
) -> None:
    """Extract audio track from video.

    Args:
        video_path: Path to video file
        output_path: Path for extracted audio
        track_index: Which audio track to extract
        for_transcription: If True, converts to mono 16kHz for Whisper.
                          If False, preserves original channels/sample rate.
    """
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-map", f"0:a:{track_index}",
        "-vn",
    ]

    if for_transcription:
        # Mono 16kHz for Whisper
        cmd.extend(["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"])
    else:
        # Preserve original channels and sample rate
        cmd.extend(["-acodec", "pcm_s16le"])

    cmd.extend(["-y", str(output_path)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")


def transcribe_audio(
    audio_path: Path,
    model_name: str = "base",
    device: str = "cpu"
) -> list[dict]:
    """Transcribe audio using Whisper and return word-level timestamps."""
    logging.info(f"Loading Whisper model '{model_name}' on {device}...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    logging.info("Transcribing audio (this may take a while)...")
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
    )

    words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                })

    logging.info(f"Transcribed {len(words)} words")
    return words


def match_profanity(
    words: list[dict],
    profanity_list: list[dict],
    combined_regex: re.Pattern
) -> list[dict]:
    """Match transcribed words against profanity list using regex."""
    detections = []

    # Check each word against all profanity patterns
    for word_info in words:
        word_clean = word_info["word"].strip(".,!?;:'\"")
        
        # Optimization: Quick check with combined regex
        if not combined_regex.search(word_clean):
            continue

        for entry in profanity_list:
            if entry["regex"].search(word_clean):
                detections.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "matched_id": entry["id"],
                    "severity": entry["severity"],
                    "tags": entry["tags"],
                })
                break  # Only match first profanity entry per word

    # Sort by start time
    detections.sort(key=lambda x: x["start"])

    return detections


def merge_overlapping_ranges(
    ranges: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Merge overlapping time ranges."""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def build_mute_ranges(
    detections: list[dict],
    padding_before_ms: int,
    padding_after_ms: int
) -> list[tuple[float, float]]:
    """Build list of time ranges to mute with padding."""
    ranges = []
    padding_before = padding_before_ms / 1000.0
    padding_after = padding_after_ms / 1000.0

    for detection in detections:
        start = max(0, detection["start"] - padding_before)
        end = detection["end"] + padding_after
        ranges.append((start, end))

    return merge_overlapping_ranges(ranges)


def build_ffmpeg_filter(mute_ranges: list[tuple[float, float]], volume_boost: float = 1.0) -> str:
    """Build ffmpeg volume filter string to mute specified ranges and optionally boost volume."""
    filters = []

    # Add volume boost if specified
    if volume_boost != 1.0:
        filters.append(f"volume={volume_boost}")

    # Add mute filters for each profanity range
    for start, end in mute_ranges:
        filters.append(f"volume=enable='between(t,{start:.3f},{end:.3f})':volume=0")

    if not filters:
        return "anull"

    return ",".join(filters)


def create_clean_audio(
    input_audio: Path,
    output_audio: Path,
    mute_ranges: list[tuple[float, float]],
    volume_boost: float = 1.0
) -> None:
    """Create clean audio file with muted sections and optional volume boost."""
    filter_str = build_ffmpeg_filter(mute_ranges, volume_boost)

    cmd = [
        "ffmpeg",
        "-i", str(input_audio),
        "-af", filter_str,
        "-y",
        str(output_audio),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio filtering failed: {result.stderr}")


def remux_video(
    original_video: Path,
    clean_audio: Path,
    output_video: Path,
    track_title: str = "English (Clean)",
    mode: str = "add",
    replace_existing_clean: bool = False
) -> None:
    """Add clean audio track to video file.

    Args:
        mode: "add" = extra track, "default" = extra track set as default,
              "replace" = remove original audio
        replace_existing_clean: If True, remove any existing clean tracks before adding new one
    """
    # Get number of existing streams
    probe_cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(original_video),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    num_audio_streams = sum(1 for s in streams if s["codec_type"] == "audio")

    if mode == "replace":
        # Replace original audio - map video, subtitles, and clean audio only
        cmd = [
            "ffmpeg",
            "-i", str(original_video),
            "-i", str(clean_audio),
            "-map", "0:v",  # Video streams
            "-map", "0:s?",  # Subtitle streams (if any)
            "-map", "1:a",  # Clean audio only
            "-c", "copy",  # Copy all codecs
            "-c:a:0", "aac",  # Encode clean audio as AAC
            "-c:a:0", "aac",  # Encode clean audio as AAC
            "-metadata:s:a:0", f"title={track_title}",
            "-metadata:s:a:0", f"handler_name={track_title}",
            "-metadata:s:a:0", "language=eng",
            "-disposition:a:0", "default",
            "-y",
            str(output_video),
        ]
    else:
        # "add" or "default" - keep original audio and add clean track
        if replace_existing_clean:
            # Remove previously added clean tracks before adding new one
            # Get detailed stream info including metadata
            audio_streams = [s for s in streams if s["codec_type"] == "audio"]

            cmd = [
                "ffmpeg",
                "-i", str(original_video),
                "-i", str(clean_audio),
            ]

            # Map video streams
            cmd.extend(["-map", "0:v"])

            # Map only audio streams that don't have the clean track title
            kept_audio_count = 0
            for idx, stream in enumerate(audio_streams):
                title = stream.get("tags", {}).get("title", "")
                if title != track_title:
                    cmd.extend(["-map", f"0:a:{idx}"])
                    kept_audio_count += 1

            # Map subtitle streams
            cmd.extend(["-map", "0:s?"])

            # Add new clean audio
            cmd.extend(["-map", "1:a"])

            # Set codecs and metadata
            cmd.extend([
                "-c", "copy",
                f"-c:a:{kept_audio_count}", "aac",
                f"-c:a:{kept_audio_count}", "aac",
                f"-metadata:s:a:{kept_audio_count}", f"title={track_title}",
                f"-metadata:s:a:{kept_audio_count}", f"handler_name={track_title}",
                f"-metadata:s:a:{kept_audio_count}", "language=eng",
            ])

            if mode == "default":
                # Set clean track as default, remove default from others
                for i in range(kept_audio_count):
                    cmd.extend([f"-disposition:a:{i}", "0"])
                cmd.extend([f"-disposition:a:{kept_audio_count}", "default"])

            cmd.extend(["-y", str(output_video)])
        else:
            # Original behavior - keep everything
            cmd = [
                "ffmpeg",
                "-i", str(original_video),
                "-i", str(clean_audio),
                "-map", "0",  # Copy all streams from original
                "-map", "1:a",  # Add clean audio
                "-c", "copy",  # Copy all codecs
                "-c:a:" + str(num_audio_streams), "aac",  # Encode new audio as AAC
                "-c:a:" + str(num_audio_streams), "aac",  # Encode new audio as AAC
                "-metadata:s:a:" + str(num_audio_streams), f"title={track_title}",
                "-metadata:s:a:" + str(num_audio_streams), f"handler_name={track_title}",
                "-metadata:s:a:" + str(num_audio_streams), "language=eng",
            ]

            if mode == "default":
                # Set clean track as default, remove default from others
                for i in range(num_audio_streams):
                    cmd.extend([f"-disposition:a:{i}", "0"])
                cmd.extend([f"-disposition:a:{num_audio_streams}", "default"])

            cmd.extend(["-y", str(output_video)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Video remuxing failed: {result.stderr}")


def write_log(
    log_path: Path,
    video_path: Path,
    detections: list[dict],
    mute_ranges: list[tuple[float, float]]
) -> None:
    """Write detection log file."""
    with open(log_path, "w") as f:
        f.write(f"Profanity Detection Log\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Total detections: {len(detections)}\n")
        f.write(f"Muted ranges: {len(mute_ranges)}\n\n")

        f.write("Detections:\n")
        f.write("-" * 50 + "\n")
        for d in detections:
            f.write(
                f"  [{d['start']:.2f}s - {d['end']:.2f}s] "
                f"\"{d['word']}\" (severity: {d['severity']}, tags: {d['tags']})\n"
            )

        f.write("\nMuted Ranges:\n")
        f.write("-" * 50 + "\n")
        for start, end in mute_ranges:
            f.write(f"  {start:.2f}s - {end:.2f}s ({(end-start)*1000:.0f}ms)\n")


def process_video(
    video_path: Path,
    config: dict,
    profanity_list: list[dict],
    combined_regex: re.Pattern,
    dry_run: bool = False,
    transcript_path: Path = None,
    replace_clean: bool = False,
    skip_clean: bool = False,
    logger: logging.Logger = None
) -> dict:
    """Process a single video file."""
    log = logger or logging.getLogger(__name__)

    log.info(f"Processing: {video_path.name}")

    # Get audio stream info early
    try:
        audio_streams = get_audio_info(video_path)
    except Exception as e:
        log.error(f"Failed to get audio info: {e}")
        return {}

    # Check for existing clean track
    clean_title = config["clean_track_title"]
    existing_clean_track = False
    for stream in audio_streams:
        tags = stream.get("tags", {})
        title = tags.get("title")
        handler = tags.get("handler_name")
        if clean_title == title or clean_title == handler:
            existing_clean_track = True
            break
    
    if existing_clean_track:
        if replace_clean:
            log.info("Found existing clean track - overwriting due to --replace-clean")
        elif skip_clean:
            log.info("Found existing clean track - skipping due to --skip-clean")
            return {}
        elif not dry_run:
            # Interactive prompt
            print(f"\nExisting clean track found in: {video_path.name}")
            print("1. Overwrite existing track")
            print("2. Skip file")
            print("3. Create new track (duplicate)")
            while True:
                choice = input("Select an option [1-3]: ").strip()
                if choice == "1":
                    replace_clean = True
                    log.info("User selected: Overwrite")
                    break
                elif choice == "2":
                    log.info("User selected: Skip")
                    return {}
                elif choice == "3":
                    log.info("User selected: Create new track")
                    break
                
    # Determine audio track to process
    track_index = config["audio_track_index"]
    if track_index == 0:
        # Try to find English track automatically
        eng_index = get_english_track_index(audio_streams)
        if eng_index != -1:
            track_index = eng_index
            log.info(f"Automatically selected English audio track at index {track_index}")
        else:
            log.info(f"No English track found, using default index {track_index}")
    else:
        log.info(f"Using configured audio track index {track_index}")

    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_for_transcription = temp_path / "audio_mono.wav"
        audio_full_quality = temp_path / "audio_full.wav"
        clean_audio = temp_path / "clean.wav"

        # Extract mono audio for Whisper transcription
        log.info("Extracting audio for transcription...")
        extract_audio(video_path, audio_for_transcription, track_index, for_transcription=True)

        # Transcribe
        words = transcribe_audio(
            audio_for_transcription,
            config["whisper_model"],
            config["whisper_device"],
        )

        # Save transcript if requested
        if transcript_path:
            with open(transcript_path, "w") as f:
                for w in words:
                    f.write(f"[{w['start']:.2f}s - {w['end']:.2f}s] {w['word']}\n")
            log.info(f"Transcript saved to: {transcript_path}")

        # Match profanity
        log.info("Detecting profanity...")
        detections = match_profanity(words, profanity_list, combined_regex)
        log.info(f"Found {len(detections)} profanity instances")

        # Build mute ranges
        mute_ranges = build_mute_ranges(
            detections,
            config["padding_before_ms"],
            config["padding_after_ms"],
        )

        if dry_run:
            log.info("Dry run - not creating output files")
            for d in detections:
                log.info(f"  [{d['start']:.2f}s] \"{d['word']}\" (severity: {d['severity']})")
            return {
                "detections": detections,
                "mute_ranges": mute_ranges,
                "output_path": None,
            }

        if not detections:
            log.info("No profanity detected - skipping file")
            return {
                "detections": [],
                "mute_ranges": [],
                "output_path": None,
            }

        # Extract full quality audio (preserves original channels: stereo, 5.1, etc.)
        log.info("Extracting full quality audio...")
        extract_audio(video_path, audio_full_quality, track_index, for_transcription=False)

        # Create clean audio with muted profanity
        log.info("Creating clean audio track...")
        volume_boost = config.get("volume_boost", 1.0)
        create_clean_audio(audio_full_quality, clean_audio, mute_ranges, volume_boost)

        # Remux video - write to temp file first, then replace original
        log.info("Adding clean audio track to video...")
        temp_output = temp_path / f"output{video_path.suffix}"
        remux_video(video_path, clean_audio, temp_output, config["clean_track_title"], config["clean_track_mode"], replace_clean)

        # Replace original with new file
        log.info("Replacing original file...")
        shutil.move(str(temp_output), str(video_path))

        log.info(f"Updated: {video_path}")

        # Write log file
        if config["log_detections"]:
            log_path = video_path.with_suffix(".profanity.log")
            write_log(log_path, video_path, detections, mute_ranges)
            log.info(f"Log: {log_path}")

        return {
            "detections": detections,
            "mute_ranges": mute_ranges,
            "video_path": video_path,
        }


def cleanup_tracks(video_path: Path, config: dict, logger: logging.Logger = None) -> None:
    """Interactively cleanup audio tracks."""
    log = logger or logging.getLogger(__name__)
    
    try:
        streams = get_audio_info(video_path)
    except Exception as e:
        log.error(f"Failed to read audio info for {video_path}: {e}")
        return

    if len(streams) <= 1:
        log.info(f"Skipping {video_path.name}: Only 1 audio track")
        return

    print(f"\nScanning: {video_path.name}")
    print("Found audio tracks:")
    print(f"{'Idx':<5} {'Lang':<10} {'Title/Handler':<30} {'Codec':<10}")
    print("-" * 60)
    
    for i, s in enumerate(streams):
        tags = s.get("tags", {})
        lang = tags.get("language", "und")
        title = tags.get("title") or tags.get("handler_name", "N/A")
        codec = s.get("codec_name", "unknown")
        print(f"{i:<5} {lang:<10} {title[:30]:<30} {codec:<10}")

    print("-" * 60)
    print("Options:")
    print(" - Enter comma-separated indices to KEEP (e.g. '0,2')")
    print(" - 'a' or 'all' to keep all (skip)")
    print(" - 's' or 'skip' to skip")
    
    choice = input("Keep tracks: ").strip().lower()
    
    if choice in ('a', 'all', 's', 'skip', ''):
        log.info("Skipping cleanup")
        return

    try:
        keep_indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
    except ValueError:
        log.error("Invalid input. Skipping.")
        return
    
    if len(keep_indices) == len(streams):
        log.info("Keeping all tracks. Skipping.")
        return

    # Sort indices to match stream order? Actually ffmpeg map order matters.
    # We map 0:v (video), 0:s? (subtitles), and then specific audio streams.
    
    log.info(f"Remuxing to keep audio tracks: {keep_indices}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / f"cleanup_{video_path.name}"
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-map", "0:v?",     # Video (optional)
            "-map", "0:s?",    # Subtitles
            "-c", "copy",      # Copy all codecs
        ]
        
        # Add audio maps
        for idx in keep_indices:
            if 0 <= idx < len(streams):
                cmd.extend(["-map", f"0:a:{idx}"])
            else:
                log.warning(f"Invalid track index {idx} ignored")
        
        cmd.extend(["-y", str(temp_output)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"Cleanup failed: {result.stderr}")
            return
            
        # Replace original
        shutil.move(str(temp_output), str(video_path))
        log.info(f"Cleanup complete: {video_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate clean audio tracks for videos by muting profanity"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Video file or directory to process",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show detections without creating output files",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process directories recursively",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--transcript", "-t",
        type=Path,
        help="Save transcript to file (for debugging)",
    )
    parser.add_argument(
        "--replace-clean", "-o",
        action="store_true",
        help="Remove previously added clean tracks before adding new one (overwrite)",
    )

    parser.add_argument(
        "--skip-clean", "-s",
        action="store_true",
        help="Skip files that already have a clean audio track",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Interactive mode to cleanup duplicate audio tracks",
    )

    args = parser.parse_args()

    # Determine log directory based on input
    log_dir = None
    if args.input.exists():
        log_dir = args.input if args.input.is_dir() else args.input.parent

    logger = setup_logging(args.verbose, log_dir)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Load config
    config = load_config(args.config)
    logger.debug(f"Config: {config}")

    # Load profanity list
    script_dir = Path(__file__).parent
    profanity_path = script_dir / config["profanity_file"]
    if not profanity_path.exists():
        logger.error(f"Profanity file not found: {profanity_path}")
        sys.exit(1)

    profanity_list, combined_regex = load_profanity_list(profanity_path, config["min_severity"])
    logger.info(f"Loaded {len(profanity_list)} profanity entries (severity >= {config['min_severity']})")

    # Get list of files to process
    video_extensions = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".webm"}
    files_to_process = []

    if args.input.is_file():
        files_to_process = [args.input]
    elif args.input.is_dir():
        if args.recursive:
            files_to_process = [
                f for f in args.input.rglob("*")
                if f.suffix.lower() in video_extensions
            ]
        else:
            files_to_process = [
                f for f in args.input.iterdir()
                if f.suffix.lower() in video_extensions
            ]
    else:
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    if not files_to_process:
        logger.error("No video files found to process")
        sys.exit(1)

    logger.info(f"Found {len(files_to_process)} video(s) to process")


    
    # Process each file
    for video_path in files_to_process:
        try:
            if args.cleanup:
                cleanup_tracks(video_path, config, logger)
            else:
                process_video(video_path, config, profanity_list, combined_regex, args.dry_run, args.transcript, args.replace_clean, args.skip_clean, logger)
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
