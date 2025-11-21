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


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
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


def load_profanity_list(profanity_path: Path, min_severity: int = 1) -> list[dict]:
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

    return filtered


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
    profanity_list: list[dict]
) -> list[dict]:
    """Match transcribed words against profanity list using regex."""
    detections = []

    # Check each word against all profanity patterns
    for word_info in words:
        word_clean = word_info["word"].strip(".,!?;:'\"")

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
    track_title: str = "English (Clean)"
) -> None:
    """Add clean audio track to video file."""
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

    cmd = [
        "ffmpeg",
        "-i", str(original_video),
        "-i", str(clean_audio),
        "-map", "0",  # Copy all streams from original
        "-map", "1:a",  # Add clean audio
        "-c", "copy",  # Copy all codecs
        "-c:a:" + str(num_audio_streams), "aac",  # Encode new audio as AAC
        "-metadata:s:a:" + str(num_audio_streams), f"title={track_title}",
        "-y",
        str(output_video),
    ]

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
    dry_run: bool = False,
    transcript_path: Path = None,
    logger: logging.Logger = None
) -> dict:
    """Process a single video file."""
    log = logger or logging.getLogger(__name__)

    log.info(f"Processing: {video_path.name}")

    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_for_transcription = temp_path / "audio_mono.wav"
        audio_full_quality = temp_path / "audio_full.wav"
        clean_audio = temp_path / "clean.wav"

        # Extract mono audio for Whisper transcription
        log.info("Extracting audio for transcription...")
        extract_audio(video_path, audio_for_transcription, config["audio_track_index"], for_transcription=True)

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
        detections = match_profanity(words, profanity_list)
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
        extract_audio(video_path, audio_full_quality, config["audio_track_index"], for_transcription=False)

        # Create clean audio with muted profanity
        log.info("Creating clean audio track...")
        volume_boost = config.get("volume_boost", 1.0)
        create_clean_audio(audio_full_quality, clean_audio, mute_ranges, volume_boost)

        # Remux video - write to temp file first, then replace original
        log.info("Adding clean audio track to video...")
        temp_output = temp_path / f"output{video_path.suffix}"
        remux_video(video_path, clean_audio, temp_output, config["clean_track_title"])

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
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
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
        "--transcript",
        type=Path,
        help="Save transcript to file (for debugging)",
    )

    args = parser.parse_args()

    logger = setup_logging(args.verbose)

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

    profanity_list = load_profanity_list(profanity_path, config["min_severity"])
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
            process_video(video_path, config, profanity_list, args.dry_run, args.transcript, logger)
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
