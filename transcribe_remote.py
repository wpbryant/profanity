#!/usr/bin/env python3
"""
Remote Whisper Transcription Client

Sends audio to a remote whisper-asr-webservice instance and returns
word-level timestamps in the same format as profanity_filter.transcribe_audio().

Compatible with: onerahmet/openai-whisper-asr-webservice (ahmetoner/whisper-asr-webservice)
API endpoint: POST /asr
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import requests


def transcribe_remote(
    audio_path: Path,
    api_url: str,
    language: str = "en",
    encode: bool = True,
    timeout: int = 600,
) -> list[dict]:
    """Transcribe audio via remote Whisper ASR webservice.

    Args:
        audio_path: Path to audio file (WAV recommended).
        api_url: Base URL of the webservice (e.g. "http://192.168.1.135:9000").
        language: Language code for transcription.
        encode: Let the server re-encode audio through ffmpeg. Set False only
                if the audio is already in a raw format the server expects.
        timeout: Request timeout in seconds (transcription can be slow).

    Returns:
        List of dicts with keys: word (str), start (float), end (float).

    Raises:
        ConnectionError: If the server is unreachable.
        RuntimeError: If the server returns an error.
    """
    endpoint = f"{api_url.rstrip('/')}/asr"

    logging.info(f"Sending audio to remote Whisper at {endpoint}...")

    with open(audio_path, "rb") as f:
        files = {"audio_file": (audio_path.name, f)}
        params = {
            "encode": str(encode).lower(),
            "task": "transcribe",
            "language": language,
            "word_timestamps": "true",
            "output": "json",
        }

        response = requests.post(
            endpoint,
            files=files,
            params=params,
            timeout=timeout,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Remote transcription failed (HTTP {response.status_code}): {response.text}"
        )

    # The server returns JSON as a streaming text/plain response
    try:
        data = response.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON response from server: {response.text[:500]}")

    # Parse segments -> words into the same format as local transcribe_audio()
    words = []
    for segment in data.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })

    logging.info(f"Remote transcription returned {len(words)} words")
    return words


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file using a remote Whisper ASR webservice",
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Audio file to transcribe (WAV recommended)",
    )
    parser.add_argument(
        "--url",
        default="http://192.168.1.135:9000",
        help="Base URL of the Whisper ASR webservice (default: http://192.168.1.135:9000)",
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--no-encode",
        action="store_true",
        help="Skip server-side ffmpeg encoding",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Save output as JSON to file",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.audio.exists():
        logging.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    try:
        words = transcribe_remote(
            args.audio,
            args.url,
            language=args.language,
            encode=not args.no_encode,
            timeout=args.timeout,
        )
    except (ConnectionError, RuntimeError) as e:
        logging.error(e)
        sys.exit(1)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(words, f, indent=2)
        logging.info(f"Saved to: {args.output}")
    else:
        for w in words:
            print(f"[{w['start']:.2f}s - {w['end']:.2f}s] {w['word']}")


if __name__ == "__main__":
    main()
