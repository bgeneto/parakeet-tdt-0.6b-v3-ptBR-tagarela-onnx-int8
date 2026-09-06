#!/usr/bin/env python3
"""Example client for transcribing audio files using the Parakeet STT API."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from urllib import error, request


def transcribe_via_urllib(
    url: str,
    file_path: Path,
    response_format: str = "verbose_json",
    language: str | None = None,
    api_key: str | None = None,
) -> str:
    """Send multipart form upload using Python standard library without external dependencies."""
    boundary = f"----WebKitFormBoundary{os.urandom(16).hex()}"
    filename = file_path.name
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        mime_type = "application/octet-stream"

    body = bytearray()

    # file field
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(
            "utf-8"
        )
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    with file_path.open("rb") as f:
        body.extend(f.read())
    body.extend(b"\r\n")

    # response_format field
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        'Content-Disposition: form-data; name="response_format"\r\n\r\n'.encode("utf-8")
    )
    body.extend(f"{response_format}\r\n".encode("utf-8"))

    # language field (optional)
    if language:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend('Content-Disposition: form-data; name="language"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{language}\r\n".encode("utf-8"))

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        url=url,
        data=bytes(body),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=300) as resp:
            return resp.read().decode("utf-8")
    except error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        sys.exit(f"HTTP Error {e.code}: {err_msg}")
    except error.URLError as e:
        sys.exit(f"Failed to connect to server at {url}: {e.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using the Parakeet STT API."
    )
    parser.add_argument("audio_file", type=Path, help="Path to the audio or video file")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/v1/audio/transcriptions",
        help="API URL (default: http://localhost:8080/v1/audio/transcriptions)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="verbose_json",
        choices=["json", "verbose_json", "text"],
        help="Response format (default: verbose_json)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language hint echoed in verbose_json. Decoding is multilingual; this is not passed to the model.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY", ""),
        help="API key (Authorization: Bearer). Defaults to the API_KEY env var.",
    )

    args = parser.parse_args()

    if not args.audio_file.is_file():
        sys.exit(f"Error: file '{args.audio_file}' does not exist.")

    print(f"Uploading '{args.audio_file}' ({args.audio_file.stat().st_size / 1024:.1f} KB) to {args.url}...")
    t0 = time.perf_counter()
    raw_response = transcribe_via_urllib(
        url=args.url,
        file_path=args.audio_file,
        response_format=args.format,
        language=args.language,
        api_key=args.api_key.strip() or None,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n--- Result (Request took {elapsed:.2f}s) ---")
    if args.format == "text":
        print(raw_response)
    else:
        try:
            parsed = json.loads(raw_response)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(raw_response)


if __name__ == "__main__":
    main()
