#!/usr/bin/env python3
"""Helper script to download the TAGARELA Parakeet ONNX INT8 model from Hugging Face."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO = "calneymgp/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8"
DEFAULT_DEST = "./models/parakeet"


def download_model(repo_id: str, local_dir: str | Path) -> None:
    dest = Path(local_dir).resolve()
    print(f"Downloading model repository '{repo_id}' to '{dest}'...")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "Error: huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )

    # Enable faster downloads if hf_transfer is installed
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print(f"Model successfully downloaded to: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Parakeet TDT 0.6B v3 pt-BR TAGARELA ONNX INT8 model."
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO,
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=DEFAULT_DEST,
        help=f"Local destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()
    download_model(args.repo, args.dest)


if __name__ == "__main__":
    main()
