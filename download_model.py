#!/usr/bin/env python3
"""Download the TAGARELA Parakeet ONNX model from Hugging Face.

USE_QUANTIZATION=true  → INT8  (calneymgp/...-onnx-int8)
USE_QUANTIZATION=false → FP32  (alefiury/...-TAGARELA-onnx)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DEST = "./models/parakeet"
MARKER_NAME = ".parakeet-variant"

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


@dataclass(frozen=True)
class ModelVariant:
    key: str
    repo_id: str
    revision: str
    quantization: str | None
    model_id: str
    encoder_file: str


VARIANTS: dict[str, ModelVariant] = {
    "int8": ModelVariant(
        key="int8",
        repo_id="calneymgp/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8",
        revision="7d84392553633a8e5bdca7eccb5ae25467e9572f",
        quantization="int8",
        model_id="parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8",
        encoder_file="encoder-model.int8.onnx",
    ),
    "fp32": ModelVariant(
        key="fp32",
        repo_id="alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx",
        revision="f97e702671c4dc14344da4ef7a3c07ba94b279fc",
        quantization=None,
        model_id="parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx",
        encoder_file="encoder-model.onnx",
    ),
}


def use_quantization_enabled(raw: str | None = None) -> bool:
    """Parse USE_QUANTIZATION. Default true (INT8). Empty/unset counts as true."""
    if raw is None:
        raw = os.environ.get("USE_QUANTIZATION", "true")
    value = raw.strip().lower()
    if not value:
        return True
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"USE_QUANTIZATION must be true or false, got {raw!r}")


def resolve_variant(use_quant: bool | str | None = None) -> ModelVariant:
    if isinstance(use_quant, bool):
        enabled = use_quant
    else:
        enabled = use_quantization_enabled(use_quant)
    return VARIANTS["int8" if enabled else "fp32"]


def _marker_value(repo_id: str, revision: str | None) -> str:
    return f"{repo_id}@{revision or 'main'}"


def is_model_ready(
    local_dir: str | Path,
    encoder_file: str,
    repo_id: str,
    revision: str | None,
) -> bool:
    """True when the encoder is on disk and (if present) the variant marker matches."""
    dest = Path(local_dir).resolve()
    encoder = dest / encoder_file
    if not encoder.is_file() or encoder.stat().st_size == 0:
        return False
    marker = dest / MARKER_NAME
    if not marker.is_file():
        return True
    return marker.read_text(encoding="utf-8").strip() == _marker_value(repo_id, revision)


def download_model(
    repo_id: str,
    local_dir: str | Path,
    revision: str | None = None,
    encoder_file: str | None = None,
    force: bool = False,
) -> None:
    dest = Path(local_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and encoder_file
        and is_model_ready(dest, encoder_file, repo_id, revision)
    ):
        print(f"Model already present at '{dest}' ({encoder_file}); skipping download.")
        return

    rev = f"@{revision}" if revision else ""
    print(f"Downloading model repository '{repo_id}{rev}' to '{dest}'...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "Error: huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    kwargs: dict = {
        "repo_id": repo_id,
        "local_dir": str(dest),
        "ignore_patterns": ["*.md", ".gitattributes", ".hf", ".hf/**", MARKER_NAME],
    }
    if revision:
        kwargs["revision"] = revision
    token = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    ).strip()
    if token:
        kwargs["token"] = token
        print("Hugging Face Hub: authenticated (HF_TOKEN)")
    else:
        print(
            "Hugging Face Hub: unauthenticated. Set HF_TOKEN to avoid rate limits "
            "(https://huggingface.co/settings/tokens)."
        )

    snapshot_download(**kwargs)
    (dest / MARKER_NAME).write_text(_marker_value(repo_id, revision), encoding="utf-8")
    print(f"Model successfully downloaded to: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Parakeet TDT 0.6B v3 pt-BR TAGARELA ONNX weights. "
            "USE_QUANTIZATION=true (INT8) or false (FP32 alefiury)."
        )
    )
    parser.add_argument(
        "--use-quantization",
        type=str,
        default=os.environ.get("USE_QUANTIZATION", "true"),
        help="true = INT8 (calneymgp), false = FP32 (alefiury). Default: env USE_QUANTIZATION or true.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("MODEL_REPO", ""),
        help="Override Hugging Face repository ID (default: derived from USE_QUANTIZATION).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=os.environ.get("MODEL_REVISION", ""),
        help="Override git revision/commit (default: pinned revision for the selected variant).",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=os.environ.get("MODEL_DIR", DEFAULT_DEST),
        help=f"Local destination directory (default: MODEL_DIR or {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the encoder is already present.",
    )
    args = parser.parse_args()

    try:
        variant = resolve_variant(args.use_quantization)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    repo_id = args.repo.strip() or variant.repo_id
    revision = args.revision.strip() or (None if args.repo.strip() else variant.revision)
    print(
        f"USE_QUANTIZATION={variant.key == 'int8'} variant={variant.key} "
        f"encoder={variant.encoder_file}"
    )
    download_model(
        repo_id,
        args.dest,
        revision=revision,
        encoder_file=None if args.repo.strip() else variant.encoder_file,
        force=args.force,
    )


if __name__ == "__main__":
    main()
