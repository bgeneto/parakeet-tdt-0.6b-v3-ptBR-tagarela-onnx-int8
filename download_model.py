#!/usr/bin/env python3
"""Download Parakeet TDT ONNX weights from Hugging Face.

MODEL_VARIANT selects the checkpoint:

  pt-BR          TAGARELA FP32 (GPU) — alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx
  pt-BR-INT8     TAGARELA INT8 — calneymgp/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8
  multilanguage  NVIDIA Parakeet TDT 0.6B v3 (25 European languages).
                 ONNX export of nvidia/parakeet-tdt-0.6b-v3:
                 istupakov/parakeet-tdt-0.6b-v3-onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DEST = "./models/parakeet"
MARKER_NAME = ".parakeet-variant"
ENV_VAR = "MODEL_VARIANT"
LEGACY_ENV_VAR = "USE_QUANTIZATION"
DEFAULT_VARIANT_KEY = "pt-BR"

# Case-insensitive aliases → canonical MODEL_VARIANT key.
# true/false keep the old USE_QUANTIZATION mapping.
_ALIASES: dict[str, str] = {
    "pt-br": "pt-BR",
    "ptbr": "pt-BR",
    "pt-br-int8": "pt-BR-INT8",
    "ptbr-int8": "pt-BR-INT8",
    "pt-br-cpu": "pt-BR-INT8",
    "ptbr-cpu": "pt-BR-INT8",
    "multilanguage": "multilanguage",
    "multilingual": "multilanguage",
    "nvidia": "multilanguage",
    "true": "pt-BR-INT8",
    "1": "pt-BR-INT8",
    "yes": "pt-BR-INT8",
    "on": "pt-BR-INT8",
    "y": "pt-BR-INT8",
    "int8": "pt-BR-INT8",
    "false": "pt-BR",
    "0": "pt-BR",
    "no": "pt-BR",
    "off": "pt-BR",
    "n": "pt-BR",
    "fp32": "pt-BR",
}

_HELP = (
    "pt-BR          — TAGARELA FP32 for GPU "
    "(alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx)\n"
    "  pt-BR-INT8     — TAGARELA INT8 (CPU / smaller files) "
    "(calneymgp/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8)\n"
    "  multilanguage  — NVIDIA Parakeet TDT 0.6B v3, 25 European languages "
    "(istupakov/parakeet-tdt-0.6b-v3-onnx, from nvidia/parakeet-tdt-0.6b-v3)"
)


@dataclass(frozen=True)
class ModelVariant:
    key: str
    repo_id: str
    revision: str
    quantization: str | None
    model_id: str
    encoder_file: str
    required_files: tuple[str, ...]
    extra_ignore: tuple[str, ...] = field(default_factory=tuple)


VARIANTS: dict[str, ModelVariant] = {
    "pt-BR": ModelVariant(
        key="pt-BR",
        repo_id="alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx",
        revision="f97e702671c4dc14344da4ef7a3c07ba94b279fc",
        quantization=None,
        model_id="parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx",
        encoder_file="encoder-model.onnx",
        required_files=("encoder-model.onnx", "encoder-model.onnx.data"),
    ),
    "pt-BR-INT8": ModelVariant(
        key="pt-BR-INT8",
        repo_id="calneymgp/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8",
        revision="7d84392553633a8e5bdca7eccb5ae25467e9572f",
        quantization="int8",
        model_id="parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8",
        encoder_file="encoder-model.int8.onnx",
        required_files=("encoder-model.int8.onnx",),
    ),
    "multilanguage": ModelVariant(
        key="multilanguage",
        repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
        revision="8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
        quantization=None,
        model_id="parakeet-tdt-0.6b-v3",
        encoder_file="encoder-model.onnx",
        required_files=("encoder-model.onnx", "encoder-model.onnx.data"),
        extra_ignore=("*.int8.onnx",),
    ),
}


def variant_keys() -> tuple[str, ...]:
    return tuple(VARIANTS)


def _alias_key(raw: str) -> str | None:
    return _ALIASES.get(raw.strip().lower().replace("_", "-"))


def parse_variant_key(raw: str | None = None) -> str:
    """Resolve MODEL_VARIANT (or legacy USE_QUANTIZATION) to a canonical key."""
    if raw is None:
        raw = os.environ.get(ENV_VAR, "").strip()
        if not raw:
            raw = os.environ.get(LEGACY_ENV_VAR, "").strip()
        if not raw:
            return DEFAULT_VARIANT_KEY
    value = raw.strip()
    if not value:
        return DEFAULT_VARIANT_KEY
    key = _alias_key(value)
    if key is None:
        raise ValueError(
            f"{ENV_VAR} must be one of: {', '.join(variant_keys())} (got {raw!r}).\n"
            f"  {_HELP}"
        )
    return key


def resolve_variant(raw: str | None = None) -> ModelVariant:
    return VARIANTS[parse_variant_key(raw)]


def _marker_value(repo_id: str, revision: str | None) -> str:
    return f"{repo_id}@{revision or 'main'}"


def is_model_ready(
    local_dir: str | Path,
    required_files: tuple[str, ...] | str,
    repo_id: str,
    revision: str | None,
) -> bool:
    """True when required weight files are on disk and (if present) the variant marker matches."""
    dest = Path(local_dir).resolve()
    names = (required_files,) if isinstance(required_files, str) else required_files
    for name in names:
        path = dest / name
        if not path.is_file() or path.stat().st_size == 0:
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
    required_files: tuple[str, ...] | None = None,
    extra_ignore: tuple[str, ...] = (),
    force: bool = False,
) -> None:
    dest = Path(local_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    files = required_files
    if files is None and encoder_file:
        files = (encoder_file,)
    if (
        not force
        and files
        and is_model_ready(dest, files, repo_id, revision)
    ):
        shown = files[0]
        print(f"Model already present at '{dest}' ({shown}); skipping download.")
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

    ignore = [
        "*.md",
        ".gitattributes",
        ".hf",
        ".hf/**",
        MARKER_NAME,
        *extra_ignore,
    ]
    kwargs: dict = {
        "repo_id": repo_id,
        "local_dir": str(dest),
        "ignore_patterns": ignore,
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
            "Download Parakeet TDT ONNX weights. "
            f"{ENV_VAR}={DEFAULT_VARIANT_KEY} | pt-BR-INT8 | multilanguage."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Checkpoints:\n  {_HELP}",
    )
    parser.add_argument(
        "--variant",
        "--model-variant",
        dest="variant",
        type=str,
        default=None,
        help=(
            f"Checkpoint: {', '.join(variant_keys())}. "
            f"Default: env {ENV_VAR} (legacy {LEGACY_ENV_VAR} still maps true/false) "
            f"or {DEFAULT_VARIANT_KEY}."
        ),
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("MODEL_REPO", ""),
        help=f"Override Hugging Face repository ID (default: derived from {ENV_VAR}).",
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
        variant = resolve_variant(args.variant)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    custom_repo = args.repo.strip()
    repo_id = custom_repo or variant.repo_id
    revision = args.revision.strip() or (None if custom_repo else variant.revision)
    print(
        f"{ENV_VAR}={variant.key} repo={repo_id} encoder={variant.encoder_file}"
    )
    download_model(
        repo_id,
        args.dest,
        revision=revision,
        encoder_file=None if custom_repo else variant.encoder_file,
        required_files=None if custom_repo else variant.required_files,
        extra_ignore=() if custom_repo else variant.extra_ignore,
        force=args.force,
    )


if __name__ == "__main__":
    main()
