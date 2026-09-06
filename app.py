#!/usr/bin/env python3
"""Parakeet TDT 0.6B v3 (ONNX) — STT production server."""
from __future__ import annotations

import asyncio
import ctypes
import gc
import logging
import os
import re
import secrets
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import onnx_asr
import onnxruntime as ort
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from download_model import ENV_VAR as MODEL_VARIANT_ENV
from download_model import resolve_variant

LOG = logging.getLogger("stt")

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/models/parakeet")
# onnx-asr type for this checkpoint's config.json — not user-configurable.
MODEL_ARCH = "nemo-conformer-tdt"
try:
    _MODEL = resolve_variant()
except ValueError as exc:
    raise SystemExit(f"Error: {exc}") from exc
QUANTIZATION = _MODEL.quantization
MODEL_ID = os.environ.get("MODEL_ID", "").strip() or _MODEL.model_id
SR = 16000
# Long-form TDT: 30s windows saturate the encoder better than 20s.
# TDT decode is still O(T); fewer windows = less overlap/kernel-launch tax.
MAX_CHUNK_S = float(os.environ.get("MAX_CHUNK_S", "30"))
CHUNK_OVERLAP_S = float(os.environ.get("CHUNK_OVERLAP_S", "1.0"))
CHUNK_CONTEXT_S = float(os.environ.get("CHUNK_CONTEXT_S", "0.5"))
CHUNK_LOOKBACK_S = float(os.environ.get("CHUNK_LOOKBACK_S", "2.0"))
MIN_CHUNK_S = float(os.environ.get("MIN_CHUNK_S", "0.5"))
CHUNKING = os.environ.get("CHUNKING", "window").strip().lower()
PREPROCESS_ON_GPU = os.environ.get("PREPROCESS_ON_GPU", "1").lower() not in {"0", "false", "no"}
_SPACE_RE = re.compile(r"\A\s|\s\B|(\s)\b")
GPU_ID = int(os.environ.get("GPU_ID", "0"))
GPU_MEM_LIMIT_GB = float(os.environ.get("GPU_MEM_LIMIT_GB", "6"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
SLEEP_IDLE_SECONDS = float(os.environ.get("SLEEP_IDLE_SECONDS", "60"))
LOAD_AT_STARTUP = os.environ.get("LOAD_AT_STARTUP", "1").lower() not in {"0", "false", "no"}
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "512"))
API_KEY = os.environ.get("API_KEY", "").strip()
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/stt"))
ALLOWED_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}

engine = None  # set in lifespan → AsrEngine
gpu_lock: asyncio.Semaphore | None = None
_inflight = 0
_last_used = 0.0
_idle_task: asyncio.Task | None = None


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).lower() not in {"0", "false", "no"}


def _preload_ort_cuda() -> None:
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(cuda=True, cudnn=True)
    except TypeError:
        preload()
    except Exception as exc:
        LOG.warning("ORT preload_dlls falhou: %s", exc)


def _cudart():
    for name in ("libcudart.so.12", "libcudart.so.13", "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def cuda_mem_used_mb(device_id: int = GPU_ID) -> float | None:
    lib = _cudart()
    if lib is None:
        return None
    try:
        lib.cudaSetDevice.argtypes = [ctypes.c_int]
        lib.cudaSetDevice.restype = ctypes.c_int
        lib.cudaMemGetInfo.restype = ctypes.c_int
        if lib.cudaSetDevice(int(device_id)) != 0:
            return None
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        if lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
        return (total.value - free.value) / (1024 * 1024)
    except Exception:
        return None


def cuda_release_device(device_id: int = GPU_ID) -> None:
    """Return VRAM to the driver. Must run only after ORT sessions are destroyed."""
    gc.collect()
    lib = _cudart()
    if lib is None:
        LOG.warning("libcudart não encontrada — VRAM pode permanecer no processo")
        return
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    lib.cudaDeviceReset.restype = ctypes.c_int
    rc_set = lib.cudaSetDevice(int(device_id))
    lib.cudaDeviceSynchronize()
    rc_reset = lib.cudaDeviceReset()
    if rc_set != 0 or rc_reset != 0:
        LOG.warning("cudaDeviceReset rc set=%s reset=%s", rc_set, rc_reset)
    else:
        LOG.info("CUDA device %s reset (VRAM devolvida ao driver)", device_id)


def _providers() -> list:
    mem = int(GPU_MEM_LIMIT_GB * 1024 * 1024 * 1024)
    cuda_opts = {
        "device_id": GPU_ID,
        "arena_extend_strategy": os.environ.get("ORT_ARENA_EXTEND", "kNextPowerOfTwo"),
        "gpu_mem_limit": mem,
        # HEURISTIC: wake-from-idle must not pay EXHAUSTIVE conv search every time.
        "cudnn_conv_algo_search": os.environ.get("CUDNN_CONV_ALGO_SEARCH", "HEURISTIC"),
        "do_copy_in_default_stream": True,
        "cudnn_conv_use_max_workspace": os.environ.get("CUDNN_CONV_MAX_WORKSPACE", "1"),
        "cudnn_conv1d_pad_to_nc1d": "1",
    }
    avail = ort.get_available_providers()
    LOG.info("ORT providers disponíveis: %s", avail)
    if "CUDAExecutionProvider" in avail:
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    LOG.warning("CUDA EP indisponível — caindo para CPU (lento)")
    return ["CPUExecutionProvider"]


def _session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_THREADS", "4"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_THREADS", "2"))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    return so


class AsrEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None
        self._using_cuda = False

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        with self._lock:
            self._load_unlocked()

    def unload(self) -> None:
        with self._lock:
            self._unload_unlocked()

    def try_sleep(self, idle_s: float) -> bool:
        if idle_s <= 0:
            return False
        with self._lock:
            if _inflight != 0 or self.model is None:
                return False
            if (time.monotonic() - _last_used) < idle_s:
                return False
            LOG.info("idle %.0fs — descarregando modelo da GPU", idle_s)
            self._unload_unlocked()
            return True

    def recognize_pcm(self, pcm: np.ndarray) -> Any:
        with self._lock:
            self._load_unlocked()
            pcm = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
            return self.model.recognize(pcm, sample_rate=SR)

    def _load_unlocked(self) -> None:
        if self.model is not None:
            return
        _preload_ort_cuda()
        t0 = time.perf_counter()
        providers = _providers()
        self._using_cuda = any(
            (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in providers
        )
        cpu_ep = ["CPUExecutionProvider"]
        encoder = Path(MODEL_DIR) / _MODEL.encoder_file
        if not encoder.is_file():
            raise FileNotFoundError(
                f"Missing {encoder} ({MODEL_VARIANT_ENV}={_MODEL.key}). "
                f"Expected {_MODEL.repo_id}. Weights are under {MODEL_DIR} "
                "(Compose bind-mounts ./models/parakeet by default). "
                "Restart to let entrypoint.sh download, or run download_model.py "
                f"with the same {MODEL_VARIANT_ENV}."
            )
        quant = QUANTIZATION
        pre_on_gpu = PREPROCESS_ON_GPU and self._using_cuda
        pre_cfg = {
            "providers": providers if pre_on_gpu else cpu_ep,
            "use_numpy_preprocessors": not pre_on_gpu,
            "use_conv_preprocessors": pre_on_gpu,
            "max_concurrent_workers": 1,
        }
        self.model = onnx_asr.load_model(
            MODEL_ARCH,
            MODEL_DIR,
            quantization=quant,
            sess_options=_session_options(),
            providers=providers,
            preprocessor_config=pre_cfg,
            resampler_config={"providers": cpu_ep},
        ).with_timestamps()
        # Warm a couple of lengths so cuDNN/CUDA kernels exist before the first request.
        for sec in (1.0, min(4.0, MAX_CHUNK_S)):
            n = max(SR, int(SR * sec))
            _ = self.model.recognize(np.zeros(n, dtype=np.float32), sample_rate=SR)
        used = cuda_mem_used_mb()
        extra = f" | VRAM ~{used:.0f} MiB" if used is not None else ""
        LOG.info(
            "modelo pronto em %.1fs | chunk=%.0fs overlap=%.1fs pre=%s | %s%s",
            time.perf_counter() - t0,
            MAX_CHUNK_S,
            CHUNK_OVERLAP_S,
            "cuda-conv" if pre_on_gpu else "cpu-numpy",
            _MODEL.key,
            extra,
        )

    def _unload_unlocked(self) -> None:
        if self.model is None:
            return
        self.model = None
        gc.collect()
        if self._using_cuda and _env_flag("CUDA_DEVICE_RESET", "1"):
            cuda_release_device(GPU_ID)
            LOG.info("modelo descarregado (cudaDeviceReset)")
        else:
            LOG.info("modelo descarregado")


def ffmpeg_pcm_proc(src: Path) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-analyzeduration", "20M", "-probesize", "20M",
        "-i", str(src),
        "-map", "0:a:0",
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(SR),
        "-c:a", "pcm_s16le", "-f", "s16le",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 16
    )


def _pcm16le(raw: bytes) -> np.ndarray:
    if len(raw) < 2:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(raw, dtype=np.int16, count=len(raw) // 2)
    return np.ascontiguousarray(pcm.astype(np.float32) * (1.0 / 32768.0))


def iter_audio(
    src: Path,
    max_s: float = MAX_CHUNK_S,
    overlap_s: float = CHUNK_OVERLAP_S,
    min_s: float = MIN_CHUNK_S,
) -> Iterator[tuple[float, float, np.ndarray]]:
    """ffmpeg → 16 kHz mono. Yields (start, end, pcm) in file time, pcm includes left context."""
    if CHUNKING == "vad":
        yield from _iter_vad(src, max_s, min_s)
        return
    yield from _iter_windows(src, max_s, overlap_s, min_s)


def _close_ffmpeg(proc: subprocess.Popen, emitted: bool) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        _, err = proc.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        err = b""
    if proc.returncode not in (0, None, -9) and not emitted:
        msg = (err or b"").decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"ffmpeg falhou (rc={proc.returncode}): {msg}")


def _frame_rms(pcm: np.ndarray, sr: int, win_s: float = 0.03) -> tuple[np.ndarray, int]:
    win = max(1, int(sr * win_s))
    n = pcm.size // win
    if n <= 0:
        return np.zeros(0, dtype=np.float32), win
    frames = pcm[: n * win].reshape(n, win)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12), win


def _silence_cut(pcm: np.ndarray, sr: int, search_from: int, search_to: int, max_rms: float = 0.012) -> int | None:
    """Index of the quietest 30 ms frame in [search_from, search_to), or None if never quiet."""
    search_from = max(0, search_from)
    search_to = min(pcm.size, search_to)
    if search_to - search_from < int(sr * 0.06):
        return None
    rms, win = _frame_rms(pcm[search_from:search_to], sr)
    if rms.size == 0:
        return None
    k = int(np.argmin(rms))
    if float(rms[k]) > max_rms:
        return None
    return search_from + k * win


def _iter_windows(
    src: Path, max_s: float, overlap_s: float, min_s: float
) -> Iterator[tuple[float, float, np.ndarray]]:
    """Silence-aligned ~max_s windows. Hard cuts keep `overlap_s` of audio for midpoint stitch."""
    max_n = max(1, int(SR * max_s))
    overlap_n = max(int(SR * 0.2), int(SR * min(overlap_s, max_s - min_s)))
    context_n = max(0, int(SR * min(CHUNK_CONTEXT_S, overlap_s)))
    look_n = max(overlap_n, int(SR * CHUNK_LOOKBACK_S))
    min_n = int(SR * min_s)
    read_n = max(1, int(SR * 0.25))
    proc = ffmpeg_pcm_proc(src)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout indisponível")
    carry = np.zeros(0, dtype=np.float32)
    file_samples = 0
    emitted = False
    try:
        while True:
            raw = proc.stdout.read(read_n * 2)
            if not raw:
                break
            pcm = _pcm16le(raw)
            if pcm.size == 0:
                break
            carry = np.concatenate([carry, pcm]) if carry.size else pcm
            file_samples += pcm.size
            while carry.size >= max_n:
                origin = file_samples - carry.size
                target = max_n
                min_keep = max(min_n, int(0.55 * max_n))
                search_from = min_keep
                search_to = target
                look_from = max(search_from, target - look_n)
                cut = _silence_cut(carry, SR, look_from, search_to)
                if cut is None:
                    cut = target
                    keep_from = max(0, cut - overlap_n)
                else:
                    keep_from = max(0, cut - context_n)
                window = carry[:cut]
                start = origin / SR
                yield start, start + window.size / SR, window
                emitted = True
                carry = carry[keep_from:]
        if carry.size >= min_n:
            origin = file_samples - carry.size
            start = origin / SR
            yield start, start + carry.size / SR, carry
            emitted = True
    finally:
        _close_ffmpeg(proc, emitted)


def _iter_vad(
    src: Path,
    max_s: float,
    min_s: float,
    frame_ms: int = 30,
    hangover_ms: int = 1500,
    pad_ms: int = 300,
    abs_thresh: float = 0.008,
) -> Iterator[tuple[float, float, np.ndarray]]:
    """Optional energy VAD. Default hangover is 1.5s so conversational pauses stay in-context."""
    proc = ffmpeg_pcm_proc(src)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout indisponível")
    frame_n = int(SR * frame_ms / 1000)
    hang_frames = max(1, hangover_ms // frame_ms)
    pad_n = int(SR * pad_ms / 1000)
    max_n = int(SR * max_s)
    min_n = int(SR * min_s)
    bytes_frame = frame_n * 2
    noise = 0.003
    in_speech = False
    sil_run = 0
    buf: list[np.ndarray] = []
    buf_n = 0
    t_samples = 0
    utt_start = 0.0
    emitted = False
    pre_roll = np.zeros(0, dtype=np.float32)

    def emit(end_samples: int, arr: np.ndarray):
        if arr.size < min_n:
            return None
        return max(0.0, utt_start), end_samples / SR, arr

    try:
        while True:
            raw = proc.stdout.read(bytes_frame)
            if not raw:
                break
            pcm = _pcm16le(raw)
            if pcm.size == 0:
                break
            n = pcm.size
            rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12))
            noise = 0.995 * noise + 0.005 * rms
            is_sp = rms > max(abs_thresh, 3.2 * noise)
            if is_sp and not in_speech:
                in_speech = True
                sil_run = 0
                pad = pre_roll[-pad_n:] if pre_roll.size else np.zeros(0, dtype=np.float32)
                utt_start = max(0.0, (t_samples - pad.size) / SR)
                buf = [pad, pcm] if pad.size else [pcm]
                buf_n = pad.size + n
            elif in_speech:
                buf.append(pcm)
                buf_n += n
                if is_sp:
                    sil_run = 0
                else:
                    sil_run += 1
                    if sil_run >= hang_frames and buf_n >= min_n:
                        item = emit(t_samples + n, np.concatenate(buf))
                        if item:
                            emitted = True
                            yield item
                        in_speech = False
                        buf, buf_n = [], 0
                if buf_n >= max_n:
                    arr = np.concatenate(buf)
                    cut = max(min_n, arr.size - int(SR * 1.0))
                    item = emit(t_samples + n - (arr.size - cut), arr[:cut])
                    if item:
                        emitted = True
                        yield item
                    rest = arr[cut:]
                    buf = [rest] if rest.size else []
                    buf_n = rest.size
                    utt_start = (t_samples + n - rest.size) / SR
                    in_speech = buf_n > 0
                    sil_run = 0
            elif pad_n > 0:
                pre_roll = np.concatenate([pre_roll, pcm])[-pad_n:]
            t_samples += n
        if in_speech and buf_n >= min_n:
            item = emit(t_samples, np.concatenate(buf))
            if item:
                emitted = True
                yield item
        if not emitted and t_samples >= min_n:
            if proc.poll() is None:
                proc.kill()
            yield from _iter_windows(src, max_s, CHUNK_OVERLAP_S, min_s)
    finally:
        _close_ffmpeg(proc, emitted)


def _token_words(
    tokens: list[str] | None,
    timestamps: list[float] | None,
    offset: float,
) -> list[dict]:
    if not tokens or not timestamps:
        return []
    words: list[dict] = []
    cur = ""
    start: float | None = None
    last = offset
    n = min(len(tokens), len(timestamps))
    for i in range(n):
        tok = tokens[i]
        ts = offset + float(timestamps[i])
        is_new = tok.startswith(" ") or tok.startswith("\u2581")
        piece = tok.replace("\u2581", " ").strip()
        if is_new and cur:
            words.append({"word": cur, "start": round(start or last, 3), "end": round(last, 3)})
            cur = piece
            start = ts
        else:
            if start is None:
                start = ts
            cur += piece if not cur else (piece if is_new else tok.strip())
        last = ts
    if cur:
        words.append({"word": cur, "start": round(start or offset, 3), "end": round(last, 3)})
    return words


def _tokens_to_text(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return _SPACE_RE.sub(lambda m: " " if m.group(1) else "", "".join(tokens)).strip()


def _unwrap_asr(result: Any) -> Any:
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result


def _snap_cut(timestamps: list[float], tokens: list[str], t: float, radius: float = 0.25) -> float:
    """Prefer a word-boundary token near t so SentencePiece words are not split."""
    best_t, best_d = t, radius + 1.0
    found = False
    for ts, tok in zip(timestamps, tokens):
        d = abs(ts - t)
        if d > radius:
            continue
        boundary = tok.startswith(" ") or tok.startswith("\u2581")
        score = d if boundary else d + radius
        if score < best_d:
            best_t, best_d, found = ts, score, True
    return best_t if found else t


def _slice_hypothesis(
    result: Any,
    t_from: float,
    t_until: float | None,
) -> tuple[str, list[float], list[str], list[float]]:
    result = _unwrap_asr(result)
    if result is None:
        return "", [], [], []
    if isinstance(result, str):
        return result.strip(), [], [], []
    tokens = list(getattr(result, "tokens", None) or [])
    timestamps = [float(t) for t in (getattr(result, "timestamps", None) or [])]
    logprobs = [float(x) for x in (getattr(result, "logprobs", None) or [])]
    n = min(len(tokens), len(timestamps))
    tokens, timestamps = tokens[:n], timestamps[:n]
    logprobs = logprobs[:n] if logprobs else []
    if not tokens:
        raw = getattr(result, "text", None)
        return (raw.strip() if isinstance(raw, str) else ""), [], [], []
    lo = _snap_cut(timestamps, tokens, t_from) if t_from > 0 else t_from
    hi = _snap_cut(timestamps, tokens, t_until) if t_until is not None else None
    keep = [
        i
        for i, ts in enumerate(timestamps)
        if ts >= lo and (hi is None or ts < hi)
    ]
    tokens = [tokens[i] for i in keep]
    timestamps = [timestamps[i] for i in keep]
    logprobs = [logprobs[i] for i in keep] if logprobs else []
    return _tokens_to_text(tokens), timestamps, tokens, logprobs


def transcribe_file(
    path: Path, want_words: bool = False, language: str | None = None
) -> dict:
    """Overlap is owned once: previous window keeps [0, mid), next keeps [mid, end)."""
    if engine is None:
        raise RuntimeError("engine indisponível")
    t0 = time.perf_counter()
    parts: list[str] = []
    segments: list[dict] = []
    words_out: list[dict] = []
    last_end = 0.0
    pending: tuple[float, float, Any, float] | None = None

    def emit(start: float, end: float, raw: Any, t_from: float, t_until: float | None) -> None:
        nonlocal last_end
        text, ts, tokens, logprobs = _slice_hypothesis(raw, t_from, t_until)
        last_end = end
        if not text:
            return
        if ts:
            seg_start = start + ts[0]
            seg_end = start + ts[-1]
        else:
            seg_start = start + t_from
            seg_end = end if t_until is None else start + t_until
        avg_lp = float(np.mean(logprobs)) if logprobs else 0.0
        parts.append(text)
        segments.append(
            {
                "id": len(segments),
                "seek": int(seg_start * 100),
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": text,
                "tokens": tokens,
                "temperature": 0.0,
                "avg_logprob": round(avg_lp, 5),
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
        )
        if want_words:
            words_out.extend(_token_words(tokens, ts, start))

    for start, end, pcm in iter_audio(path):
        raw = engine.recognize_pcm(pcm)
        if pending is not None:
            p_start, p_end, p_raw, p_from = pending
            overlap = p_end - start
            if overlap > 0.05:
                p_until = (p_end - p_start) - overlap / 2.0
                cur_from = overlap / 2.0
            else:
                p_until = None
                cur_from = 0.0
            emit(p_start, p_end, p_raw, p_from, p_until)
            pending = (start, end, raw, cur_from)
        else:
            pending = (start, end, raw, 0.0)
    if pending is not None:
        p_start, p_end, p_raw, p_from = pending
        emit(p_start, p_end, p_raw, p_from, None)

    elapsed = time.perf_counter() - t0
    duration = last_end
    rtf = (elapsed / duration) if duration > 0 else 0.0
    out = {
        "task": "transcribe",
        "text": " ".join(parts).strip(),
        "language": language,
        "duration": round(duration, 3),
        "processing_time": round(elapsed, 3),
        "realtime_factor": round(rtf, 4),
        "segments": segments,
        "model": MODEL_ID,
    }
    if want_words:
        out["words"] = words_out
    return out


def _srt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _vtt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _to_srt(result: dict) -> str:
    lines: list[str] = []
    for i, seg in enumerate(result.get("segments") or [], 1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(seg['start'])} --> {_srt_ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def _to_vtt(result: dict) -> str:
    lines = ["WEBVTT", ""]
    for seg in result.get("segments") or []:
        lines.append(f"{_vtt_ts(seg['start'])} --> {_vtt_ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


async def _idle_watcher() -> None:
    global engine
    interval = min(5.0, max(1.0, SLEEP_IDLE_SECONDS / 4.0)) if SLEEP_IDLE_SECONDS > 0 else 5.0
    while True:
        try:
            await asyncio.sleep(interval)
            if SLEEP_IDLE_SECONDS <= 0 or engine is None:
                continue
            if engine.try_sleep(SLEEP_IDLE_SECONDS):
                LOG.info("GPU em idle sleep (próxima request recarrega o modelo)")
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("idle watcher")


def _touch_busy() -> None:
    global _inflight, _last_used
    _inflight += 1
    _last_used = time.monotonic()


def _touch_idle() -> None:
    global _inflight, _last_used
    _inflight = max(0, _inflight - 1)
    _last_used = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, gpu_lock, _idle_task, _last_used
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _preload_ort_cuda()
    gpu_lock = asyncio.Semaphore(MAX_CONCURRENT)
    engine = AsrEngine()
    if LOAD_AT_STARTUP:
        await asyncio.to_thread(engine.load)
    _last_used = time.monotonic()
    if SLEEP_IDLE_SECONDS > 0:
        _idle_task = asyncio.create_task(_idle_watcher(), name="stt-idle-sleep")
        LOG.info("idle sleep ativo: %.0fs sem requests → unload GPU", SLEEP_IDLE_SECONDS)
    if API_KEY:
        LOG.info("API key ativa — transcrição exige Authorization: Bearer … ou X-API-Key")
    else:
        LOG.warning("API_KEY vazio — /v1/audio/transcriptions e /transcribe estão públicos")
    yield
    if _idle_task is not None:
        _idle_task.cancel()
        try:
            await _idle_task
        except asyncio.CancelledError:
            pass
    if engine is not None:
        await asyncio.to_thread(engine.unload)


app = FastAPI(
    title="Parakeet STT pt-BR",
    version="1.1.0",
    lifespan=lifespan,
)
if _env_flag("CORS_ENABLE", "1"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def _check_api_key(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    """Require API_KEY on transcription routes when the env var is set.

    Accepts ``Authorization: Bearer <key>`` (OpenAI SDK) or ``X-API-Key``.
    Health/ready stay unauthenticated for Docker probes.
    """
    if not API_KEY:
        return
    token = (x_api_key or "").strip()
    if not token and authorization:
        scheme, _, remainder = authorization.partition(" ")
        if scheme.lower() == "bearer":
            token = remainder.strip()
    if not token or not secrets.compare_digest(token, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="API key inválida",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if engine is None:
        raise HTTPException(503, "servidor inicializando")
    used = cuda_mem_used_mb()
    return {
        "status": "ready",
        "model_loaded": engine.loaded,
        "model_variant": _MODEL.key,
        "sleep_idle_seconds": SLEEP_IDLE_SECONDS,
        "providers": ort.get_available_providers(),
        "vram_used_mb": None if used is None else round(used, 1),
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
    }


async def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio.bin").suffix or ".bin"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    written = 0
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await file.read(8 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"arquivo excede {MAX_UPLOAD_MB:g} MB")
                f.write(chunk)
        if dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "arquivo vazio")
        return dest
    except HTTPException:
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"falha ao gravar upload: {e}") from e


def _parse_granularities(raw: list[str] | str | None) -> set[str]:
    if raw is None:
        return {"segment"}
    if isinstance(raw, str):
        items = [raw]
    else:
        items = list(raw)
    out: set[str] = set()
    for item in items:
        for part in str(item).split(","):
            g = part.strip().lower()
            if g:
                out.add(g)
    return out or {"segment"}


@app.post("/v1/audio/transcriptions")
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
    timestamp_granularities: str | None = Form(None),
    _: None = Depends(_check_api_key),
):
    del model, prompt, temperature  # accepted for OpenAI clients; unused
    # `language` is metadata only. TDT has no language-id input (unlike Whisper).
    # pt-BR / pt-BR-INT8 decode as Portuguese; multilanguage auto-picks among
    # the 25 European languages of NVIDIA Parakeet TDT 0.6B v3.
    lang = language.strip() if isinstance(language, str) and language.strip() else None
    if gpu_lock is None or engine is None:
        raise HTTPException(503, "servidor inicializando")
    fmt = (response_format or "json").lower()
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(400, f"response_format inválido: {fmt}")
    grains = _parse_granularities(timestamp_granularities)
    dest = await _save_upload(file)
    _touch_busy()
    try:
        async with gpu_lock:
            result = await asyncio.to_thread(
                transcribe_file, dest, "word" in grains, lang
            )
        if fmt == "text":
            return PlainTextResponse(result["text"])
        if fmt == "srt":
            return Response(_to_srt(result), media_type="text/plain; charset=utf-8")
        if fmt == "vtt":
            return Response(_to_vtt(result), media_type="text/vtt; charset=utf-8")
        if fmt == "verbose_json":
            return JSONResponse(result)
        return JSONResponse({"text": result["text"]})
    except RuntimeError as e:
        raise HTTPException(422, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        LOG.exception("transcribe failed")
        raise HTTPException(500, f"erro interno: {e}") from e
    finally:
        _touch_idle()
        dest.unlink(missing_ok=True)
