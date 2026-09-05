#!/usr/bin/env python3
"""Parakeet TDT 0.6B v3 pt-BR (ONNX INT8) — STT production server."""
from __future__ import annotations

import asyncio
import ctypes
import gc
import logging
import os
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

LOG = logging.getLogger("stt")

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/models/parakeet")
# Generic TDT type matches this repo's config.json (not the NVIDIA Hub id).
MODEL_ARCH = os.environ.get("MODEL_ARCH", "nemo-conformer-tdt")
QUANTIZATION = os.environ.get("QUANTIZATION", "int8")
LANGUAGE = os.environ.get("LANGUAGE", "pt-BR")
MODEL_ID = os.environ.get("MODEL_ID", "parakeet-tdt-0.6b-v3-ptBR")
SR = 16000
MAX_CHUNK_S = float(os.environ.get("MAX_CHUNK_S", "25"))
MIN_CHUNK_S = float(os.environ.get("MIN_CHUNK_S", "0.25"))
GPU_ID = int(os.environ.get("GPU_ID", "0"))
GPU_MEM_LIMIT_GB = float(os.environ.get("GPU_MEM_LIMIT_GB", "4"))
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
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": mem,
        # HEURISTIC: wake-from-idle must not pay EXHAUSTIVE conv search every time.
        "cudnn_conv_algo_search": os.environ.get("CUDNN_CONV_ALGO_SEARCH", "HEURISTIC"),
        "do_copy_in_default_stream": True,
        "cudnn_conv_use_max_workspace": os.environ.get("CUDNN_CONV_MAX_WORKSPACE", "0"),
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
            if pcm.dtype != np.float32:
                pcm = pcm.astype(np.float32, copy=False)
            if pcm.ndim > 1:
                pcm = pcm.reshape(-1)
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
        quant = QUANTIZATION.strip() or None
        self.model = onnx_asr.load_model(
            MODEL_ARCH,
            MODEL_DIR,
            quantization=quant,
            sess_options=_session_options(),
            providers=providers,
            preprocessor_config={
                "providers": cpu_ep,
                "use_numpy_preprocessors": True,
                "max_concurrent_workers": 1,
            },
            resampler_config={"providers": cpu_ep},
        ).with_timestamps()
        _ = self.model.recognize(np.zeros(SR, dtype=np.float32), sample_rate=SR)
        used = cuda_mem_used_mb()
        extra = f" | VRAM ~{used:.0f} MiB" if used is not None else ""
        LOG.info("modelo pronto em %.1fs | %s | %s%s", time.perf_counter() - t0, MODEL_DIR, quant, extra)

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


def iter_utterances(
    src: Path,
    max_s: float = MAX_CHUNK_S,
    min_s: float = MIN_CHUNK_S,
    frame_ms: int = 30,
    hangover_ms: int = 400,
    pad_ms: int = 200,
    abs_thresh: float = 0.010,
) -> Iterator[tuple[float, float, np.ndarray]]:
    """Decode ffmpeg → 16 kHz mono e fatia por energia (VAD) sem carregar o arquivo inteiro."""
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
        start = max(0.0, utt_start)
        end = end_samples / SR
        return start, end, arr

    try:
        while True:
            raw = proc.stdout.read(bytes_frame)
            if not raw:
                break
            if len(raw) < 2:
                break
            n = len(raw) // 2
            pcm = np.frombuffer(raw, dtype=np.int16, count=n).astype(np.float32)
            pcm /= 32768.0
            rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12))
            noise = 0.995 * noise + 0.005 * rms
            thresh = max(abs_thresh, 3.2 * noise)
            is_sp = rms > thresh

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
                        arr = np.concatenate(buf)
                        item = emit(t_samples + n, arr)
                        if item:
                            emitted = True
                            yield item
                        in_speech = False
                        buf, buf_n = [], 0
                if buf_n >= max_n:
                    arr = np.concatenate(buf)
                    cut = _cut_at_silence(arr, SR, thresh)
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
            else:
                if pad_n > 0:
                    pre_roll = np.concatenate([pre_roll, pcm])[-pad_n:]
            t_samples += n

        if in_speech and buf_n >= min_n:
            arr = np.concatenate(buf)
            item = emit(t_samples, arr)
            if item:
                emitted = True
                yield item

        if not emitted and t_samples >= min_n:
            if proc.poll() is None:
                proc.kill()
            yield from _naive_chunks(src, max_s)
    finally:
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


def _cut_at_silence(arr: np.ndarray, sr: int, thresh: float, lookback_s: float = 2.0) -> int:
    win = int(sr * 0.03)
    look = min(arr.size, int(sr * lookback_s))
    region = arr[-look:]
    min_keep = int(sr * MIN_CHUNK_S)
    fallback = max(min_keep, int(arr.size * 0.85))
    if region.size < win * 4:
        return fallback
    hop = win
    best_i, best_rms = None, 1.0
    for i in range(0, region.size - win, hop):
        sl = region[i : i + win]
        r = float(np.sqrt(np.mean(sl * sl) + 1e-12))
        if r < best_rms:
            best_rms, best_i = r, i
    if best_i is None or best_rms > thresh:
        return fallback
    cut = arr.size - look + best_i
    return cut if cut > min_keep else fallback


def _naive_chunks(src: Path, max_s: float) -> Iterator[tuple[float, float, np.ndarray]]:
    proc = ffmpeg_pcm_proc(src)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout indisponível")
    step = int(SR * max_s) * 2
    t = 0.0
    try:
        while True:
            raw = proc.stdout.read(step)
            if not raw:
                break
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            dur = pcm.size / SR
            if dur < MIN_CHUNK_S:
                break
            yield t, t + dur, pcm
            t += dur
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


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


def transcribe_file(path: Path, want_words: bool = False) -> dict:
    if engine is None:
        raise RuntimeError("engine indisponível")
    t0 = time.perf_counter()
    parts: list[str] = []
    segments: list[dict] = []
    words_out: list[dict] = []
    last_end = 0.0
    for start, end, pcm in iter_utterances(path):
        result = engine.recognize_pcm(pcm)
        text = (getattr(result, "text", None) or str(result) or "").strip()
        ts = getattr(result, "timestamps", None)
        tokens = getattr(result, "tokens", None)
        logprobs = getattr(result, "logprobs", None)
        last_end = end
        if not text:
            continue
        if ts:
            seg_start = start + float(ts[0])
            seg_end = start + float(ts[-1])
        else:
            seg_start, seg_end = start, end
        avg_lp = float(np.mean(logprobs)) if logprobs else 0.0
        parts.append(text)
        segments.append(
            {
                "id": len(segments),
                "seek": int(seg_start * 100),
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": text,
                "tokens": tokens or [],
                "temperature": 0.0,
                "avg_logprob": round(avg_lp, 5),
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
        )
        if want_words:
            words_out.extend(_token_words(tokens, ts, start))
    elapsed = time.perf_counter() - t0
    duration = last_end
    rtf = (elapsed / duration) if duration > 0 else 0.0
    out = {
        "task": "transcribe",
        "text": " ".join(parts).strip(),
        "language": LANGUAGE,
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


async def _check_api_key(authorization: str | None = Header(None)) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(401, "API key inválida")


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
    del model, language, prompt, temperature  # accepted for OpenAI clients; unused
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
            result = await asyncio.to_thread(transcribe_file, dest, "word" in grains)
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
