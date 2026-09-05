#!/usr/bin/env python3
"""Parakeet TDT 0.6B v3 pt-BR (ONNX INT8) — STT production server."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx_asr
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

LOG = logging.getLogger("stt")

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/models/parakeet")
MODEL_ARCH = os.environ.get("MODEL_ARCH", "nemo-parakeet-tdt-0.6b-v3")
QUANTIZATION = os.environ.get("QUANTIZATION", "int8")
LANGUAGE = os.environ.get("LANGUAGE", "pt-BR")
SR = 16000
MAX_CHUNK_S = float(os.environ.get("MAX_CHUNK_S", "25"))
MIN_CHUNK_S = float(os.environ.get("MIN_CHUNK_S", "0.25"))
GPU_MEM_LIMIT_GB = float(os.environ.get("GPU_MEM_LIMIT_GB", "4"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/stt"))

engine = None
gpu_lock: asyncio.Semaphore | None = None


def _providers() -> list:
    mem = int(GPU_MEM_LIMIT_GB * 1024 * 1024 * 1024)
    cuda_opts = {
        "device_id": int(os.environ.get("GPU_ID", "0")),
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": mem,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
        "cudnn_conv_use_max_workspace": "1",
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
        self.model = None

    def load(self) -> None:
        t0 = time.perf_counter()
        self.model = onnx_asr.load_model(
            MODEL_ARCH,
            MODEL_DIR,
            quantization=QUANTIZATION,
            sess_options=_session_options(),
            providers=_providers(),
            cpu_preprocessing=True,
        )
        _ = self.recognize_pcm(np.zeros(SR, dtype=np.float32))
        LOG.info("modelo pronto em %.1fs | %s", time.perf_counter() - t0, MODEL_DIR)

    def recognize_pcm(self, pcm: np.ndarray) -> str:
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32, copy=False)
        if pcm.ndim > 1:
            pcm = pcm.reshape(-1)
        text = self.model.recognize(pcm)
        if isinstance(text, (list, tuple)):
            text = " ".join(str(x) for x in text)
        return (text or "").strip()


def ffmpeg_pcm_proc(src: Path) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-analyzeduration", "200M", "-probesize", "200M",
        "-i", str(src),
        "-map", "0:a:0?",
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
    assert proc.stdout is not None
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

    def emit(end_samples: int, arr: np.ndarray):
        if arr.size < min_n:
            return None
        start = max(0.0, utt_start - pad_ms / 1000.0)
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
                utt_start = t_samples / SR
                buf = [pcm]
                buf_n = n
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
                    cut = _cut_at_silence(arr, SR)
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
            t_samples += n

        if in_speech and buf_n >= min_n:
            arr = np.concatenate(buf)
            item = emit(t_samples, arr)
            if item:
                emitted = True
                yield item

        if not emitted and t_samples >= min_n:
            proc.wait(timeout=5)
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


def _cut_at_silence(arr: np.ndarray, sr: int, lookback_s: float = 2.0) -> int:
    win = int(sr * 0.03)
    look = min(arr.size, int(sr * lookback_s))
    region = arr[-look:]
    if region.size < win * 4:
        return arr.size
    hop = win
    best_i, best_rms = region.size, 1.0
    for i in range(0, region.size - win, hop):
        sl = region[i : i + win]
        r = float(np.sqrt(np.mean(sl * sl) + 1e-12))
        if r < best_rms:
            best_rms, best_i = r, i
    cut = arr.size - look + best_i
    return cut if cut > int(sr * MIN_CHUNK_S) else arr.size


def _naive_chunks(src: Path, max_s: float) -> Iterator[tuple[float, float, np.ndarray]]:
    proc = ffmpeg_pcm_proc(src)
    assert proc.stdout is not None
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
        proc.communicate(timeout=8)


def transcribe_file(path: Path) -> dict:
    t0 = time.perf_counter()
    parts: list[str] = []
    segments: list[dict] = []
    last_end = 0.0
    for start, end, pcm in iter_utterances(path):
        text = engine.recognize_pcm(pcm)
        last_end = end
        if not text:
            continue
        parts.append(text)
        segments.append(
            {"id": len(segments), "start": round(start, 3), "end": round(end, 3), "text": text}
        )
    elapsed = time.perf_counter() - t0
    duration = last_end
    rtf = (elapsed / duration) if duration > 0 else 0.0
    return {
        "text": " ".join(parts).strip(),
        "language": LANGUAGE,
        "duration": round(duration, 3),
        "processing_time": round(elapsed, 3),
        "realtime_factor": round(rtf, 4),
        "segments": segments,
        "model": "parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, gpu_lock
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    gpu_lock = asyncio.Semaphore(MAX_CONCURRENT)
    engine = AsrEngine()
    engine.load()
    yield


app = FastAPI(
    title="Parakeet STT pt-BR",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if engine is None or engine.model is None:
        raise HTTPException(503, "modelo não carregado")
    return {"status": "ready", "providers": ort.get_available_providers()}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": "parakeet-tdt-0.6b-v3-ptBR", "object": "model", "owned_by": "local"}],
    }


async def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio.bin").suffix or ".bin"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await file.read(8 << 20)
                if not chunk:
                    break
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


@app.post("/v1/audio/transcriptions")
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    if gpu_lock is None:
        raise HTTPException(503, "servidor inicializando")
    dest = await _save_upload(file)
    try:
        async with gpu_lock:
            result = await asyncio.to_thread(transcribe_file, dest)
        fmt = (response_format or "json").lower()
        if fmt == "text":
            return PlainTextResponse(result["text"])
        if fmt == "verbose_json":
            return JSONResponse(result)
        return JSONResponse({"text": result["text"], "duration": result["duration"]})
    except RuntimeError as e:
        raise HTTPException(422, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        LOG.exception("transcribe failed")
        raise HTTPException(500, f"erro interno: {e}") from e
    finally:
        dest.unlink(missing_ok=True)
