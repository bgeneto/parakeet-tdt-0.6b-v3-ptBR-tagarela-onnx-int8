# syntax=docker/dockerfile:1
# Layer order: rare → frequent. pip only rebuilds when requirements.txt changes.
#
# Final image is nvidia/cuda:*-base (cudart only). cuBLAS / cuFFT / cuRAND / NVRTC /
# cuDNN are copied from the cudnn-runtime image. NPP, NCCL, cuSOLVER, cuSPARSE,
# nvJPEG and cuFile are omitted — libonnxruntime_providers_cuda.so does not link them.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cuda-libs
RUN mkdir -p /opt/cuda-slim/lib64 /opt/cuda-slim/cudnn \
    && cp -a /usr/local/cuda/lib64/libcublas.so.12* /opt/cuda-slim/lib64/ \
    && cp -a /usr/local/cuda/lib64/libcublasLt.so.12* /opt/cuda-slim/lib64/ \
    && cp -a /usr/local/cuda/lib64/libcufft.so.11* /opt/cuda-slim/lib64/ \
    && cp -a /usr/local/cuda/lib64/libcurand.so.10* /opt/cuda-slim/lib64/ \
    && cp -a /usr/local/cuda/lib64/libnvrtc.so.12* /opt/cuda-slim/lib64/ \
    && cp -a /usr/local/cuda/lib64/libnvrtc-builtins.so* /opt/cuda-slim/lib64/ \
    && cp -a /usr/local/cuda/lib64/libnvJitLink.so.12* /opt/cuda-slim/lib64/ \
    && cp -a /usr/local/cuda/lib64/libnvfatbin.so.12* /opt/cuda-slim/lib64/ \
    && cp -a /usr/lib/x86_64-linux-gnu/libcudnn*.so.9* /opt/cuda-slim/cudnn/

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Build/runtime invariants only. Tunables (SLEEP_IDLE_SECONDS, GPU_MEM_LIMIT_…)
# live at the bottom so editing them does not invalidate apt/pip/model layers.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

COPY --from=cuda-libs /opt/cuda-slim/lib64/ /usr/local/cuda/lib64/
COPY --from=cuda-libs /opt/cuda-slim/cudnn/ /usr/lib/x86_64-linux-gnu/
RUN ldconfig

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        ffmpeg ca-certificates \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && apt-get purge -y python3-pip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 1000 --create-home stt \
    && mkdir -p /app /opt/models /tmp/stt \
    && chown stt:stt /app /tmp/stt

WORKDIR /app

# --- Python deps: checksum of this file is the only pip cache key ---
COPY requirements.txt .
# Cache mount: image stays cache-free; rebuilds of *this* layer reuse wheels.
# Quote the GPU spec: unquoted > / < are shell redirects.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y onnxruntime || true \
    && pip install --no-cache-dir --force-reinstall --no-deps "onnxruntime-gpu>=1.22.0,<1.27.0" \
    && python -c "\
import glob, os, subprocess, sys
import onnxruntime as o
getattr(o, 'preload_dlls', lambda **k: None)()
p = o.get_available_providers()
print(p)
assert 'CUDAExecutionProvider' in p, p
sos = glob.glob('/opt/venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so')
assert sos, 'cuda EP .so missing'
out = subprocess.check_output(['ldd', sos[0]], text=True)
missing = [ln for ln in out.splitlines() if 'not found' in ln]
if missing:
    sys.stderr.write('\n'.join(missing) + '\n')
    raise SystemExit('CUDA EP has unresolved libraries')
" \
    && find /opt/venv -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/venv -type f \( -name '*.pyi' -o -name '*.pyc' \) -delete \
    && chown -R stt:stt /opt/venv

# --- Model: independent of app.py; Hub blob cache is a mount, not an image layer ---
RUN --mount=type=cache,target=/root/.cache/huggingface \
    HF_HOME=/root/.cache/huggingface python -c "\
from huggingface_hub import snapshot_download;\
snapshot_download(\
    repo_id='calneymgp/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8',\
    revision='7d84392553633a8e5bdca7eccb5ae25467e9572f',\
    local_dir='/opt/models/parakeet',\
    ignore_patterns=['*.md', '.gitattributes'],\
)" \
    && chown -R stt:stt /opt/models

# --- App code: this is the only layer that changes on typical edits ---
COPY --chown=stt:stt app.py /app/app.py

# Runtime tunables (override with compose/.env / docker run -e). Do not bake API_KEY here.
ENV MODEL_DIR=/opt/models/parakeet \
    MODEL_ARCH=nemo-conformer-tdt \
    QUANTIZATION=int8 \
    LANGUAGE=pt-BR \
    MODEL_ID=parakeet-tdt-0.6b-v3-ptBR \
    UPLOAD_DIR=/tmp/stt \
    GPU_ID=0 \
    GPU_MEM_LIMIT_GB=6 \
    CUDNN_CONV_ALGO_SEARCH=HEURISTIC \
    CUDNN_CONV_MAX_WORKSPACE=1 \
    ORT_ARENA_EXTEND=kNextPowerOfTwo \
    PREPROCESS_ON_GPU=1 \
    CHUNKING=window \
    MAX_CHUNK_S=30 \
    CHUNK_OVERLAP_S=1.0 \
    CHUNK_CONTEXT_S=0.5 \
    CHUNK_LOOKBACK_S=2.0 \
    MIN_CHUNK_S=0.5 \
    MAX_UPLOAD_MB=512 \
    MAX_CONCURRENT=1 \
    SLEEP_IDLE_SECONDS=60 \
    LOAD_AT_STARTUP=1 \
    CUDA_DEVICE_RESET=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    ORT_INTRA_THREADS=4 \
    ORT_INTER_THREADS=2 \
    HOST=0.0.0.0 \
    PORT=8080 \
    LOG_LEVEL=INFO \
    CORS_ENABLE=1 \
    CORS_ORIGINS=* \
    API_KEY=""

USER stt
EXPOSE 8080

# Liveness only — /ready may report model_loaded=false after idle sleep.
HEALTHCHECK --interval=30s --timeout=8s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8080'), timeout=5)"

STOPSIGNAL SIGINT
CMD ["sh", "-c", "exec uvicorn app:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8080}\" --workers 1 --loop uvloop --http httptools --timeout-keep-alive 30 --access-log"]
