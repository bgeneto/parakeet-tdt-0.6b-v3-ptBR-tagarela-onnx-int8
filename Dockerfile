# Parakeet TDT 0.6B v3 pt-BR ONNX INT8 — produção, RTX 3090 (sm_86)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    ORT_INTRA_THREADS=4 \
    MODEL_DIR=/opt/models/parakeet \
    UPLOAD_DIR=/tmp/stt \
    GPU_MEM_LIMIT_GB=4 \
    MAX_CHUNK_S=25 \
    MAX_CONCURRENT=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .
# onnx-asr puxa onnxruntime CPU; forçamos só o GPU wheel
RUN pip install -r requirements.txt \
    && pip uninstall -y onnxruntime || true \
    && pip install --force-reinstall --no-deps onnxruntime-gpu>=1.20.0,<1.23.0 \
    && python -c "import onnxruntime as o; print(o.get_available_providers())"

# Modelo imutável na imagem (air-gap friendly)
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "calneymgp/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx-int8",
    local_dir="/opt/models/parakeet",
    local_dir_use_symlinks=False,
)
PY

COPY app.py /app/app.py

RUN useradd --system --uid 1000 --create-home stt \
    && mkdir -p /tmp/stt \
    && chown -R stt:stt /app /opt/models /tmp/stt /opt/venv

USER stt
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=8s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"

STOPSIGNAL SIGINT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--loop", "uvloop", "--http", "httptools", \
     "--timeout-keep-alive", "30", "--access-log"]
