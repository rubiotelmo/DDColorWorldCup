FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="" \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.0+cpu
RUN pip install --no-cache-dir fastapi==0.111.0 uvicorn[standard]==0.30.1 python-multipart==0.0.9 numpy==1.26.4 opencv-python-headless==4.10.0.84 transformers==4.41.2

COPY app.py best_wcddcolor_team_regularized_tiny.pt ./
COPY wcddcolor ./wcddcolor
COPY basicsr ./basicsr

RUN python - <<'PY'
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
PY

ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
