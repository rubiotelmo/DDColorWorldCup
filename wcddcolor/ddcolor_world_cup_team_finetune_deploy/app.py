from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

from wcddcolor.model import WorldCupDDColor


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_wcddcolor_team_regularized_tiny.pt"
DEVICE = torch.device("cpu")
IMG_SIZE = 256
CHROMA_SCALE = 0.8

model = WorldCupDDColor(
    encoder_name="convnext-t",
    input_size=(IMG_SIZE, IMG_SIZE),
    num_output_channels=2,
    num_queries=100,
    last_norm="Spectral",
    do_normalize=False,
).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/colorize")
async def colorize(image: UploadFile = File(...), team1: str = Form(...), team2: str = Form(...)):
    data = np.frombuffer(await image.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return Response("invalid image", status_code=400)

    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    l_orig = lab[:, :, :1]

    rgb_256 = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    l_256 = cv2.cvtColor(rgb_256, cv2.COLOR_RGB2Lab)[:, :, :1]
    gray_lab = np.concatenate([l_256, np.zeros_like(l_256), np.zeros_like(l_256)], axis=2)
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_Lab2RGB)
    x = torch.from_numpy(gray_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    with torch.inference_mode():
        ab = model(x, teams=[(team1, team2)]) * CHROMA_SCALE
        ab = F.interpolate(ab, size=(height, width), mode="bilinear", align_corners=False)

    out_lab = np.concatenate([l_orig, ab[0].cpu().numpy().transpose(1, 2, 0)], axis=2).astype(np.float32)
    out_rgb = np.clip(cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB), 0.0, 1.0)
    out_bgr = cv2.cvtColor((out_rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", out_bgr)
    if not ok:
        return Response("could not encode output", status_code=500)
    return Response(encoded.tobytes(), media_type="image/png")
