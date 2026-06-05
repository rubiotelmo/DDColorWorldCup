# WC-DDColor Team Deploy

Self-contained Docker bundle for serving the team-conditioned World Cup DDColor model with FastAPI on CPU.

The directory contains the API code, the model checkpoint, and the minimal local source files required by the model. You can zip this directory and share it as a standalone deploy bundle.

## Prerequisites

- Docker
- Network access while building the image for Python packages and the CLIP text model cache
- No network is required at container runtime

## Build

Run from this directory:

```bash
docker build -t wcddcolor-team-api .
```

## Run

```bash
docker run --rm -p 8000:8000 wcddcolor-team-api
```

## Test

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/colorize \
  -F "image=@/path/to/image.jpg" \
  -F "team1=Brazil" \
  -F "team2=Germany" \
  --output colorized.png
```
