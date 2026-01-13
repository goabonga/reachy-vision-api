---
title: Reachy Vision API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Reachy Vision API

**FastAPI-based YOLOv8 vision API for Reachy robots.**

Reachy Vision API is a lightweight object detection service built with **FastAPI** and **YOLOv8**, designed to run on **Hugging Face Spaces (Docker)** or locally, and to be easily integrated with **Reachy robots** or any backend.

---

## ✨ Features

- 🔍 Object detection powered by **YOLOv8**
- ⚡ FastAPI HTTP API (simple & stateless)
- 🐳 Hugging Face **Docker Space** compatible
- 🧠 CPU-friendly (`yolov8n` by default)
- 🤖 Ready to integrate with **Reachy Mini**
- 📦 Dependency management with **uv**

---

## 📡 API Endpoints

### Health check
```
GET /health
```

Response:
```json
{ "status": "ok" }
```

---

### Object detection
```
POST /detect
```

**Request**
- `multipart/form-data`
- Field: `file` (image)

**Example**
```bash
curl -X POST \
  -F "file=@image.jpg" \
  http://localhost:7860/detect
```

**Response**
```json
{
  "num_detections": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.92,
      "bbox_xyxy": [120.3, 45.1, 380.7, 620.9]
    }
  ]
}
```

---

## 🚀 Deployment (Hugging Face Space)

Recommended setup:

- **Space type**: `Docker`
- **Hardware**: CPU (default) or GPU
- **Exposed port**: `7860`

### Repository structure

```
reachy-vision-api/
├── app.py              # FastAPI application
├── Dockerfile          # Docker image definition
├── pyproject.toml      # Project configuration and dependencies
├── uv.lock             # Lockfile for reproducible builds
├── .gitignore          # Git ignore rules
├── tests/              # Test suite
│   ├── __init__.py
│   ├── conftest.py     # Pytest fixtures
│   └── test_api.py     # API tests
└── README.md
```

Once pushed, the Space will automatically build and expose:
```
https://<username>-<space-name>.hf.space
```

---

## 🐳 Docker (local run)

```bash
docker build -t reachy-vision-api .
docker run -p 7860:7860 reachy-vision-api
```

---

## 📦 Dependencies

Dependencies are managed using **uv**.

Main dependencies:
- `fastapi`
- `uvicorn`
- `ultralytics`
- `pillow`
- `python-multipart`

The lockfile (`uv.lock`) ensures reproducible builds.

---

## 🛠️ Development

Install dev dependencies:
```bash
uv sync --extra dev
```

### Tools

- **ruff** - Linter and formatter
- **mypy** - Static type checker
- **pytest** - Testing framework
- **pytest-cov** - Code coverage

### Run tests
```bash
uv run pytest
```

Coverage report is generated in `htmlcov/` and displayed in terminal.

### Lint and format
```bash
uv run ruff check .
uv run ruff format .
```

### Type checking
```bash
uv run mypy app.py
```

### Release workflow

This project uses [commitizen](https://commitizen-tools.github.io/commitizen/) for versioning and changelog generation.

To trigger a new release, push a commit to `main` with the message `chore: release a new version`:

```bash
git commit --allow-empty -m "chore: release a new version"
git push origin main
```

This will:
1. Bump the version based on conventional commits
2. Generate/update the CHANGELOG
3. Create a GitHub Release
4. Sync to Hugging Face Space

---

## 🤖 Usage with Reachy

This API is designed to be called from:
- Reachy Mini
- A central VPS backend
- Another Hugging Face Space

Typical flow:
1. Capture image from Reachy camera
2. Send image to `/detect`
3. Use detections for interaction, navigation, or reasoning

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.