import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger("uvicorn.error")

model: YOLO | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    logger.info("Model loaded successfully!")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Reachy Vision API", lifespan=lifespan)


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        raise RuntimeError("Model not loaded")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append(
                {
                    "class_id": int(box.cls),
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox_xyxy": [float(x) for x in box.xyxy[0]],
                }
            )

    return {"num_detections": len(detections), "detections": detections}
