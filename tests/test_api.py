import io
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

import app as app_module


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_detect_returns_detections(client):
    img = Image.new("RGB", (640, 480), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post("/detect", files={"file": ("test.png", buffer, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert "num_detections" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_detect_with_jpeg(client):
    img = Image.new("RGB", (640, 480), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/detect", files={"file": ("test.jpg", buffer, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "num_detections" in data
    assert "detections" in data


def test_detect_missing_file(client):
    response = client.post("/detect")
    assert response.status_code == 422


def test_detect_response_structure(client):
    img = Image.new("RGB", (100, 100), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post("/detect", files={"file": ("test.png", buffer, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["num_detections"], int)
    assert data["num_detections"] >= 0
    assert data["num_detections"] == len(data["detections"])

    for detection in data["detections"]:
        assert "class_id" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert "bbox_xyxy" in detection
        assert isinstance(detection["class_id"], int)
        assert isinstance(detection["class_name"], str)
        assert 0 <= detection["confidence"] <= 1
        assert len(detection["bbox_xyxy"]) == 4


def test_detect_model_not_loaded(client):
    """Test that detect raises error when model is not loaded."""
    original_model = app_module.model
    app_module.model = None

    try:
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            client.post("/detect", files={"file": ("test.png", buffer, "image/png")})
    finally:
        app_module.model = original_model


def test_detect_with_detections(client):
    """Test detection with mocked YOLO results containing detections."""
    original_model = app_module.model

    # Create mock model
    mock_model = MagicMock()
    mock_model.names = {0: "person", 1: "car"}

    # Create mock box
    mock_box = MagicMock()
    mock_box.cls = torch.tensor([0])
    mock_box.conf = torch.tensor([0.95])
    mock_box.xyxy = torch.tensor([[100.0, 150.0, 300.0, 400.0]])

    # Create mock result
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model.return_value = [mock_result]
    app_module.model = mock_model

    try:
        img = Image.new("RGB", (640, 480), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.post(
            "/detect", files={"file": ("test.png", buffer, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["num_detections"] == 1
        assert len(data["detections"]) == 1
        assert data["detections"][0]["class_id"] == 0
        assert data["detections"][0]["class_name"] == "person"
        assert data["detections"][0]["confidence"] == pytest.approx(0.95)
        assert data["detections"][0]["bbox_xyxy"] == pytest.approx(
            [100.0, 150.0, 300.0, 400.0]
        )
    finally:
        app_module.model = original_model
