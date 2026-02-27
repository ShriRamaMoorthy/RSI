import io
import time
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# let's define class names
CLASS_NAMES = {
    0: "product",
    1: "empty_slot",
    2: "misplaced",
    3: "price_tag",
}

CLASS_COLORS = {
    "product": (34,197,94),
    "empty_slot": (239,68,68),
    "misplaced": (234,179,8),
    "price_tag": (99,102,241),
}

ALERT_CLASSES = {"empty_slot","misplaced"}

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]   # [x1,y1,x2,y2] normalized 0-1
    is_alert: bool

@dataclass
class ShelAnalysis:
    total_slots: int
    empty_slots: int
    misplaced_items: int
    products_items: int
    stock_level_pct: float
    alerts: list[str]
    detections: list[Detection]
    annotated_image_b64: str
    inference_ms: float
    timestamp: str


class ShelfInferenceEngine:
    def __init__(self,model_path:str="models/best.pt"):
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self._load_model()
        