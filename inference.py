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

    def _load_model(self):
        # Load YOLO else Fall back if not found.
        path = Path(self.model_path)
        if not path.exists():
            print(f" Model not found at {path}. Using pretrained YOLOv8s for demo")
            self.model=YOLO("yolov8s.pt")
        else:
            self.model=YOLO(str(path))
            print(f"Model loaded from {path}")

    def _images_from_bytes(self,image_bytes:bytes)-> np.ndarray:
        # converts raw bytes to opencv BGR array
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_BAYER_BG2BGR)
    
    def _annonate_image(self,img:np.ndarray,detections:list[Detection]) -> np.ndarray:
        h,w = img.shape[:2]
        annotated = img.copy()

        for det in detections:
            x1, y1, x2, y2 = [
                int(det.bbox[0]*w),int(det.bbox[1]*h),
                int(det.bbox[2]*w),int(det.bbox[3]*h)
            ]
            color = CLASS_COLORS.get(det.class_name, (255,255,255))
            thickness = 3 if det.is_alert else 2

            # Box
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,thickness)

            # Background Label
            label = f"{det.class_name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.55,1)
            cv2.rectangle(annotated, (x1,y1-th-8), (x1+tw+6,y1),color, -1)

            # Label text
            cv2.putText(
                annotated, label, (x1+3,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,25,255),1, cv2.LINE_AA
            )

        return annotated
    
    def _image_to_b64(self,img:np.ndarray) -> str:
        # Encode OpenCV image to base64 JPEG string
        _ , buffer = cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,88])
        return base64.b64encode(buffer).decode("utf-8")
    
    def _build_alerts(self,detections:list[Detection],analysis:dict) -> list[str]:
        # Generated human readable messages
        alerts = []
        if analysis['empty_slots'] > 0:
            alerts.append(
                f" {analysis['empty_slots']} empty solt(s) detected - restock required"
            )
        if analysis['misplaced_items'] > 0:
            alerts.append(
                f"{analysis['misplaced_items']} misplaced item(s) - planogram violation"
            )
        if analysis['stock_level_pct'] < 30:
            alerts.append(
                f"Critical stock level : {analysis['stock_level_pct']:.0f}% - urgent restock"
            )
        if analysis['stock_level_pct'] < 50:
            alerts.append(
                f"Low stock level: {analysis['stock_level_pct']:.0f}% - schedule restock"
            )
        if not alerts:
            alerts.append(" Shelf is fully stocked and compliant")
        return alerts
    
    def analyze(
            self,
            image_bytes: bytes,
            conf_threshold: float=0.35,
            iou_threshold: float=0.45,
    ) -> ShelAnalysis:
        # Like a main entry point. Takes raw images bytes , returns full shelf analysis
        from datetime import datetime,timezone

        t0 = time.perf_counter()

        # Decode image
        img = self._images_from_bytes(image_bytes)
        h,w = img.shape[:2]

        # Run inference
        results =self.model.predict(
            source = img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )[0]

        # Parse Detection
        detections : list[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            x1,y1,x2,y2 = box.xyxyn[0].tolist()

            detections.append(Detection(
                class_id = cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=[x1,y1,x2,y2],
                is_alerts=cls_name in ALERT_CLASSES,
            ))

        # Analytics
        empty    = sum(1 for d in detections if d.class_name == "empty_slot")
        misplace = sum(1 for d in detections if d.class_name == "misplaced")
        products = sum(1 for d in detections if d.class_name == "product")
        
        total = empty + products + misplace
        stock_pct = (products/total*100) if total > 0 else 100.0

        analysis_data = {
            "empty_slots": empty,
            "misplaced_items":misplace,
            "products_detected":products,
            "stock_level_pts":round(stock_pct,1),
        }

        alerts = self._build_alerts(detections, analysis_data)


        # Annonate image
        annotated = self._annonate_image(img,detections)
        img_b64 = self._image_to_b64(annotated)

        inference_ms = round((time.perf_counter()-t0)*1000,1)

        return ShelAnalysis(
            total_slots=total,
            empty_slots=empty,
            misplaced_items=misplace,
            products_items=products,
            stock_level_pct=round(stock_pct,1),
            alerts=alerts,
            detections=detections,
            annotated_image_b64=img_b64,
            inference_ms=inference_ms,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
