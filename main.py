# Main App

import json
from typing import Optional

from fastapi import FastAPI, File , UploadFile , Form, Depends , HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from inference import ShelfInferenceEngine
from database import(
    create_tables, get_db,
    save_scan, get_scan_history, get_analytics_summary,
    ScanRecord
)


app = FastAPI(
    title = "Retail Shelf Intelligence API",
    description = "Real-time shelf analysis: empty slot detection, misplaced items, stock level tracking.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

engine_instance: Optional[ShelfInferenceEngine] = None

@app.on_event("startup")
async def startup():
    global engine_instance
    create_tables()
    engine_instance = ShelfInferenceEngine(model_path="models/best/pt")
    print("API server ready")


class DetectionOut(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]
    is_alert:bool


class ShelfAnalysisOut(BaseModel):
    scan_id: int
    total_slots: int
    empty_slots: int
    misplaced_items: int
    products_detected: int
    stock_level_pct: float
    alerts: list[str]
    detections: list[DetectionOut]
    annotated_image_b64: str
    inference_ms: float
    timestamp: str


class ScanHistoryItem(BaseModel):
    id: int
    store_id: Optional[str]
    aisle: Optional[str]
    stock_level_pct: float
    empty_slots: int
    misplaced_items: int
    has_alerts: bool
    inference_ms: float
    created_at: str

    class Config:
        from_attributes = True


class AnalyticsSummary(BaseModel):
    total_scans: int
    avg_stock_level_pct: float=0
    avg_empty_slots: float=0
    avg_misplaced: float=0
    total_alerts: int=0


@app.get("/health",tags=['System'])
async def health_check():
    return {"status":"ok", "model_loaded":engine_instance is not None}

@app.post("/analyze",response_model=ShelfAnalysisOut, tags=["Detection"])
async def analyze_shlef(
    file : UploadFile = File(... , description="Shelf image(IPEG/PNG)"),
    store_id: Optional[str] = Form(None, description="Store identifier"),
    aisle: Optional[str] = Form(None, descriptions="Aisle label e.g. 'A3'"),
    conf_threshold: float = Form(0.35, ge=0.1, le=0.95),
    db: Session = Depends(get_db),
):
    """
    Main endpoint. You upload a shelf image -> get detections, alerts and annotated image.
    Returns base64-encoded annotated image for display in the Flutter app.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail = "Only JPEG, PNG and WEBP images are supported.",
        )
    
    image_bytes = await file.read()

    # Inference
    analysis = engine_instance.analyze(image_bytes, conf_threshold=conf_threshold)

    # Persist to DB
    record = save_scan(db, analysis, store_id=store_id, aisle=aisle)

    # Build response
    detections_out = [
        DetectionOut(
            class_name = d.class_name,
            confidence=round(d.confidence,4),
            bbox=d.bbox,
            is_alert=d.is_alert
        )
        for d in analysis.detections
    ]

    return ShelfAnalysisOut(
        scan_id=record.id,
        total_slots=analysis.total_slots,
        empty_slots=analysis.empty_slots,
        misplaced_items=analysis.misplaced_items,
        products_detected=analysis.products_detected,
        stock_level_pct=analysis.stock_level_pct,
        alerts=analysis.alerts,
        detections=detections_out,
        annotated_image_b64=analysis.annotated_image_b64,
        inference_ms=analysis.inference_ms,
        timestamp=analysis.timestamp,
    )

@app.get("/history", response_model=list[ScanHistoryItem],tags=["History"])
async def get_history(
    limit: int = 50,
    store_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    # Retrieve recent scan history (shown in Flutter history screen)
    records = get_scan_history(db,limit=limit, store_id=store_id)
    return [
        ScanHistoryItem(
            id=r.id,
            store_id=r.store_id,
            aisle=r.aisle,
            stock_level_pct=r.stock_level_pct,
            empty_slots=r.empty_slots,
            misplaced_items=r.misplaced_items,
            has_alerts=r.has_alerts,
            inference_ms=r.inference_ms,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]

@app.get("/history/{scan_id}",tags=["History"])
async def get_scan_detail(scan_id: int, db:Session= Depends(get_db)):
    # Get full details for a single scan
    record = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Scan not Found")
    
    return {
        "id": record.id,
        "store_id": record.store_id,
        "aisle": record.aisle,
        "stock_level_pct": record.stock_level_pct,
        "empty_slots": record.empty_slots,
        "misplaced_items": record.misplaced_items,
        "has_alerts": record.has_alerts,
        "alerts": json.loads(record.alerts_json or "[]"),
        "detections": json.loads(record.detections_json or "[]"),
        "annotated_image_b64": record.annotated_image,
        "inference_ms": record.inference_ms,
        "created_at": record.created_at.isoformat(),
    }

@app.get("/analytics",response_model=AnalyticsSummary, tags=["Analytics"])
async def gett_analytics(
    store_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    # Aggregate analytics for the dashboard screen
    summary = get_analytics_summary(db, store_id=store_id)
    return AnalyticsSummary(**summary)

@app.delete("/history/{scan_id}", tags=["History"])
async def delete_scan(scan_id:int, db:Session=Depends(get_db)):
    # Delete a scan record
    record = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()
    if not record:
        raise HTTPException(status_code=404, details="Scan Not Found")
    db.delete(record)
    db.commit()
    return {"message":f"Scan {scan_id} deleted..."}