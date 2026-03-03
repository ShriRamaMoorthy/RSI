import json
from datetime import datetime , timezone
from typing import Optional

from sqlalchemy import(
    Column , Integer , Float , String, Text,
    DateTime , Boolean , create_engine, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session


# Setup
DATABASE_URL = "sqlite:///./shelf_intelligence.db"

engine = create_engine(
    DATABASE_URL,
    connect_args = {"check_same_thread":False}
)

# Enable WAL mode for better concurrent read performance
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base = declarative_base()

# Models
class ScanRecord(Base):
    # Each shelf scan result stored here
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(String(64), index=True, nullable=True)
    aisle = Column(String(64),nullable=True)
    total_slots = Column(Integer)
    empty_slots = Column(Integer)
    misplaced_items = Column(Integer)
    products_detected = Column(Integer)
    stock_level_pct = Column(Float)
    inference_ms = Column(Float)
    has_alerts = Column(Boolean,default=False)
    alerts_json = Column(Text)
    detections_json = Column(Text)
    annotated_json = Column(Text)
    created_at = Column(DateTime, default=lambda:datetime.now(timezone.utc))


class AlertLog(Base):
    # Critical alerts logged separatley for dashboard view
    __tablename__ = "alerts_logs"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, index=True)
    store_id = Column(String(64), nullable=True)
    alert_type = Column(String(64))
    severity = Column(String(16))
    message = Column(Text)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    # FastAPI dependency - yield DB session
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



def save_scan(db:Session , analysis, store_id: str = None, aisle:str=None) -> ScanRecord:
    # Persist a Shelf Analysis result to the database
    detections_data = [
        {
            "class_name": d.class_name,
            "confidence": round(d.confidence, 4),
            "bbox": d.bbox,
            "is_alert": d.is_alert,
        }
        for d in analysis.detections
    ]

    record = ScanRecord(
        store_id = store_id,
        aisle = aisle,
        total_slots=analysis.total_slots,
        empty_slots=analysis.empty_slots,
        misplaced_items=analysis.products_detected,
        products_detected=analysis.products_detected,
        stock_level_pct=analysis.stock_level_pct,
        inference_ms=analysis.inference_ms,
        has_alerts=any(d.is_alert for d in analysis.detections),
        alerts_json=json.dumps(analysis.alerts),
        detections_json=json.dumps(detections_data),
        annotated_image=analysis.annotated_image_b64,
    )
    db.add(record)
    db.flush()

    for alert in analysis.alerts:
        if "🔴" in alert:
            severity = "critical"
        elif "🟡" in alert:
            severity = "warning"
        else:
            severity = "info"


        if severity in ("critical","warning"):
            alert_type = "empty_slot" if "empty" in alert.lower() else \
                        "misplaced" if "misplaced" in alert.lower() else "low_stock"
            
            log = AlertLog(
                scan_id=record.id,
                store_id=store_id,
                alert_type=alert_type,
                severity=severity,
                message=alert,
                )
            db.add(log)

    db.commit()
    db.refresh(record)
    return record


def get_scan_history(db: Session, limit:int=50, store_id:str=None):
    # Retrieve recent scan history
    q = db.query(ScanRecord).order_by(ScanRecord.created_at.desc())
    if store_id:
        q = q.filter(ScanRecord.store_id==store_id)
    return q.limit(limit).all()


def get_analytics_summary(db: Session, store_id:str=None):
    # Aggregate stats for dashboard
    from sqlalchemy import func

    q=db.query(ScanRecord)
    if store_id:
        q = q.filter(ScanRecord.store_id==store_id)

    total_scans = q.count()
    if total_scans == 0:
        return {"total_scans":0}
    
    agg = q.with_entities(
        func.avg(ScanRecord.stock_level_pct).label("avg_stock"),
        func.avg(ScanRecord.empty_slots).label("avg_empty"),
        func.avg(ScanRecord.misplaced_items).label("avg_misplaced"),
        func.sum(ScanRecord.has_alerts.cast(Integer)).label("total_alerts"),
    ).first()

    return{
        "total_scans":total_scans,
        "avg_stock_level_pct":round(agg.avg_stock or 0,1),
        "avg_empty_slots":round(agg.avg_misplaced or 0,1),
        "total_alerts":int(agg.total_alerts or 0),
        
    }