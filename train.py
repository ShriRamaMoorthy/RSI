import os
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

CONFIG = {
    '''
    Model:
    nano = fastest
    small = balanced 
    medium = accurate
    large = best
    '''

    "model_size": "yolov8s.pt",
    "dataset_yaml": "dataset/dataset.yaml",
    "epochs":100,
    "batch_size":16,
    "image_size":640,
    "patience":20,
    "lr0":0.01,
    "lrf":0.01,
    "weigth_delay":0.0005,
    "warmup_epochs":3,
    "device":"0" if torch.cuda.is_available() else "cpu",
    "workers":4,
    "project":"runs/train",
    "name":f"shelf_detector_{datetime.now().strftime('%Y%m%d_%H%M')}",
    "save_period":10,
    "exist_pk":True,
}

def train():
    print("="*60)
    print("  RETAIL SHELF INTEELIGENCE - Model Training")
    print("="*60)
    print(f"\n  Device  : {CONFIG['device']}")
    print(f"  Model  : {CONFIG['model_size']}")
    print(f"  Epoch  : {CONFIG['epochs']}")
    print(f"  Batch  : {CONFIG['batch_size']}")
    print(f"  ImgSize  : {CONFIG['image_size']}")

    model = YOLO(CONFIG["model_size"])

    results = model.train(
        data=CONFIG["dataset_yaml"],
        epochs=CONFIG['epochs'],
        batch=CONFIG['batch_size'],
        imgsz=CONFIG['image_size'],
        patience=CONFIG['patience'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        weight_decay=CONFIG['weight_decay'],
        warmup_epochs=CONFIG['warmup_epochs'],
        device=CONFIG['device'],
        workers=CONFIG['workers'],
        project=CONFIG['project'],
        name=CONFIG['name'],
        save_period=CONFIG['save_period'],
        exist_ok=CONFIG['exist_ok'],

        # Augmentation overrides
        degress=5.0,
        fliplr=0.5,
        mosiac=1.0,
        mixup=0.1,

        # Logging
        plots=True,
        val=True,
        verbose=True
    )

    best_model_path = Path(CONFIG["project"]) / CONFIG['name'] / "weigths" / "best.pt"
    print(f"\n Training Complete...")
    print(f"Best Model: {best_model_path}")

    return model, best_model_path

# evaluation
def evaluate(model_path:str):
    print("\n Running Evaluation...")
    model = YOLO(model_path)
    metrics=model.val(
        data=CONFIG['dataset_yaml'],
        imgsz=CONFIG['image_size'],
        device=CONFIG['device'],
        plots=True,
        save_json=True,
    )

    results = {
        "mAP50" : round(metrics.box.map50,4),
        "mAP50-95" : round(metrics.box.map, 4),
        "precision" : round(metrics.box.mp, 4),
        "recall" : round(metrics.box.mr, 4),
    }

    print("\n Metrics:")
    for k,v in results.items():
        print(f"    {k:12s}: {v:.4f}")

    metrics_path = Path(model_path).parent.parent / "metrics.json"
    with open(metrics_path,"w") as f:
        json.dump(results,f,indent=2)
    print(f"\n  Metrics saved to: {metrics_path}")
    return results


# Export to ONNX for FASTAPI deployment
def export_to_onnx(model_path:str):
    print("Exporting to ONNX...")
    model=YOLO(model_path)
    export_path = model.export(
        format="onnx",
        imgsz=CONFIG["image_size"],
        simplify=True,
        opset=17,
        dynamic=False,
    )
    print(f"ONNX model saved to: {export_path}")

    backend_models = Path("../backend/models")
    backend_models.mkdir(parents=True,exist_ok=True)
    shutil.copy(model_path,backend_models/"best.pt")
    shutil.copy(f"Model copied to: {backend_models.resolve()}")
    return export_path


def test_inference(model_path:str, test_image:str):
    print(f"Test Inference on: {test_image}")
    model = YOLO(model_path)
    results = model.predict(
        source=test_image,
        conf=0.25,
        iou=0.45,
        save=True,
        save_txt=True,
        projects="runs/predict",
        name='test',
    )
    for r in results:
        print(f" Detected {len(r.boxes)} objects")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"    Class {cls}: {conf:.2%} confidence")


if __name__ == "__main__":
    model, best_path = train()
    evaluate(str(best_path))
    export_to_onnx(str(best_path))

    print("\n" + "="*60)
    print("All done! Next step: Run the FASTAPI backend")
    print("cd../backend && uvicorn main:app --reload")
    print("="*60)