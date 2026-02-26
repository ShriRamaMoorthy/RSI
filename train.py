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

    )