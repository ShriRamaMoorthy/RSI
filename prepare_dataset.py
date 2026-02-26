import os
import shutil
import random
import yaml
from pathlib import Path

DATASET_ROOT = Path("dataset")
SPLITS = {"train":0.75 , "val":0.15, "test":0.10}

CLASSES = [
    "product",
    "empty_slot",
    "misplaced",
    "price_tag",
]

# directory setup
def create_yolo_structure():
    for split in ["train","val","test"]:
        (DATASET_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)
    print("YOLOv8 directory structuring created.")

# Dataset YAML
def create_dataset_yaml():
    config={
        "path":str(DATASET_ROOT.resolve()),
        "train":"train/images",
        "val":"val/images",
        "test":"test/images",
        "nc":len(CLASSES),
        "names":CLASSES,
    }
    yaml_path = DATASET_ROOT / "dataset.yaml"
    with open(yaml_path,"w") as f:
        yaml.dump(config,f,default_flow_style=False)
    print(f"dataset yaml created at {yaml_path}")
    return yaml_path

def split_dataset(raw_images_dir:str, raw_labels_dir:str):
    # splits raw labeled images into train/val/test
    images = sorted(Path(raw_images_dir).glob("*jpg")) + \
             sorted(Path(raw_images_dir).glob("*.png"))
    
    random.seed(42)
    random.shuffle(images)

    n = len(images)
    n_train = int(n*SPLITS["train"])
    n_val = int(n*SPLITS["val"])

    splits_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:],
    }

    for split_name , split_images in splits_map.items():
        for img_path in split_images:
            # let's copy images
            dst_img = DATASET_ROOT / split_name / "images" / img_path.name
            shutil.copy(img_path,dst_img)

            # let's copy labels
            label_path = Path(raw_labels_dir) / (img_path.stem + ".txt")
            if label_path.exists():
                dst_lbl = DATASET_ROOT / split_name / "labels" / label_path.name
                shutil.copy(label_path, dst_lbl)
        print(f"Dataset split : {n_train} train || {n_val} val || {n-n_train-n_val} test")

# create a config file for YOLO to handle augmentation during training
def create_augmentation_config():
    hyp = {
        # Geometric
        "degrees" : 5.0,  # rotation
        "translate" : 0.1, # translation
        "scale" : 0.5, # scale
        "shear" : 2.0, # shear
        "flipud" : 0.0, # vertical flip
        "fliplr" : 0.5, # horizontal flip

        # Photometric 
        "hsv_h" : 0.015,
        "hsv_s" : 0.7,
        "hsv_v" : 0.4,
        "mosaic" : 1.0,
        "mixup" : 0.1,
        "copy_paste" : 0.1,
    }

    hyp_path = DATASET_ROOT / "hyp.yaml"
    with open(hyp_path,"w") as f:
        yaml.dump(hyp,f,default_flow_style=False)
    print(f"Augmentation config saved to {hyp_path}")


# Let's verify label format
def verify_labels(split="train",num_samples=5):
    # on spot YOLO label files check.
    label_dir = DATASET_ROOT / split / "labels"
    label_files = list(label_dir.glob("*.txt"))[:num_samples]
    print(f"Verifying {num_samples} label files from '{split}'...")

    for lf in label_files:
        with open(lf) as f:
            lines = f.readlines()
        print(f"{lf.name}: {len(lines)} annotations")
        for line in lines[:2]:
            parts = line.strip().split()
            assert len(parts)==5 , f"Bad label format in {lf.name}: {line}"
            cls_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
            assert cls_id < len(CLASSES), f"Class ID {cls_id} out of range"
            assert all(0 <= c <= 1 for c in coords), f"Coords out of [0,1]: {coords}"
    print("All sampled labels look valid!")


if __name__ == "__main__":
    print("="*60)
    print(" RETAIL SHELF INTELLIGENCE - Dataset Preparation")
    print("="*60)

    create_yolo_structure()
    yaml_path = create_dataset_yaml()
    create_augmentation_config()

    print(f"\n dataset.yaml is at: {yaml_path}")
    
     
