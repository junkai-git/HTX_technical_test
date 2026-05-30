from ultralytics import YOLO
from pathlib import Path
import yaml

def training_model(
    data_yaml: str,
    project_dir: str,
    model_weights: str = "yolo11n.pt",
    epochs: int = 50):

    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_weights)  
    model.train(
        data=data_yaml,
        epochs=epochs, # number of runs
        imgsz=1280, # scaling of image
        batch=-1, # number of image per traininig step
        workers=0, # dataloader workers??
        device=0, # run on gpu
        project=project_dir,
        exist_ok = True )
    

# ============ EDIT THESE ============
OUT_YAML = Path(r"C:\Users\intern\Desktop\Crowd Simulation\kaggle data\VisDrone2019-VID-train\data.yaml")

TRAIN_IMAGES_DIR = Path(r"C:\Users\intern\Desktop\Crowd Simulation\kaggle data\VisDrone2019-VID-train\train")
VAL_IMAGES_DIR   = Path(r"C:\Users\intern\Desktop\Crowd Simulation\kaggle data\VisDrone2019-VID-train\val")

# Put your class names here (order matters)
NAMES = [
    "human"
]
# ===================================

def rel_or_abs(base: Path, target: Path) -> str:
    """Return a nice yaml path: relative to yaml file if possible, else absolute."""
    try:
        return target.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return target.resolve().as_posix()

def main():
    # Basic checks
    if not TRAIN_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Train images folder not found: {TRAIN_IMAGES_DIR}")
    if not VAL_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Val images folder not found: {VAL_IMAGES_DIR}")
    if not NAMES:
        raise ValueError("NAMES is empty. Add at least 1 class name.")

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)

    # YOLO yaml typically uses 'path' as dataset root + relative train/val paths
    dataset_root = OUT_YAML.parent

    data = {
        "path": dataset_root.resolve().as_posix(),
        "train": rel_or_abs(dataset_root, TRAIN_IMAGES_DIR),
        "val": rel_or_abs(dataset_root, VAL_IMAGES_DIR),
        "names": {i: n for i, n in enumerate(NAMES)},
    }

    with OUT_YAML.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print("Wrote:", OUT_YAML.resolve())
    print("train:", TRAIN_IMAGES_DIR.resolve())
    print("val:  ", VAL_IMAGES_DIR.resolve())
    print("classes:", len(NAMES))

if __name__ == "__main__":
    main()
