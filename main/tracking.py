from ultralytics import YOLO
import csv
from pathlib import Path

def track_mp4(model_path: str | None, input_path: str, output_path: str):
    #if no custom weight is selected, use default YOLO model
    if not model_path:
        model_path = "yolo11n.pt"

    model = YOLO(model_path)

    INPUT_VIDEO = Path(input_path)
    OUTPUT = Path(output_path)
    RUN_DIR = OUTPUT / f"{INPUT_VIDEO.stem}_tracking"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    CSV_OUTPUT = RUN_DIR / f"{INPUT_VIDEO.stem}.csv"

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "track_id",
            "x1", "y1", "x2", "y2",
            "centre_x", "centre_y"
        ])

        for frame_idx, res in enumerate(model.track(
            source=INPUT_VIDEO,
            show=True,
            save=True,
            tracker="bytetrack.yaml",
            conf=0.5,
            iou=0.5,
            classes = [0],
            project=OUTPUT,
            name=f"{INPUT_VIDEO.stem}_tracking",
            exist_ok=True,
            stream=True
        )):
            boxes = res.boxes

            if boxes is None or boxes.id is None:
                continue

            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()

            for track_id, (x1, y1, x2, y2) in zip(ids, xyxy):
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                writer.writerow([
                    frame_idx,
                    int(track_id),
                    float(x1), float(y1), float(x2), float(y2),
                    float(cx), float(cy)
                ])

            f.flush()

    print("CSV saved to:", CSV_OUTPUT)
    
    return {
        "run_folder": str(RUN_DIR),
        "csv": str(CSV_OUTPUT),
        "model_used": str(model_path)
    }