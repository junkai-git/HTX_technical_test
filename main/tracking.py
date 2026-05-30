from ultralytics import YOLO
import csv
from pathlib import Path

def track_mp4(model_path : str, input_path : str, output_path :str):
    # ===== USER SETTINGS =====
    model = YOLO(model_path)  
    INPUT_VIDEO = Path(input_path)
    OUTPUT = Path(output_path)
    CSV_OUTPUT = OUTPUT / (INPUT_VIDEO.stem + ".csv")

    # =========================


    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "track_id",
            "x1", "y1", "x2", "y2",
            "centre_x" , "centre_y"
        ])

        # results is a list of Results objects, one per frame
        for frame_idx, res in enumerate(model.track(
        source=INPUT_VIDEO,               # or 0 for webcam, or a youtube url
        show=True,                        # show window with tracking
        save=True,                        # save annotated video
        tracker="bytetrack.yaml",         # or "botsort.yaml" (default is BoT-SORT)(BoT-SORT has more options while ByteTrack is a good baseline)
        conf=0.5,
        iou=0.5,
        project = Path(OUTPUT),
        exist_ok = True,
        stream=True
        )):
            boxes = res.boxes

            # If no detections or no tracking IDs, skip this frame
            if boxes is None or boxes.id is None:
                continue

            # Move tensors to cpu + convert to python types
            ids   = boxes.id.int().cpu().tolist()
            xyxy  = boxes.xyxy.cpu().tolist()

            for track_id, (x1, y1, x2, y2) in zip(ids, xyxy):

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2)/ 2.0

                writer.writerow([
                    frame_idx,
                    int(track_id),
                    float(x1), float(y1), float(x2), float(y2),
                    float(cx), float(cy)
                ])

            f.flush()

    print("CSV saved to:", CSV_OUTPUT)
