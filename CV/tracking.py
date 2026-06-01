from ultralytics import YOLO
import csv
import math
from pathlib import Path

import cv2


def get_video_fps(input_video: Path) -> float:
    """Read FPS from the input video. Falls back to 30 FPS if unavailable."""
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 0:
        fps = 30.0

    return float(fps)


def fmt(value, ndigits=6):
    """Format values nicely for CSV. Blank if value is None."""
    if value is None:
        return ""
    return f"{float(value):.{ndigits}f}"


def track_mp4(model_path: str | None, input_path: str, output_path: str):
    # If no custom weight is selected, use default YOLO model
    if not model_path:
        model_path = "yolo11n.pt"

    model = YOLO(model_path)

    INPUT_VIDEO = Path(input_path)
    OUTPUT = Path(output_path)

    RUN_NAME = f"{INPUT_VIDEO.stem}_tracking"
    RUN_DIR = OUTPUT / RUN_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    CSV_OUTPUT = RUN_DIR / f"{INPUT_VIDEO.stem}.csv"

    # Read the actual video FPS so speed can be calculated in pixels/second
    fps = get_video_fps(INPUT_VIDEO)

    # Stores the previous known position of each track_id:
    # track_id -> (previous_frame_idx, previous_centre_x, previous_centre_y)
    previous_positions = {}

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "time_sec",
            "people_count",
            "track_id",
            "x1", "y1", "x2", "y2",
            "centre_x", "centre_y",
            "distance_px",
            "speed_px_per_second",
        ])

        for frame_idx, res in enumerate(model.track(
            source=INPUT_VIDEO,
            show=True,
            save=True,
            tracker="bytetrack.yaml",
            conf=0.5,
            iou=0.5,
            classes=[0],  # only track humans/person for COCO YOLO models
            project=OUTPUT,
            name=RUN_NAME,
            exist_ok=True,
            stream=True
        )):
            time_sec = frame_idx / fps

            boxes = res.boxes

            # If no detections or no tracking IDs, still write one row for this frame
            if boxes is None or boxes.id is None:
                writer.writerow([
                    frame_idx,
                    fmt(time_sec),
                    0,
                    "",
                    "", "", "", "",
                    "", "",
                    "",
                    "",
                ])
                f.flush()
                continue

            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()

            people_count = len(ids)

            for track_id, (x1, y1, x2, y2) in zip(ids, xyxy):
                track_id = int(track_id)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                distance_px = None
                speed_px_per_second = None

                # Calculate speed only if this same track_id appeared before
                if track_id in previous_positions:
                    prev_frame_idx, prev_cx, prev_cy = previous_positions[track_id]

                    frame_gap = frame_idx - prev_frame_idx

                    if frame_gap > 0:
                        distance_px = math.hypot(cx - prev_cx, cy - prev_cy)
                        time_gap_sec = frame_gap / fps
                        speed_px_per_second = distance_px / time_gap_sec

                # Update previous position for this track_id
                previous_positions[track_id] = (frame_idx, cx, cy)

                writer.writerow([
                    frame_idx,
                    fmt(time_sec),
                    people_count,
                    track_id,
                    fmt(x1), fmt(y1), fmt(x2), fmt(y2),
                    fmt(cx), fmt(cy),
                    fmt(distance_px),
                    fmt(speed_px_per_second),
                ])

            f.flush()

    print("CSV saved to:", CSV_OUTPUT)
    print("Tracking results saved in:", RUN_DIR)
    print("Video FPS used for speed calculation:", fps)

    return {
        "run_folder": str(RUN_DIR),
        "csv": str(CSV_OUTPUT),
        "model_used": str(model_path),
        "fps": fps,
    }
