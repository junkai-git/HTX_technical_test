import cv2
from pathlib import Path
from collections import defaultdict

def ROI(input_path, out_dir):
    input_path = Path(input_path)
    out_dir = Path(out_dir)

    #ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    #user selects ROI on the first frame
    x, y, w, h = cv2.selectROI(
        "Select ROI", first_frame,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyAllWindows()

    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")
    print("FPS:", fps)

    ext = input_path.suffix if input_path.suffix else ".mp4"
    out_file = out_dir / f"{input_path.stem}_cropped{ext}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"VideoWriter failed to open. Output path: {out_file}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = frame[y:y+h, x:x+w]
        out.write(cropped)
        i += 1
        print(f"\r{i}/{total_frames}", end="")
    print()  
    
    cap.release()
    out.release()

    print("Saved cropped video to:", out_file)
    return str(out_file)


def convert_to_yolo(IN_TXT : Path, OUT_DIR : Path, IMG_W : int, IMG_H : int, CLASS_ID : 0):
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # frame_id -> list of (x, y, w, h)
    boxes_by_frame = defaultdict(list)

    # 1) Read and group by frame_id
    for line in IN_TXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        frame_id = int(parts[0])
        
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])

        # Skip invalid boxes
        if w <= 1 or h <= 1:
            continue

        boxes_by_frame[frame_id].append((x, y, w, h))

    # 2) Write one YOLO label file per frame
    for frame_id, boxes in boxes_by_frame.items():
        yolo_lines = []
        for x, y, w, h in boxes:
            # Convert top-left to center
            xc = x + w / 2
            yc = y + h / 2

            # Normalize to 0..1
            xc /= IMG_W
            yc /= IMG_H
            w  /= IMG_W
            h  /= IMG_H

            # (Optional) clamp to [0,1] just in case
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))

            yolo_lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # Name matches common frame naming: 000001.txt, 000002.txt, ...
        out_path = OUT_DIR / f"{frame_id:06d}.txt"
        out_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

    print(f"Done! Wrote {len(boxes_by_frame)} YOLO label files to: {OUT_DIR.resolve()}")