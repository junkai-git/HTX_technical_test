import os
import csv
import math
from collections import deque, defaultdict
from pathlib import Path

import cv2
import numpy as np


#ROI polygon: only motion inside this polygon is analyzed to get rid of unwanted motions
#put points in (x, y)
ROI_POLYGON = np.array([
    # Example:
    # [100, 250],
    # [1800, 250],
    # [1900, 1000],
    # [50, 1000],
], dtype=np.int32)

#if ROI_POLYGON is empty, entire frame is used
USE_FULL_FRAME_IF_NO_POLYGON = True

#grid cell size in pixels
CELL_SIZE = 20

#minimum ratio of valid ROI pixels inside a cell before we use it
MIN_VALID_RATIO = 0.20

#ignore cells with very small motion (pixels/frame)
MIN_SPEED_PX_PER_FRAME = 0.30

#arrow drawing scale for visualization
ARROW_SCALE = 8.0

#smoothing of overall crowd speed over time
SMOOTHING_WINDOW = 5

#save annotated video
SAVE_VIDEO = True

#show preview while running
SHOW_PREVIEW = True

# Farneback optical flow parameters
FB_PYR_SCALE = 0.5
FB_LEVELS = 3
FB_WINSIZE = 21
FB_ITERATIONS = 3
FB_POLY_N = 5
FB_POLY_SIGMA = 1.2

#optional light blur before optical flow to ignore camera instability
GAUSSIAN_BLUR = 3  # set 0 to disable



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def make_roi_mask(frame_shape, roi_polygon, use_full_if_empty=True):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if roi_polygon is not None and len(roi_polygon) >= 3:
        cv2.fillPoly(mask, [roi_polygon.astype(np.int32)], 255)
    else:
        if use_full_if_empty:
            mask[:] = 255
        else:
            raise ValueError("ROI polygon has fewer than 3 points.")

    return mask


def preprocess_gray(frame_bgr, gaussian_blur=GAUSSIAN_BLUR):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gaussian_blur and gaussian_blur > 0:
        k = int(gaussian_blur)
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def compute_dense_flow(
    prev_gray,
    curr_gray,
    *,
    pyr_scale=FB_PYR_SCALE,
    levels=FB_LEVELS,
    winsize=FB_WINSIZE,
    iterations=FB_ITERATIONS,
    poly_n=FB_POLY_N,
    poly_sigma=FB_POLY_SIGMA,
):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=0,
    )
    return flow  # shape: (H, W, 2), flow[...,0]=dx, flow[...,1]=dy


def angle_deg_from_vector(dx, dy):
    """Return angle in degrees from a motion vector.

    0 degrees = right, 90 degrees = down in image coordinates,
    -90 degrees = up, 180/-180 degrees = left.
    """
    return math.degrees(math.atan2(dy, dx))


def circular_stats_degrees(angles_deg):
    """Circular mean/variance for direction angles.

    This avoids the common problem where -179 degrees and +179 degrees
    are incorrectly treated as very far apart.
    """
    angles = [a for a in angles_deg if a is not None and not math.isnan(a)]
    if not angles:
        return {
            "mean_deg": None,
            "variance": None,
            "std_deg": None,
            "resultant_length": None,
        }

    radians = np.deg2rad(np.array(angles, dtype=float))
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    resultant_length = float(np.hypot(mean_cos, mean_sin))

    mean_deg = math.degrees(math.atan2(mean_sin, mean_cos))
    circular_variance = 1.0 - resultant_length

    if resultant_length > 0:
        circular_std_rad = math.sqrt(-2.0 * math.log(resultant_length))
        circular_std_deg = math.degrees(circular_std_rad)
    else:
        circular_std_deg = 180.0

    return {
        "mean_deg": mean_deg,
        "variance": circular_variance,
        "std_deg": circular_std_deg,
        "resultant_length": resultant_length,
    }


def normal_stats(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return {"mean": None, "variance": None}

    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "variance": float(np.var(arr)),  # population variance
    }


def summarize_grid_flow(flow, roi_mask, cell_size, min_valid_ratio, min_speed_px_per_frame):
    h, w = roi_mask.shape[:2]
    cell_rows = []

    active_dx = []
    active_dy = []
    active_speed = []

    for y0 in range(0, h, cell_size):
        for x0 in range(0, w, cell_size):
            y1 = min(h, y0 + cell_size)
            x1 = min(w, x0 + cell_size)

            cell_mask = roi_mask[y0:y1, x0:x1]
            valid = cell_mask > 0

            total_pixels = valid.size
            valid_pixels = int(valid.sum())

            if total_pixels == 0:
                continue

            valid_ratio = valid_pixels / total_pixels
            if valid_ratio < min_valid_ratio:
                continue

            cell_flow = flow[y0:y1, x0:x1]
            dx_vals = cell_flow[..., 0][valid]
            dy_vals = cell_flow[..., 1][valid]

            if dx_vals.size == 0:
                continue

            # Median is more robust than mean for noisy crowd flow
            dx = float(np.median(dx_vals))
            dy = float(np.median(dy_vals))
            speed_pf = float(np.hypot(dx, dy))  # pixels/frame

            if speed_pf < min_speed_px_per_frame:
                continue

            cx = int((x0 + x1) / 2)
            cy = int((y0 + y1) / 2)

            row = {
                "cell_x0": x0,
                "cell_y0": y0,
                "cell_x1": x1,
                "cell_y1": y1,
                "cx": cx,
                "cy": cy,
                "dx_pf": dx,
                "dy_pf": dy,
                "speed_pf": speed_pf,
                "direction_deg": angle_deg_from_vector(dx, dy),
                "valid_ratio": valid_ratio,
            }
            cell_rows.append(row)

            active_dx.append(dx)
            active_dy.append(dy)
            active_speed.append(speed_pf)

    if len(active_speed) == 0:
        global_stats = {
            "num_active_cells": 0,
            "mean_dx_pf": 0.0,
            "mean_dy_pf": 0.0,
            "mean_speed_pf": 0.0,
            "median_speed_pf": 0.0,
            "dominant_angle_deg": None,
        }
    else:
        mean_dx_pf = float(np.mean(active_dx))
        mean_dy_pf = float(np.mean(active_dy))
        mean_speed_pf = float(np.mean(active_speed))
        median_speed_pf = float(np.median(active_speed))
        dominant_angle_deg = float(angle_deg_from_vector(mean_dx_pf, mean_dy_pf))

        global_stats = {
            "num_active_cells": len(active_speed),
            "mean_dx_pf": mean_dx_pf,
            "mean_dy_pf": mean_dy_pf,
            "mean_speed_pf": mean_speed_pf,
            "median_speed_pf": median_speed_pf,
            "dominant_angle_deg": dominant_angle_deg,
        }

    return cell_rows, global_stats


def draw_roi(frame, roi_polygon, roi_mask):
    vis = frame.copy()
    if roi_polygon is not None and len(roi_polygon) >= 3:
        cv2.polylines(vis, [roi_polygon.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
    return vis


def draw_grid_flow(frame, cell_rows, roi_polygon, roi_mask, fps, smoothed_speed_pf, direction_deg, arrow_scale=ARROW_SCALE):
    vis = draw_roi(frame, roi_polygon, roi_mask)

    outside = roi_mask == 0
    if outside.any():
        overlay = vis.copy()
        overlay[outside] = (overlay[outside] * 0.35).astype(np.uint8)
        vis = overlay

    for row in cell_rows:
        cx = row["cx"]
        cy = row["cy"]
        dx = row["dx_pf"]
        dy = row["dy_pf"]

        end_x = int(round(cx + dx * arrow_scale))
        end_y = int(round(cy + dy * arrow_scale))

        cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
        cv2.circle(vis, (cx, cy), 2, (0, 255, 255), -1)

    speed_ps = smoothed_speed_pf * fps
    direction_text = "N/A" if direction_deg is None else f"{direction_deg:.1f} deg"

    text_lines = [
        f"Active cells: {len(cell_rows)}",
        f"Smoothed crowd speed: {speed_ps:.3f} px/s",
        f"Flow direction: {direction_text}",
    ]

    y = 30
    for line in text_lines:
        cv2.putText(
            vis,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    return vis


def save_mask_preview(mask, out_path):
    preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_path), preview)


def fmt(value, ndigits=6):
    """Format numbers for CSV while keeping missing values blank."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return f"{float(value):.{ndigits}f}"


def write_per_second_summary(path, frame_records):
    by_second = defaultdict(list)
    for record in frame_records:
        by_second[record["second"]].append(record)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "second",
            "num_frames",
            "mean_speed_px_s",
            "variance_speed_px_s2",
            "mean_direction_deg_circular",
            "direction_circular_variance",
            "direction_circular_std_deg",
        ])

        for second in sorted(by_second):
            records = by_second[second]
            speeds = [r["mean_speed_ps"] for r in records]
            directions = [r["direction_deg"] for r in records if r["direction_deg"] is not None]

            speed_stats = normal_stats(speeds)
            direction_stats = circular_stats_degrees(directions)

            writer.writerow([
                second,
                len(records),
                fmt(speed_stats["mean"]),
                fmt(speed_stats["variance"]),
                fmt(direction_stats["mean_deg"]),
                fmt(direction_stats["variance"]),
                fmt(direction_stats["std_deg"]),
            ])


def write_overall_summary(path, frame_records, fps):
    all_speeds = [r["mean_speed_ps"] for r in frame_records]
    active_records = [r for r in frame_records if r["num_active_cells"] > 0]
    active_speeds = [r["mean_speed_ps"] for r in active_records]
    directions = [r["direction_deg"] for r in active_records if r["direction_deg"] is not None]

    all_speed_stats = normal_stats(all_speeds)
    active_speed_stats = normal_stats(active_speeds)
    direction_stats = circular_stats_degrees(directions)

    total_frames = len(frame_records)
    active_frames = len(active_records)
    duration_sec = total_frames / fps if fps else 0.0

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_processed_frames", total_frames])
        writer.writerow(["active_motion_frames", active_frames])
        writer.writerow(["duration_sec", fmt(duration_sec)])
        writer.writerow(["fps", fmt(fps)])
        writer.writerow(["mean_speed_px_s_all_frames", fmt(all_speed_stats["mean"])])
        writer.writerow(["variance_speed_px_s2_all_frames", fmt(all_speed_stats["variance"])])
        writer.writerow(["mean_speed_px_s_active_frames", fmt(active_speed_stats["mean"])])
        writer.writerow(["variance_speed_px_s2_active_frames", fmt(active_speed_stats["variance"])])
        writer.writerow(["overall_direction_deg_circular", fmt(direction_stats["mean_deg"])])
        writer.writerow(["direction_circular_variance", fmt(direction_stats["variance"])])
        writer.writerow(["direction_circular_std_deg", fmt(direction_stats["std_deg"])])
        writer.writerow(["direction_resultant_length", fmt(direction_stats["resultant_length"])])


def run_optical_flow(
    video_path,
    out_dir,
    *,
    roi_polygon=None,
    use_full_frame_if_no_polygon=USE_FULL_FRAME_IF_NO_POLYGON,
    cell_size=CELL_SIZE,
    min_valid_ratio=MIN_VALID_RATIO,
    min_speed_px_per_frame=MIN_SPEED_PX_PER_FRAME,
    arrow_scale=ARROW_SCALE,
    smoothing_window=SMOOTHING_WINDOW,
    save_video=SAVE_VIDEO,
    show_preview=SHOW_PREVIEW,
    gaussian_blur=GAUSSIAN_BLUR,
):
    video_path = Path(video_path)

    selected_parent_dir = Path(out_dir)
    out_dir = selected_parent_dir / "optical flow"
    ensure_dir(out_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Could not find video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame from video.")

    h, w = first_frame.shape[:2]

    if roi_polygon is None:
        roi_polygon = ROI_POLYGON.copy() if ROI_POLYGON is not None else np.array([], dtype=np.int32)
    else:
        roi_polygon = np.array(roi_polygon, dtype=np.int32)

    if roi_polygon is not None and len(roi_polygon) >= 3:
        roi_polygon = np.round(roi_polygon.astype(np.float32)).astype(np.int32)

    roi_mask = make_roi_mask(first_frame.shape, roi_polygon, use_full_frame_if_no_polygon)

    base_name = video_path.stem
    mask_preview_path = out_dir / f"{base_name}_roi_mask.png"
    arrow_csv_path = out_dir / f"{base_name}_arrow_flow_by_frame.csv"
    per_second_csv_path = out_dir / f"{base_name}_per_second_speed_summary.csv"
    overall_csv_path = out_dir / f"{base_name}_overall_flow_summary.csv"
    out_video_path = out_dir / f"{base_name}_flow_annotated.mp4"

    save_mask_preview(roi_mask, mask_preview_path)
    print("Saved mask preview:", mask_preview_path)

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"VideoWriter failed to open. Output path: {out_video_path}")

    prev_gray = preprocess_gray(first_frame, gaussian_blur=gaussian_blur)
    speed_buffer = deque(maxlen=smoothing_window)
    frame_records = []

    with open(arrow_csv_path, "w", newline="", encoding="utf-8") as f_arrow:
        arrow_writer = csv.writer(f_arrow)
        arrow_writer.writerow([
            "frame_idx",
            "time_sec",
            "second",
            "arrow_id",
            "cell_x0",
            "cell_y0",
            "cell_x1",
            "cell_y1",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "dx_px_frame",
            "dy_px_frame",
            "speed_px_frame",
            "dx_px_s",
            "dy_px_s",
            "speed_px_s",
            "direction_deg",
            "valid_ratio",
            "frame_num_active_cells",
            "frame_mean_speed_px_s",
            "frame_direction_deg",
        ])

        frame_idx = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_gray = preprocess_gray(frame, gaussian_blur=gaussian_blur)
            flow = compute_dense_flow(curr_gray=curr_gray, prev_gray=prev_gray)

            cell_rows, global_stats = summarize_grid_flow(
                flow=flow,
                roi_mask=roi_mask,
                cell_size=cell_size,
                min_valid_ratio=min_valid_ratio,
                min_speed_px_per_frame=min_speed_px_per_frame,
            )

            mean_speed_pf = global_stats["mean_speed_pf"]
            mean_speed_ps = mean_speed_pf * fps
            direction_deg = global_stats["dominant_angle_deg"]
            mean_dx_ps = global_stats["mean_dx_pf"] * fps
            mean_dy_ps = global_stats["mean_dy_pf"] * fps

            speed_buffer.append(mean_speed_pf)
            smoothed_speed_pf = float(np.mean(speed_buffer)) if len(speed_buffer) > 0 else 0.0
            smoothed_speed_ps = smoothed_speed_pf * fps

            time_sec = frame_idx / fps
            second = int(time_sec)

            frame_records.append({
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "second": second,
                "num_active_cells": global_stats["num_active_cells"],
                "mean_speed_pf": mean_speed_pf,
                "mean_speed_ps": mean_speed_ps,
                "smoothed_speed_ps": smoothed_speed_ps,
                "direction_deg": direction_deg,
                "mean_dx_ps": mean_dx_ps,
                "mean_dy_ps": mean_dy_ps,
            })

            for arrow_id, row in enumerate(cell_rows):
                start_x = row["cx"]
                start_y = row["cy"]
                end_x = start_x + row["dx_pf"] * arrow_scale
                end_y = start_y + row["dy_pf"] * arrow_scale

                arrow_writer.writerow([
                    frame_idx,
                    fmt(time_sec),
                    second,
                    arrow_id,
                    row["cell_x0"],
                    row["cell_y0"],
                    row["cell_x1"],
                    row["cell_y1"],
                    fmt(start_x),
                    fmt(start_y),
                    fmt(end_x),
                    fmt(end_y),
                    fmt(row["dx_pf"]),
                    fmt(row["dy_pf"]),
                    fmt(row["speed_pf"]),
                    fmt(row["dx_pf"] * fps),
                    fmt(row["dy_pf"] * fps),
                    fmt(row["speed_pf"] * fps),
                    fmt(row["direction_deg"]),
                    fmt(row["valid_ratio"]),
                    global_stats["num_active_cells"],
                    fmt(mean_speed_ps),
                    fmt(direction_deg),
                ])

            annotated = draw_grid_flow(
                frame=frame,
                cell_rows=cell_rows,
                roi_polygon=roi_polygon,
                roi_mask=roi_mask,
                fps=fps,
                smoothed_speed_pf=smoothed_speed_pf,
                direction_deg=direction_deg,
                arrow_scale=arrow_scale,
            )

            if writer is not None:
                writer.write(annotated)

            if show_preview:
                cv2.imshow("Crowd Flow", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

            prev_gray = curr_gray
            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print("Saved annotated video:", out_video_path)

    if show_preview:
        cv2.destroyAllWindows()

    write_per_second_summary(per_second_csv_path, frame_records)
    write_overall_summary(overall_csv_path, frame_records, fps)

    print("Saved arrow-level flow CSV:", arrow_csv_path)
    print("Saved per-second speed summary CSV:", per_second_csv_path)
    print("Saved overall flow summary CSV:", overall_csv_path)
    print("Done.")

    return {
        "mask_preview": str(mask_preview_path),
        "arrow_csv": str(arrow_csv_path),
        "per_second_csv": str(per_second_csv_path),
        "overall_csv": str(overall_csv_path),
        "annotated_video": str(out_video_path) if save_video else None,
    }


if __name__ == "__main__":
    # Optional direct test. For normal UI usage, import this file and call run_optical_flow().
    example_video = input("Enter video path: ").strip().strip('"')
    example_out_dir = input("Enter output folder: ").strip().strip('"')
    run_optical_flow(example_video, example_out_dir)
