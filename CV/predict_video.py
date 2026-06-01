import os
import sys
import csv
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

# ----------------------------
# Make imports + relative paths work (pretrained/, util/, models/)
# ----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import util.misc as utils
from models import build_model


# ============================================================
VIDEO_PATH = r"C:\Users\intern\Desktop\Crowd Simulation\test_cropped.mp4"
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "SHA_model.pth")

OUT_DIR = os.path.join(ROOT, "video_outputs")
OUT_VIDEO_PATH = os.path.join(OUT_DIR, "test_pred_overlay.mp4")
OUT_CSV_PATH = os.path.join(OUT_DIR, "test_pred_points.csv")
OUT_COUNTS_CSV_PATH = os.path.join(OUT_DIR, "test_frame_counts.csv")

DEVICE = "cuda"  # "cuda" or "cpu"
SCORE_THR = 0.8

FRAME_STRIDE = 1          # 1 = every frame, 2 = every 2nd frame, etc.
MAX_FRAMES = 0            # 0 = no limit, else stop after N processed frames

DRAW_POINTS = True
SHOW_SPLIT_MAP = False    # red/blue overlay; slow and optional
SPLIT_ALPHA = 0.7         # overlay strength if SHOW_SPLIT_MAP=True


# ============================================================
# Model config (match PET defaults)
# ============================================================
cfg = SimpleNamespace(
    backbone="vgg16_bn",
    position_embedding="sine",
    dec_layers=2,
    dim_feedforward=512,
    hidden_dim=256,
    dropout=0.0,
    nheads=8,
    set_cost_class=1.0,
    set_cost_point=0.05,
    ce_loss_coef=1.0,
    point_loss_coef=5.0,
    eos_coef=0.5,

    # fields build_model may expect
    dataset_file="SHA",
    data_path="./data/ShanghaiTech/PartA",
    device=DEVICE,
    seed=42,
    resume=CHECKPOINT_PATH,
    vis_dir="",
    num_workers=2,
    world_size=1,
    dist_url="env://",
)

# ImageNet normalization (same as PET)
TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_frame_as_samples(bgr_frame: np.ndarray, device: torch.device):
    """Convert BGR numpy frame -> NestedTensor samples."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    img_t = TRANSFORM(pil)
    samples = utils.nested_tensor_from_tensor_list([img_t]).to(device)
    return samples


@torch.no_grad()
def predict_points_on_frame(model, bgr_frame, device, score_thr=0.5):
    """
    Returns:
      xs, ys, ss: numpy arrays of x, y pixel coords and scores
      split_map: None or a HxW float numpy mask (0/1)
    """
    samples = load_frame_as_samples(bgr_frame, device)
    H, W = samples.tensors.shape[-2], samples.tensors.shape[-1]

    outputs = model(samples, test=True)

    probs = torch.softmax(outputs["pred_logits"][0], dim=-1)  # (Q,C)
    if probs.shape[-1] == 2:
        scores = probs[:, 1]
    else:
        scores = probs[:, :-1].max(dim=-1).values  # ignore no-object

    keep = scores > score_thr
    pts = outputs["pred_points"][0][keep]   # (N,2) normalized (y,x)
    ss = scores[keep]

    ys = (pts[:, 0] * H).detach().cpu().numpy()
    xs = (pts[:, 1] * W).detach().cpu().numpy()
    ss = ss.detach().cpu().numpy()

    split_map = None
    if SHOW_SPLIT_MAP and ("split_map_raw" in outputs):
        # split_map_raw usually smaller; we resize later
        split_map = (outputs["split_map_raw"][0].detach().cpu().squeeze(0) > 0.5).float().numpy()

    return xs, ys, ss, split_map


def draw_overlay(bgr_frame, xs, ys, split_map=None):
    """Draw green points and optional split-map overlay onto frame."""
    out = bgr_frame.copy()

    # optional red/blue overlay (JET)
    if SHOW_SPLIT_MAP and split_map is not None:
        h, w = out.shape[:2]
        sm = (split_map * 255).astype(np.uint8)
        sm = cv2.applyColorMap(sm, cv2.COLORMAP_JET)
        sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
        out = (sm * SPLIT_ALPHA + out).astype(np.uint8)

    # draw points last so they stay visible
    for x, y in zip(xs, ys):
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)

    return out


def main():
    global DEVICE
    os.makedirs(OUT_DIR, exist_ok=True)

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> switching to CPU")
        DEVICE = "cpu"
        cfg.device = "cpu"

    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device(DEVICE)

    # Build model
    model, _criterion = build_model(cfg)
    model.to(device).eval()

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # If you skip frames, write output at reduced fps so playback speed looks normal
    fps_out = fps_in / FRAME_STRIDE if FRAME_STRIDE > 0 else fps_in

    writer = None
    if DRAW_POINTS:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps_out, (w, h))

    # CSV outputs
    points_f = open(OUT_CSV_PATH, "w", newline="", encoding="utf-8")
    points_w = csv.writer(points_f)
    points_w.writerow(["frame_idx", "time_sec", "x", "y", "score"])

    counts_f = open(OUT_COUNTS_CSV_PATH, "w", newline="", encoding="utf-8")
    counts_w = csv.writer(counts_f)
    counts_w.writerow(["frame_idx", "time_sec", "count"])

    frame_idx = 0
    processed = 0

    print("Running video inference...")
    print("Input FPS:", fps_in, "Output FPS:", fps_out, "Stride:", FRAME_STRIDE)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # process every Nth frame
        if FRAME_STRIDE > 1 and (frame_idx % FRAME_STRIDE != 0):
            frame_idx += 1
            continue

        t = frame_idx / fps_in

        xs, ys, ss, split_map = predict_points_on_frame(model, frame, device, SCORE_THR)

        # write counts
        counts_w.writerow([frame_idx, f"{t:.3f}", int(len(xs))])

        # write all points
        for x, y, s in zip(xs, ys, ss):
            points_w.writerow([frame_idx, f"{t:.3f}", float(x), float(y), float(s)])

        # output video frame with overlay
        if DRAW_POINTS and writer is not None:
            out_frame = draw_overlay(frame, xs, ys, split_map=split_map)
            writer.write(out_frame)

        processed += 1
        if processed % 10 == 0:
            print(f"Processed frames: {processed} (video frame idx: {frame_idx})")

        if MAX_FRAMES and processed >= MAX_FRAMES:
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    points_f.close()
    counts_f.close()

    print("Done.")
    print("Saved overlay video:", OUT_VIDEO_PATH if DRAW_POINTS else "(disabled)")
    print("Saved points CSV:", OUT_CSV_PATH)
    print("Saved counts CSV:", OUT_COUNTS_CSV_PATH)


if __name__ == "__main__":
    main()