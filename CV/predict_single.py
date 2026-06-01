import os , csv , json
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms
from types import SimpleNamespace
import util.misc as utils
from models import build_model

IMG_PATH = r"C:\Users\junka\Downloads\stadiumcrowd.webp"
CHECKPOINT_PATH = r"C:\Users\junka\Desktop\HTX_technical_test\main\checkpoints\SHA_model.pth"
VIS_DIR = r"C:\Users\junka\Desktop"  
DEVICE = "cuda"  
SCORE_THR = 0.5  
SHOW_SPLIT_MAP = False   # True = show red/blue overlay, False = disable
SPLIT_ALPHA = 0.9        # opacity of split map overlay (0.0 to 1.0)

MODEL_BACKBONE = "vgg16_bn"
POSITION_EMBEDDING = "sine"     # "sine" | "learned" | "fourier"
DEC_LAYERS = 2
DIM_FEEDFORWARD = 512
HIDDEN_DIM = 256
DROPOUT = 0.0
NHEADS = 8

SET_COST_CLASS = 1.0
SET_COST_POINT = 0.05
CE_LOSS_COEF = 1.0
POINT_LOSS_COEF = 5.0
EOS_COEF = 0.5


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, pred_yx_px, vis_dir, img_path, split_map=None):
    """
    Visualize predictions by drawing green points.
    pred_yx_px: list of list of [y, x] points in pixel coordinates, one list per image in batch.
    """
    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask

    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # draw predictions (green)
        size = 3
        for p in pred_yx_px[idx]:
            y, x = p[0], p[1]
            sample_vis = cv2.circle(sample_vis, (int(x), int(y)), size, (0, 255, 0), -1)

        # draw split map if available
        if SHOW_SPLIT_MAP and split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * SPLIT_ALPHA + sample_vis

        # save image (crop invalid padding)
        if vis_dir:
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h + 1, :valid_w + 1]

            name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(vis_dir, f"{name}_pred{len(pred_yx_px[idx])}.jpg")
            cv2.imwrite(out_path, sample_vis)
            print("Saved:", out_path)


def load_image_rgb_pil(img_path: str) -> Image.Image:
    """
    Tries OpenCV first; falls back to PIL if needed.
    Returns RGB PIL Image.
    """
    img = cv2.imread(img_path)
    if img is not None:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # fallback (helps for formats OpenCV may not support in your build)
    return Image.open(img_path).convert("RGB")


@torch.no_grad()
def evaluate_single_image(model, img_path, device, vis_dir="vis_single", score_thr=0.5):
    model.eval()

    # Choose an output directory even if vis_dir is empty
    out_dir = vis_dir if vis_dir else "pred_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # load image
    img = load_image_rgb_pil(img_path)

    # transform image (ImageNet norm)
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(img)
    samples = utils.nested_tensor_from_tensor_list([img_t]).to(device)
    img_h, img_w = samples.tensors.shape[-2:]  # (H, W)

    # inference
    outputs = model(samples, test=True)

    # scores (object probability)
    probs = torch.softmax(outputs["pred_logits"][0], dim=-1)  # (Q, C)
    if probs.shape[-1] == 2:
        scores = probs[:, 1]
    else:
        scores = probs[:, :-1].max(dim=-1).values  # ignore no-object

    # points in normalized (y, x)
    points_norm = outputs["pred_points"][0]  # (Q, 2)

    # threshold
    keep = scores > score_thr
    points_keep = points_norm[keep]
    scores_keep = scores[keep]

    print(f"Predicted count (score>{score_thr}): {points_keep.shape[0]}")
    if scores_keep.numel() > 0:
        print(f"Score range: {scores_keep.min().item():.3f} to {scores_keep.max().item():.3f}")

    # convert to pixel coords
    ys = (points_keep[:, 0] * img_h).detach().cpu().numpy()
    xs = (points_keep[:, 1] * img_w).detach().cpu().numpy()
    ss = scores_keep.detach().cpu().numpy()

    base = os.path.splitext(os.path.basename(img_path))[0]
    csv_path = os.path.join(out_dir, f"{base}_pred_points.csv")
    json_path = os.path.join(out_dir, f"{base}_pred_points.json")

    # save CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "score"])
        for x, y, s in zip(xs, ys, ss):
            w.writerow([float(x), float(y), float(s)])

    # save JSON
    data = [{"x": float(x), "y": float(y), "score": float(s)} for x, y, s in zip(xs, ys, ss)]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved:", csv_path)
    print("Saved:", json_path)

    # visualization (green points + optional split-map overlay)
    if vis_dir:
        points_px_yx = [[ [float(y), float(x)] for x, y in zip(xs, ys) ]]  # list for batch size 1

        split_map = None
        if SHOW_SPLIT_MAP and ("split_map_raw" in outputs):
            split_map = (outputs["split_map_raw"][0].detach().cpu().squeeze(0) > 0.5).float().numpy()

        visualization(samples, points_px_yx, vis_dir, img_path, split_map=split_map)



def main():
    global DEVICE

    # Auto-fallback if CUDA not available
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> switching DEVICE to cpu")
        DEVICE = "cpu"

    # Basic path checks
    if not os.path.isfile(IMG_PATH):
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device(DEVICE)

    # Build a config object (no argparse; just values build_model expects)
    cfg = SimpleNamespace(
        backbone=MODEL_BACKBONE,
        position_embedding=POSITION_EMBEDDING,
        dec_layers=DEC_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        nheads=NHEADS,
        set_cost_class=SET_COST_CLASS,
        set_cost_point=SET_COST_POINT,
        ce_loss_coef=CE_LOSS_COEF,
        point_loss_coef=POINT_LOSS_COEF,
        eos_coef=EOS_COEF,
        # unused for single-image, but some code paths expect them to exist:
        dataset_file="SHA",
        data_path="./data/ShanghaiTech/PartA",
        device=DEVICE,
        seed=42,    
        resume=str(CHECKPOINT_PATH),
        vis_dir=str(VIS_DIR),
        num_workers=2,
        world_size=1,
        dist_url="env://",
    )

    model, _criterion = build_model(cfg)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)

    evaluate_single_image(model, IMG_PATH, device, vis_dir=VIS_DIR, score_thr=SCORE_THR)


if __name__ == "__main__":
    main()