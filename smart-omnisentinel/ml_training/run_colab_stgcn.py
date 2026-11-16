"""
ml_training/run_colab_stgcn.py
------------------------------
End-to-end cloud training runner for ST-GCN on Google Colab or Kaggle GPU.

Steps:
  1. Detects RWF-2000 dataset in data/
  2. Extracts 51-dim features (17 keypoints x 3: x, y, confidence) using YOLOv8n-pose
  3. Trains ST-GCN model with spatial-temporal graph convolutions & augmentation
  4. Exports best model to ml/checkpoints/stgcn_best.pt

Usage on Colab:
  !python ml_training/run_colab_stgcn.py --epochs 40 --batch_size 32
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.st_gcn import STGCNClassifier
from ml.training.train_stgcn import STGCNDataset, train_epoch, validate


def find_rwf_dataset(search_dir: str = "data") -> Path:
    for root, dirs, _ in os.walk(search_dir):
        if "train" in dirs and "val" in dirs:
            if (Path(root) / "train" / "Fight").is_dir():
                return Path(root)
    # Check default paths
    for cand in ["data/RWF-2000", "data/rwf-2000", "data"]:
        p = Path(cand)
        if (p / "train" / "Fight").is_dir():
            return p
    raise FileNotFoundError(f"Could not find RWF-2000 dataset in {search_dir}")


def extract_51dim_features(
    video_path: str,
    pose_model: YOLO,
    device: str,
    num_frames: int = 16,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, 51), dtype=np.float32)

    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames = []
    current_idx = 0
    pos = 0

    while cap.isOpened() and pos < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx == indices[pos]:
            frames.append(frame)
            pos += 1
        current_idx += 1
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    results = pose_model.predict(frames, verbose=False, device=device)
    window = []
    for res in results:
        feat = np.zeros(51, dtype=np.float32)
        if res.keypoints is not None and len(res.keypoints) > 0:
            kp = res.keypoints.xyn[0].cpu().numpy()  # (17, 2)
            feat[:34] = kp.flatten()[:34]
            if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None:
                conf = res.keypoints.conf[0].cpu().numpy()
                feat[34:51] = conf[:17]
            else:
                feat[34:51] = 1.0
        window.append(feat)

    return np.array(window, dtype=np.float32)  # (16, 51)


def extract_dataset(dataset_root: Path, processed_dir: Path, pose_model: YOLO, device: str):
    print("\n--- Step 1: Extracting 51-dim Keypoints (X, Y, Confidence) ---")
    for split in ["train", "val"]:
        for cls in ["normal", "violent"]:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

        split_dir = dataset_root / split

        # NonFight -> normal
        nonfight_vids = glob.glob(str(split_dir / "NonFight" / "*.avi")) + glob.glob(str(split_dir / "NonFight" / "*.mp4"))
        for v in tqdm(nonfight_vids, desc=f"{split} NonFight (Normal)"):
            out_file = processed_dir / split / "normal" / (Path(v).stem + ".npy")
            if not out_file.exists():
                feat = extract_51dim_features(v, pose_model, device)
                np.save(out_file, feat)

        # Fight -> violent
        fight_vids = glob.glob(str(split_dir / "Fight" / "*.avi")) + glob.glob(str(split_dir / "Fight" / "*.mp4"))
        for v in tqdm(fight_vids, desc=f"{split} Fight (Violent)"):
            out_file = processed_dir / split / "violent" / (Path(v).stem + ".npy")
            if not out_file.exists():
                feat = extract_51dim_features(v, pose_model, device)
                np.save(out_file, feat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--skip_extraction", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"  SmartOmniSentinel ST-GCN End-to-End Cloud Trainer")
    print(f"  Device: {device.upper()} | Epochs: {args.epochs}")
    print(f"=======================================================")

    processed_dir = Path("data/processed_stgcn")

    if not args.skip_extraction:
        dataset_root = find_rwf_dataset(args.data_dir)
        print(f"Dataset found at: {dataset_root}")
        pose_model = YOLO("yolov8n-pose.pt").to(device)
        extract_dataset(dataset_root, processed_dir, pose_model, device)
    else:
        print("Skipping extraction; using existing features in data/processed_stgcn")

    train_ds = STGCNDataset(str(processed_dir / "train"), augment=True)
    val_ds = STGCNDataset(str(processed_dir / "val"), augment=False)
    print(f"\nLoaded {len(train_ds)} train samples and {len(val_ds)} validation samples.")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = STGCNClassifier(in_channels=3, num_classes=2, proximity_dim=0).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs("ml/checkpoints", exist_ok=True)
    best_f1 = 0.0
    ckpt_path = "ml/checkpoints/stgcn_best.pt"

    print("\n--- Step 2: Training Spatio-Temporal Graph Convolutions (ST-GCN) ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, torch.device(device))
        val_m = validate(model, val_loader, criterion, torch.device(device))
        scheduler.step()
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs} [{dt:.1f}s] - "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.3f} "
            f"P: {val_m['precision']:.3f} R: {val_m['recall']:.3f} F1: {val_m['f1_score']:.3f}"
        )

        if val_m["f1_score"] > best_f1:
            best_f1 = val_m["f1_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "f1_score": best_f1,
                    "val_metrics": val_m,
                },
                ckpt_path,
            )
            print(f"  >>> Saved new best ST-GCN model: {ckpt_path} (F1: {best_f1:.4f})")

    print(f"\n=======================================================")
    print(f"  Training Complete! Best ST-GCN F1-Score: {best_f1:.4f}")
    print(f"  Saved Checkpoint: {ckpt_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
