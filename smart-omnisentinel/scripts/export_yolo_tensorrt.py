"""
scripts/export_yolo_tensorrt.py
-------------------------------
Day 1 Deliverable: Export YOLO models to TensorRT Engine for Jetson / NVIDIA GPU.

Usage:
  python scripts/export_yolo_tensorrt.py --model yolov8n-pose.pt --half
  python scripts/export_yolo_tensorrt.py --model ml/checkpoints/yolov8n.pt --half
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def export_tensorrt_engine(
    model_path: str,
    output_dir: str = "models",
    half: bool = True,
    device: int = 0,
    workspace_gb: float = 2.0,
) -> None:
    print(f"\n=======================================================")
    print(f"  SmartOmniSentinel — YOLO to TensorRT Engine Exporter")
    print(f"  Source Model: {model_path}")
    print(f"  Precision:    {'FP16 (Half)' if half else 'FP32'}")
    print(f"  Device:       CUDA:{device}")
    print(f"=======================================================\n")

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(model_path).stem

    # Method 1: Ultralytics native TensorRT engine exporter
    try:
        from ultralytics import YOLO
        import torch

        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available on this host.")
            print("TensorRT engines must be built on the target GPU (e.g. Jetson Orin / NVIDIA RTX).")
            print(f"Exporting to ONNX instead for hardware-accelerated ONNX Runtime...")
            model = YOLO(model_path)
            onnx_path = model.export(format="onnx", imgsz=640, dynamic=True)
            target_onnx = os.path.join(output_dir, f"{stem}.onnx")
            shutil.copy(onnx_path, target_onnx)
            print(f"Successfully saved ONNX model: {target_onnx}")
            return

        print("Exporting via Ultralytics engine exporter...")
        model = YOLO(model_path)
        engine_path = model.export(
            format="engine",
            imgsz=640,
            half=half,
            device=device,
            workspace=workspace_gb,
        )
        target_engine = os.path.join(output_dir, f"{stem}.engine")
        if os.path.exists(engine_path):
            shutil.copy(engine_path, target_engine)
            print(f"SUCCESS: TensorRT engine exported to {target_engine}")
        return

    except Exception as e:
        print(f"Ultralytics engine export failed or not supported: {e}")

    # Method 2: Fallback to trtexec CLI
    onnx_file = os.path.join(output_dir, f"{stem}.onnx")
    if not os.path.exists(onnx_file):
        print(f"Generating intermediate ONNX model: {onnx_file}")
        from ultralytics import YOLO
        model = YOLO(model_path)
        p = model.export(format="onnx", imgsz=640, dynamic=False)
        shutil.copy(p, onnx_file)

    target_engine = os.path.join(output_dir, f"{stem}.engine")
    fp16_flag = "--fp16" if half else ""
    cmd = f"trtexec --onnx={onnx_file} --saveEngine={target_engine} {fp16_flag} --memPoolSize=workspace:{int(workspace_gb * 1024)}"

    print(f"\nExecuting trtexec command:\n  {cmd}\n")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"SUCCESS: TensorRT engine exported to {target_engine}")
    except Exception as exc:
        print(f"trtexec execution failed ({exc}).")
        print("Note: To run on Jetson, transfer models/*.onnx to Jetson and run the trtexec command directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    export_tensorrt_engine(
        model_path=args.model,
        output_dir=args.output_dir,
        half=args.half,
        device=args.device,
    )

