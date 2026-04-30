"""
ml/export/export_onnx.py
--------------------------
Export the trained TemporalClassifier to ONNX format.
ONNX enables deployment with onnxruntime (CPU or GPU) without
requiring PyTorch on the inference host.

On Jetson: use export_tensorrt.py after this to convert ONNX → TensorRT.

Usage:
    python -m ml.export.export_onnx \
        --checkpoint ml/checkpoints/temporal_v1.pt \
        --output ml/checkpoints/temporal_v1.onnx \
        --validate
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

WINDOW_SIZE = 16
FEATURE_DIM = 55
BATCH_SIZE = 1


def export(
    checkpoint_path: str,
    output_path: str,
    validate: bool = True,
    opset_version: int = 17,
) -> None:
    import torch
    from ml.models.temporal_model import TemporalClassifier

    print(f"Exporting TemporalClassifier to ONNX")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Output:      {output_path}")
    print(f"  Opset:       {opset_version}")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    # Load model
    model = TemporalClassifier()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"  Model loaded. Epoch: {state.get('epoch', 'N/A')}")

    # Dummy input with correct shape
    dummy_input = torch.randn(BATCH_SIZE, WINDOW_SIZE, FEATURE_DIM)

    # Dynamic axes: allow variable batch size at inference
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    print(f"  ONNX export complete: {output_path}")

    # Validate ONNX graph
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX graph validated ✓")

    if validate:
        _validate_parity(model, output_path, dummy_input)


def _validate_parity(
    torch_model,
    onnx_path: str,
    dummy_input,
) -> None:
    """Check that ONNX output matches PyTorch output within tolerance."""
    import torch
    import onnxruntime as ort

    print("\nValidating output parity (PyTorch vs ONNX)...")

    # PyTorch output
    with torch.no_grad():
        pt_output = torch_model(dummy_input).numpy()

    # ONNX output
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_output = sess.run(
        None, {"input": dummy_input.numpy()}
    )[0]

    max_diff = float(np.abs(pt_output - ort_output).max())
    mean_diff = float(np.abs(pt_output - ort_output).mean())

    print(f"  Max absolute difference:  {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("  ✓ PASS — outputs match within tolerance (1e-4)")
    elif max_diff < 1e-3:
        print("  ⚠  WARN — small numerical drift; acceptable for FP32 inference")
    else:
        print("  ✗ FAIL — significant output mismatch; check model or opset version")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TemporalClassifier to ONNX")
    parser.add_argument("--checkpoint", type=str, default="ml/checkpoints/temporal_v1.pt")
    parser.add_argument("--output", type=str, default="ml/checkpoints/temporal_v1.onnx")
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.validate, args.opset)
