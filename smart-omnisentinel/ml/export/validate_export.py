"""
ml/export/validate_export.py
------------------------------
Validates that an ONNX export matches PyTorch output within tolerance.
Run after export_onnx.py to confirm the export is correct before deployment.

Usage:
    python -m ml.export.validate_export \
        --checkpoint ml/checkpoints/temporal_v1.pt \
        --onnx ml/checkpoints/temporal_v1.onnx \
        --n_tests 100
"""
from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.models.temporal_model import TemporalClassifier, WINDOW_SIZE, FEATURE_DIM


def validate(
    checkpoint_path: str,
    onnx_path: str,
    n_tests: int = 100,
    tolerance: float = 1e-4,
) -> bool:
    """
    Run N random inputs through both PT and ONNX and compare outputs.
    Returns True if all tests pass within tolerance.
    """
    import onnxruntime as ort

    print(f"Validating ONNX export...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  ONNX:       {onnx_path}")
    print(f"  Tests:      {n_tests}")
    print(f"  Tolerance:  {tolerance}\n")

    # Load PyTorch model
    model = TemporalClassifier()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load ONNX session
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_diff = 0.0
    failures = 0

    for i in range(n_tests):
        # Random input
        x_np = np.random.randn(1, WINDOW_SIZE, FEATURE_DIM).astype(np.float32)
        x_pt = torch.tensor(x_np)

        # PyTorch output (softmax probabilities)
        with torch.no_grad():
            pt_logits = model(x_pt)
            pt_probs = torch.softmax(pt_logits, dim=-1).numpy()

        # ONNX output
        ort_logits = session.run(None, {input_name: x_np})[0]
        ort_probs = np.exp(ort_logits) / np.exp(ort_logits).sum(axis=-1, keepdims=True)

        diff = np.abs(pt_probs - ort_probs).max()
        max_diff = max(max_diff, diff)

        if diff > tolerance:
            failures += 1
            print(f"  FAIL test {i+1}: max diff = {diff:.2e}")

    print(f"Results: {n_tests - failures}/{n_tests} passed")
    print(f"Max difference: {max_diff:.2e}")

    if failures == 0:
        print(f"\n✓ All tests passed (tolerance: {tolerance:.0e})")
        return True
    else:
        print(f"\n✗ {failures} tests failed")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ml/checkpoints/temporal_v1.pt")
    parser.add_argument("--onnx", default="ml/checkpoints/temporal_v1.onnx")
    parser.add_argument("--n_tests", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    for path in [args.checkpoint, args.onnx]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    ok = validate(args.checkpoint, args.onnx, args.n_tests, args.tolerance)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
