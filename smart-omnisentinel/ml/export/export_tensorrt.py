"""
ml/export/export_tensorrt.py
------------------------------
Convert ONNX model to TensorRT engine for Jetson deployment.
Requires: TensorRT installed (comes with JetPack on Jetson).

Steps:
    1. First run export_onnx.py to get .onnx file
    2. Run this script on the Jetson device (not x86)
    3. TensorRT engine is hardware-specific — build ON the target device

Usage (on Jetson):
    python -m ml.export.export_tensorrt \
        --onnx ml/checkpoints/temporal_v1.onnx \
        --output ml/checkpoints/temporal_v1.trt \
        --fp16

Note: INT8 quantization requires a calibration dataset and is not
recommended for student/prototype deployments due to accuracy risk.
Stick with FP16 which gives ~2x speedup on Jetson Tensor Cores with
no accuracy loss.
"""
from __future__ import annotations

import argparse
import os
import sys

from core.logger import get_logger

logger = get_logger(__name__)


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    max_batch_size: int = 1,
    workspace_gb: float = 1.0,
) -> None:
    print(f"\nSmartOmniSentinel — TensorRT Export")
    print(f"ONNX input:  {onnx_path}")
    print(f"TRT output:  {output_path}")
    print(f"Mode:        {'FP16' if fp16 else 'FP32'}")
    print(f"Max batch:   {max_batch_size}")
    print(f"Workspace:   {workspace_gb}GB\n")

    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: TensorRT not found.")
        print("TensorRT is available on Jetson devices with JetPack installed.")
        print("On x86, install: pip install tensorrt")
        print("\nAlternative: Use trtexec command line tool:")
        print(f"  trtexec --onnx={onnx_path} --saveEngine={output_path} --fp16")
        sys.exit(1)

    logger_trt = trt.Logger(trt.Logger.INFO)

    with trt.Builder(logger_trt) as builder:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(workspace_gb * (1 << 30))
        )

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 mode enabled")
        else:
            print("  FP16 not available — using FP32")

        # Parse ONNX network
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with builder.create_network(network_flags) as network:
            parser = trt.OnnxParser(network, logger_trt)

            print(f"Parsing ONNX model: {onnx_path}")
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(f"  ONNX parse error: {parser.get_error(error)}")
                    sys.exit(1)
            print("✓ ONNX parsed successfully")

            # Optimization profile for dynamic batch
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            # Input shape: (batch, window_size=16, feature_dim=55)
            profile.set_shape(
                input_name,
                min=(1, 16, 55),
                opt=(1, 16, 55),
                max=(max_batch_size, 16, 55),
            )
            config.add_optimization_profile(profile)

            print("Building TensorRT engine (this may take several minutes)...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                print("ERROR: Engine build failed")
                sys.exit(1)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(serialized_engine)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✓ TensorRT engine saved: {output_path} ({size_mb:.1f}MB)")
        print("\nNext steps:")
        print("  - Set INFERENCE__USE_TENSORRT=true in .env")
        print("  - The inference service will load the .trt engine on startup")
        print("  - Benchmark with: python scripts/benchmark_inference.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="ml/checkpoints/temporal_v1.onnx")
    parser.add_argument("--output", default="ml/checkpoints/temporal_v1.trt")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--fp32", dest="fp16", action="store_false")
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--workspace_gb", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX file not found: {args.onnx}")
        print("Run export_onnx.py first.")
        sys.exit(1)

    export_tensorrt(
        onnx_path=args.onnx,
        output_path=args.output,
        fp16=args.fp16,
        max_batch_size=args.max_batch_size,
        workspace_gb=args.workspace_gb,
    )


if __name__ == "__main__":
    main()
