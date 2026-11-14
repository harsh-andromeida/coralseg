"""
Optional fp16 copies of the CoralSCOP/SAM ONNX models (image encoder and mask decoder) for ONNX Runtime CUDA.

Both models are compute-bound in fp32 on mid-range GPUs (T4: ~410 ms encoder + ~880 ms decoder per keyframe).
fp16 lets tensor cores do the work. Converted files are cached next to the originals; inputs/outputs stay fp32
(`keep_io_types`) so callers are unchanged. Uses ONNX Runtime's own converter, which handles the attention Cast
nodes that the generic onnxconverter-common converter mistypes.
"""

import os
from typing import Any, Optional

import onnxruntime as ort


def build_fp16_session(fp32_path: str, providers: Any, sess_options: Optional[ort.SessionOptions] = None) -> ort.InferenceSession:
    fp16_path = os.path.splitext(fp32_path)[0] + "_fp16_ort.onnx"
    if not os.path.isfile(fp16_path):
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        model = onnx.load(fp32_path)
        model16 = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
        onnx.save(model16, fp16_path)
    return ort.InferenceSession(fp16_path, sess_options=sess_options, providers=providers)


build_fp16_encoder_session = build_fp16_session  # backwards-compatible name
