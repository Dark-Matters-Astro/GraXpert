"""ONNX Runtime acceleration for the sample-free background method."""

from __future__ import annotations

import threading

import numpy as np
import onnxruntime as ort

from graxpert.resource_utils import resource_path


MODEL_PATH = resource_path("models/sample_free_gaussian.onnx")
_session = None
_session_lock = threading.Lock()


def get_sample_free_session():
    """Return the shared ONNX Runtime session for sample-free filtering.

    This graph deliberately uses the CPU execution provider.  CoreML supports
    only a small subset of its dynamic CumSum/Slice graph and would move data
    between CoreML and CPU six times per inference.  ORT's optimized CPU graph
    is both faster and portable across GraXpert's supported platforms.
    """

    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                options = ort.SessionOptions()
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                options.log_severity_level = 3
                _session = ort.InferenceSession(
                    MODEL_PATH,
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
    return _session


def run_sample_free_gaussian(image, radius):
    """Run one complete three-pass box-Gaussian approximation in ONNX."""

    source = np.ascontiguousarray(image, dtype=np.float32)
    result = get_sample_free_session().run(
        ["background"],
        {
            "image": source[np.newaxis, np.newaxis, :, :],
            "radius": np.asarray([radius], dtype=np.int64),
        },
    )[0]
    return result[0, 0]
