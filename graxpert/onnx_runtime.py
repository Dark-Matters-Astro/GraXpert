"""Shared ONNX Runtime import with telemetry disabled before session creation."""

# Must precede the ONNX import: the runtime may initialize telemetry on import.
import graxpert.runtime_environment  # noqa: F401

import onnxruntime as ort


# Complement the early environment setting with the public API, including for
# Windows ETW. This alone is too late to prevent POSIX initialization events.
ort.disable_telemetry_events()
