# ONNX Runtime telemetry and shutdown

GraXpert disables ONNX Runtime telemetry on every platform. This avoids a
reproduced macOS shutdown abort in the bundled Microsoft 1DS telemetry worker.
The image calculation itself is unchanged.

`graxpert/runtime_environment.py` sets `ORT_DISABLE_TELEMETRY=1` before ONNX
Runtime loads. It uses only the standard library and is both imported at the
start of `main.py` and registered as an early runtime hook in the PyInstaller
specifications. Spawned helper processes inherit this environment. An existing
value of `0` is deliberately overridden.

All ONNX consumers import `ort` through `graxpert/onnx_runtime.py`. This shared
module configures the environment before importing ONNX Runtime and calls its
public `disable_telemetry_events()` API before any GraXpert inference session
is created. The API complements the POSIX environment setting for other
providers, including Windows ETW. Calling the API alone is too late to prevent
POSIX initialization events. Older runtime versions may not implement the
environment switch; this change does not claim to repair every runtime version.

References:
- https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
- https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.disable_telemetry_events
- https://pyinstaller.org/en/stable/when-things-go-wrong.html#changing-runtime-behavior

## Validation

A controlled test on macOS with the installed GraXpert 3.2.0.dev1 application
processed the same RGB M42 FITS image 20 times per arm, alternating arms:

- Environment variable absent: 9 shutdown crash reports in 20 runs.
- `ORT_DISABLE_TELEMETRY=1` before launch: 0 crash reports in 20 runs.
- All 40 output images were pixel-identical.
- Every main-process exit code was 0, including runs with a helper crash.
- A captured crashed PID was identified as `multiprocessing.resource_tracker`.
- All nine reports showed the same ONNX telemetry mutex failure.

This A/B test used an external environment setting on the existing application,
not a newly packaged application containing this patch. It validates the
workaround on that installation. Tests in `test_runtime_environment.py` cover
early startup ordering, overriding a previous value, subprocess inheritance,
and the public API call. The Linux/Windows platform cases in those tests are
simulated platform values, not native OS runs.

Before release, build the patched application and repeat CLI and PixInsight
invocations on macOS, and run native Windows/Linux smoke tests. Check for new
helper crash reports as well as output files and main-process exit status.
