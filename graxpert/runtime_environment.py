"""Early process configuration; also used as a PyInstaller runtime hook.

Keep this module standard-library-only: it must run before ONNX Runtime is
imported, including when a frozen app starts multiprocessing helper processes.
"""

import os


# Prevent POSIX telemetry initialization, including in spawned helper processes.
# Set this on every platform before ORT loads; the shared runtime module also
# disables telemetry through the public API for providers such as Windows ETW.
# Override an existing value of 0 to keep application shutdown deterministic.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
