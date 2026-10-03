"""Check early initialization and child inheritance in fresh interpreters."""

import json
from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
@pytest.mark.parametrize("initial", [None, "0", "1"])
def test_telemetry_environment_and_child_inheritance(platform, initial):
    env = os.environ.copy()
    env.pop("ORT_DISABLE_TELEMETRY", None)
    if initial is not None:
        env["ORT_DISABLE_TELEMETRY"] = initial
    script = '''
import json, os, runpy, subprocess, sys
sys.platform = sys.argv[1]
runpy.run_path("graxpert/runtime_environment.py")
assert "onnxruntime" not in sys.modules
child = subprocess.check_output([sys.executable, "-c", "import os,json; print(json.dumps(os.environ.get('ORT_DISABLE_TELEMETRY')))"])
print(json.dumps([os.environ.get("ORT_DISABLE_TELEMETRY"), json.loads(child)]))
'''
    result = subprocess.check_output([sys.executable, "-c", script, platform], cwd=ROOT, env=env, text=True)
    expected = "1"
    assert json.loads(result) == [expected, expected]


def test_main_configures_environment_before_loading_onnx_consumer():
    # Stop at the first ONNX consumer; do not require model credentials or GUI.
    script = '''
import importlib.abc, os, runpy, sys
sys.platform = "darwin"
os.environ["ORT_DISABLE_TELEMETRY"] = "0"
class ReachedConsumer(Exception): pass
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "graxpert.ai_model_handling":
            assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
            assert "onnxruntime" not in sys.modules
            raise ReachedConsumer()
sys.meta_path.insert(0, Guard())
try:
    runpy.run_path("graxpert/main.py", run_name="test_startup")
except ReachedConsumer:
    pass
else:
    raise AssertionError("ONNX consumer was not reached")
'''
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_shared_runtime_disables_telemetry_before_use(tmp_path):
    # Substitute only ORT; exercise GraXpert's actual shared import in a clean
    # process and verify the environment was configured before ORT imports.
    fake = tmp_path / "onnxruntime.py"
    fake.write_text('''
import os
assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
disabled = False
def disable_telemetry_events():
    global disabled
    disabled = True
''')
    script = '''
import os, sys
os.environ["ORT_DISABLE_TELEMETRY"] = "0"
sys.path.insert(0, sys.argv[1])
from graxpert.onnx_runtime import ort
assert ort.disabled
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=ROOT, check=True)
