"""Launch the source checkout with isolated local test data."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import appdirs


TEST_HOME = PROJECT_ROOT / ".local-test-data"
CONFIG_HOME = TEST_HOME / "config"
DATA_HOME = TEST_HOME / "data"
LOG_HOME = TEST_HOME / "logs"
for directory in (CONFIG_HOME, DATA_HOME, LOG_HOME):
    directory.mkdir(parents=True, exist_ok=True)

appdirs.user_config_dir = lambda *args, **kwargs: str(CONFIG_HOME)
appdirs.user_data_dir = lambda *args, **kwargs: str(DATA_HOME)
appdirs.user_log_dir = lambda *args, **kwargs: str(LOG_HOME)

import graxpert.ai_model_handling as ai_model_handling


# The sample-free test does not need access to GraXpert's model-download bucket.
ai_model_handling.client = None

from graxpert.main import main
from graxpert.mp_logging import configure_logging


if __name__ == "__main__":
    configure_logging()
    main()
