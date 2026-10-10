"""Integration checks for the GraXpert background-extraction entry point."""

import numpy as np

from graxpert.background_extraction import extract_background
from graxpert.sample_free_background import (
    SampleFreeParameters,
    extract_sample_free_background,
)


class ProgressRecorder:
    def __init__(self):
        self.total = 0

    def update(self, amount):
        assert isinstance(amount, int)
        self.total += amount


def test_sample_free_dispatch_matches_core_and_completes_progress():
    y, x = np.mgrid[0:64, 0:80]
    base = 0.1 + 0.002 * x - 0.001 * y
    object_signal = 0.2 * np.exp(-((x - 42) ** 2 + (y - 30) ** 2) / 90.0)
    mono = (base + object_signal).astype(np.float32)
    image = np.stack((mono, mono * 0.9, mono * 1.1), axis=-1)
    expected_source = image.copy()
    parameters = SampleFreeParameters(downsample=2)
    _, expected_corrected, expected_simplified_model = extract_sample_free_background(
        expected_source,
        parameters=parameters,
        correction="subtract",
        return_simplified_model=True,
    )
    progress = ProgressRecorder()
    components = {}

    background = extract_background(
        image,
        np.empty((0, 2)),
        "Sample-free",
        0.0,
        1,
        25,
        "thin_plate",
        3,
        "Subtraction",
        None,
        progress=progress,
        ai_gpu_acceleration=False,
        sample_free_parameters=parameters,
        sample_free_components=components,
    )

    assert background.shape == image.shape
    assert np.allclose(image, np.clip(expected_corrected, 0.0, 1.0))
    assert np.allclose(components["simplified_model"], expected_simplified_model)
    assert progress.total == 100
