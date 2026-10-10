"""Tests for the automatic sample-free background model."""

import threading
import unittest

import numpy as np

from graxpert.sample_free_background import (
    SampleFreeParameters,
    _ag_gauss,
    _ag_gauss_numpy,
    _ag_sigma,
    _box_blur_edge,
    _poly_fit,
    estimate_channel_background,
    extract_sample_free_background,
)
from graxpert.sample_free_onnx import get_sample_free_session


def synthetic_scene(height=128, width=160, seed=20260903):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    xn = x / (width - 1)
    yn = y / (height - 1)

    background = (
        0.12
        + 0.32 * xn
        - 0.12 * yn
        + 0.035 * np.sin(2.0 * np.pi * (0.7 * xn + 0.25 * yn))
    )
    nebula = 0.16 * np.exp(
        -(
            (x - 0.55 * width) ** 2 / (2 * (0.18 * width) ** 2)
            + (y - 0.48 * height) ** 2 / (2 * (0.24 * height) ** 2)
        )
    )
    stars = np.zeros((height, width), dtype=np.float64)
    for _ in range(28):
        sx = rng.uniform(0, width - 1)
        sy = rng.uniform(0, height - 1)
        amplitude = rng.uniform(0.08, 0.45)
        sigma = rng.uniform(0.7, 1.7)
        stars += amplitude * np.exp(
            -((x - sx) ** 2 + (y - sy) ** 2) / (2 * sigma * sigma)
        )
    signal = nebula + stars
    image = background + signal + rng.normal(0.0, 0.003, (height, width))
    object_mask = (nebula > 0.025) | (stars > 0.015)
    return image.astype(np.float32), background.astype(np.float32), object_mask


class SampleFreeBackgroundTests(unittest.TestCase):
    def test_box_blur_matches_replicated_edge_definition(self):
        source = np.arange(35, dtype=np.float32).reshape(5, 7)
        for axis in (0, 1):
            result = _box_blur_edge(source, radius=2, axis=axis)
            expected = np.empty_like(source)
            for y in range(source.shape[0]):
                for x in range(source.shape[1]):
                    values = []
                    for offset in range(-2, 3):
                        yy = min(max(y + offset, 0), source.shape[0] - 1)
                        xx = min(max(x + offset, 0), source.shape[1] - 1)
                        values.append(source[yy, x] if axis == 0 else source[y, xx])
                    expected[y, x] = np.mean(values)
            self.assertTrue(np.allclose(result, expected, atol=2e-6))

    def test_gaussian_approximation_preserves_constant(self):
        source = np.full((31, 47), 0.25, dtype=np.float32)
        result = _ag_gauss(source, _ag_sigma(4, 2))
        self.assertTrue(np.allclose(result, source, atol=2e-6))

    def test_onnx_gaussian_matches_numpy_reference(self):
        source = np.random.default_rng(20260905).normal(
            size=(127, 173)
        ).astype(np.float32)
        for radius in (1, 3, 11, 31):
            sigma = _ag_sigma(radius, 3)
            expected = _ag_gauss_numpy(source, sigma)
            result = _ag_gauss(source, sigma)
            self.assertTrue(np.allclose(result, expected, rtol=2e-5, atol=2e-6))

    def test_sample_free_onnx_session_uses_cpu_provider(self):
        self.assertEqual(
            get_sample_free_session().get_providers(), ["CPUExecutionProvider"]
        )

    def test_polynomial_degree_one_recovers_plane(self):
        y, x = np.mgrid[0:43, 0:61]
        source = (0.2 + 0.003 * x - 0.002 * y).astype(np.float32)
        mask = np.ones_like(source, dtype=bool)
        mask[10:20, 20:40] = False
        model = _poly_fit(source, mask, degree=1)
        self.assertLess(float(np.max(np.abs(model - source))), 2e-6)

    def test_simplified_model_returns_finite_rgb_results(self):
        image, _, _ = synthetic_scene(64, 80)
        rgb = np.stack((image, image * 0.9, image * 1.1), axis=-1)
        parameters = SampleFreeParameters(simplified=True, degree=1, downsample=2)
        background, corrected, simplified_model = extract_sample_free_background(
            rgb, parameters, return_simplified_model=True
        )
        self.assertEqual(background.shape, rgb.shape)
        self.assertEqual(corrected.shape, rgb.shape)
        self.assertEqual(simplified_model.shape, rgb.shape)
        self.assertTrue(np.all(np.isfinite(background)))
        self.assertTrue(np.all(np.isfinite(corrected)))
        self.assertTrue(np.all(np.isfinite(simplified_model)))
        self.assertGreater(float(np.max(np.abs(background - simplified_model))), 1e-5)

    def test_parallel_rgb_matches_independent_channels_and_progress_is_monotonic(self):
        image, _, _ = synthetic_scene(64, 80)
        rgb = np.stack((image, image * 0.9, image * 1.1), axis=-1)
        parameters = SampleFreeParameters(simplified=True, degree=1, downsample=2)
        expected = np.stack(
            [
                estimate_channel_background(rgb[:, :, channel], parameters)
                for channel in range(3)
            ],
            axis=-1,
        )
        progress_values = []
        callback_threads = set()
        main_thread = threading.get_ident()

        def record_progress(value, stage):
            progress_values.append(value)
            callback_threads.add(threading.get_ident())

        background, _ = extract_sample_free_background(
            rgb,
            parameters,
            progress=record_progress,
        )
        self.assertTrue(np.array_equal(background, expected))
        self.assertEqual(callback_threads, {main_thread})
        self.assertTrue(
            all(
                earlier <= later
                for earlier, later in zip(progress_values, progress_values[1:])
            )
        )
        self.assertEqual(progress_values[-1], 1.0)

    def test_disabled_simplified_model_returns_no_component(self):
        image, _, _ = synthetic_scene(64, 80)
        background, corrected, simplified_model = extract_sample_free_background(
            image,
            SampleFreeParameters(simplified=False, downsample=2),
            return_simplified_model=True,
        )
        self.assertEqual(background.shape, image.shape)
        self.assertEqual(corrected.shape, image.shape)
        self.assertIsNone(simplified_model)

    def test_tiny_image_falls_back_from_downsampling(self):
        y, x = np.mgrid[0:5, 0:7]
        image = (0.2 + 0.01 * x + 0.02 * y).astype(np.float32)
        parameters = SampleFreeParameters(simplified=True, degree=1, downsample=4)
        background, corrected = extract_sample_free_background(image, parameters)
        self.assertEqual(background.shape, image.shape)
        self.assertEqual(corrected.shape, image.shape)
        self.assertTrue(np.all(np.isfinite(background)))
        self.assertTrue(np.all(np.isfinite(corrected)))

    def test_simplified_model_improves_strong_gradient(self):
        image, truth, object_mask = synthetic_scene(96, 128)
        regular, _ = extract_sample_free_background(
            image, SampleFreeParameters(simplified=False, downsample=2)
        )
        simplified, _ = extract_sample_free_background(
            image, SampleFreeParameters(simplified=True, degree=1, downsample=2)
        )
        clear = ~object_mask
        regular_rmse = float(np.sqrt(np.mean((regular[clear] - truth[clear]) ** 2)))
        simplified_rmse = float(
            np.sqrt(np.mean((simplified[clear] - truth[clear]) ** 2))
        )
        self.assertLess(simplified_rmse, regular_rmse * 0.35)

    def test_nonfinite_source_pixels_are_preserved(self):
        image, _, _ = synthetic_scene(64, 80)
        image[5, 7] = np.nan
        parameters = SampleFreeParameters(simplified=True, degree=1, downsample=2)
        background, corrected = extract_sample_free_background(image, parameters)
        self.assertTrue(np.all(np.isfinite(background)))
        self.assertTrue(np.isnan(corrected[5, 7]))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            SampleFreeParameters(degree=0).validate()
        with self.assertRaises(ValueError):
            SampleFreeParameters(downsample=3).validate()


if __name__ == "__main__":
    unittest.main()
