"""Automatic sample-free background extraction adapted from Siril.

This module follows Siril's automatic-gradient implementation, including the
revised simplified mode introduced on 2026-09-03.  In simplified mode a robust
polynomial is fitted first; a multiscale model is then fitted to the residual.

Original implementation and current source:
  Copyright (C) 2005-2011 Francois Meyer
  Copyright (C) 2012-2026 team free-astro (see Siril's AUTHORS file)
  https://gitlab.com/free-astro/siril/-/blob/master/src/algos/background_extraction.c

Simplified-model revision by Cyril Richard and the Siril team:
  https://gitlab.com/free-astro/siril/-/commit/5555df299f96923551bcc093c0e5e1a022648061

The GraXpert team thanks Cyril Richard and the Siril team for their work and
for supporting continued collaboration between both projects.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Callable

import numpy as np

from graxpert.sample_free_onnx import (
    get_sample_free_session,
    run_sample_free_gaussian,
)


AG_HIGH_K = 2.0
AG_LOW_K = 4.0
AG_N_ITER = 20
AG_PASSES = 3

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class SampleFreeParameters:
    """Parameters corresponding to Siril's automatic-model defaults."""

    scale: float = 5.0
    smoothness: float = 1.0
    protect: bool = True
    protect_threshold: float = 0.05
    protect_amount: float = 0.5
    simplified: bool = True
    degree: int = 1
    downsample: int = 4

    def validate(self):
        if not 1.0 <= self.scale <= 10.0:
            raise ValueError("scale must be between 1 and 10")
        if self.smoothness < 0.0:
            raise ValueError("smoothness must be non-negative")
        if not 0.0 <= self.protect_threshold <= 1.0:
            raise ValueError("protect_threshold must be between 0 and 1")
        if not 0.0 <= self.protect_amount <= 1.0:
            raise ValueError("protect_amount must be between 0 and 1")
        if not 1 <= self.degree <= 6:
            raise ValueError("degree must be between 1 and 6")
        if self.downsample not in (1, 2, 4, 8):
            raise ValueError("downsample must be one of 1, 2, 4, 8")


def _lround_positive(value):
    return int(math.floor(value + 0.5))


def _ag_sigma(radius, passes):
    if radius < 1:
        return 0.0
    return math.sqrt(passes * radius * (radius + 1) / 3.0)


def _box_blur_edge(image, radius, axis):
    """Apply one running-sum box blur with replicated edge pixels."""

    if radius < 1:
        return image.astype(np.float32, copy=True)

    if axis == 1:
        padded = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
        cumulative = np.cumsum(padded, axis=1, dtype=np.float64)
        cumulative = np.pad(cumulative, ((0, 0), (1, 0)), mode="constant")
        width = 2 * radius + 1
        result = cumulative[:, width:] - cumulative[:, :-width]
    elif axis == 0:
        padded = np.pad(image, ((radius, radius), (0, 0)), mode="edge")
        cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
        cumulative = np.pad(cumulative, ((1, 0), (0, 0)), mode="constant")
        width = 2 * radius + 1
        result = cumulative[width:, :] - cumulative[:-width, :]
    else:
        raise ValueError("axis must be 0 or 1")

    result *= 1.0 / (2 * radius + 1)
    return result.astype(np.float32, copy=False)


def _ag_gauss_radius(sigma):
    radius = _lround_positive(
        (-1.0 + math.sqrt(1.0 + 12.0 * sigma * sigma / AG_PASSES)) / 2.0
    )
    return radius


def _ag_gauss_numpy(image, sigma):
    """NumPy reference for Siril's three-pass box Gaussian approximation."""

    radius = _ag_gauss_radius(sigma)
    if sigma < 0.25 or radius < 1:
        return image.astype(np.float32, copy=True)

    result = image.astype(np.float32, copy=True)
    for _ in range(AG_PASSES):
        result = _box_blur_edge(result, radius, axis=1)
        result = _box_blur_edge(result, radius, axis=0)
    return result


def _ag_gauss(image, sigma):
    """Run Siril's three-pass box Gaussian approximation with ONNX Runtime."""

    radius = _ag_gauss_radius(sigma)
    if sigma < 0.25 or radius < 1:
        return image.astype(np.float32, copy=True)
    return run_sample_free_gaussian(image, radius)


def _mad_sigma(values):
    if values.size == 0:
        return 0.0, 1e-12
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, 1.4826 * mad + 1e-12


def _inpaint_lowpass(image, mask, radius, n_fill=10, passes=2):
    kept = int(np.count_nonzero(mask))
    if kept == 0:
        return np.zeros_like(image, dtype=np.float32)

    sigma = _ag_sigma(radius, passes)
    if kept == mask.size:
        return _ag_gauss(image, sigma)

    known_mean = float(np.mean(image[mask], dtype=np.float64))
    filled = np.where(mask, image, known_mean).astype(np.float32)
    for _ in range(n_fill):
        smoothed = _ag_gauss(filled, sigma)
        filled = np.where(mask, image, smoothed).astype(np.float32)
    return _ag_gauss(filled, sigma)


def _structure_mask(residual, model_radius, protect_threshold, protect_amount):
    detected = (residual > protect_threshold).astype(np.float32)
    if not np.any(detected):
        return np.zeros_like(detected, dtype=bool)

    grow_radius = max(
        1, _lround_positive(model_radius * (0.5 + protect_amount))
    )
    grown = _ag_gauss(detected, _ag_sigma(grow_radius, 2))
    cutoff = (1.0 - protect_amount) * 0.5 + 1e-3
    return grown > cutoff


def _poly_exponents(degree):
    return [
        (x_power, y_power)
        for x_power in range(degree + 1)
        for y_power in range(degree - x_power + 1)
    ]


def _polynomial_design_block(x_normalized, y_normalized, exponents):
    rows = y_normalized.size
    width = x_normalized.size
    design = np.empty((rows * width, len(exponents)), dtype=np.float64)
    for column, (x_power, y_power) in enumerate(exponents):
        term = (
            np.power(y_normalized, y_power)[:, None]
            * np.power(x_normalized, x_power)[None, :]
        )
        design[:, column] = term.ravel()
    return design


def _poly_fit(image, mask, degree, row_block=128):
    height, width = image.shape
    exponents = _poly_exponents(degree)
    terms = len(exponents)
    x_normalized = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    y_normalized = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    ata = np.zeros((terms, terms), dtype=np.float64)
    atb = np.zeros(terms, dtype=np.float64)

    for y_start in range(0, height, row_block):
        y_stop = min(height, y_start + row_block)
        design = _polynomial_design_block(
            x_normalized, y_normalized[y_start:y_stop], exponents
        )
        selected = mask[y_start:y_stop].ravel()
        selected_design = design[selected]
        selected_values = image[y_start:y_stop].ravel()[selected].astype(np.float64)
        if selected_values.size:
            ata += selected_design.T @ selected_design
            atb += selected_design.T @ selected_values

    ata.flat[:: terms + 1] += 1e-9
    try:
        cholesky = np.linalg.cholesky(ata)
        coefficients = np.linalg.solve(
            cholesky.T, np.linalg.solve(cholesky, atb)
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError("polynomial model is singular") from error

    model = np.empty_like(image, dtype=np.float32)
    for y_start in range(0, height, row_block):
        y_stop = min(height, y_start + row_block)
        design = _polynomial_design_block(
            x_normalized, y_normalized[y_start:y_stop], exponents
        )
        model[y_start:y_stop] = (design @ coefficients).reshape(
            y_stop - y_start, width
        )
    return model


def _estimate_background(
    image,
    radius,
    parameters,
    use_polynomial,
    progress=None,
    progress_base=0.0,
    progress_span=1.0,
):
    radius = max(1, radius)
    keep = np.ones(image.shape, dtype=bool)

    def fit(mask):
        if use_polynomial:
            return _poly_fit(image, mask, parameters.degree)
        return _inpaint_lowpass(image, mask, radius)

    model = fit(keep)
    previous_count = image.size
    minimum_keep = max(16, int(0.02 * image.size))

    for iteration in range(AG_N_ITER):
        residual = image - model
        reference = residual[keep]
        if reference.size == 0:
            reference = residual.ravel()
        median, sigma = _mad_sigma(reference)
        high = median + AG_HIGH_K * sigma
        low = median - AG_LOW_K * sigma
        new_keep = (residual <= high) & (residual >= low)

        if parameters.protect:
            protected = _structure_mask(
                residual - median,
                radius,
                parameters.protect_threshold,
                parameters.protect_amount,
            )
            new_keep[protected] = False

        kept_count = int(np.count_nonzero(new_keep))
        if kept_count < minimum_keep:
            rank = min(minimum_keep, image.size - 1)
            threshold = float(np.partition(residual.ravel(), rank)[rank])
            new_keep = residual <= threshold
            kept_count = int(np.count_nonzero(new_keep))

        model = fit(new_keep)
        change = abs(kept_count - previous_count) / image.size
        keep = new_keep
        previous_count = kept_count

        if progress is not None:
            progress(
                progress_base + progress_span * (iteration + 1) / AG_N_ITER,
                "polynomial" if use_polynomial else "multiscale",
            )
        if iteration > 0 and change < 1e-4:
            break

    if not use_polynomial and parameters.smoothness > 0.0:
        smoothing_radius = max(
            1, _lround_positive(radius * parameters.smoothness)
        )
        model = _ag_gauss(model, _ag_sigma(smoothing_radius, AG_PASSES))
    return model


def _downsample_area(image, factor):
    height, width = image.shape
    small_height = height // factor
    small_width = width // factor
    cropped = image[: small_height * factor, : small_width * factor]
    return cropped.reshape(
        small_height, factor, small_width, factor
    ).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def _resize_bilinear(image, width, height):
    """Bilinear resize with corner-aligned coordinates, as used by Siril."""

    source_height, source_width = image.shape
    if (source_width, source_height) == (width, height):
        return image.astype(np.float32, copy=True)

    x_float = (
        np.arange(width, dtype=np.float64) * (source_width - 1) / (width - 1)
        if width > 1
        else np.zeros(1, dtype=np.float64)
    )
    x0 = x_float.astype(np.int64)
    x1 = np.minimum(x0 + 1, source_width - 1)
    x_weight = (x_float - x0).astype(np.float32)
    result = np.empty((height, width), dtype=np.float32)

    for y in range(height):
        y_float = y * (source_height - 1) / (height - 1) if height > 1 else 0.0
        y0 = int(y_float)
        y1 = min(y0 + 1, source_height - 1)
        y_weight = np.float32(y_float - y0)
        top = image[y0, x0] * (1.0 - x_weight) + image[y0, x1] * x_weight
        bottom = image[y1, x0] * (1.0 - x_weight) + image[y1, x1] * x_weight
        result[y] = top * (1.0 - y_weight) + bottom * y_weight
    return result


def estimate_channel_background(
    image, parameters, progress=None, return_simplified_model=False
):
    """Estimate the background of one two-dimensional image channel."""

    parameters.validate()
    if image.ndim != 2:
        raise ValueError("estimate_channel_background expects a 2-D array")
    if min(image.shape) < 2:
        raise ValueError("image must be at least 2 x 2 pixels")

    working = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(working)
    if not np.any(finite):
        raise ValueError("image contains no finite pixels")
    if not np.all(finite):
        replacement = float(np.median(working[finite]))
        working = np.where(finite, working, replacement)

    factor = parameters.downsample
    if working.shape[0] // factor < 2 or working.shape[1] // factor < 2:
        factor = 1
    small = _downsample_area(working, factor)

    radius = max(
        1, _lround_positive(parameters.scale / 100.0 * min(small.shape))
    )

    simplified_model = None
    if parameters.simplified:
        polynomial = _estimate_background(
            small,
            radius,
            parameters,
            use_polynomial=True,
            progress=progress,
            progress_base=0.0,
            progress_span=0.35,
        )
        flattened = small - polynomial
        multiscale = _estimate_background(
            flattened,
            radius,
            parameters,
            use_polynomial=False,
            progress=progress,
            progress_base=0.35,
            progress_span=0.55,
        )
        small_model = polynomial + multiscale
        if return_simplified_model:
            simplified_model = _resize_bilinear(
                polynomial, width=working.shape[1], height=working.shape[0]
            )
    else:
        small_model = _estimate_background(
            small,
            radius,
            parameters,
            use_polynomial=False,
            progress=progress,
            progress_base=0.0,
            progress_span=0.9,
        )

    background = _resize_bilinear(
        small_model, width=working.shape[1], height=working.shape[0]
    )
    if return_simplified_model:
        return background, simplified_model
    return background


def extract_sample_free_background(
    image,
    parameters=None,
    correction="subtract",
    progress=None,
    return_simplified_model=False,
):
    """Return ``(background, corrected)`` for a GraXpert channels-last image."""

    if parameters is None:
        parameters = SampleFreeParameters()
    parameters.validate()

    source_image = np.asarray(image)
    is_monochrome_2d = source_image.ndim == 2
    if is_monochrome_2d:
        channels_last = source_image[:, :, None]
    elif source_image.ndim == 3:
        channels_last = source_image
    else:
        raise ValueError("only 2-D and channels-last 3-D images are supported")

    channels_last = channels_last.astype(np.float64, copy=False)
    background = np.empty(channels_last.shape, dtype=np.float32)
    corrected = np.empty(channels_last.shape, dtype=np.float32)
    simplified_model = (
        np.empty(channels_last.shape, dtype=np.float32)
        if return_simplified_model and parameters.simplified
        else None
    )
    num_channels = channels_last.shape[-1]

    progress_values = np.zeros(num_channels, dtype=np.float64)
    progress_lock = threading.Lock()
    progress_messages = SimpleQueue()

    def process_channel(channel):
        channel_progress = None
        if progress is not None:
            def channel_progress(value, stage):
                with progress_lock:
                    progress_values[channel] = max(
                        progress_values[channel], min(1.0, value)
                    )
                    combined = float(np.mean(progress_values))
                    message = (
                        combined,
                        "channel {}: {}".format(channel + 1, stage),
                    )
                if num_channels == 1:
                    progress(*message)
                else:
                    # Tk callbacks must run on the main thread. The caller
                    # drains this queue while waiting for channel workers.
                    progress_messages.put(message)

        source = channels_last[:, :, channel]
        if return_simplified_model:
            model, channel_simplified_model = estimate_channel_background(
                source,
                parameters,
                channel_progress,
                return_simplified_model=True,
            )
            if simplified_model is not None:
                simplified_model[:, :, channel] = channel_simplified_model
        else:
            model = estimate_channel_background(source, parameters, channel_progress)
        background[:, :, channel] = model
        level = float(np.median(model))

        if correction == "divide":
            denominator = np.maximum(model, 1e-6)
            corrected[:, :, channel] = source / denominator * level
        elif correction == "subtract":
            corrected[:, :, channel] = source - model + level
        else:
            raise ValueError("correction must be 'subtract' or 'divide'")

        corrected[:, :, channel][~np.isfinite(source)] = np.nan

    if num_channels == 1:
        process_channel(0)
    else:
        # Initialize once before workers enter the shared, thread-safe session.
        get_sample_free_session()
        with ThreadPoolExecutor(
            max_workers=min(3, num_channels),
            thread_name_prefix="sample-free",
        ) as channel_pool:
            pending = {
                channel_pool.submit(process_channel, channel)
                for channel in range(num_channels)
            }
            while pending:
                completed, pending = wait(
                    pending,
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                if progress is not None:
                    while True:
                        try:
                            progress(*progress_messages.get_nowait())
                        except Empty:
                            break
                for future in completed:
                    future.result()

        if progress is not None:
            while True:
                try:
                    progress(*progress_messages.get_nowait())
                except Empty:
                    break

    if progress is not None:
        progress(1.0, "complete")

    if is_monochrome_2d:
        background = background[:, :, 0]
        corrected = corrected[:, :, 0]
        if simplified_model is not None:
            simplified_model = simplified_model[:, :, 0]

    if return_simplified_model:
        return background, corrected, simplified_model
    return background, corrected
