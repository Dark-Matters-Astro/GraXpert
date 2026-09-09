"""Stress-test GraXpert's ONNX-backed sample-free background extraction.

The test intentionally keeps all results in memory only. It checks repeated
full-resolution runs, numerical determinism, callback thread affinity, a
parameter matrix, ONNX/reference agreement and an optional real Tk callback.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graxpert.sample_free_background import (
    SampleFreeParameters,
    _ag_gauss,
    _ag_gauss_numpy,
    _ag_sigma,
    extract_sample_free_background,
)
from graxpert.sample_free_onnx import get_sample_free_session


def _rss_mb():
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        success = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            raise OSError("GetProcessMemoryInfo failed")
        return counters.WorkingSetSize / (1024.0 * 1024.0)

    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)

    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip()) / 1024.0


def _digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        if array is None:
            digest.update(b"<none>")
            continue
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _load_rgb(path):
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    if data.ndim == 2:
        return data[:, :, np.newaxis]
    if data.ndim == 3 and data.shape[0] <= 4:
        return np.moveaxis(data, 0, -1)
    if data.ndim == 3:
        return data
    raise ValueError(f"Unsupported FITS shape: {data.shape}")


def _run_extraction(image, parameters, include_simplified=True):
    callback_thread_ids = set()
    progress_values = []
    main_thread_id = threading.get_ident()

    def progress(value, stage):
        callback_thread_ids.add(threading.get_ident())
        progress_values.append(value)

    started = time.perf_counter()
    result = extract_sample_free_background(
        image,
        parameters=parameters,
        progress=progress,
        return_simplified_model=include_simplified,
    )
    elapsed = time.perf_counter() - started

    if callback_thread_ids != {main_thread_id}:
        raise AssertionError(
            f"Progress callback escaped main thread: {callback_thread_ids}"
        )
    if not progress_values or progress_values[-1] != 1.0:
        raise AssertionError("Progress did not finish at 100 percent")
    if any(
        earlier > later
        for earlier, later in zip(progress_values, progress_values[1:])
    ):
        raise AssertionError("Progress moved backwards")

    arrays = result if isinstance(result, tuple) else (result,)
    for array in arrays:
        if array is not None and not np.all(np.isfinite(array)):
            raise AssertionError("Extraction returned non-finite values")
    return result, elapsed


def test_onnx_reference():
    rng = np.random.default_rng(20260905)
    cases = [
        ((17, 23), 1),
        ((127, 173), 3),
        ((257, 311), 11),
        ((513, 769), 49),
        ((1025, 1537), 120),
    ]
    maximum = 0.0
    mean_errors = []
    for shape, radius in cases:
        image = rng.normal(size=shape).astype(np.float32)
        sigma = _ag_sigma(radius, 3)
        reference = _ag_gauss_numpy(image, sigma)
        actual = _ag_gauss(image, sigma)
        difference = np.abs(reference - actual)
        maximum = max(maximum, float(np.max(difference)))
        mean_errors.append(float(np.mean(difference)))
        if not np.allclose(actual, reference, rtol=2e-5, atol=2e-6):
            raise AssertionError(f"ONNX mismatch at shape={shape}, radius={radius}")
    return {
        "cases": len(cases),
        "maximum_absolute_error": maximum,
        "maximum_mean_absolute_error": max(mean_errors),
    }


def test_repeated_full_image(image, runs):
    parameters = SampleFreeParameters()
    baseline_digest = None
    elapsed_values = []
    rss_values = []
    for run_number in range(1, runs + 1):
        result, elapsed = _run_extraction(image, parameters)
        run_digest = _digest(*result)
        if baseline_digest is None:
            baseline_digest = run_digest
        elif run_digest != baseline_digest:
            raise AssertionError(f"Non-deterministic full result in run {run_number}")
        del result
        gc.collect()
        rss = _rss_mb()
        elapsed_values.append(elapsed)
        rss_values.append(rss)
        print(
            f"full_run={run_number}/{runs} seconds={elapsed:.3f} "
            f"rss_mb={rss:.1f} digest={run_digest[:12]}",
            flush=True,
        )

    # Ignore allocator settling during the first two runs, then reject a clear
    # retained-memory growth trend. A stable high-water mark is acceptable.
    settled = rss_values[min(2, len(rss_values) - 1) :]
    retained_growth = settled[-1] - min(settled)
    if len(settled) >= 5 and retained_growth > 384.0:
        raise AssertionError(
            f"Resident memory grew by {retained_growth:.1f} MB after settling"
        )
    return {
        "runs": runs,
        "digest": baseline_digest,
        "seconds_min": min(elapsed_values),
        "seconds_median": float(np.median(elapsed_values)),
        "seconds_max": max(elapsed_values),
        "rss_mb_min": min(rss_values),
        "rss_mb_max": max(rss_values),
        "rss_mb_retained_growth_after_settling": retained_growth,
    }


def test_parameter_matrix(image):
    crop = np.ascontiguousarray(image[:1025, :1537, :])
    cases = [
        SampleFreeParameters(downsample=1),
        SampleFreeParameters(downsample=2),
        SampleFreeParameters(downsample=4),
        SampleFreeParameters(downsample=8),
        SampleFreeParameters(
            scale=1.0,
            smoothness=0.0,
            protect=False,
            simplified=False,
            downsample=1,
        ),
        SampleFreeParameters(
            scale=10.0,
            smoothness=3.0,
            protect=True,
            protect_threshold=0.0,
            protect_amount=1.0,
            simplified=True,
            degree=6,
            downsample=2,
        ),
        SampleFreeParameters(
            scale=5.5,
            smoothness=0.5,
            protect=True,
            protect_threshold=1.0,
            protect_amount=0.0,
            simplified=True,
            degree=3,
            downsample=4,
        ),
    ]
    results = []
    for case_number, parameters in enumerate(cases, 1):
        result, elapsed = _run_extraction(crop, parameters)
        digest = _digest(*result)
        del result
        gc.collect()
        repeated_result, repeated_elapsed = _run_extraction(crop, parameters)
        repeated_digest = _digest(*repeated_result)
        if repeated_digest != digest:
            raise AssertionError(
                f"Non-deterministic parameter result in case {case_number}"
            )
        results.append(
            {
                "parameters": asdict(parameters),
                "seconds": [elapsed, repeated_elapsed],
                "digest": digest,
                "rss_mb": _rss_mb(),
            }
        )
        del repeated_result
        gc.collect()
        print(
            f"parameter_case={case_number}/{len(cases)} "
            f"seconds={elapsed:.3f},{repeated_elapsed:.3f} "
            f"downsample={parameters.downsample} digest={digest[:12]}",
            flush=True,
        )
    return {"crop_shape": crop.shape, "cases": results}


def test_mono_and_odd_shapes(image):
    cases = [
        np.ascontiguousarray(image[:1537, :2049, 0]),
        np.ascontiguousarray(image[:513, :769, :1]),
        np.random.default_rng(17).normal(size=(65, 97, 3)).astype(np.float32),
    ]
    records = []
    for case_number, case in enumerate(cases, 1):
        parameters = SampleFreeParameters(downsample=1)
        result, elapsed = _run_extraction(case, parameters)
        digest = _digest(*result)
        records.append(
            {"shape": case.shape, "seconds": elapsed, "digest": digest}
        )
        del result
        gc.collect()
        print(
            f"shape_case={case_number}/{len(cases)} shape={case.shape} "
            f"seconds={elapsed:.3f} digest={digest[:12]}",
            flush=True,
        )
    return records


def test_tk_progress(image):
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    variable = tk.DoubleVar(root, value=0.0)
    callback_thread_ids = set()
    main_thread_id = threading.get_ident()

    def progress(value, stage):
        callback_thread_ids.add(threading.get_ident())
        variable.set(value)
        root.update_idletasks()

    crop = np.ascontiguousarray(image[:513, :769, :])
    started = time.perf_counter()
    result = extract_sample_free_background(
        crop,
        SampleFreeParameters(downsample=2),
        progress=progress,
        return_simplified_model=True,
    )
    elapsed = time.perf_counter() - started
    final_progress = variable.get()
    root.destroy()
    if callback_thread_ids != {main_thread_id}:
        raise AssertionError("Tk progress callback ran outside the main thread")
    if final_progress != 1.0:
        raise AssertionError(f"Tk progress ended at {final_progress}")
    digest = _digest(*result)
    del result
    gc.collect()
    print(
        f"tk_progress=ok seconds={elapsed:.3f} digest={digest[:12]}",
        flush=True,
    )
    return {"seconds": elapsed, "digest": digest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--full-runs", type=int, default=20)
    parser.add_argument("--tk", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.full_runs < 1:
        raise ValueError("--full-runs must be positive")

    started = time.perf_counter()
    image = _load_rgb(args.image)
    print(
        f"image={args.image} shape={image.shape} initial_rss_mb={_rss_mb():.1f}",
        flush=True,
    )
    session = get_sample_free_session()
    report = {
        "status": "running",
        "image": str(args.image),
        "image_shape": image.shape,
        "onnx_providers": session.get_providers(),
        "onnx_reference": test_onnx_reference(),
        "full_image_repetition": test_repeated_full_image(image, args.full_runs),
        "parameter_matrix": test_parameter_matrix(image),
        "shape_matrix": test_mono_and_odd_shapes(image),
    }
    if args.tk:
        report["tk_progress"] = test_tk_progress(image)
    report["status"] = "passed"
    report["total_seconds"] = time.perf_counter() - started
    report["final_rss_mb"] = _rss_mb()

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
