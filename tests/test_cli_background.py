"""CLI contract and precedence checks for sample-free extraction."""

import argparse
from types import SimpleNamespace

import pytest

from graxpert.cli_background import add_background_arguments, apply_background_arguments


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("filename")
    result.add_argument("-cli", action="store_true")
    result.add_argument("-correction")
    add_background_arguments(result)
    return result


def test_pixinsight_arguments_and_explicit_false_zero_override_preferences():
    args = parser().parse_args("PixInsight.xisf -cli -interpolation sample_free -sample_free_scale 5 -sample_free_smoothness 0 -sample_free_protect false -sample_free_protect_threshold 0.05 -sample_free_protect_amount 0 -sample_free_simplified false -sample_free_downsample 1 -sample_free_degree 2 -correction Subtraction".split())
    prefs = SimpleNamespace(interpol_type_option="RBF", sample_free_protect=True, sample_free_simplified=True)
    apply_background_arguments(args, prefs)
    assert prefs.interpol_type_option == "Sample-free"
    assert prefs.sample_free_protect is False
    assert prefs.sample_free_simplified is False
    assert prefs.sample_free_smoothness == 0
    assert prefs.sample_free_protect_amount == 0
    assert prefs.sample_free_downsample == 1
    assert prefs.sample_free_degree == 2


def test_omitted_options_preserve_preferences():
    prefs = SimpleNamespace(interpol_type_option="AI", sample_free_scale=7)
    apply_background_arguments(parser().parse_args(["image.fits"]), prefs)
    assert vars(prefs) == {"interpol_type_option": "AI", "sample_free_scale": 7}


@pytest.mark.parametrize("method", ["sample_free", "Sample-free", "SAMPLE_FREE"])
def test_method_aliases(method):
    assert parser().parse_args(["image.fits", "--interpolation", method]).interpolation == "Sample-free"


@pytest.mark.parametrize("option,value", [
    ("scale", "0"), ("scale", "11"), ("scale", "nan"),
    ("smoothness", "-1"), ("smoothness", "inf"),
    ("protect", "yes"), ("simplified", "0"),
    ("protect_threshold", "1.1"), ("protect_amount", "-0.1"),
    ("degree", "0"), ("degree", "7"), ("downsample", "3"),
])
def test_invalid_values_rejected(option, value):
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["image.fits", "-sample_free_" + option, value])
    assert error.value.code == 2


@pytest.mark.parametrize("extension", ["fits", "xisf"])
@pytest.mark.parametrize("use_preferences", [False, True])
def test_cli_processes_image_without_samples_or_ai(tmp_path, monkeypatch, extension, use_preferences):
    import importlib
    import json
    import sys
    import types

    import numpy as np
    from astropy.io import fits
    from xisf import XISF

    # Distribution-only bucket configuration is irrelevant to sample-free tests.
    secrets = types.ModuleType("graxpert.s3_secrets")
    for name in ("bge_bucket_name", "denoise_bucket_name", "deconvolution_object_bucket_name", "deconvolution_stars_bucket_name"):
        setattr(secrets, name, "unused-test-bucket")
    monkeypatch.setitem(sys.modules, "graxpert.s3_secrets", secrets)
    main = importlib.import_module("graxpert.main")
    cmdline = importlib.import_module("graxpert.cmdline_tools")
    monkeypatch.setattr(main, "collect_available_versions", lambda *args: ([], []))

    def no_ai(*args):
        pytest.fail("Sample-free must not load an AI model")
    monkeypatch.setattr(cmdline.BGECmdlineTool, "get_ai_version", no_ai)
    y, x = np.mgrid[:32, :40]
    data = (0.2 + x * 0.002 + y * 0.001).astype(np.float32)
    source = tmp_path / ("input." + extension)
    if extension == "fits":
        fits.writeto(source, data)
    else:
        XISF.write(str(source), data[:, :, None])
    argv = ["GraXpert", str(source), "-cli", "-output", "result", "-bg", "-gpu", "false", "-correction", "Subtraction"]
    if use_preferences:
        config = tmp_path / "preferences.json"
        config.write_text(json.dumps({"interpol_type_option": "Sample-free", "sample_free_downsample": 1}))
        argv += ["-preferences_file", str(config)]
    else:
        argv += ["-interpolation", "sample_free", "-sample_free_scale", "5", "-sample_free_smoothness", "1.0", "-sample_free_protect", "true", "-sample_free_protect_threshold", "0.05", "-sample_free_protect_amount", "0.50", "-sample_free_simplified", "false", "-sample_free_downsample", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    main.main()
    output = tmp_path / ("result." + extension)
    background = tmp_path / ("result_background." + extension)
    assert output.exists() and background.exists()
    result = fits.getdata(output) if extension == "fits" else XISF(str(output)).read_image(0)[:, :, 0]
    assert result.shape == data.shape
    assert np.isfinite(result).all()
    from graxpert.sample_free_background import SampleFreeParameters, extract_sample_free_background
    from graxpert.preferences import Prefs
    parameters = SampleFreeParameters(downsample=1, simplified=Prefs().sample_free_simplified if use_preferences else False)
    _, expected = extract_sample_free_background(data[:, :, None].copy(), parameters, correction="subtract")
    assert np.allclose(result, np.clip(expected[:, :, 0], 0, 1), atol=1e-6)
