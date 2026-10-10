"""Background-extraction CLI options, independent of GUI and model downloads."""

import argparse
import math


SAMPLE_FREE_OPTIONS = (
    "scale", "smoothness", "protect", "protect_threshold", "protect_amount",
    "simplified", "degree", "downsample",
)


def _boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value.lower() == "true"


def _number(minimum, maximum=None):
    def parse(value):
        try:
            result = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError("expected a number")
        if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
            bounds = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
            raise argparse.ArgumentTypeError(f"expected a finite number {bounds}")
        return result
    return parse


def _interpolation(value):
    methods = {"ai": "AI", "rbf": "RBF", "kriging": "Kriging", "splines": "Splines", "sample-free": "Sample-free"}
    try:
        return methods[value.lower().replace("_", "-")]
    except KeyError:
        raise argparse.ArgumentTypeError("expected AI, RBF, Kriging, Splines or sample_free")


def add_background_arguments(parser):
    parser.add_argument("-interpolation", "--interpolation", type=_interpolation, default=None,
                        help="Extraction method: AI, RBF, Kriging, Splines or sample_free (default: AI, or preferences file)")
    definitions = (
        ("scale", _number(1, 10), None, "Model scale, 1–10"),
        ("smoothness", _number(0), None, "Sample-free smoothing, >= 0"),
        ("protect", _boolean, None, "Protect structures: true or false"),
        ("protect_threshold", _number(0, 1), None, "Structure protection threshold, 0–1"),
        ("protect_amount", _number(0, 1), None, "Structure protection amount, 0–1"),
        ("simplified", _boolean, None, "Fit a polynomial before multiscale extraction: true or false"),
        ("degree", int, range(1, 7), "Polynomial degree, 1–6"),
        ("downsample", int, (1, 2, 4, 8), "Downsampling factor: 1, 2, 4 or 8"),
    )
    for name, value_type, choices, help_text in definitions:
        option = "sample_free_" + name
        parser.add_argument("-" + option, "--" + option, type=value_type, choices=choices,
                            default=None, help=help_text + "; overrides preferences file")


def apply_background_arguments(args, preferences):
    """Only explicit CLI values override loaded preferences, including false/zero."""
    if getattr(args, "interpolation", None) is not None:
        preferences.interpol_type_option = args.interpolation
    for name in SAMPLE_FREE_OPTIONS:
        option = "sample_free_" + name
        value = getattr(args, option, None)
        if value is not None:
            setattr(preferences, option, value)
