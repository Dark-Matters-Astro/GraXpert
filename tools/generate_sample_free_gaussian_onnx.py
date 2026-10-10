"""Generate the ONNX model used by the sample-free Gaussian approximation.

The generated graph performs Siril's three pairs of horizontal and vertical
box blurs.  The box radius remains a runtime input, so one small graph covers
all image sizes and Sample-free scale settings.

The ``onnx`` package is only needed to regenerate the model.  GraXpert itself
loads the generated file with its existing ``onnxruntime`` dependency.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sample_free_gaussian.onnx"
OPSET = 17


def _constant(name, values, data_type=TensorProto.INT64):
    return helper.make_tensor(name, data_type, [len(values)], values)


def _blur(nodes, value, radius, axis, pass_number):
    prefix = f"pass{pass_number}_{'horizontal' if axis == 3 else 'vertical'}"
    edge_pads = f"{prefix}_edge_pads"
    if axis == 3:
        pad_inputs = ["zeros3", radius, "zeros3", radius]
    else:
        pad_inputs = ["zeros2", radius, "zeros3", radius, "zeros1"]
    nodes.append(helper.make_node("Concat", pad_inputs, [edge_pads], axis=0))

    padded = f"{prefix}_edge_padded"
    nodes.append(helper.make_node("Pad", [value, edge_pads], [padded], mode="edge"))

    cumulative = f"{prefix}_cumulative"
    nodes.append(
        helper.make_node(
            "CumSum",
            [padded, f"axis{axis}"],
            [cumulative],
        )
    )

    cumulative_zero = f"{prefix}_cumulative_zero"
    nodes.append(
        helper.make_node(
            "Pad",
            [cumulative, f"leading_pad_axis{axis}"],
            [cumulative_zero],
            mode="constant",
        )
    )

    window = f"{prefix}_window"
    nodes.append(helper.make_node("Mul", [radius, "two"], [f"{window}_twice_radius"]))
    nodes.append(helper.make_node("Add", [f"{window}_twice_radius", "one"], [window]))
    negative_window = f"{prefix}_negative_window"
    nodes.append(helper.make_node("Neg", [window], [negative_window]))

    upper = f"{prefix}_upper"
    lower = f"{prefix}_lower"
    nodes.append(
        helper.make_node(
            "Slice",
            [cumulative_zero, window, "maximum", f"axis_vector{axis}", "one"],
            [upper],
        )
    )
    nodes.append(
        helper.make_node(
            "Slice",
            [cumulative_zero, "zero", negative_window, f"axis_vector{axis}", "one"],
            [lower],
        )
    )

    summed = f"{prefix}_summed"
    divisor = f"{prefix}_divisor"
    output = f"{prefix}_output"
    nodes.append(helper.make_node("Sub", [upper, lower], [summed]))
    nodes.append(
        helper.make_node("Cast", [window], [divisor], to=TensorProto.FLOAT)
    )
    nodes.append(helper.make_node("Div", [summed, divisor], [output]))
    return output


def generate_model(path=MODEL_PATH):
    nodes = []
    value = "image"
    for pass_number in range(1, 4):
        value = _blur(nodes, value, "radius", axis=3, pass_number=pass_number)
        value = _blur(nodes, value, "radius", axis=2, pass_number=pass_number)

    nodes.append(helper.make_node("Identity", [value], ["background"]))
    initializers = [
        _constant("zero", [0]),
        _constant("one", [1]),
        _constant("two", [2]),
        _constant("maximum", [2**63 - 1]),
        _constant("zeros1", [0]),
        _constant("zeros2", [0, 0]),
        _constant("zeros3", [0, 0, 0]),
        _constant("axis2", [2]),
        _constant("axis3", [3]),
        _constant("axis_vector2", [2]),
        _constant("axis_vector3", [3]),
        _constant("leading_pad_axis2", [0, 0, 1, 0, 0, 0, 0, 0]),
        _constant("leading_pad_axis3", [0, 0, 0, 1, 0, 0, 0, 0]),
    ]
    graph = helper.make_graph(
        nodes,
        "GraXpert sample-free three-pass box Gaussian",
        [
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, [1, 1, "height", "width"]
            ),
            helper.make_tensor_value_info("radius", TensorProto.INT64, [1]),
        ],
        [
            helper.make_tensor_value_info(
                "background", TensorProto.FLOAT, [1, 1, "height", "width"]
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="GraXpert",
        producer_version="3.0.2-sample-free",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.ir_version = 10
    model.doc_string = (
        "Three-pass separable box approximation used by the GPL-3.0-or-later "
        "sample-free background method adapted from Siril."
    )
    helper.set_model_props(
        model,
        {
            "license": "GPL-3.0-or-later",
            "source": (
                "https://gitlab.com/free-astro/siril/-/blob/master/"
                "src/algos/background_extraction.c"
            ),
        },
    )
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    print(generate_model())
