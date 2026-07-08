"""Tests for Aloff-Wallach N11 endpoint smoothness conditions."""

from __future__ import annotations

import experiments.aloff_wallach_endpoint_smoothness
from experiments.aloff_wallach import endpoint_smoothness


def test_endpoint_models_record_the_two_singular_weights() -> None:
    """The two CP2 singular orbits should have different normal weights."""
    models = {model.label: model for model in endpoint_smoothness.endpoint_models()}
    assert models["real_rp2"].normal_weight == 1
    assert models["null_conic_cp1"].normal_weight == 2
    assert models["real_rp2"].collapse_combination == "theta = base_3 + fiber_3"


def test_principal_variables_are_rewritten_without_dropping_dimensions() -> None:
    """The endpoint chart should start from all 19 principal coefficients."""
    terms = endpoint_smoothness.principal_variable_smooth_terms()
    assert len(terms) == 19
    assert terms["x6"] == [
        {
            "coefficient": -0.5,
            "basis": "normal_radial^normal_angular^surviving_axis",
            "collapse_shift": 1,
        }
    ]
    assert terms["y1"] == [
        {
            "coefficient": 0.5,
            "basis": "base_1^base_2^surviving_axis",
            "collapse_shift": 0,
        },
        {
            "coefficient": 0.5,
            "basis": "normal_angular^base_1^base_2",
            "collapse_shift": 1,
        },
    ]
    assert terms["y2"][0]["coefficient"] == -0.5
    assert terms["y2"][1]["coefficient"] == 0.5


def test_zero_order_conditions_give_four_endpoint_constants() -> None:
    """Smooth endpoint values should be a four-constant chart."""
    parameterization = endpoint_smoothness.zero_order_parameterization()
    assert parameterization["dimension"] == 4
    assert parameterization["x_values"]["x1(0)"] == "0"
    assert parameterization["x_values"]["x7(0)"] == "0"
    assert parameterization["y_values"] == {
        "y1(0)": "A",
        "y2(0)": "-A",
        "y3(0)": "B",
        "y4(0)": "C",
        "y5(0)": "B",
        "y6(0)": "C",
        "y7(0)": "-C",
        "y8(0)": "B",
        "y9(0)": "-C",
        "y10(0)": "B",
        "y11(0)": "D",
        "y12(0)": "-D",
    }


def test_jet_dimensions_follow_the_singular_isotropy_weights() -> None:
    """The representation calculation should distinguish RP2 and CP1 ends."""
    rows = {
        (row["endpoint"], row["max_order"]): row
        for row in endpoint_smoothness.jet_dimension_table(max_order=2)
    }
    assert rows[("real_rp2", 0)]["allowed_dimension"] == 4
    assert rows[("real_rp2", 1)]["allowed_dimension"] == 9
    assert rows[("real_rp2", 2)]["allowed_dimension"] == 21
    assert rows[("null_conic_cp1", 0)]["allowed_dimension"] == 4
    assert rows[("null_conic_cp1", 1)]["allowed_dimension"] == 13
    assert rows[("null_conic_cp1", 2)]["allowed_dimension"] == 23


def test_endpoint_smoothness_summary_and_shim() -> None:
    """The command summary should expose the parameter-count conclusion."""
    summary = endpoint_smoothness.build_summary(max_order=1)
    assert summary["version"] == "aloff-wallach-n11-endpoint-smoothness-v1"
    assert summary["principal_variable_count"] == 19
    assert summary["algebraic_su3_constraint_count"] == 3
    assert summary["zero_order_parameterization"]["dimension"] == 4
    assert "four zeroth-order constants" in summary["parameter_count_note"]
    assert experiments.aloff_wallach_endpoint_smoothness.main is endpoint_smoothness.main
