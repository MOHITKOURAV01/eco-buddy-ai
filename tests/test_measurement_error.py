"""Tests for the measurement error engine.

The claim under test is that noise in a predictor biases a slope toward zero by
a calculable factor, and that dividing by the reliability takes it back out. So
the load-bearing test generates an outcome from a *known* true predictor, fits
it against a noisily-reported version of that predictor, and asserts both
halves: the naive slope is attenuated by roughly the theoretical lambda, and
the corrected slope lands back on the truth.

The second claim is about the direction of the bias. Testing attenuation at one
noise level would prove nothing; the tests assert the relationship — more noise,
lower lambda, larger correction, and never a slope pushed further from zero.

The refusals are tested as hard as the arithmetic. A correction applied when
the classical assumption has failed is wrong in an unknown direction, which is
worse than the attenuated estimate it replaces, and that is the failure mode
this module exists to prevent.
"""

import math
import os
import sqlite3
import tempfile

import pytest

from src.utils import measurement_error
from src.utils.measurement_error import (
    DEFAULT_CONFIDENCE,
    DIFFERENTIAL_ALPHA,
    ENGINE_VERSION,
    GOOD_RELIABILITY,
    MIN_REGRESSION_POINTS,
    MIN_REPEAT_PAIRS,
    MIN_VALIDATION_PAIRS,
    POOR_RELIABILITY,
    RELIABILITY_FLOOR,
    WHIPPLE_ROUGH,
    CalibrationError,
    analyse,
    apply_calibration,
    attenuation_table,
    betai,
    build_record,
    chi_square_p,
    correlation,
    delete_analysis,
    demo_records,
    demo_regression,
    differential_error_test,
    disattenuate_correlation,
    disattenuate_slope,
    get_analyses,
    get_measurement_notes,
    heaping_diagnostics,
    make_component,
    ols,
    propagate_to_total,
    regression_calibration,
    reliability_band,
    reliability_from_repeats,
    reliability_from_validation,
    save_analysis,
    simex,
    summarise,
    t_cdf,
    two_sided_p,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _theoretical_lambda(true_sd, error_sd):
    return true_sd**2 / (true_sd**2 + error_sd**2)


@pytest.fixture
def temp_db(monkeypatch):
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    monkeypatch.setattr(measurement_error, "DB_NAME", path)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# build_record
# ---------------------------------------------------------------------------


def test_build_record_keeps_the_reported_value():
    record = build_record("a", 120.0)
    assert record["reported"] == pytest.approx(120.0)
    assert record["has_repeat"] is False
    assert record["has_validation"] is False


def test_build_record_flags_a_repeat():
    record = build_record("a", 120.0, repeat=118.0)
    assert record["has_repeat"] is True
    assert record["repeat"] == pytest.approx(118.0)


def test_build_record_flags_a_validated_value():
    record = build_record("a", 120.0, validated=125.0)
    assert record["has_validation"] is True


def test_build_record_defaults_the_category():
    assert build_record("a", 1.0)["category"] == "general"


def test_build_record_rejects_a_non_numeric_report():
    with pytest.raises(CalibrationError, match="non-numeric reported"):
        build_record("a", "banana")


def test_build_record_rejects_a_non_numeric_repeat():
    with pytest.raises(CalibrationError, match="non-numeric repeat"):
        build_record("a", 1.0, repeat="banana")


def test_build_record_rejects_a_non_numeric_validation():
    with pytest.raises(CalibrationError, match="non-numeric validated"):
        build_record("a", 1.0, validated=float("nan"))


def test_build_record_rejects_infinity():
    with pytest.raises(CalibrationError, match="non-numeric reported"):
        build_record("a", float("inf"))


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


def test_ols_recovers_an_exact_line():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0 + 2.0 * x for x in xs]
    fit = ols(xs, ys)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(3.0)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_ols_reports_a_slope_standard_error():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [1.0, 2.5, 2.0, 4.5, 4.0, 6.5]
    assert ols(xs, ys)["slope_se"] > 0


def test_ols_rejects_mismatched_lengths():
    with pytest.raises(CalibrationError, match="same length"):
        ols([1.0, 2.0], [1.0])


def test_ols_rejects_too_few_points():
    with pytest.raises(CalibrationError, match="at least %d points" % MIN_REGRESSION_POINTS):
        ols([1.0, 2.0], [1.0, 2.0])


def test_ols_rejects_a_constant_predictor():
    with pytest.raises(CalibrationError, match="does not vary"):
        ols([2.0] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_correlation_of_a_perfect_line_is_one():
    assert correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_correlation_of_a_perfect_inverse_is_minus_one():
    assert correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)


def test_correlation_rejects_a_constant_variable():
    with pytest.raises(CalibrationError, match="does not vary"):
        correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Reliability from repeats
# ---------------------------------------------------------------------------


def test_repeat_reliability_is_one_when_the_repeats_agree_exactly():
    records = [
        build_record("r%d" % i, 100.0 + i * 10.0, repeat=100.0 + i * 10.0)
        for i in range(12)
    ]
    result = reliability_from_repeats(records)
    assert result["reliability"] == pytest.approx(1.0)
    assert result["error_variance"] == pytest.approx(0.0)


def test_repeat_reliability_recovers_a_known_error_variance():
    """Every pair differs by exactly 20, so var(d) = 0 and the halving is exact."""
    records = [
        build_record("r%d" % i, 100.0 + i * 50.0 + 10.0, repeat=100.0 + i * 50.0 - 10.0)
        for i in range(12)
    ]
    result = reliability_from_repeats(records)
    # A constant difference has zero variance, so the estimated error is zero.
    assert result["error_variance"] == pytest.approx(0.0)
    # But the mean difference is 20, and that is reported separately as a bias.
    assert result["mean_difference"] == pytest.approx(20.0)


def test_repeat_reliability_falls_as_the_recalls_disagree_more():
    weights = []
    for error in (50.0, 200.0, 500.0):
        records = demo_records(count=400, error_sd=error, repeat_share=1.0, seed=21)
        weights.append(reliability_from_repeats(records)["reliability"])
    assert weights == sorted(weights, reverse=True)


def test_repeat_reliability_is_near_the_theoretical_value():
    records = demo_records(
        count=1500, true_sd=400.0, error_sd=300.0, repeat_share=1.0, seed=31
    )
    result = reliability_from_repeats(records)
    assert result["reliability"] == pytest.approx(
        _theoretical_lambda(400.0, 300.0), abs=0.06
    )


def test_repeat_reliability_needs_enough_pairs():
    records = [build_record("r%d" % i, 100.0, repeat=100.0) for i in range(3)]
    with pytest.raises(CalibrationError, match="at least %d records" % MIN_REPEAT_PAIRS):
        reliability_from_repeats(records)


def test_repeat_reliability_reports_a_degenerate_variable():
    """Disagreement larger than the between-unit spread means no signal at all."""
    records = demo_records(
        count=400, true_sd=10.0, error_sd=600.0, repeat_share=1.0, seed=41
    )
    result = reliability_from_repeats(records)
    assert result["degenerate"] is True
    assert result["reliability"] == 0.0
    assert "no usable between-unit signal" in result["note"]


def test_repeat_reliability_flags_a_systematic_difference():
    records = [
        build_record("r%d" % i, 100.0 + i * 30.0 + 40.0, repeat=100.0 + i * 30.0)
        for i in range(20)
    ]
    result = reliability_from_repeats(records)
    assert result["mean_difference"] == pytest.approx(40.0)


def test_repeat_reliability_reports_a_standard_error():
    records = demo_records(count=300, repeat_share=1.0, seed=51)
    assert reliability_from_repeats(records)["reliability_se"] > 0


# ---------------------------------------------------------------------------
# Reliability from a validation subsample
# ---------------------------------------------------------------------------


def test_the_calibration_slope_is_the_reliability():
    """cov(X, W) = var(T), so the slope of W on X is var(T)/var(X) = lambda."""
    records = demo_records(count=1200, validation_share=1.0, seed=61)
    result = reliability_from_validation(records)
    assert result["slope"] == pytest.approx(result["reliability"])


def test_validation_reliability_is_near_the_theoretical_value():
    records = demo_records(
        count=2000, true_sd=400.0, error_sd=300.0, validation_share=1.0, seed=71
    )
    result = reliability_from_validation(records)
    assert result["reliability"] == pytest.approx(
        _theoretical_lambda(400.0, 300.0), abs=0.05
    )


def test_validation_reliability_is_one_when_reports_are_perfect():
    records = [
        build_record("r%d" % i, 100.0 + i * 25.0, validated=100.0 + i * 25.0)
        for i in range(20)
    ]
    assert reliability_from_validation(records)["reliability"] == pytest.approx(1.0)


def test_validation_reliability_needs_enough_pairs():
    records = [build_record("r%d" % i, float(i), validated=float(i)) for i in range(4)]
    with pytest.raises(
        CalibrationError, match="at least %d records" % MIN_VALIDATION_PAIRS
    ):
        reliability_from_validation(records)


def test_a_calibration_slope_outside_the_unit_interval_is_flagged():
    """A reliability above one is impossible; it means an input is wrong."""
    records = [
        build_record("r%d" % i, 100.0 + i * 10.0, validated=100.0 + i * 30.0)
        for i in range(15)
    ]
    result = reliability_from_validation(records)
    assert result["out_of_range"] is True
    assert "outside (0, 1]" in result["note"]


def test_regression_calibration_refuses_an_out_of_range_slope():
    records = [
        build_record("r%d" % i, 100.0 + i * 10.0, validated=100.0 + i * 30.0)
        for i in range(15)
    ]
    with pytest.raises(CalibrationError, match="outside"):
        regression_calibration(records)


def test_apply_calibration_maps_onto_the_trusted_scale():
    calibration = {"intercept": 10.0, "slope": 0.5}
    assert apply_calibration(calibration, [0.0, 100.0]) == [10.0, 60.0]


def test_apply_calibration_rejects_non_numeric_input():
    with pytest.raises(CalibrationError, match="non-numeric"):
        apply_calibration({"intercept": 0.0, "slope": 1.0}, ["x"])


def test_calibration_recovers_the_truth_on_average():
    records = demo_records(count=1500, validation_share=1.0, seed=81)
    calibration = regression_calibration(records)
    reported = [record["reported"] for record in records]
    truth = [record["validated"] for record in records]
    calibrated = apply_calibration(calibration, reported)
    naive_error = sum(abs(reported[i] - truth[i]) for i in range(len(truth)))
    calibrated_error = sum(abs(calibrated[i] - truth[i]) for i in range(len(truth)))
    assert calibrated_error < naive_error


# ---------------------------------------------------------------------------
# The differential-error gate
# ---------------------------------------------------------------------------


def test_classical_error_passes_the_differential_test():
    records = demo_records(count=800, validation_share=1.0, seed=91)
    result = differential_error_test(records)
    assert result["classical"] is True
    assert result["p_value"] > DIFFERENTIAL_ALPHA


def test_error_that_tracks_the_truth_fails_the_test():
    records = demo_records(
        count=800, validation_share=1.0, differential_slope=0.5, seed=101
    )
    result = differential_error_test(records)
    assert result["differential"] is True
    assert "does not apply" in result["headline"]


def test_the_differential_test_reports_the_direction():
    over = differential_error_test(
        demo_records(count=600, validation_share=1.0, differential_slope=0.5, seed=111)
    )
    under = differential_error_test(
        demo_records(count=600, validation_share=1.0, differential_slope=-0.5, seed=111)
    )
    assert "over-reported" in over["headline"]
    assert "under-reported" in under["headline"]


def test_the_differential_test_needs_validated_records():
    with pytest.raises(CalibrationError, match="at least %d validated" % MIN_VALIDATION_PAIRS):
        differential_error_test(demo_records(count=50, validation_share=0.0))


def test_a_stricter_alpha_makes_the_test_harder_to_fail():
    records = demo_records(
        count=400, validation_share=1.0, differential_slope=0.08, seed=121
    )
    lenient = differential_error_test(records, alpha=0.20)
    strict = differential_error_test(records, alpha=0.0001)
    assert lenient["p_value"] == pytest.approx(strict["p_value"])
    assert strict["classical"] or not lenient["classical"]


# ---------------------------------------------------------------------------
# Disattenuation — the load-bearing behaviour
# ---------------------------------------------------------------------------


def test_the_correction_recovers_a_known_slope():
    """Outcome generated from the truth, fitted on the report, corrected back."""
    records = demo_records(
        count=1200, true_sd=400.0, error_sd=300.0, validation_share=1.0, seed=131
    )
    example = demo_regression(records, true_slope=0.5)
    reliability = reliability_from_validation(records)

    observed = example["fit_on_reported"]["slope"]
    corrected = disattenuate_slope(
        observed,
        example["fit_on_reported"]["slope_se"],
        reliability["reliability"],
        reliability_se=reliability["reliability_se"],
    )["corrected_slope"]

    assert abs(observed - 0.5) > abs(corrected - 0.5), (
        "The correction should move the estimate toward the truth: "
        "observed %.4f, corrected %.4f, truth 0.5" % (observed, corrected)
    )
    assert corrected == pytest.approx(0.5, abs=0.06)


def test_the_naive_slope_is_attenuated_by_roughly_lambda():
    records = demo_records(
        count=1500, true_sd=400.0, error_sd=300.0, validation_share=1.0, seed=141
    )
    example = demo_regression(records, true_slope=0.5)
    expected = _theoretical_lambda(400.0, 300.0)
    assert example["attenuation_observed"] == pytest.approx(expected, abs=0.10)


def test_attenuation_never_pushes_a_slope_away_from_zero():
    for seed in (151, 152, 153, 154):
        records = demo_records(count=600, validation_share=1.0, seed=seed)
        example = demo_regression(records, true_slope=0.5)
        assert abs(example["fit_on_reported"]["slope"]) <= abs(
            example["fit_on_truth"]["slope"]
        ) + 1e-9


def test_the_correction_factor_is_one_over_lambda():
    result = disattenuate_slope(0.4, 0.02, 0.8)
    assert result["correction_factor"] == pytest.approx(1.25)
    assert result["corrected_slope"] == pytest.approx(0.5)


def test_a_lower_reliability_means_a_larger_correction():
    factors = [disattenuate_slope(0.4, 0.02, lam)["correction_factor"] for lam in (0.9, 0.6, 0.3)]
    assert factors == sorted(factors)


def test_the_delta_method_widens_beyond_dividing_the_error():
    result = disattenuate_slope(0.4, 0.02, 0.7, reliability_se=0.05)
    assert result["corrected_se"] > result["naive_corrected_se"]
    assert result["se_inflation_from_lambda"] > 1.0


def test_a_known_reliability_leaves_the_error_undivided_only_by_lambda():
    result = disattenuate_slope(0.4, 0.02, 0.7, reliability_se=0.0)
    assert result["corrected_se"] == pytest.approx(result["naive_corrected_se"])


def test_the_corrected_slope_sits_inside_its_interval():
    result = disattenuate_slope(0.4, 0.02, 0.7, reliability_se=0.05)
    assert result["lower"] <= result["corrected_slope"] <= result["upper"]


def test_intervals_widen_as_confidence_rises():
    narrow = disattenuate_slope(0.4, 0.02, 0.7, 0.05, confidence=0.80)
    wide = disattenuate_slope(0.4, 0.02, 0.7, 0.05, confidence=0.99)
    assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])


def test_a_reliability_below_the_floor_is_refused():
    with pytest.raises(CalibrationError, match="below the floor"):
        disattenuate_slope(0.4, 0.02, RELIABILITY_FLOOR / 2.0)


def test_a_zero_reliability_is_refused():
    with pytest.raises(CalibrationError, match="nothing to correct"):
        disattenuate_slope(0.4, 0.02, 0.0)


def test_a_reliability_above_one_is_refused():
    with pytest.raises(CalibrationError, match="above one is not possible"):
        disattenuate_slope(0.4, 0.02, 1.4)


def test_an_unsupported_confidence_is_refused():
    with pytest.raises(CalibrationError, match="Confidence must be"):
        disattenuate_slope(0.4, 0.02, 0.8, confidence=0.77)


# ---------------------------------------------------------------------------
# Correlations attenuate twice
# ---------------------------------------------------------------------------


def test_a_correlation_corrects_by_the_geometric_mean():
    result = disattenuate_correlation(0.36, 0.81, 0.64)
    assert result["corrected"] == pytest.approx(0.36 / math.sqrt(0.81 * 0.64))


def test_two_noisy_variables_attenuate_more_than_one():
    one_noisy = disattenuate_correlation(0.4, 0.7, 1.0)["correction_factor"]
    both_noisy = disattenuate_correlation(0.4, 0.7, 0.7)["correction_factor"]
    assert both_noisy > one_noisy


def test_an_impossible_correction_is_reported_not_clamped():
    result = disattenuate_correlation(0.9, 0.4, 0.4)
    assert result["impossible"] is True
    assert abs(result["corrected"]) > 1.0
    assert "contradiction" in result["headline"]


def test_a_correlation_outside_the_unit_interval_is_refused():
    with pytest.raises(CalibrationError, match=r"must be in \[-1, 1\]"):
        disattenuate_correlation(1.4, 0.8, 0.8)


def test_a_reliability_outside_the_unit_interval_is_refused():
    with pytest.raises(CalibrationError, match="must be in"):
        disattenuate_correlation(0.4, 0.0, 0.8)


def test_perfect_reliabilities_leave_the_correlation_alone():
    assert disattenuate_correlation(0.4, 1.0, 1.0)["corrected"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# SIMEX
# ---------------------------------------------------------------------------


def test_simex_moves_the_slope_away_from_zero():
    records = demo_records(count=800, validation_share=1.0, seed=161)
    example = demo_regression(records, true_slope=0.5)
    result = simex(example["reported"], example["outcome"], 300.0**2)
    assert abs(result["corrected_slope"]) > abs(result["naive_slope"])


def test_simex_slopes_fall_as_more_noise_is_added():
    records = demo_records(count=800, validation_share=1.0, seed=171)
    example = demo_regression(records, true_slope=0.5)
    result = simex(example["reported"], example["outcome"], 300.0**2)
    slopes = [point["slope"] for point in result["curve"]]
    assert slopes[0] > slopes[-1]


def test_simex_is_deterministic_for_a_seed():
    records = demo_records(count=400, validation_share=1.0, seed=181)
    example = demo_regression(records, true_slope=0.5)
    first = simex(example["reported"], example["outcome"], 90000.0, seed=5)
    second = simex(example["reported"], example["outcome"], 90000.0, seed=5)
    assert first["corrected_slope"] == pytest.approx(second["corrected_slope"])


def test_simex_refuses_a_zero_error_variance():
    with pytest.raises(CalibrationError, match="nothing to extrapolate"):
        simex([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 0.0)


def test_simex_refuses_a_negative_error_variance():
    with pytest.raises(CalibrationError, match="cannot be negative"):
        simex([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], -1.0)


def test_simex_refuses_a_single_replicate():
    with pytest.raises(CalibrationError, match="at least two replicates"):
        simex([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 1.0, replicates=1)


def test_simex_refuses_a_non_positive_level():
    with pytest.raises(CalibrationError, match="levels must be positive"):
        simex([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 1.0, lambdas=(0.0, 1.0))


# ---------------------------------------------------------------------------
# Heaping
# ---------------------------------------------------------------------------


def test_unheaped_values_pass():
    values = [record["reported"] for record in demo_records(count=500, seed=191)]
    result = heaping_diagnostics(values)
    assert result["heaped"] is False
    assert result["whipple_index"] < WHIPPLE_ROUGH


def test_values_rounded_to_fifty_are_caught():
    values = [
        record["reported"] for record in demo_records(count=500, heap_to=50, seed=201)
    ]
    result = heaping_diagnostics(values)
    assert result["heaped"] is True
    assert result["whipple_index"] == pytest.approx(500.0)
    assert result["effective_precision"] == 50


def test_the_whipple_index_of_clean_data_is_near_one_hundred():
    values = [record["reported"] for record in demo_records(count=2000, seed=211)]
    assert heaping_diagnostics(values)["whipple_index"] == pytest.approx(100.0, abs=25.0)


def test_digit_counts_sum_to_the_sample_size():
    values = [record["reported"] for record in demo_records(count=300, seed=221)]
    result = heaping_diagnostics(values)
    assert sum(result["digit_counts"]) == result["n"]


def test_heaping_reports_round_multiple_shares():
    values = [
        record["reported"] for record in demo_records(count=400, heap_to=25, seed=231)
    ]
    shares = heaping_diagnostics(values)["round_multiple_share"]
    assert shares[25] > 0.95
    assert shares[5] > 0.95


def test_heaping_needs_enough_values():
    with pytest.raises(CalibrationError, match="at least %d values" % MIN_REPEAT_PAIRS):
        heaping_diagnostics([1.0, 2.0, 3.0])


def test_heaping_rejects_non_numeric_values():
    with pytest.raises(CalibrationError, match="non-numeric"):
        heaping_diagnostics([1.0] * 10 + ["x"])


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------


def test_random_errors_partly_cancel_in_a_sum():
    components = [make_component("c%d" % i, 1000.0, error_sd=100.0) for i in range(9)]
    result = propagate_to_total(components)
    assert result["random_sd"] == pytest.approx(300.0)
    assert result["sum_of_component_sds"] == pytest.approx(900.0)
    assert result["cancellation_factor"] == pytest.approx(1.0 / 3.0)


def test_systematic_bias_does_not_cancel():
    components = [make_component("c%d" % i, 1000.0, bias=-100.0) for i in range(9)]
    result = propagate_to_total(components)
    assert result["systematic_bias"] == pytest.approx(-900.0)
    assert result["corrected_total"] == pytest.approx(9900.0)


def test_more_components_dilute_random_error_relative_to_bias():
    def relative(count):
        components = [
            make_component("c%d" % i, 1000.0, error_sd=100.0, bias=-100.0)
            for i in range(count)
        ]
        result = propagate_to_total(components)
        return result["random_sd"] / abs(result["systematic_bias"])

    assert relative(4) > relative(25)


def test_bias_domination_is_flagged():
    small = propagate_to_total(
        [make_component("a", 1000.0, error_sd=500.0, bias=-10.0)]
    )
    large = propagate_to_total(
        [make_component("a", 1000.0, error_sd=10.0, bias=-500.0)]
    )
    assert small["bias_dominates"] is False
    assert large["bias_dominates"] is True


def test_totals_refuse_an_empty_component_list():
    with pytest.raises(CalibrationError, match="No components"):
        propagate_to_total([])


def test_totals_refuse_a_negative_error_standard_deviation():
    with pytest.raises(CalibrationError, match="cannot be negative"):
        propagate_to_total([make_component("a", 100.0, error_sd=-1.0)])


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------


def test_analyse_prefers_validation_over_repeats():
    result = analyse(demo_records(count=600, seed=241))
    assert result["source"] == "validation"


def test_analyse_falls_back_to_repeats_without_validation():
    result = analyse(demo_records(count=600, validation_share=0.0, seed=251))
    assert result["source"] == "repeats"


def test_analyse_blocks_without_any_reliability_source():
    records = demo_records(count=600, validation_share=0.0, repeat_share=0.0, seed=261)
    result = analyse(records)
    assert result["blocked"]
    assert "guessed reliability" in result["blocked"]
    assert result["correction"] is None


def test_analyse_blocks_on_differential_error():
    records = demo_records(count=600, differential_slope=0.5, seed=271)
    result = analyse(records, slope=0.4, slope_se=0.02)
    assert result["blocked"]
    assert result["correction"] is None


def test_analyse_returns_a_correction_when_given_a_slope():
    result = analyse(demo_records(count=600, seed=281), slope=0.4, slope_se=0.02)
    assert result["correction"] is not None
    assert result["correction"]["corrected_slope"] > 0.4


def test_analyse_carries_the_engine_version():
    assert analyse(demo_records(count=200))["engine_version"] == ENGINE_VERSION


def test_analyse_counts_the_records_it_had():
    records = demo_records(count=250, seed=291)
    result = analyse(records)
    assert result["records"] == 250
    assert result["with_repeat"] == sum(1 for r in records if r["has_repeat"])
    assert result["with_validation"] == sum(1 for r in records if r["has_validation"])


def test_analyse_uses_the_default_confidence():
    assert analyse(demo_records(count=200))["confidence"] == DEFAULT_CONFIDENCE


def test_analyse_refuses_an_empty_record_set():
    with pytest.raises(CalibrationError, match="No records"):
        analyse([])


def test_analyse_refuses_raw_mappings():
    with pytest.raises(CalibrationError, match="build_record"):
        analyse([{"value": 1.0}])


def test_analyse_blocks_rather_than_correcting_below_the_floor():
    """A very noisy variable produces a lambda under the floor, and no number."""
    records = demo_records(count=800, true_sd=100.0, error_sd=900.0, seed=301)
    result = analyse(records, slope=0.4, slope_se=0.02)
    assert result["correction"] is None
    assert result["blocked"]


# ---------------------------------------------------------------------------
# Bands, notes and summaries
# ---------------------------------------------------------------------------


def test_reliability_bands_use_the_documented_cuts():
    assert reliability_band(GOOD_RELIABILITY) == "good"
    assert reliability_band(POOR_RELIABILITY) == "usable"
    assert reliability_band(RELIABILITY_FLOOR) == "poor"
    assert reliability_band(RELIABILITY_FLOOR / 2.0) == "unusable"


def test_attenuation_table_marks_what_cannot_be_corrected():
    rows = attenuation_table()
    assert any(row["correctable"] is False for row in rows)
    assert all(
        row["correctable"] == (row["reliability"] >= RELIABILITY_FLOOR) for row in rows
    )


def test_attenuation_table_understatement_matches_the_reliability():
    for row in attenuation_table(slope=1.0):
        assert row["observed_slope"] == pytest.approx(row["reliability"])


def test_notes_lead_with_the_headline():
    result = analyse(demo_records(count=300, seed=311))
    assert get_measurement_notes(result)[0] == result["headline"]


def test_notes_explain_a_block():
    records = demo_records(count=400, validation_share=0.0, repeat_share=0.0, seed=321)
    notes = get_measurement_notes(analyse(records))
    assert any("not a missing result" in note for note in notes)


def test_notes_flag_heaping():
    result = analyse(demo_records(count=500, heap_to=50, seed=331))
    assert any("heaped on multiples" in note for note in get_measurement_notes(result))


def test_notes_warn_that_repeats_flatter_the_reliability():
    result = analyse(demo_records(count=500, validation_share=0.0, seed=341))
    assert any("share whatever bias" in note for note in get_measurement_notes(result))


def test_summary_is_one_line():
    summary = summarise(analyse(demo_records(count=900, seed=352)))
    assert "\n" not in summary
    assert "lambda" in summary


def test_the_differential_gate_blocks_at_about_its_nominal_rate():
    """A 10% alpha should falsely block about 10% of clean datasets.

    Materially more than that would mean the test is mis-calibrated and the
    module is refusing corrections it should be making; materially less would
    mean it is not catching differential error either.
    """
    blocked = sum(
        1
        for seed in range(120)
        if analyse(demo_records(count=300, seed=seed)).get("blocked")
    )
    assert 0.02 <= blocked / 120.0 <= 0.20


def test_summary_says_so_when_blocked():
    records = demo_records(count=400, validation_share=0.0, repeat_share=0.0, seed=361)
    assert summarise(analyse(records)).startswith("blocked:")


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def test_t_cdf_is_a_half_at_zero():
    assert t_cdf(0.0, 10) == pytest.approx(0.5)


def test_two_sided_p_of_zero_is_one():
    assert two_sided_p(0.0, 10) == pytest.approx(1.0)


def test_two_sided_p_falls_as_the_statistic_grows():
    assert two_sided_p(4.0, 10) < two_sided_p(1.0, 10)


def test_betai_endpoints():
    assert betai(2.0, 3.0, 0.0) == pytest.approx(0.0)
    assert betai(2.0, 3.0, 1.0) == pytest.approx(1.0)


def test_chi_square_p_of_zero_is_one():
    assert chi_square_p(0.0, 9) == pytest.approx(1.0)


def test_chi_square_p_falls_as_the_statistic_grows():
    assert chi_square_p(30.0, 9) < chi_square_p(5.0, 9)


def test_chi_square_p_near_the_known_five_percent_point():
    """The 0.05 critical value on 9 df is 16.919."""
    assert chi_square_p(16.919, 9) == pytest.approx(0.05, abs=0.002)


def test_chi_square_rejects_zero_degrees_of_freedom():
    with pytest.raises(CalibrationError, match="must be positive"):
        chi_square_p(5.0, 0)


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def test_demo_records_are_deterministic_for_a_seed():
    first = demo_records(count=50, seed=999)
    second = demo_records(count=50, seed=999)
    assert [r["reported"] for r in first] == [r["reported"] for r in second]


def test_demo_records_honour_the_count():
    assert len(demo_records(count=77)) == 77


def test_demo_records_respect_the_validation_share():
    records = demo_records(count=1000, validation_share=0.0)
    assert all(not record["has_validation"] for record in records)


def test_demo_regression_needs_validated_records():
    with pytest.raises(CalibrationError, match="Not enough validated"):
        demo_regression(demo_records(count=100, validation_share=0.0))


def test_demo_regression_recovers_the_true_slope_from_the_truth():
    records = demo_records(count=1200, validation_share=1.0, seed=371)
    example = demo_regression(records, true_slope=0.5)
    assert example["fit_on_truth"]["slope"] == pytest.approx(0.5, abs=0.03)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_read_back_an_analysis(temp_db):
    result = analyse(demo_records(count=300, seed=381))
    analysis_id = save_analysis("user-1", result, label="Energy", category="energy_kwh")
    assert analysis_id is not None

    analyses = get_analyses("user-1")
    assert len(analyses) == 1
    assert analyses[0]["label"] == "Energy"
    assert analyses[0]["category"] == "energy_kwh"
    assert analyses[0]["payload"]["engine_version"] == ENGINE_VERSION


def test_a_blocked_analysis_is_stored_as_blocked(temp_db):
    records = demo_records(count=400, validation_share=0.0, repeat_share=0.0, seed=391)
    save_analysis("user-1", analyse(records))
    assert get_analyses("user-1")[0]["blocked"] is True


def test_analyses_are_scoped_to_their_user(temp_db):
    save_analysis("user-1", analyse(demo_records(count=200)))
    assert get_analyses("user-2") == []


def test_saving_without_a_user_is_a_no_op(temp_db):
    assert save_analysis("", analyse(demo_records(count=200))) is None


def test_saving_a_non_result_is_a_no_op(temp_db):
    assert save_analysis("user-1", {}) is None


def test_delete_removes_only_the_named_analysis(temp_db):
    result = analyse(demo_records(count=200))
    first = save_analysis("user-1", result, label="one")
    save_analysis("user-1", result, label="two")
    assert delete_analysis("user-1", first) is True
    remaining = get_analyses("user-1")
    assert len(remaining) == 1
    assert remaining[0]["label"] == "two"


def test_delete_refuses_another_users_analysis(temp_db):
    analysis_id = save_analysis("user-1", analyse(demo_records(count=200)))
    assert delete_analysis("user-2", analysis_id) is False
    assert len(get_analyses("user-1")) == 1


def test_delete_without_a_user_is_false(temp_db):
    assert delete_analysis("", 1) is False


def test_reads_without_a_user_are_empty(temp_db):
    assert get_analyses(None) == []


def test_analyses_come_back_newest_first(temp_db):
    result = analyse(demo_records(count=200))
    save_analysis("user-1", result, label="older")
    save_analysis("user-1", result, label="newer")
    assert [entry["label"] for entry in get_analyses("user-1")] == ["newer", "older"]


def test_storage_failure_is_swallowed_not_raised(monkeypatch):
    """A dashboard must render when the database is unavailable."""

    def explode(*_args, **_kwargs):
        raise sqlite3.Error("disk is on fire")

    monkeypatch.setattr(measurement_error, "_connect", explode)
    assert save_analysis("user-1", analyse(demo_records(count=200))) is None
    assert get_analyses("user-1") == []
    assert delete_analysis("user-1", 1) is False


def test_a_corrupt_payload_reads_back_as_empty(temp_db):
    save_analysis("user-1", analyse(demo_records(count=200)))
    with sqlite3.connect(temp_db) as conn:
        conn.execute("UPDATE measurement_error_analyses SET payload = 'not json'")
    assert get_analyses("user-1")[0]["payload"] == {}
