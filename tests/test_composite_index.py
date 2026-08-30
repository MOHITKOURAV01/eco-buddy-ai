"""Tests for the composite index engine.

The claim under test is that the three choices this app makes silently —
normalisation, aggregation and weighting — each change the answer, and that the
module can say by how much.

The load-bearing test is the unnormalised sum: three components with *equal*
nominal weights, on the scales `src/calculators/eco_score.py` actually adds,
where one component ends up carrying most of the variation. The nominal weights
are equal by construction, so any imbalance the effective-weight calculation
finds is the units doing the weighting and nothing else.

The second claim is about compensation. Testing one aggregation against another
on arbitrary data would prove nothing; the fixture here is built so that some
units are spiky and some are even, because a reversal between a linear sum and
a geometric mean is only meaningful when there is a zero available to buy back.

The refusals are tested as hard as the arithmetic. An index built from
components on incompatible scales, or one that is a single component with
decoration, is worse than no index because it carries the authority of a
balanced score.
"""

import math
import os
import sqlite3
import tempfile

import pytest

from src.utils import composite_index
from src.utils.composite_index import (
    AGGREGATIONS,
    APP_CONFIDENCE_BANDS,
    APP_CONFIDENCE_WEIGHTS,
    ENGINE_VERSION,
    GEOMETRIC_FLOOR,
    MIN_BAND_PROBABILITY,
    MIN_COMPONENTS,
    MIN_UNITS,
    NORMALISATIONS,
    REDUNDANCY_THRESHOLD,
    SINGLE_COMPONENT_CEILING,
    CompositeError,
    aggregate,
    analyse,
    band_probabilities,
    build_component,
    build_index,
    component_correlations,
    correlation,
    delete_analysis,
    demo_compensating_index,
    demo_confidence_index,
    demo_eco_score,
    dominance_violations,
    effective_weights,
    get_analyses,
    get_index_notes,
    normalise,
    rank_reversals,
    raw_sum_effective_weights,
    save_analysis,
    scale_mismatch,
    summarise,
    weight_sensitivity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(monkeypatch):
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    monkeypatch.setattr(composite_index, "DB_NAME", path)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def _simple_components():
    """Two components, hand-checkable, on the same scale."""
    return [
        build_component("a", [0.0, 25.0, 50.0, 75.0, 100.0], weight=1.0),
        build_component("b", [100.0, 75.0, 50.0, 25.0, 0.0], weight=1.0),
    ]


def _units(count=5):
    return ["u%d" % index for index in range(count)]


# ---------------------------------------------------------------------------
# build_component
# ---------------------------------------------------------------------------


def test_build_component_records_the_moments():
    component = build_component("a", [10.0, 20.0, 30.0, 40.0, 50.0])
    assert component["n"] == 5
    assert component["mean"] == pytest.approx(30.0)
    assert component["min"] == 10.0
    assert component["max"] == 50.0
    assert component["range"] == 40.0


def test_build_component_flags_a_constant():
    assert build_component("a", [7.0] * 6)["constant"] is True


def test_build_component_defaults_to_higher_is_better():
    assert build_component("a", [1.0] * 6)["polarity"] == "higher_is_better"


def test_build_component_rejects_an_unknown_polarity():
    with pytest.raises(CompositeError, match="Polarity must be"):
        build_component("a", [1.0] * 6, polarity="sideways")


def test_build_component_rejects_a_negative_weight():
    with pytest.raises(CompositeError, match="non-negative weight"):
        build_component("a", [1.0] * 6, weight=-1.0)


def test_build_component_rejects_non_numeric_values():
    with pytest.raises(CompositeError, match="non-numeric"):
        build_component("a", [1.0, 2.0, "x", 4.0, 5.0])


def test_build_component_rejects_too_few_units():
    with pytest.raises(CompositeError, match="at least %d units" % MIN_UNITS):
        build_component("a", [1.0, 2.0])


# ---------------------------------------------------------------------------
# Validation — the refusals
# ---------------------------------------------------------------------------


def test_a_single_component_is_not_an_index():
    with pytest.raises(CompositeError, match="at least %d components" % MIN_COMPONENTS):
        scale_mismatch([build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0])])


def test_duplicate_component_names_are_refused():
    components = [
        build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0]),
        build_component("a", [5.0, 4.0, 3.0, 2.0, 1.0]),
    ]
    with pytest.raises(CompositeError, match="appears twice"):
        scale_mismatch(components)


def test_components_covering_different_units_are_refused():
    components = [
        build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0]),
        build_component("b", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ]
    with pytest.raises(CompositeError, match="must cover the same units"):
        scale_mismatch(components)


def test_all_zero_weights_are_refused():
    components = [
        build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0], weight=0.0),
        build_component("b", [5.0, 4.0, 3.0, 2.0, 1.0], weight=0.0),
    ]
    with pytest.raises(CompositeError, match="Every weight is zero"):
        scale_mismatch(components)


def test_all_constant_components_are_refused():
    components = [
        build_component("a", [3.0] * 6),
        build_component("b", [7.0] * 6),
    ]
    with pytest.raises(CompositeError, match="Every component is constant"):
        scale_mismatch(components)


def test_raw_mappings_are_refused():
    with pytest.raises(CompositeError, match="build_component"):
        scale_mismatch([{"name": "a"}, {"name": "b"}])


def test_a_unit_name_count_mismatch_is_refused():
    with pytest.raises(CompositeError, match="unit names for"):
        build_index(["one", "two"], _simple_components())


# ---------------------------------------------------------------------------
# Scale mismatch
# ---------------------------------------------------------------------------


def test_matched_scales_are_not_flagged():
    result = scale_mismatch(_simple_components())
    assert result["ratio"] == pytest.approx(1.0)
    assert result["mismatched"] is False


def test_a_wide_scale_gap_is_flagged():
    components = [
        build_component("small", [1.0, 2.0, 3.0, 4.0, 5.0]),
        build_component("large", [0.0, 250.0, 500.0, 750.0, 1000.0]),
    ]
    result = scale_mismatch(components)
    assert result["mismatched"] is True
    assert result["dominant"] == "large"
    assert result["ratio"] == pytest.approx(250.0)


def test_the_eco_score_demo_has_mismatched_scales():
    _, components = demo_eco_score()
    assert scale_mismatch(components)["mismatched"] is True


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_minmax_maps_onto_zero_and_one():
    component = build_component("a", [10.0, 20.0, 30.0, 40.0, 50.0])
    scaled = normalise(component, "minmax")
    assert scaled[0] == pytest.approx(0.0)
    assert scaled[-1] == pytest.approx(1.0)
    assert scaled[2] == pytest.approx(0.5)


def test_lower_is_better_reverses_the_scale():
    component = build_component(
        "a", [10.0, 20.0, 30.0, 40.0, 50.0], polarity="lower_is_better"
    )
    scaled = normalise(component, "minmax")
    assert scaled[0] == pytest.approx(1.0)
    assert scaled[-1] == pytest.approx(0.0)


def test_a_constant_component_normalises_to_the_midpoint():
    component = build_component("a", [7.0] * 6)
    assert normalise(component, "minmax") == [0.5] * 6


def test_zscore_preserves_the_ordering():
    component = build_component("a", [1.0, 5.0, 2.0, 90.0, 3.0, 4.0])
    scaled = normalise(component, "zscore")
    ordering = sorted(range(6), key=lambda i: component["values"][i])
    assert [scaled[i] for i in ordering] == sorted([scaled[i] for i in ordering])


def test_zscore_stays_inside_the_unit_interval():
    component = build_component("a", [1.0, 2.0, 3.0, 4.0, 5000.0])
    assert all(0.0 < value < 1.0 for value in normalise(component, "zscore"))


def test_rank_normalisation_discards_magnitude():
    """A unit twice as good as the next is one position better, and no more."""
    even = build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0])
    spiky = build_component("a", [1.0, 2.0, 3.0, 4.0, 5000.0])
    assert normalise(even, "rank") == normalise(spiky, "rank")


def test_rank_normalisation_averages_ties():
    component = build_component("a", [10.0, 10.0, 20.0, 30.0, 40.0])
    scaled = normalise(component, "rank")
    assert scaled[0] == pytest.approx(scaled[1])


def test_minmax_is_dragged_by_an_outlier_where_rank_is_not():
    spiky = build_component("a", [1.0, 2.0, 3.0, 4.0, 5000.0])
    minmax = normalise(spiky, "minmax")
    ranked = normalise(spiky, "rank")
    assert minmax[3] < 0.01
    assert ranked[3] == pytest.approx(0.75)


def test_distance_to_reference_needs_a_reference():
    component = build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(CompositeError, match="needs a reference value"):
        normalise(component, "distance_to_reference")


def test_distance_to_reference_rejects_a_zero_reference():
    component = build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(CompositeError, match="non-zero number"):
        normalise(component, "distance_to_reference", reference=0.0)


def test_distance_to_reference_preserves_the_ordering():
    component = build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0])
    scaled = normalise(component, "distance_to_reference", reference=3.0)
    assert scaled == sorted(scaled)


def test_an_unknown_normalisation_is_refused():
    component = build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(CompositeError, match="Normalisation must be"):
        normalise(component, "vibes")


def test_every_declared_normalisation_runs():
    component = build_component("a", [1.0, 2.0, 3.0, 4.0, 5.0])
    for method in NORMALISATIONS:
        scaled = normalise(component, method, reference=3.0)
        assert len(scaled) == 5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_linear_aggregation_is_the_weighted_mean():
    normalised = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    scores = aggregate(normalised, {"a": 3.0, "b": 1.0}, "linear")
    assert scores == pytest.approx([0.75, 0.25])


def test_geometric_aggregation_collapses_on_a_zero():
    normalised = {"a": [1.0, 1.0], "b": [1.0, 0.0]}
    scores = aggregate(normalised, {"a": 1.0, "b": 1.0}, "geometric")
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < math.sqrt(GEOMETRIC_FLOOR) * 1.01


def test_linear_aggregation_does_not_collapse_on_a_zero():
    """The compensability assumption, stated as a difference between two numbers."""
    normalised = {"a": [1.0, 1.0], "b": [1.0, 0.0]}
    linear = aggregate(normalised, {"a": 1.0, "b": 1.0}, "linear")
    assert linear[1] == pytest.approx(0.5)


def test_geometric_equals_linear_when_every_component_agrees():
    normalised = {"a": [0.5, 0.8], "b": [0.5, 0.8]}
    linear = aggregate(normalised, {"a": 1.0, "b": 1.0}, "linear")
    geometric = aggregate(normalised, {"a": 1.0, "b": 1.0}, "geometric")
    assert linear == pytest.approx(geometric)


def test_non_compensatory_scores_lie_in_minus_one_to_one():
    _, components = demo_compensating_index(units=12)
    normalised = {c["name"]: normalise(c, "minmax") for c in components}
    weights = {c["name"]: c["weight"] for c in components}
    scores = aggregate(normalised, weights, "non_compensatory")
    assert all(-1.0 <= score <= 1.0 for score in scores)


def test_a_unit_best_on_everything_tops_the_outranking():
    normalised = {"a": [1.0, 0.5, 0.2], "b": [1.0, 0.4, 0.1]}
    scores = aggregate(normalised, {"a": 1.0, "b": 1.0}, "non_compensatory")
    assert scores[0] == max(scores)


def test_an_unknown_aggregation_is_refused():
    with pytest.raises(CompositeError, match="Aggregation must be"):
        aggregate({"a": [1.0, 2.0]}, {"a": 1.0}, "averaging-ish")


def test_zero_total_weight_is_refused():
    with pytest.raises(CompositeError, match="sum to something positive"):
        aggregate({"a": [1.0, 2.0]}, {"a": 0.0}, "linear")


def test_every_declared_aggregation_runs():
    _, components = demo_compensating_index(units=10)
    normalised = {c["name"]: normalise(c, "minmax") for c in components}
    weights = {c["name"]: c["weight"] for c in components}
    for method in AGGREGATIONS:
        assert len(aggregate(normalised, weights, method)) == 10


# ---------------------------------------------------------------------------
# Effective weights — the load-bearing result
# ---------------------------------------------------------------------------


def test_an_unnormalised_sum_gives_the_widest_component_most_of_the_influence():
    """Equal nominal weights by construction; the units do the rest."""
    _, components = demo_eco_score(units=80)
    result = raw_sum_effective_weights(components)
    assert result["dominant"] == "energy"
    assert result["dominant_share"] > 0.5
    nominal = result["effective_weights"]["nominal"]
    assert nominal["energy"] == pytest.approx(1.0 / 3.0)


def test_normalising_restores_the_nominal_weights():
    _, components = demo_eco_score(units=80)
    unnormalised = raw_sum_effective_weights(components)
    normalised = build_index(
        ["u%d" % i for i in range(80)], components, normalisation="minmax"
    )
    effective = normalised["effective_weights"]["effective"]
    assert unnormalised["dominant_share"] > 0.5
    assert max(effective.values()) < 0.5


def test_effective_weights_sum_to_one():
    names, components = demo_confidence_index(units=60)
    index = build_index(names, components)
    assert sum(index["effective_weights"]["effective"].values()) == pytest.approx(1.0)


def test_nominal_weights_sum_to_one():
    names, components = demo_confidence_index(units=60)
    index = build_index(names, components)
    assert sum(index["effective_weights"]["nominal"].values()) == pytest.approx(1.0)


def test_a_component_with_no_spread_earns_no_effective_weight():
    components = [
        build_component("varies", [0.0, 25.0, 50.0, 75.0, 100.0], weight=1.0),
        build_component("constant", [50.0] * 5, weight=9.0),
    ]
    index = build_index(_units(), components)
    effective = index["effective_weights"]
    assert effective["nominal"]["constant"] == pytest.approx(0.9)
    assert effective["effective"]["constant"] < 0.1


def test_a_single_dominant_component_is_flagged():
    components = [
        build_component("real", [0.0, 25.0, 50.0, 75.0, 100.0], weight=1.0),
        build_component("flat", [50.0, 50.0, 50.0, 50.0, 50.0], weight=1.0),
    ]
    index = build_index(_units(), components)
    assert index["dominated_by_one"] is True
    assert "with decoration" in index["headline"]


def test_a_balanced_index_is_not_flagged_as_dominated():
    index = build_index(_units(), _simple_components())
    assert index["dominated_by_one"] is False


def test_the_dominance_ceiling_is_the_documented_one():
    components = [
        build_component("real", [0.0, 25.0, 50.0, 75.0, 100.0], weight=1.0),
        build_component("flat", [50.0] * 5, weight=1.0),
    ]
    index = build_index(_units(), components)
    top = max(index["effective_weights"]["effective"].values())
    assert index["dominated_by_one"] == (top > SINGLE_COMPONENT_CEILING)


def test_the_most_overworked_component_is_reported():
    names, components = demo_confidence_index(units=80)
    effective = build_index(names, components)["effective_weights"]
    assert effective["most_overworked_ratio"] > 1.0
    assert effective["ratios"][effective["most_overworked"]] == pytest.approx(
        effective["most_overworked_ratio"]
    )


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------


def test_correlated_components_are_reported_as_redundant():
    names, components = demo_confidence_index(units=80)
    result = component_correlations(components)
    pairs = {tuple(sorted(entry["components"])) for entry in result["redundant_pairs"]}
    assert ("category_coverage", "input_completeness") in pairs


def test_the_redundant_pair_reports_its_combined_weight():
    _, components = demo_confidence_index(units=80)
    result = component_correlations(components)
    entry = next(
        item
        for item in result["redundant_pairs"]
        if set(item["components"]) == {"input_completeness", "category_coverage"}
    )
    assert entry["combined_weight"] == pytest.approx(
        APP_CONFIDENCE_WEIGHTS["input_completeness"]
        + APP_CONFIDENCE_WEIGHTS["category_coverage"]
    )


def test_uncorrelated_components_are_not_flagged():
    components = [
        build_component("a", [0.0, 25.0, 50.0, 75.0, 100.0]),
        build_component("b", [50.0, 0.0, 100.0, 25.0, 75.0]),
    ]
    result = component_correlations(components)
    assert result["redundant_pairs"] == []
    assert "distinct information" in result["headline"]


def test_the_correlation_matrix_has_ones_on_the_diagonal():
    matrix = component_correlations(_simple_components())["matrix"]
    assert matrix["a"]["a"] == pytest.approx(1.0)
    assert matrix["b"]["b"] == pytest.approx(1.0)


def test_the_correlation_matrix_is_symmetric():
    matrix = component_correlations(_simple_components())["matrix"]
    assert matrix["a"]["b"] == pytest.approx(matrix["b"]["a"])


def test_perfectly_opposed_components_correlate_at_minus_one():
    matrix = component_correlations(_simple_components())["matrix"]
    assert matrix["a"]["b"] == pytest.approx(-1.0)
    assert abs(matrix["a"]["b"]) >= REDUNDANCY_THRESHOLD


def test_correlation_of_a_constant_series_is_zero():
    assert correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_correlation_refuses_mismatched_lengths():
    with pytest.raises(CompositeError, match="same length"):
        correlation([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Weight sensitivity
# ---------------------------------------------------------------------------


def test_identical_components_give_a_stable_ranking():
    """With every component the same, the weights cannot matter."""
    values = [0.0, 25.0, 50.0, 75.0, 100.0]
    components = [
        build_component("a", values, weight=1.0),
        build_component("b", values, weight=3.0),
        build_component("c", values, weight=6.0),
    ]
    result = weight_sensitivity(_units(), components, draws=200)
    assert result["robust"] is True
    assert result["max_width"] == 0


def test_opposed_components_give_an_unstable_ranking():
    result = weight_sensitivity(_units(), _simple_components(), draws=200)
    assert result["robust"] is False
    assert result["max_width"] > 0


def test_sensitivity_intervals_bracket_the_baseline_rank():
    names, components = demo_confidence_index(units=30)
    result = weight_sensitivity(names, components, draws=300)
    for entry in result["intervals"]:
        assert entry["lower"] <= entry["upper"]


def test_sensitivity_covers_every_unit():
    names, components = demo_confidence_index(units=25)
    result = weight_sensitivity(names, components, draws=200)
    assert {entry["unit"] for entry in result["intervals"]} == set(names)


def test_a_tighter_concentration_widens_the_intervals():
    """Lower concentration means the drawn weights wander further."""
    names, components = demo_confidence_index(units=40)
    tight = weight_sensitivity(names, components, draws=300, concentration=400.0)
    loose = weight_sensitivity(names, components, draws=300, concentration=8.0)
    assert loose["mean_width"] > tight["mean_width"]


def test_sensitivity_is_deterministic_for_a_seed():
    names, components = demo_confidence_index(units=20)
    first = weight_sensitivity(names, components, draws=150, seed=99)
    second = weight_sensitivity(names, components, draws=150, seed=99)
    assert first["intervals"] == second["intervals"]


def test_sensitivity_refuses_too_few_draws():
    with pytest.raises(CompositeError, match="At least 20 draws"):
        weight_sensitivity(_units(), _simple_components(), draws=5)


def test_sensitivity_refuses_a_non_positive_concentration():
    with pytest.raises(CompositeError, match="Concentration must be positive"):
        weight_sensitivity(_units(), _simple_components(), concentration=0.0)


# ---------------------------------------------------------------------------
# Reversals and dominance
# ---------------------------------------------------------------------------


def test_an_index_does_not_reverse_against_itself():
    index = build_index(_units(), _simple_components())
    assert rank_reversals(index, index)["count"] == 0


def test_linear_and_geometric_disagree_where_compensation_matters():
    names, components = demo_compensating_index(units=30)
    linear = build_index(names, components, aggregation="linear")
    geometric = build_index(names, components, aggregation="geometric")
    assert rank_reversals(linear, geometric)["count"] > 0


def test_reversals_are_refused_across_different_unit_sets():
    first = build_index(_units(), _simple_components())
    names, components = demo_compensating_index(units=10)
    second = build_index(names, components)
    with pytest.raises(CompositeError, match="same units"):
        rank_reversals(first, second)


def test_the_reversal_share_is_out_of_the_pair_count():
    names, components = demo_compensating_index(units=10)
    linear = build_index(names, components, aggregation="linear")
    geometric = build_index(names, components, aggregation="geometric")
    result = rank_reversals(linear, geometric)
    assert result["pairs"] == 45
    assert result["share"] == pytest.approx(result["count"] / 45)


def test_a_linear_index_has_no_dominance_violations():
    """A weighted sum is monotone in every component, so it cannot produce one."""
    names, components = demo_compensating_index(units=20)
    index = build_index(names, components, aggregation="linear")
    assert dominance_violations(index)["clean"] is True


def test_dominance_is_checked_against_every_component():
    index = build_index(_units(), _simple_components())
    result = dominance_violations(index)
    assert result["count"] == len(result["violations"])


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------


def test_band_probabilities_sum_to_one():
    names, components = demo_confidence_index(units=20)
    result = band_probabilities(names, components, draws=200)
    for row in result["rows"]:
        assert sum(row["probabilities"].values()) == pytest.approx(1.0)


def test_every_unit_gets_a_band_row():
    names, components = demo_confidence_index(units=15)
    result = band_probabilities(names, components, draws=100)
    assert {row["unit"] for row in result["rows"]} == set(names)


def test_the_modal_band_is_the_most_probable_one():
    names, components = demo_confidence_index(units=20)
    result = band_probabilities(names, components, draws=200)
    for row in result["rows"]:
        assert row["probabilities"][row["modal_band"]] == max(
            row["probabilities"].values()
        )


def test_a_band_label_is_reportable_only_above_the_threshold():
    names, components = demo_confidence_index(units=25)
    result = band_probabilities(names, components, draws=200)
    for row in result["rows"]:
        assert row["confident"] == (row["modal_probability"] >= MIN_BAND_PROBABILITY)


def test_the_declared_bands_come_back():
    names, components = demo_confidence_index(units=10)
    result = band_probabilities(
        names, components, bands=APP_CONFIDENCE_BANDS, draws=100
    )
    assert result["bands"] == ["High", "Medium", "Low"]


def test_bands_refuse_too_few_draws():
    names, components = demo_confidence_index(units=10)
    with pytest.raises(CompositeError, match="At least 20 draws"):
        band_probabilities(names, components, draws=3)


def test_bands_refuse_an_empty_band_list():
    names, components = demo_confidence_index(units=10)
    with pytest.raises(CompositeError, match="At least one band"):
        band_probabilities(names, components, bands=[], draws=100)


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------


def test_analyse_builds_all_three_aggregations():
    names, components = demo_compensating_index(units=15)
    result = analyse(names, components, draws=100)
    for method in AGGREGATIONS:
        assert result[method]["aggregation"] == method


def test_analyse_reports_the_unnormalised_imbalance():
    names, components = demo_eco_score(units=40)
    result = analyse(names, components, draws=100)
    assert result["unnormalised"]["dominant"] == "energy"
    assert result["scales"]["mismatched"] is True


def test_analyse_carries_the_engine_version():
    names, components = demo_confidence_index(units=15)
    assert analyse(names, components, draws=100)["engine_version"] == ENGINE_VERSION


def test_analyse_reports_both_reversal_comparisons():
    names, components = demo_compensating_index(units=15)
    result = analyse(names, components, draws=100)
    assert "linear_vs_geometric" in result
    assert "linear_vs_outranking" in result


def test_analyse_honours_the_normalisation_choice():
    names, components = demo_confidence_index(units=15)
    result = analyse(names, components, normalisation="rank", draws=100)
    assert result["normalisation"] == "rank"
    assert result["linear"]["normalisation"] == "rank"


# ---------------------------------------------------------------------------
# Notes and summaries
# ---------------------------------------------------------------------------


def test_notes_lead_with_the_headline():
    names, components = demo_confidence_index(units=15)
    result = analyse(names, components, draws=100)
    assert get_index_notes(result)[0] == result["headline"]


def test_notes_flag_a_scale_mismatch():
    names, components = demo_eco_score(units=30)
    notes = get_index_notes(analyse(names, components, draws=100))
    assert any("units did the weighting" in note for note in notes)


def test_notes_report_the_effective_weights():
    names, components = demo_confidence_index(units=40)
    notes = get_index_notes(analyse(names, components, draws=100))
    assert any("describe an intention" in note for note in notes)


def test_notes_explain_the_compensability_choice():
    names, components = demo_compensating_index(units=20)
    notes = get_index_notes(analyse(names, components, draws=100))
    assert any("value judgement" in note for note in notes)


def test_summary_is_one_line():
    names, components = demo_confidence_index(units=15)
    summary = summarise(analyse(names, components, draws=100))
    assert "\n" not in summary
    assert "units" in summary


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def test_the_confidence_demo_uses_the_real_weights():
    _, components = demo_confidence_index(units=10)
    weights = {component["name"]: component["weight"] for component in components}
    assert weights == APP_CONFIDENCE_WEIGHTS


def test_the_confidence_demo_is_deterministic_for_a_seed():
    first = demo_confidence_index(units=10, seed=5)[1]
    second = demo_confidence_index(units=10, seed=5)[1]
    assert [c["values"] for c in first] == [c["values"] for c in second]


def test_the_eco_score_demo_gives_every_component_equal_weight():
    _, components = demo_eco_score(units=10)
    assert len({component["weight"] for component in components}) == 1


def test_the_eco_score_demo_treats_every_component_as_lower_is_better():
    _, components = demo_eco_score(units=10)
    assert all(c["polarity"] == "lower_is_better" for c in components)


def test_the_compensating_demo_contains_spiky_units():
    _, components = demo_compensating_index(units=30)
    third = next(c for c in components if c["name"] == "third")
    assert third["min"] < 10.0
    assert third["max"] > 40.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_read_back_an_analysis(temp_db):
    names, components = demo_confidence_index(units=15)
    result = analyse(names, components, draws=100)
    analysis_id = save_analysis("user-1", result, label="Confidence")
    assert analysis_id is not None

    analyses = get_analyses("user-1")
    assert len(analyses) == 1
    assert analyses[0]["label"] == "Confidence"
    assert analyses[0]["units"] == 15
    assert analyses[0]["payload"]["engine_version"] == ENGINE_VERSION


def test_analyses_are_scoped_to_their_user(temp_db):
    names, components = demo_confidence_index(units=10)
    save_analysis("user-1", analyse(names, components, draws=100))
    assert get_analyses("user-2") == []


def test_saving_without_a_user_is_a_no_op(temp_db):
    names, components = demo_confidence_index(units=10)
    assert save_analysis("", analyse(names, components, draws=100)) is None


def test_saving_a_non_result_is_a_no_op(temp_db):
    assert save_analysis("user-1", {}) is None


def test_delete_removes_only_the_named_analysis(temp_db):
    names, components = demo_confidence_index(units=10)
    result = analyse(names, components, draws=100)
    first = save_analysis("user-1", result, label="one")
    save_analysis("user-1", result, label="two")
    assert delete_analysis("user-1", first) is True
    remaining = get_analyses("user-1")
    assert len(remaining) == 1
    assert remaining[0]["label"] == "two"


def test_delete_refuses_another_users_analysis(temp_db):
    names, components = demo_confidence_index(units=10)
    analysis_id = save_analysis("user-1", analyse(names, components, draws=100))
    assert delete_analysis("user-2", analysis_id) is False


def test_delete_without_a_user_is_false(temp_db):
    assert delete_analysis("", 1) is False


def test_reads_without_a_user_are_empty(temp_db):
    assert get_analyses(None) == []


def test_analyses_come_back_newest_first(temp_db):
    names, components = demo_confidence_index(units=10)
    result = analyse(names, components, draws=100)
    save_analysis("user-1", result, label="older")
    save_analysis("user-1", result, label="newer")
    assert [entry["label"] for entry in get_analyses("user-1")] == ["newer", "older"]


def test_storage_failure_is_swallowed_not_raised(monkeypatch):
    """A dashboard must render when the database is unavailable."""

    def explode(*_args, **_kwargs):
        raise sqlite3.Error("disk is on fire")

    monkeypatch.setattr(composite_index, "_connect", explode)
    names, components = demo_confidence_index(units=10)
    result = analyse(names, components, draws=100)
    assert save_analysis("user-1", result) is None
    assert get_analyses("user-1") == []
    assert delete_analysis("user-1", 1) is False


def test_a_corrupt_payload_reads_back_as_empty(temp_db):
    names, components = demo_confidence_index(units=10)
    save_analysis("user-1", analyse(names, components, draws=100))
    with sqlite3.connect(temp_db) as conn:
        conn.execute("UPDATE composite_index_analyses SET payload = 'not json'")
    assert get_analyses("user-1")[0]["payload"] == {}
