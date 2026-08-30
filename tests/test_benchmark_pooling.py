"""Tests for the partial pooling engine.

The claim under test is that an unpooled ranking sorts partly on history
length, and that shrinkage takes that out. So the load-bearing test builds a
panel where every household is drawn from the *same* distribution regardless of
how many observations it has — meaning any relationship between "few
observations" and "extreme rank" is a pure artefact — and asserts both halves:
the raw ranking puts short histories at the extremes, and the pooled one does
not.

The second claim is about the shrinkage weight itself. Testing lambda at one
value of n would prove nothing; the tests here assert the direction of the
relationship — monotone in n, monotone in the noise ratio — because that is the
mechanism, not a coincidence of one panel.

The refusals are tested as hard as the arithmetic. A panel too small to
estimate between-household variance that returns a pooled number anyway is the
failure mode this module exists to prevent.
"""

import math
import os
import sqlite3
import tempfile

import pytest

from src.carbon import benchmark_pooling
from src.carbon.benchmark_pooling import (
    DEFAULT_BADGES,
    DEFAULT_CONFIDENCE,
    ENGINE_VERSION,
    HIGH_RELIABILITY,
    LOW_RELIABILITY,
    MIN_HOUSEHOLDS,
    MIN_TREND_POINTS,
    PoolingError,
    badge_for,
    betai,
    build_household,
    delete_panel,
    demo_panel,
    get_panels,
    get_pooling_notes,
    percentile_of,
    pool_panel,
    rank_churn,
    rank_with_uncertainty,
    reliability,
    save_panel,
    summarise,
    t_cdf,
    trend_direction,
    two_sided_p,
    variance_components,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _balanced_panel(levels, per_household, spread=0.0):
    """A panel with hand-checkable moments.

    Each household sits at its own level and its observations are placed
    symmetrically around it, so the within-household mean is exactly the level
    and the within-household variance is exactly computable.
    """
    panel = []
    for index, level in enumerate(levels):
        observations = []
        for step in range(per_household):
            offset = spread * (1 if step % 2 == 0 else -1)
            observations.append(level + offset)
        panel.append(build_household("h%d" % index, observations))
    return panel


def _mixed_history_panel():
    """Same true level everywhere; history length is the only thing that varies."""
    panel = []
    for index in range(12):
        n = 1 if index < 6 else 24
        observations = [
            1000.0 + (300.0 if step % 2 else -300.0) for step in range(n)
        ]
        # Nudge the single-observation households to the extremes on luck alone.
        if n == 1:
            observations = [1000.0 + (600.0 if index % 2 else -600.0)]
        panel.append(build_household("h%02d" % index, observations))
    return panel


@pytest.fixture
def temp_db(monkeypatch):
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    monkeypatch.setattr(benchmark_pooling, "DB_NAME", path)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# build_household
# ---------------------------------------------------------------------------


def test_build_household_computes_mean_and_count():
    household = build_household("a", [100.0, 200.0, 300.0])
    assert household["n"] == 3
    assert household["mean"] == pytest.approx(200.0)
    assert household["own_variance"] == pytest.approx(10000.0)


def test_build_household_accepts_a_single_observation():
    """The case the module exists for. Refusing it pushes the problem upstream."""
    household = build_household("a", [500.0])
    assert household["n"] == 1
    assert household["own_variance"] is None


def test_build_household_rejects_empty_history():
    with pytest.raises(PoolingError, match="no observations"):
        build_household("a", [])


def test_build_household_rejects_non_numeric():
    with pytest.raises(PoolingError, match="non-numeric"):
        build_household("a", [100.0, "banana"])


def test_build_household_rejects_nan():
    with pytest.raises(PoolingError, match="non-numeric"):
        build_household("a", [100.0, float("nan")])


def test_build_household_rejects_infinity():
    with pytest.raises(PoolingError, match="non-numeric"):
        build_household("a", [float("inf")])


def test_build_household_defaults_label_to_identifier():
    assert build_household("hh7", [1.0])["label"] == "hh7"


# ---------------------------------------------------------------------------
# Panel validation — the refusals
# ---------------------------------------------------------------------------


def test_panel_below_minimum_households_is_refused():
    panel = _balanced_panel([1000.0, 2000.0], 4, spread=50.0)
    with pytest.raises(PoolingError, match="at least %d households" % MIN_HOUSEHOLDS):
        variance_components(panel)


def test_panel_of_all_singletons_is_refused():
    """No repeat measurements anywhere means no noise model to shrink against."""
    panel = [build_household("h%d" % i, [1000.0 + i * 100.0]) for i in range(8)]
    with pytest.raises(PoolingError, match="within-household noise cannot"):
        variance_components(panel)


def test_duplicate_household_ids_are_refused():
    panel = _balanced_panel([1000.0, 2000.0, 3000.0, 4000.0], 4, spread=50.0)
    panel[2]["id"] = panel[0]["id"]
    with pytest.raises(PoolingError, match="appears twice"):
        variance_components(panel)


def test_raw_mappings_are_refused():
    panel = _balanced_panel([1000.0, 2000.0, 3000.0, 4000.0], 4, spread=50.0)
    panel[1] = {"id": "x", "mean": 1.0, "n": 1}
    with pytest.raises(PoolingError, match="build_household"):
        variance_components(panel)


def test_unsupported_confidence_is_refused():
    panel = _balanced_panel([1000.0, 2000.0, 3000.0, 4000.0], 6, spread=100.0)
    with pytest.raises(PoolingError, match="Confidence must be"):
        pool_panel(panel, confidence=0.77)


# ---------------------------------------------------------------------------
# Variance components
# ---------------------------------------------------------------------------


def test_within_variance_recovers_a_known_spread():
    """Every household alternates +/-100 about its level, so sigma^2 is exact."""
    panel = _balanced_panel([1000.0, 2000.0, 3000.0, 4000.0], 8, spread=100.0)
    components = variance_components(panel)
    # 8 points at +/-100 about the mean: SS = 8 * 100^2 per household,
    # df = 8 - 1 = 7 per household, so sigma^2 = 80000/7 per household.
    assert components["within_variance"] == pytest.approx(80000.0 / 7.0)


def test_grand_mean_is_observation_weighted_not_household_weighted():
    """A household with 20 readings pulls the grand mean more than one with 2."""
    panel = [
        build_household("big", [1000.0] * 20),
        build_household("small_a", [4000.0, 4000.0]),
        build_household("small_b", [4000.0, 4000.0]),
        build_household("small_c", [4000.0, 4000.0]),
    ]
    components = variance_components(panel)
    household_weighted = (1000.0 + 4000.0 * 3) / 4
    observation_weighted = (1000.0 * 20 + 4000.0 * 6) / 26
    assert components["grand_mean"] == pytest.approx(observation_weighted)
    assert components["grand_mean"] != pytest.approx(household_weighted)


def test_identical_households_give_zero_between_variance():
    panel = _balanced_panel([1000.0] * 6, 6, spread=200.0)
    components = variance_components(panel)
    assert components["between_variance"] == pytest.approx(0.0, abs=1e-6)
    assert components["intraclass_correlation"] == pytest.approx(0.0, abs=1e-6)


def test_negative_between_variance_is_reported_not_hidden():
    """Households indistinguishable under the noise is a finding, not a zero."""
    panel = _balanced_panel([1000.0] * 6, 6, spread=400.0)
    components = variance_components(panel)
    assert components["negative_component"] is True
    assert components["raw_between_variance"] < 0
    assert components["between_variance"] == 0.0
    assert "not measurably different" in components["note"]


def test_well_separated_households_give_high_icc():
    panel = _balanced_panel([1000.0, 3000.0, 5000.0, 7000.0, 9000.0], 8, spread=50.0)
    components = variance_components(panel)
    assert components["intraclass_correlation"] > 0.99
    assert components["negative_component"] is False


def test_between_sd_is_the_square_root_of_between_variance():
    panel = _balanced_panel([1000.0, 3000.0, 5000.0, 7000.0], 6, spread=100.0)
    components = variance_components(panel)
    assert components["between_sd"] == pytest.approx(
        math.sqrt(components["between_variance"])
    )


def test_component_counts_are_reported():
    panel = _balanced_panel([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], 4, spread=80.0)
    components = variance_components(panel)
    assert components["households"] == 5
    assert components["observations"] == 20
    assert components["within_df"] == 15
    assert components["between_df"] == 4


# ---------------------------------------------------------------------------
# Reliability — the mechanism, not one number
# ---------------------------------------------------------------------------


def test_reliability_increases_with_observations():
    weights = [reliability(n, 1000.0, 1000.0) for n in (1, 2, 5, 10, 40)]
    assert weights == sorted(weights)
    assert weights[0] < weights[-1]


def test_reliability_approaches_one_with_a_long_history():
    assert reliability(10000, 1000.0, 1000.0) > 0.99


def test_reliability_at_one_observation_is_the_variance_ratio():
    assert reliability(1, 400.0, 1600.0) == pytest.approx(400.0 / 2000.0)


def test_reliability_decreases_as_noise_grows():
    weights = [reliability(4, 1000.0, noise) for noise in (100.0, 1000.0, 10000.0)]
    assert weights == sorted(weights, reverse=True)


def test_reliability_is_zero_without_between_household_variance():
    """Nothing to distinguish households means the group mean is the estimate."""
    assert reliability(50, 0.0, 1000.0) == 0.0


def test_reliability_is_one_without_noise():
    assert reliability(1, 1000.0, 0.0) == 1.0


def test_reliability_rejects_zero_observations():
    with pytest.raises(PoolingError, match="at least one observation"):
        reliability(0, 1000.0, 1000.0)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def test_pooled_estimate_lies_between_own_mean_and_group_mean():
    result = pool_panel(demo_panel(households=30))
    grand = result["components"]["grand_mean"]
    for entry in result["estimates"]:
        low, high = sorted((entry["raw_mean"], grand))
        assert low - 1e-6 <= entry["pooled_mean"] <= high + 1e-6


def test_pooling_is_the_stated_convex_combination():
    result = pool_panel(demo_panel(households=20))
    grand = result["components"]["grand_mean"]
    for entry in result["estimates"]:
        expected = (
            entry["reliability"] * entry["raw_mean"]
            + (1.0 - entry["reliability"]) * grand
        )
        assert entry["pooled_mean"] == pytest.approx(expected)


def test_short_histories_are_shrunk_more_than_long_ones():
    result = pool_panel(demo_panel(households=60))
    short = [e for e in result["estimates"] if e["n"] <= 2]
    long = [e for e in result["estimates"] if e["n"] >= 15]
    assert short and long
    mean_short = sum(e["shrinkage_share"] for e in short) / len(short)
    mean_long = sum(e["shrinkage_share"] for e in long) / len(long)
    assert mean_short > mean_long


def test_unpooled_extremes_are_short_histories_and_pooled_ones_are_not():
    """The load-bearing test.

    Every household is drawn from the same distribution regardless of how many
    observations it has, so a relationship between history length and rank is
    entirely an artefact of small samples. It should be present in the raw
    ranking and gone from the pooled one.
    """
    panel = demo_panel(households=120, seed=7)
    result = pool_panel(panel)
    by_id = {entry["id"]: entry for entry in result["estimates"]}

    raw_extremes = result["raw_order"][:10] + result["raw_order"][-10:]
    pooled_extremes = result["pooled_order"][:10] + result["pooled_order"][-10:]

    raw_history = sum(by_id[i]["n"] for i in raw_extremes) / len(raw_extremes)
    pooled_history = sum(by_id[i]["n"] for i in pooled_extremes) / len(pooled_extremes)

    assert raw_history < pooled_history, (
        "The unpooled extremes should be shorter histories than the pooled ones; "
        "raw mean n=%.1f, pooled mean n=%.1f" % (raw_history, pooled_history)
    )


def test_pooling_changes_the_ranking():
    result = pool_panel(demo_panel(households=80, seed=11))
    assert result["rank_churn"]["changed"] > 0
    assert result["raw_order"] != result["pooled_order"]


def test_shrinkage_share_is_zero_for_a_household_at_the_group_mean():
    panel = _balanced_panel([1000.0, 1000.0, 3000.0, 5000.0], 6, spread=100.0)
    result = pool_panel(panel)
    grand = result["components"]["grand_mean"]
    closest = min(result["estimates"], key=lambda e: abs(e["raw_mean"] - grand))
    assert abs(closest["shrinkage"]) < abs(closest["raw_mean"] - grand) + 1e-6


def test_reliability_bands_are_assigned_by_the_documented_cuts():
    result = pool_panel(demo_panel(households=60))
    for entry in result["estimates"]:
        if entry["reliability"] >= HIGH_RELIABILITY:
            assert entry["reliability_band"] == "own_data"
        elif entry["reliability"] >= LOW_RELIABILITY:
            assert entry["reliability_band"] == "mixed"
        else:
            assert entry["reliability_band"] == "group"


def test_mostly_group_flag_matches_the_low_reliability_cut():
    result = pool_panel(demo_panel(households=60))
    for entry in result["estimates"]:
        assert entry["mostly_group"] == (entry["reliability"] < LOW_RELIABILITY)


def test_intervals_widen_as_confidence_rises():
    panel = demo_panel(households=30)
    narrow = pool_panel(panel, confidence=0.80)["estimates"][0]
    wide = pool_panel(panel, confidence=0.99)["estimates"][0]
    assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])


def test_pooled_estimate_sits_inside_its_own_interval():
    result = pool_panel(demo_panel(households=40))
    for entry in result["estimates"]:
        assert entry["lower"] <= entry["pooled_mean"] <= entry["upper"]


def test_estimates_come_back_sorted_by_pooled_mean():
    result = pool_panel(demo_panel(households=40))
    values = [entry["pooled_mean"] for entry in result["estimates"]]
    assert values == sorted(values)


def test_result_carries_the_engine_version():
    assert pool_panel(demo_panel(households=20))["engine_version"] == ENGINE_VERSION


def test_default_confidence_is_used_when_unspecified():
    assert pool_panel(demo_panel(households=20))["confidence"] == DEFAULT_CONFIDENCE


# ---------------------------------------------------------------------------
# Rank churn
# ---------------------------------------------------------------------------


def test_rank_churn_is_zero_for_an_unchanged_order():
    churn = rank_churn(["a", "b", "c"], ["a", "b", "c"])
    assert churn["changed"] == 0
    assert churn["largest_move"] == 0
    assert churn["share"] == 0.0


def test_rank_churn_counts_every_moved_position():
    churn = rank_churn(["a", "b", "c"], ["c", "b", "a"])
    assert churn["changed"] == 2
    assert churn["largest_move"] == 2
    assert churn["share"] == pytest.approx(2.0 / 3.0)


def test_rank_churn_handles_an_empty_order():
    churn = rank_churn([], [])
    assert churn["changed"] == 0
    assert churn["share"] == 0.0


# ---------------------------------------------------------------------------
# Ranking with uncertainty
# ---------------------------------------------------------------------------


def _estimate(identifier, mean, half_width):
    return {
        "id": identifier,
        "label": identifier,
        "pooled_mean": mean,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def test_separated_intervals_produce_one_band_each():
    bands = rank_with_uncertainty(
        [_estimate("a", 100.0, 5.0), _estimate("b", 200.0, 5.0)]
    )
    assert len(bands) == 2
    assert all(band["separated"] for band in bands)


def test_overlapping_intervals_are_reported_as_a_tie():
    bands = rank_with_uncertainty(
        [_estimate("a", 100.0, 50.0), _estimate("b", 120.0, 50.0)]
    )
    assert len(bands) == 1
    assert bands[0]["separated"] is False
    assert set(bands[0]["ids"]) == {"a", "b"}


def test_bands_do_not_chain_transitively():
    """A overlaps B and B overlaps C, but A and C are cleanly separated.

    A running maximum would put all three in one band, and on a wide panel that
    collapses every household into a single band that says nothing. Membership
    is against the household that opened the band instead.
    """
    bands = rank_with_uncertainty(
        [
            _estimate("a", 100.0, 15.0),
            _estimate("b", 125.0, 15.0),
            _estimate("c", 150.0, 15.0),
        ]
    )
    assert len(bands) > 1
    for band in bands:
        assert not ("a" in band["ids"] and "c" in band["ids"])


def test_band_reports_its_range():
    bands = rank_with_uncertainty(
        [_estimate("a", 100.0, 50.0), _estimate("b", 130.0, 50.0)]
    )
    assert bands[0]["lowest"] == pytest.approx(100.0)
    assert bands[0]["highest"] == pytest.approx(130.0)


def test_band_numbers_are_sequential():
    bands = rank_with_uncertainty(
        [_estimate("a", 100.0, 1.0), _estimate("b", 200.0, 1.0), _estimate("c", 300.0, 1.0)]
    )
    assert [band["band"] for band in bands] == [1, 2, 3]


def test_a_wide_panel_produces_more_than_one_band():
    result = pool_panel(demo_panel(households=60, seed=3))
    assert len(result["ranking"]) > 1


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def test_percentile_of_the_lowest_household_is_zero():
    result = pool_panel(demo_panel(households=40))
    lowest = result["estimates"][0]["id"]
    assert percentile_of(lowest, result)["pooled_percentile"] == pytest.approx(0.0)


def test_percentile_interval_brackets_the_point_estimate():
    result = pool_panel(demo_panel(households=40))
    for entry in result["estimates"][:10]:
        place = percentile_of(entry["id"], result)
        assert place["percentile_low"] <= place["pooled_percentile"] + 1e-9
        assert place["pooled_percentile"] <= place["percentile_high"] + 1e-9


def test_percentile_carries_reliability_and_history_length():
    result = pool_panel(demo_panel(households=30))
    entry = result["estimates"][5]
    place = percentile_of(entry["id"], result)
    assert place["n"] == entry["n"]
    assert place["reliability"] == pytest.approx(entry["reliability"])


def test_percentile_of_an_unknown_household_is_refused():
    result = pool_panel(demo_panel(households=20))
    with pytest.raises(PoolingError, match="not in this panel"):
        percentile_of("nobody", result)


def test_percentile_reports_the_unpooled_figure_alongside():
    result = pool_panel(demo_panel(households=50, seed=5))
    gaps = [
        abs(
            percentile_of(entry["id"], result)["pooled_percentile"]
            - percentile_of(entry["id"], result)["raw_percentile"]
        )
        for entry in result["estimates"]
    ]
    assert max(gaps) > 0.0


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def test_trend_below_the_minimum_returns_insufficient_data_not_stable():
    """'Stable' and 'we have no idea' are different claims and must not collide."""
    household = build_household("a", [1000.0, 900.0])
    trend = trend_direction(household, 10000.0)
    assert trend["direction"] == "insufficient_data"
    assert trend["slope"] is None
    assert "not the same as 'stable'" in trend["headline"]


def test_minimum_trend_points_is_respected_exactly():
    household = build_household("a", [1000.0] * MIN_TREND_POINTS)
    assert trend_direction(household, 100.0)["direction"] != "insufficient_data"


def test_a_clear_decline_against_low_noise_is_improving():
    household = build_household("a", [3000.0, 2800.0, 2600.0, 2400.0, 2200.0])
    trend = trend_direction(household, 100.0)
    assert trend["direction"] == "improving"
    assert trend["slope"] < 0


def test_a_clear_rise_against_low_noise_is_worsening():
    household = build_household("a", [2200.0, 2400.0, 2600.0, 2800.0, 3000.0])
    assert trend_direction(household, 100.0)["direction"] == "worsening"


def test_the_same_slope_is_not_a_trend_against_high_noise():
    """The fixed 50 kg rule fires here. A noise-scaled test does not."""
    observations = [3000.0, 2940.0, 2880.0, 2820.0, 2760.0]
    household = build_household("a", observations)
    assert trend_direction(household, 100.0)["direction"] == "improving"
    assert trend_direction(household, 4000000.0)["direction"] == "stable"


def test_stable_reports_what_could_not_have_been_detected():
    household = build_household("a", [3000.0, 2990.0, 2995.0, 2985.0, 2992.0])
    trend = trend_direction(household, 250000.0)
    assert trend["direction"] == "stable"
    assert trend["minimum_detectable_slope"] > 0
    assert "could have been detected" in trend["headline"]


def test_trend_standard_error_shrinks_with_more_points():
    short = trend_direction(build_household("a", [100.0, 110.0, 120.0]), 400.0)
    long = trend_direction(
        build_household("b", [100.0 + 10.0 * i for i in range(20)]), 400.0
    )
    assert long["standard_error"] < short["standard_error"]


def test_trend_with_no_panel_noise_reports_the_exact_slope():
    household = build_household("a", [300.0, 200.0, 100.0])
    trend = trend_direction(household, 0.0)
    assert trend["direction"] == "improving"
    assert trend["p_value"] == 0.0


def test_a_flat_history_with_no_noise_is_stable():
    household = build_household("a", [100.0, 100.0, 100.0])
    assert trend_direction(household, 0.0)["direction"] == "stable"


def test_trend_alpha_is_honoured():
    observations = [3000.0, 2900.0, 2850.0, 2760.0, 2700.0]
    household = build_household("a", observations)
    strict = trend_direction(household, 90000.0, alpha=0.001)
    loose = trend_direction(household, 90000.0, alpha=0.5)
    assert strict["direction"] == "stable"
    assert loose["direction"] == "improving"


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------


def test_a_badge_is_earned_when_the_whole_interval_clears():
    estimate = {"pooled_mean": 1000.0, "lower": 900.0, "upper": 1100.0}
    verdict = badge_for(estimate)
    assert verdict["badge"] == "Platinum"
    assert verdict["reason"] == "interval_clears"


def test_the_awarded_badge_is_the_one_the_interval_supports_not_the_point():
    """Point estimate says Platinum; only Gold survives the interval."""
    estimate = {"pooled_mean": 1400.0, "lower": 800.0, "upper": 2400.0}
    verdict = badge_for(estimate)
    assert verdict["provisional"] == "Platinum"
    assert verdict["earned_outright"] == "Gold"
    assert verdict["badge"] == "Gold"


def test_a_point_estimate_alone_awards_nothing_outright():
    """Nothing is earned when the interval clears no threshold at all."""
    estimate = {"pooled_mean": 5500.0, "lower": 4000.0, "upper": 9000.0}
    verdict = badge_for(estimate)
    assert verdict["provisional"] == "Bronze"
    assert verdict["earned_outright"] is None
    assert verdict["reason"] == "point_estimate_only"


def test_an_existing_badge_is_retained_rather_than_revoked_on_noise():
    """Point estimate still inside Platinum, interval only clears Gold."""
    estimate = {"pooled_mean": 1400.0, "lower": 800.0, "upper": 2400.0}
    assert badge_for(estimate)["badge"] == "Gold"
    verdict = badge_for(estimate, current_badge="Platinum")
    assert verdict["badge"] == "Platinum"
    assert verdict["held_on_hysteresis"] is True
    assert verdict["reason"] == "retained"


def test_a_held_badge_is_upgraded_when_the_interval_supports_better():
    estimate = {"pooled_mean": 1000.0, "lower": 900.0, "upper": 1100.0}
    verdict = badge_for(estimate, current_badge="Gold")
    assert verdict["badge"] == "Platinum"
    assert verdict["reason"] == "interval_clears"


def test_a_genuine_fall_downgrades_the_badge():
    """The point estimate has left Platinum entirely, so hysteresis does not apply."""
    estimate = {"pooled_mean": 3800.0, "lower": 3600.0, "upper": 4200.0}
    verdict = badge_for(estimate, current_badge="Platinum")
    assert verdict["badge"] != "Platinum"
    assert verdict["reason"] == "downgraded"
    assert verdict["held_on_hysteresis"] is False
    assert "down from Platinum" in verdict["headline"]


def test_no_badge_above_every_threshold():
    estimate = {"pooled_mean": 20000.0, "lower": 19000.0, "upper": 21000.0}
    verdict = badge_for(estimate)
    assert verdict["badge"] is None
    assert verdict["headline"] == "No badge."


def test_hysteresis_does_not_invent_a_badge_for_a_household_that_has_none():
    estimate = {"pooled_mean": 20000.0, "lower": 19000.0, "upper": 21000.0}
    assert badge_for(estimate, current_badge="Gold")["badge"] is None


def test_badges_are_ordered_by_ceiling_regardless_of_input_order():
    estimate = {"pooled_mean": 1000.0, "lower": 900.0, "upper": 1100.0}
    shuffled = tuple(reversed(DEFAULT_BADGES))
    assert badge_for(estimate, thresholds=shuffled)["badge"] == "Platinum"


def test_a_narrower_interval_earns_a_better_badge_at_the_same_estimate():
    """More data, not better luck, is what buys the tier."""
    wide = {"pooled_mean": 2400.0, "lower": 1500.0, "upper": 3300.0}
    narrow = {"pooled_mean": 2400.0, "lower": 2350.0, "upper": 2450.0}
    assert badge_for(wide)["earned_outright"] == "Silver"
    assert badge_for(narrow)["earned_outright"] == "Gold"


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def test_t_cdf_is_a_half_at_zero():
    assert t_cdf(0.0, 10) == pytest.approx(0.5)


def test_t_cdf_is_monotone():
    values = [t_cdf(x, 8) for x in (-3.0, -1.0, 0.0, 1.0, 3.0)]
    assert values == sorted(values)


def test_two_sided_p_of_zero_is_one():
    assert two_sided_p(0.0, 10) == pytest.approx(1.0)


def test_two_sided_p_falls_as_the_statistic_grows():
    assert two_sided_p(4.0, 10) < two_sided_p(1.0, 10)


def test_two_sided_p_is_symmetric():
    assert two_sided_p(2.5, 12) == pytest.approx(two_sided_p(-2.5, 12))


def test_betai_endpoints():
    assert betai(2.0, 3.0, 0.0) == pytest.approx(0.0)
    assert betai(2.0, 3.0, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Notes and summaries
# ---------------------------------------------------------------------------


def test_notes_lead_with_the_headline():
    result = pool_panel(demo_panel(households=30))
    notes = get_pooling_notes(result)
    assert notes[0] == result["headline"]


def test_notes_report_the_rank_churn():
    result = pool_panel(demo_panel(households=60, seed=13))
    assert any("unpooled leaderboard" in note for note in get_pooling_notes(result))


def test_notes_flag_a_negative_variance_component():
    panel = _balanced_panel([1000.0] * 6, 6, spread=400.0)
    result = pool_panel(panel)
    assert any("not a bug" in note for note in get_pooling_notes(result))


def test_notes_flag_low_reliability_households():
    panel = _mixed_history_panel()
    result = pool_panel(panel)
    thin = [e for e in result["estimates"] if e["mostly_group"]]
    if thin:
        assert any("statement about the group" in n for n in get_pooling_notes(result))


def test_summary_is_one_line():
    summary = summarise(pool_panel(demo_panel(households=25)))
    assert "\n" not in summary
    assert "ICC" in summary


# ---------------------------------------------------------------------------
# Demo panel
# ---------------------------------------------------------------------------


def test_demo_panel_is_deterministic_for_a_seed():
    first = demo_panel(households=20, seed=99)
    second = demo_panel(households=20, seed=99)
    assert [h["observations"] for h in first] == [h["observations"] for h in second]


def test_demo_panel_has_uneven_history_lengths():
    lengths = {household["n"] for household in demo_panel(households=60)}
    assert len(lengths) > 3


def test_demo_panel_respects_its_history_bounds():
    panel = demo_panel(households=40, min_history=2, max_history=9)
    assert all(2 <= household["n"] <= 9 for household in panel)


def test_demo_panel_honours_the_household_count():
    assert len(demo_panel(households=17)) == 17


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_read_back_a_panel(temp_db):
    result = pool_panel(demo_panel(households=20))
    panel_id = save_panel("user-1", result, label="Block A")
    assert panel_id is not None

    panels = get_panels("user-1")
    assert len(panels) == 1
    assert panels[0]["label"] == "Block A"
    assert panels[0]["households"] == 20
    assert panels[0]["payload"]["engine_version"] == ENGINE_VERSION


def test_panels_are_scoped_to_their_user(temp_db):
    result = pool_panel(demo_panel(households=20))
    save_panel("user-1", result)
    assert get_panels("user-2") == []


def test_saving_without_a_user_is_a_no_op(temp_db):
    assert save_panel("", pool_panel(demo_panel(households=20))) is None


def test_saving_an_empty_result_is_a_no_op(temp_db):
    assert save_panel("user-1", {}) is None


def test_delete_removes_only_the_named_panel(temp_db):
    result = pool_panel(demo_panel(households=20))
    first = save_panel("user-1", result, label="one")
    save_panel("user-1", result, label="two")
    assert delete_panel("user-1", first) is True
    remaining = get_panels("user-1")
    assert len(remaining) == 1
    assert remaining[0]["label"] == "two"


def test_delete_refuses_another_users_panel(temp_db):
    panel_id = save_panel("user-1", pool_panel(demo_panel(households=20)))
    assert delete_panel("user-2", panel_id) is False
    assert len(get_panels("user-1")) == 1


def test_delete_without_a_user_is_false(temp_db):
    assert delete_panel("", 1) is False


def test_reads_without_a_user_are_empty(temp_db):
    assert get_panels(None) == []


def test_panels_come_back_newest_first(temp_db):
    result = pool_panel(demo_panel(households=20))
    save_panel("user-1", result, label="older")
    save_panel("user-1", result, label="newer")
    assert [entry["label"] for entry in get_panels("user-1")] == ["newer", "older"]


def test_storage_failure_is_swallowed_not_raised(monkeypatch):
    """A dashboard must render when the database is unavailable."""

    def explode(*_args, **_kwargs):
        raise sqlite3.Error("disk is on fire")

    monkeypatch.setattr(benchmark_pooling, "_connect", explode)
    assert save_panel("user-1", pool_panel(demo_panel(households=20))) is None
    assert get_panels("user-1") == []
    assert delete_panel("user-1", 1) is False


def test_a_corrupt_payload_reads_back_as_empty(temp_db):
    save_panel("user-1", pool_panel(demo_panel(households=20)))
    with sqlite3.connect(temp_db) as conn:
        conn.execute("UPDATE benchmark_pooled_panels SET payload = 'not json'")
    assert get_panels("user-1")[0]["payload"] == {}
