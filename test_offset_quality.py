"""Tests for the Carbon Offset Quality Auditor."""
import os
import tempfile

import pytest

os.environ["ECO_BUDDY_DB"] = ":memory:"

import offset_quality
from offset_quality import (
    AVOIDANCE,
    DEFAULT_REGISTRY,
    DURABILITY_REFERENCE_YEARS,
    GRADE_BANDS,
    MAX_VINTAGE_PENALTY,
    MIN_DURABILITY_FACTOR,
    PROJECT_TYPES,
    REGISTRIES,
    REMOVAL,
    SCORE_WEIGHTS,
    TYPICAL_BUFFER_SHARE,
    VINTAGE_PENALTY_PER_YEAR,
    VINTAGE_STALE_AFTER_YEARS,
    assess_credit,
    default_buffer_share,
    delete_holding,
    durability_discount,
    effective_tonnes,
    get_credit_warnings,
    get_holdings,
    get_offset_advice,
    get_project_type,
    grade_for_score,
    list_project_types,
    list_registries,
    mitigation_hierarchy,
    portfolio_summary,
    price_credibility,
    quality_score,
    recommend_portfolio,
    registry_weight,
    save_holding,
    vintage_penalty,
)

CURRENT_YEAR = 2026


@pytest.fixture(autouse=True)
def temp_db():
    """Point the module at a throwaway SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    original = offset_quality.DB_NAME
    offset_quality.DB_NAME = db_path
    yield db_path
    offset_quality.DB_NAME = original
    try:
        os.unlink(db_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def test_every_project_type_is_fully_specified():
    for name, project in PROJECT_TYPES.items():
        assert project["kind"] in (REMOVAL, AVOIDANCE), name
        assert 0 <= project["additionality"] <= 1, name
        assert 0 <= project["leakage"] <= 1, name
        assert 0 <= project["measurement"] <= 1, name
        assert 0 <= project["co_benefits"] <= 1, name
        assert project["permanence_years"] > 0, name
        assert 0 < project["floor_price"] <= project["typical_price"], name
        assert project["note"], name


def test_projects_are_listed_most_durable_first():
    durations = [project["permanence_years"] for project in list_project_types()]
    assert durations == sorted(durations, reverse=True)


def test_projects_can_be_filtered_by_kind():
    removals = list_project_types(REMOVAL)
    assert removals
    assert all(project["kind"] == REMOVAL for project in removals)
    assert len(removals) + len(list_project_types(AVOIDANCE)) == len(PROJECT_TYPES)


def test_unknown_project_type_returns_none():
    assert get_project_type("Planting a windowsill herb") is None


def test_registries_are_listed_most_scrutinised_first():
    weights = [registry["weight"] for registry in list_registries()]
    assert weights == sorted(weights, reverse=True)
    assert list_registries()[-1]["name"] == DEFAULT_REGISTRY


def test_an_unknown_registry_is_treated_as_unverified():
    assert registry_weight("My Own Certificate") == REGISTRIES[DEFAULT_REGISTRY]["weight"]
    assert registry_weight(None) == REGISTRIES[DEFAULT_REGISTRY]["weight"]


def test_score_weights_sum_to_one():
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)


def test_grade_bands_are_ordered_and_cover_zero():
    thresholds = [band[0] for band in GRADE_BANDS]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[-1] == 0


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------

def test_durability_rises_with_storage_time():
    values = [durability_discount(years) for years in (1, 10, 40, 100, 500, 1000)]
    assert values == sorted(values)


def test_geological_storage_is_not_discounted():
    assert durability_discount(DURABILITY_REFERENCE_YEARS) == 1.0
    assert durability_discount(10000) == 1.0


def test_short_lived_storage_is_heavily_discounted():
    assert durability_discount(25) < 0.55
    assert durability_discount(0) == 0.0
    assert durability_discount(1) >= MIN_DURABILITY_FACTOR


def test_durability_is_always_a_fraction():
    for years in (0, 1, 3, 30, 60, 300, 900, 10**6):
        assert 0.0 <= durability_discount(years) <= 1.0


def test_durability_survives_junk_input():
    assert durability_discount(None) == 0.0
    assert durability_discount("forever") == 0.0
    assert durability_discount(-50) == 0.0


# ---------------------------------------------------------------------------
# Vintage
# ---------------------------------------------------------------------------

def test_recent_vintages_are_not_penalised():
    assert vintage_penalty(CURRENT_YEAR, CURRENT_YEAR) == 0.0
    assert vintage_penalty(CURRENT_YEAR - VINTAGE_STALE_AFTER_YEARS, CURRENT_YEAR) == 0.0


def test_old_vintages_are_penalised_progressively():
    older = vintage_penalty(CURRENT_YEAR - 12, CURRENT_YEAR)
    oldest = vintage_penalty(CURRENT_YEAR - 20, CURRENT_YEAR)
    assert 0 < older < oldest
    assert older == pytest.approx(
        (12 - VINTAGE_STALE_AFTER_YEARS) * VINTAGE_PENALTY_PER_YEAR, abs=1e-6
    )


def test_the_vintage_penalty_is_capped():
    assert vintage_penalty(1990, CURRENT_YEAR) == MAX_VINTAGE_PENALTY


def test_vintage_penalty_survives_junk_input():
    assert vintage_penalty(None, CURRENT_YEAR) == 0.0
    assert vintage_penalty("last year", CURRENT_YEAR) == 0.0


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def test_a_credit_below_its_floor_price_is_not_credible():
    result = price_credibility(3.0, "Direct air capture with storage")
    assert result["credible"] is False
    assert result["suspiciously_cheap"] is True


def test_a_fairly_priced_credit_is_credible():
    result = price_credibility(600.0, "Direct air capture with storage")
    assert result["credible"] is True
    assert result["suspiciously_cheap"] is False
    assert result["ratio"] > 1


def test_price_credibility_needs_a_known_project():
    assert price_credibility(50, "Magic beans")["credible"] is False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_scores_are_bounded_for_every_combination():
    for project in PROJECT_TYPES:
        for registry in REGISTRIES:
            scoring = quality_score(project, registry, 2010, CURRENT_YEAR)
            assert 0.0 <= scoring["score"] <= 100.0, (project, registry)


def test_direct_air_capture_outscores_a_cheap_avoidance_credit():
    dac = quality_score("Direct air capture with storage", "Puro.earth")["score"]
    renewables = quality_score("Grid-connected wind or solar", "Verra (VCS)")["score"]
    assert dac > renewables + 20


def test_the_same_project_scores_worse_on_an_unverified_registry():
    verified = quality_score("Afforestation / reforestation", "Gold Standard")["score"]
    unverified = quality_score("Afforestation / reforestation", DEFAULT_REGISTRY)["score"]
    assert unverified < verified


def test_an_old_vintage_scores_lower_than_a_fresh_one():
    fresh = quality_score("Biochar", "Puro.earth", CURRENT_YEAR, CURRENT_YEAR)["score"]
    stale = quality_score("Biochar", "Puro.earth", 2005, CURRENT_YEAR)["score"]
    assert stale < fresh


def test_a_bigger_buffer_pool_helps_a_reversible_project():
    thin = quality_score("Afforestation / reforestation", "Verra (VCS)", buffer_share=0.0)
    thick = quality_score("Afforestation / reforestation", "Verra (VCS)", buffer_share=0.3)
    assert thick["score"] > thin["score"]


def test_unknown_projects_cannot_be_scored():
    assert quality_score("Thoughts and prayers") is None


def test_grades_follow_the_score():
    assert grade_for_score(95)["grade"] == "A"
    assert grade_for_score(70)["grade"] == "B"
    assert grade_for_score(55)["grade"] == "C"
    assert grade_for_score(40)["grade"] == "D"
    assert grade_for_score(10)["grade"] == "F"
    assert grade_for_score(-5)["grade"] == "F"
    assert grade_for_score(None)["grade"] == "F"


# ---------------------------------------------------------------------------
# Effective tonnes
# ---------------------------------------------------------------------------

def test_effective_tonnes_never_exceed_nominal():
    for project in PROJECT_TYPES:
        for registry in REGISTRIES:
            result = effective_tonnes(10, project, registry)
            assert result["effective_tonnes"] <= result["nominal_tonnes"], (project, registry)
            assert result["shortfall_tonnes"] >= 0


def test_every_delivery_factor_is_a_fraction():
    result = effective_tonnes(10, "Clean cookstoves", "Gold Standard")
    for name, value in result["factors"].items():
        assert 0.0 <= value <= 1.0, name


def test_a_high_integrity_removal_delivers_most_of_what_it_promises():
    result = effective_tonnes(10, "Direct air capture with storage", "Puro.earth")
    assert result["delivery_ratio"] > 0.7


def test_geological_storage_is_not_charged_a_reversal_buffer():
    # A buffer pool insures against fire, felling or a change of practice.
    # Carbon mineralised in rock carries none of those risks, so charging it
    # a buffer would penalise the most durable option for a risk it lacks.
    assert default_buffer_share("Direct air capture with storage") == 0.0
    assert default_buffer_share("Enhanced rock weathering") == 0.0
    assert default_buffer_share("Afforestation / reforestation") == TYPICAL_BUFFER_SHARE
    assert default_buffer_share("Soil carbon sequestration") == TYPICAL_BUFFER_SHARE


def test_an_unknown_project_falls_back_to_the_typical_buffer():
    assert default_buffer_share("Handshake agreement") == TYPICAL_BUFFER_SHARE


def test_a_weak_avoidance_credit_delivers_a_fraction():
    result = effective_tonnes(10, "Avoided deforestation (REDD+)", DEFAULT_REGISTRY)
    assert result["delivery_ratio"] < 0.1


def test_the_buffer_share_is_not_delivered_to_the_buyer():
    none_held = effective_tonnes(10, "Biochar", "Puro.earth", buffer_share=0.0)
    held_back = effective_tonnes(10, "Biochar", "Puro.earth", buffer_share=0.3)
    assert held_back["effective_tonnes"] < none_held["effective_tonnes"]


def test_zero_tonnes_yields_zero():
    result = effective_tonnes(0, "Biochar", "Puro.earth")
    assert result["effective_tonnes"] == 0.0
    assert result["delivery_ratio"] == 0.0


def test_effective_tonnes_needs_a_known_project():
    assert effective_tonnes(5, "Vibes") is None


# ---------------------------------------------------------------------------
# Whole-credit assessment
# ---------------------------------------------------------------------------

def test_assessment_reports_the_real_cost_per_tonne():
    assessment = assess_credit(
        "Avoided deforestation (REDD+)", 10, 12.0, "Verra (VCS)", 2020, CURRENT_YEAR
    )
    assert assessment["total_spend"] == 120.0
    assert assessment["cost_per_effective_tonne"] > assessment["price_per_tonne"]


def test_a_good_credit_grades_well_and_warns_little():
    assessment = assess_credit(
        "Direct air capture with storage", 1, 600.0, "Puro.earth", CURRENT_YEAR, CURRENT_YEAR
    )
    assert assessment["grade"] in ("A", "B")
    assert not any("Do not buy" in warning for warning in assessment["warnings"])


def test_an_unverified_cheap_credit_is_called_out():
    assessment = assess_credit(
        "Direct air capture with storage", 10, 2.0, DEFAULT_REGISTRY, CURRENT_YEAR, CURRENT_YEAR
    )
    warnings = " ".join(assessment["warnings"])
    assert "Nobody independent" in warnings
    assert "far below" in warnings


def test_avoidance_credits_are_always_labelled_as_such():
    assessment = assess_credit("Clean cookstoves", 5, 12.0, "Gold Standard")
    assert any("not a removal" in warning for warning in assessment["warnings"])


def test_short_lived_storage_is_flagged():
    assessment = assess_credit("Soil carbon sequestration", 5, 25.0, "Verra (VCS)")
    assert any("rented, not offset" in warning for warning in assessment["warnings"])


def test_weak_additionality_is_flagged():
    assessment = assess_credit("Grid-connected wind or solar", 5, 4.0, "Verra (VCS)")
    assert any("Weak additionality" in warning for warning in assessment["warnings"])


def test_assessment_needs_a_known_project():
    assert assess_credit("Wishful thinking", 5, 10.0) is None


def test_warnings_of_nothing_are_empty():
    assert get_credit_warnings(None) == []


# ---------------------------------------------------------------------------
# Portfolios
# ---------------------------------------------------------------------------

def test_portfolio_aggregates_tonnes_spend_and_score():
    credits = [
        assess_credit("Biochar", 2, 140.0, "Puro.earth", CURRENT_YEAR, CURRENT_YEAR),
        assess_credit("Clean cookstoves", 8, 12.0, "Gold Standard", CURRENT_YEAR, CURRENT_YEAR),
    ]
    summary = portfolio_summary(credits)
    assert summary["credit_count"] == 2
    assert summary["nominal_tonnes"] == 10.0
    assert summary["total_spend"] == pytest.approx(2 * 140 + 8 * 12, abs=0.5)
    assert summary["effective_tonnes"] < summary["nominal_tonnes"]
    assert summary["worst_credit"]["project_type"] == "Clean cookstoves"


def test_portfolio_score_is_weighted_by_tonnage():
    good = assess_credit("Biochar", 1, 140.0, "Puro.earth")
    weak = assess_credit("Grid-connected wind or solar", 99, 4.0, DEFAULT_REGISTRY)
    summary = portfolio_summary([good, weak])
    # A single excellent tonne cannot rescue ninety-nine weak ones.
    assert summary["weighted_score"] < (good["score"] + weak["score"]) / 2


def test_removal_and_durable_shares_are_reported():
    credits = [
        assess_credit("Biochar", 5, 140.0, "Puro.earth"),
        assess_credit("Avoided deforestation (REDD+)", 5, 12.0, "Verra (VCS)"),
    ]
    summary = portfolio_summary(credits)
    assert summary["removal_share_pct"] == 50.0
    assert summary["durable_share_pct"] == 50.0


def test_an_empty_portfolio_is_empty_not_broken():
    summary = portfolio_summary([])
    assert summary["credit_count"] == 0
    assert summary["effective_tonnes"] == 0.0
    assert summary["grade"] == "F"
    assert summary["worst_credit"] is None


# ---------------------------------------------------------------------------
# Mitigation hierarchy
# ---------------------------------------------------------------------------

def test_offsetting_without_reducing_is_called_out():
    result = mitigation_hierarchy(footprint_kg=8000, reduced_kg=0, offset_tonnes=8)
    assert result["status"] == "OFFSET_ONLY"
    assert "not a substitute" in result["message"]


def test_reduction_led_action_is_endorsed():
    result = mitigation_hierarchy(footprint_kg=8000, reduced_kg=3000, offset_tonnes=1)
    assert result["status"] == "REDUCTION_LED"
    assert result["reduction_share_pct"] > 50


def test_offset_heavy_action_is_flagged_but_not_scolded():
    result = mitigation_hierarchy(footprint_kg=8000, reduced_kg=500, offset_tonnes=5)
    assert result["status"] == "OFFSET_HEAVY"
    assert "cheaper and more certain" in result["message"]


def test_doing_nothing_yet_says_so():
    assert mitigation_hierarchy(8000, 0, 0)["status"] == "NOTHING_YET"


def test_residual_coverage_is_capped_at_full():
    result = mitigation_hierarchy(footprint_kg=5000, reduced_kg=1000, offset_tonnes=50)
    assert result["residual_covered_pct"] == 100.0
    assert result["over_offset"] is True


def test_reductions_cannot_exceed_the_footprint():
    result = mitigation_hierarchy(footprint_kg=5000, reduced_kg=99999, offset_tonnes=0)
    assert result["reduced_kg"] == 5000
    assert result["residual_kg"] == 0.0


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def test_a_recommendation_splits_by_the_removal_preference():
    plan = recommend_portfolio(1000, 2, removal_preference=0.5, current_year=CURRENT_YEAR)
    assert len(plan["allocations"]) == 2
    kinds = {allocation["kind"] for allocation in plan["allocations"]}
    assert kinds == {REMOVAL, AVOIDANCE}
    assert sum(allocation["share_pct"] for allocation in plan["allocations"]) == 100.0


def test_an_all_removal_preference_buys_fewer_tonnes_for_the_same_money():
    removals = recommend_portfolio(1000, 1, removal_preference=1.0, current_year=CURRENT_YEAR)
    avoidance = recommend_portfolio(1000, 1, removal_preference=0.0, current_year=CURRENT_YEAR)
    assert removals["allocations"][0]["nominal_tonnes"] < avoidance["allocations"][0]["nominal_tonnes"]


def test_an_unaffordable_target_is_reported_honestly():
    plan = recommend_portfolio(50, 100, removal_preference=1.0, current_year=CURRENT_YEAR)
    assert plan["affordable"] is False
    assert plan["shortfall_tonnes"] > 0
    assert "Reducing the emission" in plan["note"]


def test_a_recommendation_needs_a_budget_and_a_target():
    assert recommend_portfolio(0, 10)["allocations"] == []
    assert recommend_portfolio(100, 0)["affordable"] is False


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------

def test_advice_prompts_for_input_when_there_is_nothing_to_advise_on():
    assert "Assess a credit" in get_offset_advice(portfolio_summary([]))[0]


def test_advice_leads_with_the_hierarchy_when_offsetting_replaces_reducing():
    summary = portfolio_summary([assess_credit("Clean cookstoves", 10, 12.0, "Gold Standard")])
    hierarchy = mitigation_hierarchy(9000, 0, 10)
    advice = get_offset_advice(summary, hierarchy)
    assert "not a substitute" in advice[0]


def test_advice_names_the_weakest_holding():
    credits = [
        assess_credit("Biochar", 1, 140.0, "Puro.earth"),
        assess_credit("Grid-connected wind or solar", 1, 4.0, DEFAULT_REGISTRY),
    ]
    advice = " ".join(get_offset_advice(portfolio_summary(credits)))
    assert "Grid-connected wind or solar" in advice


def test_advice_respects_the_limit():
    summary = portfolio_summary([assess_credit("Clean cookstoves", 5, 12.0, "Gold Standard")])
    assert len(get_offset_advice(summary, limit=2)) == 2
    assert get_offset_advice(summary, limit=0) == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_saved_holdings_come_back_intact():
    assessment = assess_credit("Biochar", 3, 140.0, "Puro.earth", CURRENT_YEAR, CURRENT_YEAR)
    holding_id = save_holding(1, "2026 purchase", assessment)
    assert holding_id is not None

    holdings = get_holdings(1)
    assert len(holdings) == 1
    assert holdings[0]["label"] == "2026 purchase"
    assert holdings[0]["project_type"] == "Biochar"
    assert holdings[0]["kind"] == REMOVAL
    assert holdings[0]["grade"] == assessment["grade"]
    assert holdings[0]["components"]
    assert holdings[0]["permanence_years"] == 500


def test_holdings_are_scoped_to_their_owner():
    assessment = assess_credit("Clean cookstoves", 1, 12.0, "Gold Standard")
    save_holding(1, "Mine", assessment)
    save_holding(2, "Theirs", assessment)
    assert len(get_holdings(1)) == 1
    assert get_holdings(1)[0]["label"] == "Mine"


def test_an_unnamed_holding_still_saves():
    assessment = assess_credit("Clean cookstoves", 1, 12.0, "Gold Standard")
    assert save_holding(1, "", assessment) is not None
    assert get_holdings(1)[0]["label"] == "Offset purchase"


def test_a_saved_portfolio_can_be_summarised_again():
    save_holding(1, "A", assess_credit("Biochar", 2, 140.0, "Puro.earth"))
    save_holding(1, "B", assess_credit("Clean cookstoves", 4, 12.0, "Gold Standard"))
    holdings = get_holdings(1)
    total = sum(holding["nominal_tonnes"] for holding in holdings)
    assert total == 6.0


def test_deleting_a_holding_removes_only_that_holding():
    assessment = assess_credit("Clean cookstoves", 1, 12.0, "Gold Standard")
    keep = save_holding(1, "Keep", assessment)
    remove = save_holding(1, "Remove", assessment)
    assert delete_holding(remove) is True
    remaining = get_holdings(1)
    assert len(remaining) == 1
    assert remaining[0]["id"] == keep


def test_deleting_a_missing_holding_reports_failure():
    assert delete_holding(9191) is False


def test_the_holding_limit_is_honoured():
    assessment = assess_credit("Clean cookstoves", 1, 12.0, "Gold Standard")
    for index in range(6):
        save_holding(1, f"Purchase {index}", assessment)
    assert len(get_holdings(1, limit=2)) == 2
