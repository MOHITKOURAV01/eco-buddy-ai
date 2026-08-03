"""Tests for the Tree & Garden Sequestration Planner."""
import os
import tempfile

import pytest

os.environ["ECO_BUDDY_DB"] = ":memory:"

import sequestration
from sequestration import (
    DEFAULT_HORIZON_YEARS,
    DEFAULT_PLANTING_TYPE,
    DEFAULT_SURVIVAL_RATE,
    MAX_HORIZON_YEARS,
    PLANTING_TYPES,
    annual_sequestration,
    biodiversity_score,
    build_plan_summary,
    capacity_for_area,
    cumulative_curve,
    cumulative_sequestration,
    delete_planting_plan,
    design_plan,
    get_planting_plans,
    get_planting_tips,
    get_planting_type,
    growth_factor,
    list_planting_types,
    maintenance_emissions,
    mature_annual_rate,
    net_sequestration,
    offset_share,
    plan_area_used,
    plan_fits,
    save_planting_plan,
    sequestration_curve,
    years_to_offset,
)


@pytest.fixture(autouse=True)
def temp_db():
    """Point the module at a throwaway SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    original = sequestration.DB_NAME
    sequestration.DB_NAME = db_path
    yield db_path
    sequestration.DB_NAME = original
    try:
        os.unlink(db_path)
    except OSError:
        pass


SMALL_PLAN = [{"planting_type": "Medium broadleaf (birch, rowan)", "count": 3}]

MIXED_PLAN = [
    {"planting_type": "Large broadleaf (oak, beech)", "count": 2},
    {"planting_type": "Small broadleaf (hawthorn, hazel)", "count": 4},
    {"planting_type": "Hedgerow (per metre)", "count": 10},
    {"planting_type": "Shrub", "count": 6},
    {"planting_type": "Wildflower meadow (per m²)", "count": 20},
]

MONOCULTURE_PLAN = [{"planting_type": "Conifer (pine, spruce)", "count": 20}]


class TestReferenceData:
    def test_every_type_has_sane_reference_data(self):
        for name, info in PLANTING_TYPES.items():
            assert info["mature_rate_kg"] > 0, name
            assert info["years_to_maturity"] > 0, name
            assert info["spacing_m2"] > 0, name
            assert info["maintenance_kg"] >= 0, name
            assert info["note"]

    def test_oak_has_the_highest_ceiling(self):
        assert list_planting_types()[0]["name"] == "Large broadleaf (oak, beech)"

    def test_bigger_trees_need_more_space(self):
        assert (
            PLANTING_TYPES["Large broadleaf (oak, beech)"]["spacing_m2"]
            > PLANTING_TYPES["Shrub"]["spacing_m2"]
        )

    def test_bigger_trees_take_longer_to_mature(self):
        assert (
            PLANTING_TYPES["Large broadleaf (oak, beech)"]["years_to_maturity"]
            > PLANTING_TYPES["Bamboo"]["years_to_maturity"]
        )

    def test_unknown_type_falls_back(self):
        assert get_planting_type("Baobab") == PLANTING_TYPES[DEFAULT_PLANTING_TYPE]

    def test_returned_reference_data_is_a_copy(self):
        info = get_planting_type("Shrub")
        info["mature_rate_kg"] = 9999
        assert PLANTING_TYPES["Shrub"]["mature_rate_kg"] != 9999


class TestCapacity:
    def test_respects_spacing(self):
        # 300 m² at 60 m² per oak is five oaks, not six.
        assert capacity_for_area(300, "Large broadleaf (oak, beech)") == 5

    def test_partial_space_does_not_round_up(self):
        assert capacity_for_area(59, "Large broadleaf (oak, beech)") == 0
        assert capacity_for_area(119, "Large broadleaf (oak, beech)") == 1

    def test_smaller_plants_fit_more(self):
        area = 300
        assert capacity_for_area(area, "Shrub") > capacity_for_area(
            area, "Large broadleaf (oak, beech)"
        )

    def test_zero_area_fits_nothing(self):
        assert capacity_for_area(0, "Shrub") == 0

    def test_negative_area_fits_nothing(self):
        assert capacity_for_area(-100, "Shrub") == 0

    def test_garbage_area_fits_nothing(self):
        assert capacity_for_area("big", "Shrub") == 0

    def test_per_area_types_map_one_to_one(self):
        assert capacity_for_area(50, "Wildflower meadow (per m²)") == 50


class TestGrowthCurve:
    def test_year_one_is_well_below_maturity(self):
        # The single most important correction this module makes.
        assert growth_factor("Large broadleaf (oak, beech)", 1) < 0.2

    def test_factor_is_bounded(self):
        for planting_type in PLANTING_TYPES:
            for year in (0, 1, 5, 20, 50, 100):
                assert 0.0 <= growth_factor(planting_type, year) <= 1.0

    def test_factor_is_monotonically_increasing(self):
        factors = [growth_factor("Medium broadleaf (birch, rowan)", year) for year in range(1, 41)]
        assert factors == sorted(factors)

    def test_approaches_maturity_late(self):
        assert growth_factor("Medium broadleaf (birch, rowan)", 40) > 0.95

    def test_fast_types_establish_sooner(self):
        assert growth_factor("Bamboo", 5) > growth_factor("Large broadleaf (oak, beech)", 5)

    def test_annual_uptake_is_below_the_mature_rate_early(self):
        mature = PLANTING_TYPES["Large broadleaf (oak, beech)"]["mature_rate_kg"]
        assert annual_sequestration("Large broadleaf (oak, beech)", 1, 1) < mature

    def test_annual_uptake_scales_with_count(self):
        one = annual_sequestration("Shrub", 1, 10)
        ten = annual_sequestration("Shrub", 10, 10)
        assert ten == pytest.approx(one * 10, abs=0.01)

    def test_zero_count_absorbs_nothing(self):
        assert annual_sequestration("Shrub", 0, 10) == 0.0


class TestCurves:
    def test_curve_has_one_entry_per_year(self):
        assert len(sequestration_curve(SMALL_PLAN, 25)) == 25

    def test_curve_is_non_decreasing(self):
        curve = sequestration_curve(MIXED_PLAN, 40)
        assert curve == sorted(curve)

    def test_cumulative_equals_the_sum_of_annual_values(self):
        curve = sequestration_curve(MIXED_PLAN, 30)
        assert cumulative_sequestration(MIXED_PLAN, 30) == pytest.approx(sum(curve), abs=0.05)

    def test_cumulative_curve_is_monotonically_increasing(self):
        curve = cumulative_curve(MIXED_PLAN, 30)
        assert curve == sorted(curve)

    def test_cumulative_curve_ends_at_the_total(self):
        curve = cumulative_curve(MIXED_PLAN, 30)
        assert curve[-1] == pytest.approx(cumulative_sequestration(MIXED_PLAN, 30), abs=0.05)

    def test_longer_horizons_absorb_more(self):
        assert cumulative_sequestration(SMALL_PLAN, 40) > cumulative_sequestration(SMALL_PLAN, 10)

    def test_empty_plan_absorbs_nothing(self):
        assert cumulative_sequestration([], 30) == 0.0
        assert sequestration_curve([], 5) == [0.0] * 5

    def test_none_plan_is_handled(self):
        assert cumulative_sequestration(None, 10) == 0.0

    def test_zero_count_entries_are_dropped(self):
        plan = [{"planting_type": "Shrub", "count": 0}]
        assert cumulative_sequestration(plan, 10) == 0.0

    def test_horizon_is_clamped(self):
        assert len(sequestration_curve(SMALL_PLAN, 0)) == 1
        assert len(sequestration_curve(SMALL_PLAN, 9999)) == MAX_HORIZON_YEARS


class TestNetSequestration:
    def test_net_never_exceeds_gross(self):
        result = net_sequestration(MIXED_PLAN, 30)
        assert result["net_co2_kg"] <= result["gross_co2_kg"]

    def test_survival_losses_reduce_the_total(self):
        full = net_sequestration(MIXED_PLAN, 30, survival_rate=1.0)
        lossy = net_sequestration(MIXED_PLAN, 30, survival_rate=0.5)
        assert lossy["net_co2_kg"] < full["net_co2_kg"]

    def test_maintenance_is_subtracted(self):
        result = net_sequestration(MIXED_PLAN, 30, survival_rate=1.0)
        assert result["net_co2_kg"] == pytest.approx(
            result["gross_co2_kg"] - result["maintenance_co2_kg"], abs=0.05
        )

    def test_maintenance_scales_with_horizon(self):
        short = maintenance_emissions(MIXED_PLAN, 10)
        long = maintenance_emissions(MIXED_PLAN, 30)
        assert long == pytest.approx(short * 3, abs=0.05)

    def test_zero_survival_means_zero_net(self):
        assert net_sequestration(MIXED_PLAN, 30, survival_rate=0.0)["net_co2_kg"] == 0.0

    def test_survival_rate_is_clamped(self):
        assert net_sequestration(SMALL_PLAN, 10, survival_rate=5.0)["survival_rate"] == 1.0
        assert net_sequestration(SMALL_PLAN, 10, survival_rate=-1.0)["survival_rate"] == 0.0

    def test_empty_plan_is_all_zeroes(self):
        result = net_sequestration([], 30)
        assert result["gross_co2_kg"] == 0.0
        assert result["net_co2_kg"] == 0.0

    def test_high_maintenance_over_a_short_horizon_can_go_negative(self):
        # A young orchard genuinely does cost more than it absorbs at first,
        # and the module reports that rather than clamping it away.
        orchard = [{"planting_type": "Fruit tree", "count": 5}]
        assert net_sequestration(orchard, 2, survival_rate=1.0)["net_co2_kg"] < 0


class TestMatureRate:
    def test_matches_hand_calculation(self):
        info = PLANTING_TYPES["Shrub"]
        expected = (info["mature_rate_kg"] - info["maintenance_kg"]) * 10 * 1.0
        plan = [{"planting_type": "Shrub", "count": 10}]
        assert mature_annual_rate(plan, survival_rate=1.0) == pytest.approx(expected, abs=0.05)

    def test_survival_reduces_the_ceiling(self):
        assert mature_annual_rate(MIXED_PLAN, 0.5) < mature_annual_rate(MIXED_PLAN, 1.0)

    def test_empty_plan_has_no_ceiling(self):
        assert mature_annual_rate([]) == 0.0

    def test_bigger_plans_have_higher_ceilings(self):
        small = mature_annual_rate(SMALL_PLAN)
        big = mature_annual_rate(MIXED_PLAN)
        assert big > small


class TestYearsToOffset:
    def test_a_tiny_plan_never_offsets_a_real_footprint(self):
        # The honest answer for most gardens, and the reason None is allowed.
        assert years_to_offset(SMALL_PLAN, 5000.0, horizon=40) is None

    def test_a_large_plan_offsets_a_small_footprint(self):
        result = years_to_offset(MIXED_PLAN, 100.0, horizon=40)
        assert result is not None
        assert 1 <= result <= 40

    def test_zero_footprint_is_offset_immediately(self):
        assert years_to_offset(SMALL_PLAN, 0.0) == 0

    def test_a_bigger_footprint_takes_longer(self):
        soon = years_to_offset(MIXED_PLAN, 100.0, horizon=60)
        later = years_to_offset(MIXED_PLAN, 2000.0, horizon=60)
        assert later is None or later > soon

    def test_an_empty_plan_never_offsets_anything(self):
        assert years_to_offset([], 100.0) is None

    def test_a_shorter_horizon_can_turn_an_answer_into_none(self):
        assert years_to_offset(MIXED_PLAN, 800.0, horizon=1) is None

    def test_lower_survival_delays_or_prevents_the_offset(self):
        full = years_to_offset(MIXED_PLAN, 300.0, horizon=40, survival_rate=1.0)
        lossy = years_to_offset(MIXED_PLAN, 300.0, horizon=40, survival_rate=0.3)
        assert lossy is None or (full is not None and lossy >= full)


class TestOffsetShare:
    def test_share_is_bounded(self):
        for year in (1, 10, 40):
            assert 0.0 <= offset_share(MIXED_PLAN, 500.0, year) <= 100.0

    def test_share_grows_with_the_planting(self):
        early = offset_share(MIXED_PLAN, 500.0, 1)
        late = offset_share(MIXED_PLAN, 500.0, 40)
        assert late > early

    def test_zero_footprint_returns_zero(self):
        assert offset_share(MIXED_PLAN, 0.0, 20) == 0.0

    def test_empty_plan_covers_nothing(self):
        assert offset_share([], 500.0, 20) == 0.0

    def test_a_huge_footprint_is_barely_covered(self):
        assert offset_share(SMALL_PLAN, 100000.0, 40) < 1.0

    def test_share_never_goes_negative(self):
        orchard = [{"planting_type": "Fruit tree", "count": 5}]
        assert offset_share(orchard, 500.0, 1) >= 0.0


class TestDesignPlan:
    def test_suggested_plan_fits_the_space(self):
        for area in (50, 200, 1000):
            plan = design_plan(area)
            assert plan_fits(plan, area), area

    def test_capacity_goal_favours_large_trees(self):
        plan = design_plan(2000, goal="capacity")
        types = {entry["planting_type"] for entry in plan}
        assert "Large broadleaf (oak, beech)" in types

    def test_fast_goal_favours_quick_establishment(self):
        plan = design_plan(2000, goal="fast")
        types = {entry["planting_type"] for entry in plan}
        assert "Bamboo" in types or "Small broadleaf (hawthorn, hazel)" in types

    def test_unknown_goal_falls_back_to_balanced(self):
        assert design_plan(500, goal="whatever") == design_plan(500, goal="balanced")

    def test_zero_area_produces_no_plan(self):
        assert design_plan(0) == []

    def test_negative_area_produces_no_plan(self):
        assert design_plan(-100) == []

    def test_a_tiny_garden_still_gets_something_it_can_do(self):
        plan = design_plan(3)
        assert plan
        assert plan[0]["planting_type"] == "Wildflower meadow (per m²)"

    def test_bigger_gardens_get_bigger_plans(self):
        small = sum(entry["count"] for entry in design_plan(100))
        big = sum(entry["count"] for entry in design_plan(2000))
        assert big > small


class TestPlanArea:
    def test_area_used_matches_hand_calculation(self):
        plan = [{"planting_type": "Large broadleaf (oak, beech)", "count": 3}]
        assert plan_area_used(plan) == pytest.approx(180.0)

    def test_empty_plan_uses_no_space(self):
        assert plan_area_used([]) == 0.0

    def test_fits_is_true_with_room_to_spare(self):
        assert plan_fits([{"planting_type": "Shrub", "count": 2}], 100) is True

    def test_fits_is_false_when_overcrowded(self):
        assert plan_fits([{"planting_type": "Large broadleaf (oak, beech)", "count": 10}], 100) is False

    def test_exact_fit_is_allowed(self):
        assert plan_fits([{"planting_type": "Large broadleaf (oak, beech)", "count": 2}], 120) is True


class TestBiodiversityScore:
    def test_score_is_bounded(self):
        for plan in ([], SMALL_PLAN, MIXED_PLAN, MONOCULTURE_PLAN):
            assert 0.0 <= biodiversity_score(plan) <= 100.0

    def test_empty_plan_scores_zero(self):
        assert biodiversity_score([]) == 0.0

    def test_a_mix_beats_a_monoculture(self):
        assert biodiversity_score(MIXED_PLAN) > biodiversity_score(MONOCULTURE_PLAN)

    def test_natives_beat_non_natives_at_equal_diversity(self):
        native = [{"planting_type": "Small broadleaf (hawthorn, hazel)", "count": 10}]
        non_native = [{"planting_type": "Bamboo", "count": 10}]
        assert biodiversity_score(native) > biodiversity_score(non_native)

    def test_more_species_scores_higher(self):
        two = biodiversity_score(
            [
                {"planting_type": "Shrub", "count": 5},
                {"planting_type": "Hedgerow (per metre)", "count": 5},
            ]
        )
        assert biodiversity_score(MIXED_PLAN) > two

    def test_the_highest_uptake_monoculture_is_not_the_best_garden(self):
        # Optimising carbon alone gives a worse garden, which is the point.
        oaks = [{"planting_type": "Large broadleaf (oak, beech)", "count": 20}]
        assert biodiversity_score(MIXED_PLAN) > biodiversity_score(oaks)


class TestPlanSummary:
    def test_summary_is_internally_consistent(self):
        summary = build_plan_summary(MIXED_PLAN, area_m2=500, annual_footprint_kg=800, years=30)
        assert len(summary["curve"]) == 30
        assert summary["cumulative_curve"][-1] == pytest.approx(
            summary["gross_co2_kg"], abs=0.05
        )
        assert summary["net_co2_kg"] <= summary["gross_co2_kg"]

    def test_summary_flags_an_oversized_plan(self):
        summary = build_plan_summary(MIXED_PLAN, area_m2=10)
        assert summary["fits"] is False

    def test_summary_of_an_empty_plan(self):
        summary = build_plan_summary([], area_m2=100, annual_footprint_kg=500)
        assert summary["plan"] == []
        assert summary["net_co2_kg"] == 0.0
        assert summary["years_to_offset"] is None
        assert summary["biodiversity_score"] == 0.0

    def test_zero_count_entries_are_dropped_from_the_summary(self):
        plan = [
            {"planting_type": "Shrub", "count": 0},
            {"planting_type": "Shrub", "count": 3},
        ]
        assert len(build_plan_summary(plan)["plan"]) == 1


class TestTips:
    def test_empty_plan_gets_a_prompt(self):
        tips = get_planting_tips(build_plan_summary([]))
        assert len(tips) == 1
        assert "add some plantings" in tips[0].lower()

    def test_an_unachievable_offset_is_stated_plainly(self):
        summary = build_plan_summary(SMALL_PLAN, area_m2=200, annual_footprint_kg=6000, years=40)
        tips = get_planting_tips(summary, annual_footprint_kg=6000)
        assert any("will not offset" in tip.lower() for tip in tips)
        assert any("substitute" in tip.lower() for tip in tips)

    def test_an_achievable_offset_reports_the_year(self):
        summary = build_plan_summary(MIXED_PLAN, area_m2=1000, annual_footprint_kg=100, years=40)
        tips = get_planting_tips(summary, annual_footprint_kg=100)
        assert any("cancels one year" in tip.lower() for tip in tips)

    def test_the_s_curve_is_always_explained(self):
        summary = build_plan_summary(MIXED_PLAN, area_m2=1000, annual_footprint_kg=500)
        assert any("s-curve" in tip.lower() for tip in get_planting_tips(summary))

    def test_a_monoculture_is_called_out(self):
        summary = build_plan_summary(MONOCULTURE_PLAN, area_m2=1000)
        assert any("biodiversity" in tip.lower() for tip in get_planting_tips(summary))

    def test_an_oversized_plan_is_called_out(self):
        summary = build_plan_summary(MIXED_PLAN, area_m2=5)
        assert any("more than you have" in tip.lower() for tip in get_planting_tips(summary))

    def test_limit_is_respected(self):
        summary = build_plan_summary(MIXED_PLAN, area_m2=1000, annual_footprint_kg=500)
        assert len(get_planting_tips(summary, 500, limit=2)) <= 2

    def test_zero_limit_returns_nothing(self):
        summary = build_plan_summary(MIXED_PLAN, area_m2=1000)
        assert get_planting_tips(summary, 500, limit=0) == []


class TestPersistence:
    def test_save_and_load_round_trip(self):
        summary = build_plan_summary(MIXED_PLAN, area_m2=500, annual_footprint_kg=800)
        plan_id = save_planting_plan(1, "Back garden", summary, area_m2=500)
        assert plan_id is not None

        saved = get_planting_plans(1)
        assert len(saved) == 1
        assert saved[0]["plan_name"] == "Back garden"
        assert saved[0]["area_m2"] == 500
        assert len(saved[0]["plan"]) == len(MIXED_PLAN)

    def test_saved_plan_can_be_remodelled(self):
        summary = build_plan_summary(MIXED_PLAN, area_m2=500, annual_footprint_kg=800, years=30)
        save_planting_plan(1, "Mine", summary, area_m2=500)
        loaded = get_planting_plans(1)[0]
        assert cumulative_sequestration(loaded["plan"], 30) == pytest.approx(
            summary["gross_co2_kg"], abs=0.05
        )

    def test_an_unachievable_offset_is_stored_as_null(self):
        summary = build_plan_summary(SMALL_PLAN, area_m2=200, annual_footprint_kg=9999)
        save_planting_plan(1, "Hopeful", summary, area_m2=200)
        assert get_planting_plans(1)[0]["years_to_offset"] is None

    def test_blank_name_gets_a_default(self):
        summary = build_plan_summary(SMALL_PLAN)
        save_planting_plan(2, "   ", summary)
        assert get_planting_plans(2)[0]["plan_name"] == "My garden"

    def test_plans_are_scoped_per_user(self):
        summary = build_plan_summary(SMALL_PLAN)
        save_planting_plan(10, "Mine", summary)
        save_planting_plan(11, "Theirs", summary)
        assert len(get_planting_plans(10)) == 1
        assert get_planting_plans(10)[0]["plan_name"] == "Mine"

    def test_limit_is_applied(self):
        summary = build_plan_summary(SMALL_PLAN)
        for index in range(5):
            save_planting_plan(3, f"Plan {index}", summary)
        assert len(get_planting_plans(3, limit=2)) == 2

    def test_delete_removes_only_the_target(self):
        summary = build_plan_summary(SMALL_PLAN)
        first = save_planting_plan(4, "One", summary)
        save_planting_plan(4, "Two", summary)
        assert delete_planting_plan(first) is True
        remaining = get_planting_plans(4)
        assert len(remaining) == 1
        assert remaining[0]["plan_name"] == "Two"

    def test_deleting_a_missing_row_returns_false(self):
        assert delete_planting_plan(999999) is False

    def test_no_plans_for_a_new_user(self):
        assert get_planting_plans(12345) == []
