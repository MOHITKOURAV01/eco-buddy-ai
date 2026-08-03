"""Tests for the Hybrid Commute Planner."""
import os
import tempfile

import pytest

os.environ["ECO_BUDDY_DB"] = ":memory:"

import commute_planner
from commute_planner import (
    COLD_START_SHORT_TRIP_KM,
    COLD_START_SHORT_TRIP_MULTIPLIER,
    DEFAULT_GRID_INTENSITY,
    DEFAULT_MODE,
    DEFAULT_SEASON,
    HOME_WORKING_KWH,
    OFFICE_FIXED_KWH_PER_DESK_DAY,
    OFFICE_MARGINAL_KWH_PER_PERSON_DAY,
    SEASONS,
    SEASON_WEEKS,
    SHAREABLE_MODES,
    TRAVEL_MODES,
    WORKING_DAYS_PER_WEEK,
    WORKING_WEEKS_PER_YEAR,
    annual_summary,
    best_schedule,
    cold_start_penalty,
    compare_modes,
    compare_schedules,
    consolidation_benefit,
    delete_commute_plan,
    get_commute_advice,
    get_commute_plans,
    get_mode,
    home_day_emissions,
    is_shareable,
    list_modes,
    office_day_emissions,
    save_commute_plan,
    trip_emissions,
    weekly_plan,
)


@pytest.fixture(autouse=True)
def temp_db():
    """Point the module at a throwaway SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    original = commute_planner.DB_NAME
    commute_planner.DB_NAME = db_path
    yield db_path
    commute_planner.DB_NAME = original
    try:
        os.unlink(db_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def test_every_mode_is_fully_specified():
    for name, mode in TRAVEL_MODES.items():
        assert mode["kg_per_km"] >= 0, name
        assert mode["cold_start_kg"] >= 0, name
        assert 0 <= mode["winter_uplift"] <= 1, name
        assert mode["speed_kmh"] > 0, name
        assert mode["note"], name


def test_modes_are_listed_lowest_carbon_first():
    factors = [mode["kg_per_km"] for mode in list_modes()]
    assert factors == sorted(factors)
    assert list_modes()[0]["name"] in ("Bicycle", "Walking")


def test_only_private_vehicles_can_be_shared():
    for name in SHAREABLE_MODES:
        assert TRAVEL_MODES[name]["per_vehicle"] is True, name
    assert is_shareable("Bus") is False
    assert is_shareable("Suburban train") is False


def test_active_modes_are_flagged_infeasible_over_long_distances():
    modes = {mode["name"]: mode for mode in list_modes(40)}
    assert modes["Walking"]["feasible"] is False
    assert modes["Bicycle"]["feasible"] is False
    assert modes["Suburban train"]["feasible"] is True


def test_active_modes_are_feasible_over_short_distances():
    modes = {mode["name"]: mode for mode in list_modes(3)}
    assert modes["Walking"]["feasible"] is True
    assert modes["Bicycle"]["feasible"] is True


def test_an_unknown_mode_falls_back_to_the_default():
    assert get_mode("Hovercraft")["name"] == DEFAULT_MODE
    assert get_mode(None)["name"] == DEFAULT_MODE


def test_the_home_working_penalty_is_worst_in_winter():
    assert HOME_WORKING_KWH["Winter"] > HOME_WORKING_KWH["Shoulder"]
    assert HOME_WORKING_KWH["Shoulder"] > HOME_WORKING_KWH["Summer"]


def test_season_weights_cover_a_working_year():
    assert sum(SEASON_WEEKS.values()) == WORKING_WEEKS_PER_YEAR
    assert set(SEASON_WEEKS) == set(SEASONS)


# ---------------------------------------------------------------------------
# Cold starts
# ---------------------------------------------------------------------------

def test_active_modes_have_no_cold_start():
    for mode in ("Bicycle", "Walking", "Bus", "Tram or metro", "Suburban train"):
        assert cold_start_penalty(mode, 5) == 0.0


def test_a_very_short_trip_is_penalised_harder():
    short = cold_start_penalty("Petrol car", COLD_START_SHORT_TRIP_KM - 0.5)
    normal = cold_start_penalty("Petrol car", 10)
    assert short == pytest.approx(normal * COLD_START_SHORT_TRIP_MULTIPLIER, abs=1e-4)


def test_the_cold_start_is_the_same_however_far_you_then_drive():
    assert cold_start_penalty("Petrol car", 10) == cold_start_penalty("Petrol car", 60)


def test_a_zero_distance_trip_has_no_cold_start():
    assert cold_start_penalty("Petrol car", 0) == 0.0


def test_electric_cars_barely_pay_a_cold_start():
    assert cold_start_penalty("Electric car", 5) < cold_start_penalty("Petrol car", 5) / 5


# ---------------------------------------------------------------------------
# One trip
# ---------------------------------------------------------------------------

def test_a_short_car_trip_is_worse_per_kilometre_than_a_long_one():
    short = trip_emissions("Petrol car", 3)
    long_trip = trip_emissions("Petrol car", 30)
    assert short["kg_per_km"] > long_trip["kg_per_km"]
    assert short["total_kg"] < long_trip["total_kg"]


def test_trip_emissions_are_running_plus_cold_start():
    trip = trip_emissions("Diesel car", 12)
    assert trip["total_kg"] == pytest.approx(
        trip["running_kg"] + trip["cold_start_kg"], abs=1e-4
    )


def test_sharing_a_car_divides_the_emissions():
    solo = trip_emissions("Petrol car", 20, occupants=1)
    shared = trip_emissions("Petrol car", 20, occupants=2)
    assert shared["total_kg"] == pytest.approx(solo["total_kg"] / 2, abs=1e-4)


def test_sharing_a_bus_does_not_divide_anything():
    solo = trip_emissions("Bus", 20, occupants=1)
    claimed_share = trip_emissions("Bus", 20, occupants=4)
    assert claimed_share["total_kg"] == solo["total_kg"]
    assert claimed_share["occupants"] == 1


def test_winter_raises_emissions_for_modes_that_care():
    summer = trip_emissions("Electric car", 15, season="Summer")
    winter = trip_emissions("Electric car", 15, season="Winter")
    assert winter["total_kg"] > summer["total_kg"]


def test_winter_changes_nothing_for_a_bicycle():
    assert trip_emissions("Bicycle", 5, season="Winter")["total_kg"] == trip_emissions(
        "Bicycle", 5, season="Summer"
    )["total_kg"]


def test_a_zero_distance_commute_emits_nothing():
    trip = trip_emissions("Petrol car", 0)
    assert trip["total_kg"] == 0.0
    assert trip["minutes"] == 0.0


def test_junk_distances_do_not_crash_the_model():
    assert trip_emissions("Petrol car", None)["total_kg"] == 0.0
    assert trip_emissions("Petrol car", "far")["total_kg"] == 0.0
    assert trip_emissions("Petrol car", -10)["total_kg"] == 0.0


def test_travel_time_follows_the_mode_speed():
    walk = trip_emissions("Walking", 5)
    train = trip_emissions("Suburban train", 5)
    assert walk["minutes"] > train["minutes"]


# ---------------------------------------------------------------------------
# Mode comparison
# ---------------------------------------------------------------------------

def test_mode_comparison_puts_feasible_options_first():
    rows = compare_modes(30, current_mode="Petrol car")
    feasible = [row["feasible"] for row in rows]
    assert feasible == sorted(feasible, reverse=True)


def test_cycling_wins_a_short_commute_by_the_largest_margin():
    rows = {row["mode"]: row for row in compare_modes(4, current_mode="Petrol car")}
    assert rows["Bicycle"]["saving_pct"] == 100.0
    assert rows["Bicycle"]["feasible"] is True


def test_savings_are_never_negative():
    rows = compare_modes(15, current_mode="Bicycle")
    assert all(row["saving_kg"] >= 0 for row in rows)


def test_the_train_gains_ground_over_the_car_with_distance():
    near = {row["mode"]: row for row in compare_modes(5, current_mode="Petrol car")}
    far = {row["mode"]: row for row in compare_modes(50, current_mode="Petrol car")}
    assert far["Suburban train"]["saving_kg"] > near["Suburban train"]["saving_kg"]


# ---------------------------------------------------------------------------
# Office and home days
# ---------------------------------------------------------------------------

def test_office_fixed_energy_is_charged_for_open_days_not_attendance():
    full = office_day_emissions(5, office_open_days=5)
    partial = office_day_emissions(2, office_open_days=5)
    # Attending less does not recover the building's baseline energy.
    assert partial["fixed_kwh"] == full["fixed_kwh"]
    assert partial["marginal_kwh"] < full["marginal_kwh"]


def test_closing_the_office_is_what_actually_saves_the_fixed_energy():
    open_all_week = office_day_emissions(2, office_open_days=5)
    closed_three_days = office_day_emissions(2, office_open_days=2)
    assert closed_three_days["fixed_kwh"] < open_all_week["fixed_kwh"]
    assert closed_three_days["co2_kg"] < open_all_week["co2_kg"]


def test_office_energy_matches_the_documented_split():
    result = office_day_emissions(3, office_open_days=5)
    assert result["fixed_kwh"] == pytest.approx(OFFICE_FIXED_KWH_PER_DESK_DAY * 5, abs=0.01)
    assert result["marginal_kwh"] == pytest.approx(
        OFFICE_MARGINAL_KWH_PER_PERSON_DAY * 3, abs=0.01
    )


def test_the_office_cannot_be_open_fewer_days_than_you_attend():
    assert office_day_emissions(4, office_open_days=1)["office_open_days"] == 4


def test_attendance_is_capped_at_the_working_week():
    assert office_day_emissions(9)["days_attended"] == WORKING_DAYS_PER_WEEK


def test_working_from_home_is_not_free():
    home = home_day_emissions(5, "Winter")
    assert home["co2_kg"] > 0
    assert home["total_kwh"] == pytest.approx(HOME_WORKING_KWH["Winter"] * 5, abs=0.01)


def test_home_working_costs_most_in_winter():
    winter = home_day_emissions(5, "Winter")["co2_kg"]
    summer = home_day_emissions(5, "Summer")["co2_kg"]
    assert winter > summer * 2


def test_a_household_can_override_the_home_energy_figure():
    default = home_day_emissions(5, "Winter")
    efficient = home_day_emissions(5, "Winter", home_kwh_override=1.0)
    assert efficient["co2_kg"] < default["co2_kg"]


def test_a_cleaner_grid_reduces_both_buildings():
    dirty = weekly_plan(3, 20, grid_intensity=0.6)
    clean = weekly_plan(3, 20, grid_intensity=0.05)
    assert clean["total_kg"] < dirty["total_kg"]
    assert clean["commute_kg"] == dirty["commute_kg"]


# ---------------------------------------------------------------------------
# Weekly plans
# ---------------------------------------------------------------------------

def test_a_week_is_commute_plus_office_plus_home():
    plan = weekly_plan(3, 20, "Petrol car")
    assert plan["total_kg"] == pytest.approx(
        plan["commute_kg"] + plan["office"]["co2_kg"] + plan["home"]["co2_kg"], abs=0.01
    )
    assert plan["days_in_office"] + plan["days_at_home"] == WORKING_DAYS_PER_WEEK


def test_the_commute_is_counted_both_ways():
    plan = weekly_plan(1, 10, "Petrol car")
    leg = trip_emissions("Petrol car", 10)
    assert plan["commute_kg"] == pytest.approx(leg["total_kg"] * 2, abs=0.01)


def test_more_office_days_mean_more_driving():
    plans = [weekly_plan(days, 25, "Petrol car")["commute_kg"] for days in range(6)]
    assert plans == sorted(plans)


def test_a_long_car_commute_is_better_worked_from_home():
    comparison = best_schedule(45, "Petrol car")
    assert comparison["best"]["days_in_office"] == 0
    assert comparison["home_is_better"] is True


def test_a_short_walk_in_winter_can_beat_working_from_home():
    # Heating an extra home all winter can cost more than a five-minute walk.
    comparison = best_schedule(1.2, "Walking", season="Winter")
    assert comparison["best"]["days_in_office"] == WORKING_DAYS_PER_WEEK
    assert comparison["home_is_better"] is False


def test_every_schedule_from_zero_to_five_is_offered():
    schedules = compare_schedules(20)
    assert [plan["days_in_office"] for plan in schedules] == list(range(6))


def test_annual_figures_scale_by_the_working_year():
    plan = weekly_plan(3, 20)
    assert plan["annual_kg"] == pytest.approx(
        plan["total_kg"] * WORKING_WEEKS_PER_YEAR, abs=0.5
    )


def test_the_commute_share_is_a_percentage():
    plan = weekly_plan(5, 30)
    assert 0 <= plan["commute_share_pct"] <= 100


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def test_consolidating_onto_anchor_days_saves_the_buildings_baseline():
    result = consolidation_benefit(2, 20, "Petrol car")
    assert result["saving_kg"] > 0
    assert result["closed_days"] == 3
    assert result["worth_doing"] is True


def test_consolidation_saves_nothing_when_everyone_is_in_all_week():
    result = consolidation_benefit(5, 20)
    assert result["saving_kg"] == 0.0
    assert result["closed_days"] == 0
    assert result["worth_doing"] is False


def test_consolidation_does_not_change_the_commute_itself():
    result = consolidation_benefit(3, 20, "Petrol car")
    # The saving comes only from the building, not from travelling differently.
    assert result["scattered_kg"] > result["consolidated_kg"]
    assert result["annual_saving_kg"] == pytest.approx(
        result["saving_kg"] * WORKING_WEEKS_PER_YEAR, abs=0.5
    )


# ---------------------------------------------------------------------------
# Annual view
# ---------------------------------------------------------------------------

def test_the_annual_view_weights_every_season():
    summary = annual_summary(3, 20, "Petrol car")
    assert len(summary["seasons"]) == len(SEASON_WEEKS)
    assert summary["weeks_counted"] == WORKING_WEEKS_PER_YEAR
    assert summary["annual_kg"] == pytest.approx(
        sum(row["season_kg"] for row in summary["seasons"]), abs=1.0
    )


def test_winter_is_the_worst_season_for_a_home_worker():
    summary = annual_summary(0, 20, "Petrol car")
    assert summary["worst_season"] == "Winter"


def test_a_single_season_answer_would_have_been_misleading():
    winter_only = weekly_plan(0, 20, season="Winter")["total_kg"] * WORKING_WEEKS_PER_YEAR
    weighted = annual_summary(0, 20)["annual_kg"]
    assert weighted < winter_only


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------

def test_advice_prompts_for_input_when_there_is_nothing_to_advise_on():
    assert "Enter your commute" in get_commute_advice({})[0]


def test_advice_recommends_the_better_pattern():
    plan = weekly_plan(5, 45, "Petrol car")
    comparison = best_schedule(45, "Petrol car")
    advice = " ".join(get_commute_advice(plan, comparison))
    assert "in the office would be lowest" in advice


def test_advice_mentions_consolidation_when_it_helps():
    plan = weekly_plan(2, 20, "Petrol car")
    consolidation = consolidation_benefit(2, 20, "Petrol car")
    advice = " ".join(get_commute_advice(plan, consolidation=consolidation))
    assert "same days" in advice


def test_advice_respects_the_limit():
    plan = weekly_plan(3, 20)
    assert len(get_commute_advice(plan, limit=2)) == 2
    assert get_commute_advice(plan, limit=0) == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_saved_plans_come_back_intact():
    plan = weekly_plan(3, 22.5, "Suburban train", season="Winter")
    plan_id = save_commute_plan(1, "Office days", plan, 22.5)
    assert plan_id is not None

    plans = get_commute_plans(1)
    assert len(plans) == 1
    assert plans[0]["plan_name"] == "Office days"
    assert plans[0]["mode"] == "Suburban train"
    assert plans[0]["distance_km"] == 22.5
    assert plans[0]["days_in_office"] == 3
    assert plans[0]["season"] == "Winter"
    assert plans[0]["detail"]["office"]["total_kwh"] > 0


def test_plans_are_scoped_to_their_owner():
    plan = weekly_plan(2, 10)
    save_commute_plan(1, "Mine", plan, 10)
    save_commute_plan(2, "Theirs", plan, 10)
    assert len(get_commute_plans(1)) == 1
    assert get_commute_plans(1)[0]["plan_name"] == "Mine"


def test_an_unnamed_plan_still_saves():
    assert save_commute_plan(1, "   ", weekly_plan(1, 5), 5) is not None
    assert get_commute_plans(1)[0]["plan_name"] == "My commute"


def test_deleting_a_plan_removes_only_that_plan():
    plan = weekly_plan(2, 12)
    keep = save_commute_plan(1, "Keep", plan, 12)
    remove = save_commute_plan(1, "Remove", plan, 12)
    assert delete_commute_plan(remove) is True
    remaining = get_commute_plans(1)
    assert len(remaining) == 1
    assert remaining[0]["id"] == keep


def test_deleting_a_missing_plan_reports_failure():
    assert delete_commute_plan(7777) is False


def test_the_plan_limit_is_honoured():
    plan = weekly_plan(2, 12)
    for index in range(5):
        save_commute_plan(1, f"Plan {index}", plan, 12)
    assert len(get_commute_plans(1, limit=2)) == 2
