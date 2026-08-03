"""Tests for the Home Heat Loss & Retrofit Simulator."""
import os
import tempfile

import pytest

os.environ["ECO_BUDDY_DB"] = ":memory:"

import heat_retrofit
from heat_retrofit import (
    AIRTIGHTNESS_LEVELS,
    AIR_HEAT_CAPACITY,
    BASE_TEMPERATURE_C,
    CLIMATE_ZONES,
    COP_CEILING,
    COP_FLOOR,
    DAYS_IN_MONTH,
    DEFAULT_CLIMATE_ZONE,
    DEFAULT_EMITTER,
    DESIGN_INDOOR_C,
    EMITTER_TYPES,
    FUELS,
    GLAZING_TYPES,
    HEATING_SYSTEMS,
    MAX_FLOW_C,
    MIN_FLOW_C,
    REFERENCE_FLOW_C,
    RETROFIT_MEASURES,
    ROOF_TYPES,
    WALL_TYPES,
    air_changes,
    annual_degree_days,
    apply_measure,
    build_retrofit_plan,
    compare_systems,
    delete_retrofit_plan,
    emitter_capacity_w,
    estimate_envelope,
    fabric_first_check,
    get_climate,
    get_retrofit_advice,
    get_retrofit_plans,
    heat_demand_kwh,
    heat_loss_coefficient,
    list_climate_zones,
    measure_applies,
    measure_cost,
    monthly_degree_days,
    monthly_heat_demand_kwh,
    peak_heat_load_w,
    rank_measures,
    required_flow_temperature,
    running_cost_and_emissions,
    save_retrofit_plan,
    seasonal_cop,
    system_efficiency,
    u_value,
)


@pytest.fixture(autouse=True)
def temp_db():
    """Point the module at a throwaway SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    original = heat_retrofit.DB_NAME
    heat_retrofit.DB_NAME = db_path
    yield db_path
    heat_retrofit.DB_NAME = original
    try:
        os.unlink(db_path)
    except OSError:
        pass


# A typical uninsulated older house, used across the tests below.
OLD_HOUSE = {
    "wall": "Solid brick, uninsulated",
    "roof": "No loft insulation",
    "floor": "Suspended timber, uninsulated",
    "glazing": "Single glazed",
    "door": "Uninsulated timber or single-glazed",
    "airtightness": "Draughty (older, unsealed)",
}

MODERN_HOUSE = {
    "wall": "Modern insulated (current regulations)",
    "roof": "270 mm loft insulation",
    "floor": "Insulated floor",
    "glazing": "Modern double glazed",
    "door": "Insulated composite",
    "airtightness": "Well sealed",
}


@pytest.fixture
def envelope():
    return estimate_envelope(120, storeys=2, attached_walls=0)


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def test_every_climate_zone_has_twelve_months_and_a_design_temperature():
    for name, data in CLIMATE_ZONES.items():
        assert len(data["monthly_c"]) == 12, name
        assert data["design_c"] < min(data["monthly_c"]), name


def test_climate_zones_are_listed_coldest_first():
    means = [zone["mean_c"] for zone in list_climate_zones()]
    assert means == sorted(means)


def test_unknown_climate_zone_falls_back_to_the_default():
    assert get_climate("Mars") == get_climate(DEFAULT_CLIMATE_ZONE)


def test_u_values_improve_down_each_construction_table():
    assert WALL_TYPES["Solid brick, uninsulated"] > WALL_TYPES["Cavity, insulated"]
    assert WALL_TYPES["Cavity, insulated"] > WALL_TYPES["Passive-house standard"]
    assert ROOF_TYPES["No loft insulation"] > ROOF_TYPES["270 mm loft insulation"]
    assert GLAZING_TYPES["Single glazed"] > GLAZING_TYPES["Triple glazed"]


def test_an_unknown_construction_is_assumed_to_be_the_worst_in_its_class():
    # The model must never flatter a house it does not recognise.
    assert u_value("wall", "Straw bale, unknown spec") == max(WALL_TYPES.values())
    assert u_value("roof", None) == max(ROOF_TYPES.values())


def test_unknown_airtightness_is_assumed_draughty():
    assert air_changes("Hermetically sealed") == max(AIRTIGHTNESS_LEVELS.values())


def test_every_heating_system_names_a_fuel_that_exists():
    for name, system in HEATING_SYSTEMS.items():
        assert system["fuel"] in FUELS, name
        if system["variable_efficiency"]:
            assert system["carnot_quality"] > 0, name
        else:
            assert 0 < system["efficiency"] <= 1.0, name


def test_every_measure_upgrades_to_a_construction_that_exists():
    for name, measure in RETROFIT_MEASURES.items():
        element = measure["element"]
        if element == "airtightness":
            assert measure["upgrade_to"] in AIRTIGHTNESS_LEVELS, name
        else:
            assert u_value(element, measure["upgrade_to"]) > 0, name
        assert measure["note"], name


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_envelope_areas_are_derived_from_floor_area(envelope):
    assert envelope["floor_area_m2"] == 120
    assert envelope["footprint_m2"] == 60
    assert envelope["roof_m2"] == envelope["ground_floor_m2"] == 60
    assert envelope["wall_m2"] > 0
    assert envelope["glazing_m2"] > 0
    assert envelope["volume_m3"] == 300


def test_a_mid_terrace_has_less_exposed_wall_than_a_detached_house():
    detached = estimate_envelope(120, storeys=2, attached_walls=0)
    semi = estimate_envelope(120, storeys=2, attached_walls=1)
    terrace = estimate_envelope(120, storeys=2, attached_walls=2)
    assert detached["wall_m2"] > semi["wall_m2"] > terrace["wall_m2"]


def test_envelope_survives_nonsense_input():
    result = estimate_envelope("big", storeys=0, attached_walls=99)
    assert result["floor_area_m2"] == 0.0
    assert result["storeys"] == 1
    assert result["attached_walls"] == 3
    assert result["wall_m2"] >= 0


# ---------------------------------------------------------------------------
# Heat loss
# ---------------------------------------------------------------------------

def test_heat_loss_is_fabric_plus_ventilation(envelope):
    result = heat_loss_coefficient(envelope, OLD_HOUSE)
    assert result["total_w_per_k"] == pytest.approx(
        result["fabric_w_per_k"] + result["ventilation_w_per_k"], abs=0.05
    )


def test_ventilation_loss_follows_the_documented_formula(envelope):
    result = heat_loss_coefficient(envelope, OLD_HOUSE)
    expected = AIR_HEAT_CAPACITY * air_changes(OLD_HOUSE["airtightness"]) * envelope["volume_m3"]
    assert result["ventilation_w_per_k"] == pytest.approx(expected, abs=0.05)


def test_an_insulated_house_loses_far_less_heat(envelope):
    old = heat_loss_coefficient(envelope, OLD_HOUSE)["total_w_per_k"]
    modern = heat_loss_coefficient(envelope, MODERN_HOUSE)["total_w_per_k"]
    assert modern < old / 2


def test_the_worst_element_is_identified(envelope):
    result = heat_loss_coefficient(envelope, OLD_HOUSE)
    assert result["worst_element"] in result["breakdown"]
    assert result["breakdown"][result["worst_element"]] == max(result["breakdown"].values())


def test_improving_one_element_only_reduces_that_element(envelope):
    before = heat_loss_coefficient(envelope, OLD_HOUSE)["breakdown"]
    after = heat_loss_coefficient(
        envelope, apply_measure(OLD_HOUSE, "Loft insulation to 270 mm")
    )["breakdown"]
    assert after["roof"] < before["roof"]
    assert after["wall"] == before["wall"]
    assert after["ventilation"] == before["ventilation"]


# ---------------------------------------------------------------------------
# Degree days and demand
# ---------------------------------------------------------------------------

def test_degree_days_are_zero_in_months_warmer_than_the_base_temperature():
    degree_days = monthly_degree_days("Mediterranean")
    climate = get_climate("Mediterranean")
    for index, temperature in enumerate(climate["monthly_c"]):
        if temperature >= BASE_TEMPERATURE_C:
            assert degree_days[index] == 0.0


def test_degree_days_match_the_definition():
    degree_days = monthly_degree_days("Temperate maritime")
    climate = get_climate("Temperate maritime")
    expected = max(0.0, BASE_TEMPERATURE_C - climate["monthly_c"][0]) * DAYS_IN_MONTH[0]
    assert degree_days[0] == pytest.approx(expected, abs=0.05)


def test_colder_climates_have_more_degree_days():
    zones = [zone["name"] for zone in list_climate_zones()]
    degree_days = [annual_degree_days(zone) for zone in zones]
    assert degree_days == sorted(degree_days, reverse=True)


def test_a_higher_base_temperature_means_more_heating():
    assert annual_degree_days("Temperate maritime", 18.0) > annual_degree_days(
        "Temperate maritime", 15.5
    )


def test_heat_demand_follows_the_degree_day_formula():
    demand = heat_demand_kwh(200, "Temperate maritime")
    expected = 200 * annual_degree_days("Temperate maritime") * 24 / 1000
    assert demand == pytest.approx(expected, abs=0.5)


def test_monthly_demand_sums_to_the_annual_figure():
    monthly = monthly_heat_demand_kwh(200, "Cold continental")
    assert sum(monthly) == pytest.approx(heat_demand_kwh(200, "Cold continental"), abs=1.0)


def test_summer_months_need_no_heating():
    monthly = monthly_heat_demand_kwh(200, "Mediterranean")
    assert monthly[6] == 0.0  # July


def test_peak_load_uses_the_design_temperature_not_the_average():
    hlc = 250
    peak = peak_heat_load_w(hlc, "Cold continental")
    assert peak == pytest.approx(hlc * (DESIGN_INDOOR_C - get_climate("Cold continental")["design_c"]), abs=1)
    assert peak > peak_heat_load_w(hlc, "Mediterranean")


# ---------------------------------------------------------------------------
# Heat pumps
# ---------------------------------------------------------------------------

def test_heat_pump_efficiency_falls_as_flow_temperature_rises():
    cool = seasonal_cop("Air source heat pump", "Temperate maritime", 35)
    warm = seasonal_cop("Air source heat pump", "Temperate maritime", 50)
    hot = seasonal_cop("Air source heat pump", "Temperate maritime", 65)
    assert cool > warm > hot


def test_heat_pump_efficiency_falls_in_a_colder_climate():
    mild = seasonal_cop("Air source heat pump", "Mediterranean", 45)
    cold = seasonal_cop("Air source heat pump", "Cold continental", 45)
    assert mild > cold


def test_ground_source_beats_air_source_in_the_cold():
    air = seasonal_cop("Air source heat pump", "Cold continental", 45)
    ground = seasonal_cop("Ground source heat pump", "Cold continental", 45)
    assert ground > air


def test_cop_is_bounded():
    for zone in CLIMATE_ZONES:
        for flow in (25, 35, 45, 55, 65, 75):
            for system in ("Air source heat pump", "Ground source heat pump"):
                cop = seasonal_cop(system, zone, flow)
                assert COP_FLOOR <= cop <= COP_CEILING


def test_boilers_have_no_cop():
    assert seasonal_cop("Gas boiler (condensing)", "Temperate maritime", 65) is None
    assert system_efficiency("Gas boiler (condensing)") == 0.88


def test_emitter_capacity_scales_with_oversizing():
    baseline = 8000
    for name, emitter in EMITTER_TYPES.items():
        assert emitter_capacity_w(baseline, name) == pytest.approx(
            baseline * emitter["oversize_factor"], abs=1
        )


def test_unchanged_radiators_need_the_reference_flow_temperature():
    peak = 9000
    assert required_flow_temperature(peak, DEFAULT_EMITTER, peak) == pytest.approx(
        REFERENCE_FLOW_C, abs=0.2
    )


def test_insulating_the_house_lets_the_same_radiators_run_cooler():
    baseline_peak = 9000
    after = required_flow_temperature(baseline_peak / 2, DEFAULT_EMITTER, baseline_peak)
    assert after < REFERENCE_FLOW_C - 10


def test_bigger_emitters_need_cooler_water_for_the_same_load():
    peak = 9000
    temperatures = [
        required_flow_temperature(peak, name, peak) for name in EMITTER_TYPES
    ]
    assert temperatures == sorted(temperatures, reverse=True)


def test_flow_temperature_is_bounded():
    assert required_flow_temperature(200000, DEFAULT_EMITTER, 1000) == MAX_FLOW_C
    assert required_flow_temperature(1, "Underfloor heating", 50000) == MIN_FLOW_C
    assert required_flow_temperature(0, DEFAULT_EMITTER) == MIN_FLOW_C


# ---------------------------------------------------------------------------
# Running cost and system comparison
# ---------------------------------------------------------------------------

def test_fuel_use_is_demand_divided_by_efficiency():
    result = running_cost_and_emissions(10000, "Gas boiler (condensing)")
    assert result["fuel_kwh"] == pytest.approx(10000 / 0.88, abs=1)
    assert result["co2_kg"] == pytest.approx(
        result["fuel_kwh"] * FUELS["Natural gas"]["kg_co2_per_kwh"], abs=1
    )


def test_a_heat_pump_uses_less_fuel_than_resistance_heating():
    pump = running_cost_and_emissions(10000, "Air source heat pump", "Temperate maritime", 45)
    resistive = running_cost_and_emissions(10000, "Electric resistance heating")
    assert pump["fuel_kwh"] < resistive["fuel_kwh"] / 2
    assert pump["co2_kg"] < resistive["co2_kg"]


def test_a_heat_pump_on_hot_water_can_lose_to_a_gas_boiler():
    # The honest failure case: forced to 65 C in a leaky house, a heat pump's
    # advantage over gas narrows sharply and can disappear on a dirty grid.
    hot = running_cost_and_emissions(20000, "Air source heat pump", "Cold continental", 65)
    gas = running_cost_and_emissions(20000, "Gas boiler (condensing)")
    cool = running_cost_and_emissions(20000, "Air source heat pump", "Cold continental", 35)
    assert cool["co2_kg"] < gas["co2_kg"]
    assert hot["co2_kg"] > cool["co2_kg"]


def test_system_comparison_is_ranked_by_emissions():
    rows = compare_systems(12000, "Temperate maritime", 45)
    assert [row["co2_kg"] for row in rows] == sorted(row["co2_kg"] for row in rows)
    assert len(rows) == len(HEATING_SYSTEMS)


def test_fuel_price_overrides_are_respected():
    default = running_cost_and_emissions(10000, "Gas boiler (condensing)")
    expensive = running_cost_and_emissions(
        10000, "Gas boiler (condensing)", fuel_overrides={"Natural gas": {"price_per_kwh": 0.20}}
    )
    assert expensive["cost"] > default["cost"]
    assert expensive["co2_kg"] == default["co2_kg"]


def test_an_unknown_system_is_reported_as_unknown():
    assert running_cost_and_emissions(10000, "Coal fire") is None


# ---------------------------------------------------------------------------
# Measures and plans
# ---------------------------------------------------------------------------

def test_cavity_insulation_does_not_apply_to_a_solid_wall():
    assert measure_applies("Cavity wall insulation", OLD_HOUSE) is False
    cavity_house = dict(OLD_HOUSE, wall="Cavity, uninsulated")
    assert measure_applies("Cavity wall insulation", cavity_house) is True


def test_a_measure_already_done_is_not_offered_again():
    assert measure_applies("Loft insulation to 270 mm", OLD_HOUSE) is True
    assert measure_applies("Loft insulation to 270 mm", MODERN_HOUSE) is False


def test_measure_cost_scales_with_the_area_it_covers():
    small = measure_cost("Loft insulation to 270 mm", estimate_envelope(60, 1))
    large = measure_cost("Loft insulation to 270 mm", estimate_envelope(200, 1))
    assert large > small
    assert small >= RETROFIT_MEASURES["Loft insulation to 270 mm"]["fixed_cost"]


def test_draught_proofing_is_a_flat_cost(envelope):
    assert measure_cost("Draught proofing", envelope) == RETROFIT_MEASURES[
        "Draught proofing"
    ]["fixed_cost"]


def test_measures_are_ranked_by_cost_per_kwh_saved(envelope):
    ranking = rank_measures(envelope, OLD_HOUSE)
    costs = [
        row["cost_per_kwh_saved"]
        for row in ranking["measures"]
        if row["cost_per_kwh_saved"] is not None
    ]
    assert costs == sorted(costs)
    assert all(row["saved_kwh"] > 0 for row in ranking["measures"])


def test_loft_insulation_outranks_triple_glazing_in_an_old_house(envelope):
    ranking = rank_measures(envelope, OLD_HOUSE)
    order = [row["measure"] for row in ranking["measures"]]
    assert order.index("Loft insulation to 270 mm") < order.index("Upgrade to triple glazing")


def test_a_modern_house_has_little_left_to_do(envelope):
    ranking = rank_measures(envelope, MODERN_HOUSE)
    assert len(ranking["measures"]) < len(rank_measures(envelope, OLD_HOUSE)["measures"])


def test_a_plan_never_increases_demand(envelope):
    plan = build_retrofit_plan(
        envelope, OLD_HOUSE,
        ["Loft insulation to 270 mm", "Draught proofing", "Floor insulation"],
    )
    assert plan["after"]["demand_kwh"] < plan["baseline"]["demand_kwh"]
    assert plan["demand_saved_kwh"] > 0
    assert 0 < plan["demand_saved_pct"] < 100


def test_plan_steps_are_monotonically_improving(envelope):
    plan = build_retrofit_plan(
        envelope, OLD_HOUSE,
        ["Loft insulation to 270 mm", "Floor insulation", "Replace single glazing with double"],
    )
    demands = [step["demand_kwh"] for step in plan["steps"]]
    assert demands == sorted(demands, reverse=True)
    assert all(step["step_saved_kwh"] > 0 for step in plan["steps"])


def test_step_savings_add_up_to_the_total(envelope):
    plan = build_retrofit_plan(
        envelope, OLD_HOUSE, ["Loft insulation to 270 mm", "Draught proofing"]
    )
    assert sum(step["step_saved_kwh"] for step in plan["steps"]) == pytest.approx(
        plan["demand_saved_kwh"], abs=0.5
    )
    assert sum(step["cost"] for step in plan["steps"]) == pytest.approx(
        plan["total_cost"], abs=0.5
    )


def test_repeating_a_measure_cannot_claim_the_saving_twice(envelope):
    once = build_retrofit_plan(envelope, OLD_HOUSE, ["Loft insulation to 270 mm"])
    twice = build_retrofit_plan(
        envelope, OLD_HOUSE, ["Loft insulation to 270 mm", "Loft insulation to 270 mm"]
    )
    assert twice["demand_saved_kwh"] == once["demand_saved_kwh"]
    assert twice["total_cost"] == once["total_cost"]
    assert len(twice["steps"]) == 1


def test_inapplicable_measures_are_skipped_silently(envelope):
    plan = build_retrofit_plan(
        envelope, OLD_HOUSE, ["Cavity wall insulation", "Loft insulation to 270 mm"]
    )
    assert [step["measure"] for step in plan["steps"]] == ["Loft insulation to 270 mm"]


def test_an_empty_plan_changes_nothing(envelope):
    plan = build_retrofit_plan(envelope, OLD_HOUSE, [])
    assert plan["steps"] == []
    assert plan["demand_saved_kwh"] == 0.0
    assert plan["total_cost"] == 0.0
    assert plan["payback_years"] is None
    assert plan["after"]["demand_kwh"] == plan["baseline"]["demand_kwh"]


def test_payback_is_reported_when_the_saving_is_real(envelope):
    plan = build_retrofit_plan(envelope, OLD_HOUSE, ["Loft insulation to 270 mm"])
    assert plan["payback_years"] is not None
    assert plan["payback_years"] > 0


# ---------------------------------------------------------------------------
# Fabric first
# ---------------------------------------------------------------------------

def test_fabric_work_improves_the_heat_pump_that_follows_it(envelope):
    check = fabric_first_check(
        envelope, OLD_HOUSE, "Temperate continental",
        measures=[
            "Loft insulation to 270 mm",
            "Internal solid wall insulation",
            "Draught proofing",
            "Floor insulation",
        ],
    )
    assert check["cop_after_fabric"] > check["cop_now"]
    assert check["flow_after_c"] < check["flow_now_c"]
    assert check["peak_after_w"] < check["peak_now_w"]
    assert check["electricity_after_kwh"] < check["electricity_now_kwh"]
    assert check["smaller_unit_kw"] > 0
    assert "Fabric first" in check["verdict"]


def test_an_already_efficient_house_is_told_it_can_proceed(envelope):
    check = fabric_first_check(envelope, MODERN_HOUSE, "Temperate maritime", measures=[])
    assert check["cop_gain"] == 0.0
    assert "not blocking" in check["verdict"]


def test_underfloor_heating_beats_unchanged_radiators_for_a_heat_pump(envelope):
    radiators = fabric_first_check(envelope, OLD_HOUSE, emitter="Existing radiators (unchanged)")
    underfloor = fabric_first_check(envelope, OLD_HOUSE, emitter="Underfloor heating")
    assert underfloor["cop_now"] > radiators["cop_now"]


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------

def test_advice_prompts_for_input_when_there_is_nothing_to_advise_on():
    assert "Describe your home" in get_retrofit_advice({})[0]


def test_advice_names_the_worst_element(envelope):
    plan = build_retrofit_plan(envelope, OLD_HOUSE, ["Loft insulation to 270 mm"])
    advice = " ".join(get_retrofit_advice(plan))
    assert any(word in advice for word in ("walls", "roof", "windows", "draughts", "ground floor"))


def test_advice_respects_the_limit(envelope):
    plan = build_retrofit_plan(envelope, OLD_HOUSE, ["Loft insulation to 270 mm"])
    assert len(get_retrofit_advice(plan, limit=2)) == 2
    assert get_retrofit_advice(plan, limit=0) == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_saved_plans_come_back_intact(envelope):
    plan = build_retrofit_plan(envelope, OLD_HOUSE, ["Loft insulation to 270 mm"])
    plan_id = save_retrofit_plan(1, "Victorian terrace", plan, floor_area_m2=120)
    assert plan_id is not None

    plans = get_retrofit_plans(1)
    assert len(plans) == 1
    assert plans[0]["plan_name"] == "Victorian terrace"
    assert plans[0]["floor_area_m2"] == 120
    assert plans[0]["final_demand_kwh"] < plans[0]["baseline_demand_kwh"]
    assert plans[0]["detail"]["steps"]
    assert plans[0]["detail"]["fabric_after"]["roof"] == "270 mm loft insulation"


def test_plans_are_scoped_to_their_owner(envelope):
    plan = build_retrofit_plan(envelope, OLD_HOUSE, ["Draught proofing"])
    save_retrofit_plan(1, "Mine", plan)
    save_retrofit_plan(2, "Theirs", plan)
    assert len(get_retrofit_plans(1)) == 1
    assert get_retrofit_plans(1)[0]["plan_name"] == "Mine"


def test_an_unnamed_plan_still_saves(envelope):
    plan = build_retrofit_plan(envelope, OLD_HOUSE, [])
    assert save_retrofit_plan(1, "  ", plan) is not None
    assert get_retrofit_plans(1)[0]["plan_name"] == "My home"


def test_deleting_a_plan_removes_only_that_plan(envelope):
    plan = build_retrofit_plan(envelope, OLD_HOUSE, [])
    keep = save_retrofit_plan(1, "Keep", plan)
    remove = save_retrofit_plan(1, "Remove", plan)
    assert delete_retrofit_plan(remove) is True
    remaining = get_retrofit_plans(1)
    assert len(remaining) == 1
    assert remaining[0]["id"] == keep


def test_deleting_a_missing_plan_reports_failure():
    assert delete_retrofit_plan(4242) is False


def test_the_plan_limit_is_honoured(envelope):
    plan = build_retrofit_plan(envelope, OLD_HOUSE, [])
    for index in range(5):
        save_retrofit_plan(1, f"Plan {index}", plan)
    assert len(get_retrofit_plans(1, limit=3)) == 3
