"""Tests for the Flight Footprint calculator."""
import os
import tempfile

import pytest

os.environ["ECO_BUDDY_DB"] = ":memory:"

import flight_footprint
from flight_footprint import (
    AIRPORTS,
    CABIN_CLASSES,
    CLIMB_DESCENT_KM,
    CRUISE_KG_PER_KM,
    DEFAULT_CABIN,
    DEFAULT_LOAD_FACTOR,
    LTO_KG_PER_LEG,
    MEDIUM_HAUL_MAX_KM,
    PERSONAL_ANNUAL_BUDGET_KG,
    RADIATIVE_FORCING_MULTIPLIER,
    RF_RANGE,
    ROUTING_UPLIFT_KM,
    SHORT_HAUL_CABIN_CAP,
    SHORT_HAUL_MAX_KM,
    SURFACE_ALTERNATIVES,
    VIDEO_CALL_KG,
    annual_summary,
    budget_share,
    cabin_multiplier,
    compare_cabins,
    compare_routings,
    compare_to_alternatives,
    delete_trip,
    estimate_route,
    estimate_trip,
    get_airport,
    get_reduction_tips,
    get_trips,
    haul_type,
    haversine_km,
    leg_emissions,
    list_airports,
    list_cabin_classes,
    route_distance_km,
    save_trip,
    trips_within_budget,
)


@pytest.fixture(autouse=True)
def temp_db():
    """Point the module at a throwaway SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    original = flight_footprint.DB_NAME
    flight_footprint.DB_NAME = db_path
    yield db_path
    flight_footprint.DB_NAME = original
    try:
        os.unlink(db_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def test_every_airport_has_usable_coordinates():
    for code, details in AIRPORTS.items():
        assert len(code) == 3, code
        assert -90 <= details["lat"] <= 90, code
        assert -180 <= details["lon"] <= 180, code
        assert details["name"] and details["city"] and details["country"], code


def test_airport_lookup_is_case_insensitive():
    assert get_airport("lhr")["code"] == "LHR"
    assert get_airport(" jfk ")["city"] == "New York"


def test_unknown_airport_returns_none_rather_than_guessing():
    assert get_airport("ZZZ") is None
    assert get_airport("") is None
    assert get_airport(None) is None


def test_airports_are_listed_in_code_order():
    codes = [airport["code"] for airport in list_airports()]
    assert codes == sorted(codes)
    assert len(codes) == len(AIRPORTS)


def test_cabin_classes_are_ordered_by_seat_space():
    multipliers = [cabin["multiplier"] for cabin in list_cabin_classes()]
    assert multipliers == sorted(multipliers)
    assert list_cabin_classes()[0]["name"] == "Economy"


def test_economy_is_the_baseline_cabin():
    assert CABIN_CLASSES["Economy"]["multiplier"] == 1.0
    assert all(info["multiplier"] >= 1.0 for info in CABIN_CLASSES.values())


def test_radiative_forcing_multiplier_sits_inside_the_published_range():
    assert RF_RANGE[0] <= RADIATIVE_FORCING_MULTIPLIER <= RF_RANGE[1]


def test_longer_haul_aircraft_are_more_efficient_per_kilometre():
    assert CRUISE_KG_PER_KM["short"] > CRUISE_KG_PER_KM["medium"]
    assert CRUISE_KG_PER_KM["medium"] > CRUISE_KG_PER_KM["long"]


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def test_haversine_is_zero_for_the_same_point():
    assert haversine_km(51.47, -0.45, 51.47, -0.45) == 0.0


def test_haversine_matches_a_known_route():
    # London Heathrow to New York JFK is about 5,555 km great-circle.
    distance = haversine_km(51.4700, -0.4543, 40.6413, -73.7781)
    assert 5450 <= distance <= 5650


def test_haversine_is_symmetric():
    there = haversine_km(1.3644, 103.9915, -33.9399, 151.1753)
    back = haversine_km(-33.9399, 151.1753, 1.3644, 103.9915)
    assert there == back


def test_antipodal_points_are_half_the_circumference():
    distance = haversine_km(0.0, 0.0, 0.0, 180.0)
    assert 20000 <= distance <= 20040


def test_route_distance_adds_the_routing_uplift():
    great_circle = haversine_km(
        AIRPORTS["LHR"]["lat"], AIRPORTS["LHR"]["lon"],
        AIRPORTS["JFK"]["lat"], AIRPORTS["JFK"]["lon"],
    )
    assert route_distance_km("LHR", "JFK") == pytest.approx(
        great_circle + ROUTING_UPLIFT_KM, abs=0.2
    )


def test_route_distance_without_uplift_is_the_great_circle():
    assert route_distance_km("LHR", "CDG", apply_uplift=False) < route_distance_km(
        "LHR", "CDG"
    )


def test_unknown_route_returns_none_so_the_ui_can_ask_for_a_distance():
    assert route_distance_km("LHR", "ZZZ") is None
    assert route_distance_km("ZZZ", "LHR") is None


def test_haul_classification_follows_the_distance_bands():
    assert haul_type(400) == "short"
    assert haul_type(SHORT_HAUL_MAX_KM) == "short"
    assert haul_type(SHORT_HAUL_MAX_KM + 1) == "medium"
    assert haul_type(MEDIUM_HAUL_MAX_KM) == "medium"
    assert haul_type(MEDIUM_HAUL_MAX_KM + 1) == "long"


def test_haul_classification_survives_junk_input():
    assert haul_type(None) == "short"
    assert haul_type("nonsense") == "short"
    assert haul_type(-500) == "short"


# ---------------------------------------------------------------------------
# Leg emissions
# ---------------------------------------------------------------------------

def test_a_zero_distance_leg_emits_nothing():
    leg = leg_emissions(0)
    assert leg["co2_kg"] == 0.0
    assert leg["co2e_kg"] == 0.0
    assert leg["non_co2_kg"] == 0.0


def test_leg_emissions_match_the_documented_formula():
    leg = leg_emissions(2000, cabin="Economy", load_factor=DEFAULT_LOAD_FACTOR)
    cruise_km = 2000 - CLIMB_DESCENT_KM
    expected = (
        LTO_KG_PER_LEG
        + cruise_km * CRUISE_KG_PER_KM["medium"]
        + CLIMB_DESCENT_KM * CRUISE_KG_PER_KM["medium"]
    )
    assert leg["co2_kg"] == pytest.approx(expected, rel=1e-3)
    assert leg["cruise_km"] == cruise_km


def test_short_hops_burn_more_per_kilometre_than_long_hauls():
    short = leg_emissions(400)["kg_per_km"]
    medium = leg_emissions(2500)["kg_per_km"]
    long_haul = leg_emissions(9000)["kg_per_km"]
    assert short > medium > long_haul


def test_a_flight_that_never_reaches_cruise_has_no_contrail_penalty():
    leg = leg_emissions(CLIMB_DESCENT_KM - 50)
    assert leg["cruise_km"] == 0.0
    assert leg["non_co2_kg"] == 0.0
    assert leg["co2e_kg"] == leg["co2_kg"]


def test_non_co2_warming_applies_only_to_the_cruise_share():
    leg = leg_emissions(6000)
    expected = leg["cruise_kg"] * (RADIATIVE_FORCING_MULTIPLIER - 1.0)
    assert leg["non_co2_kg"] == pytest.approx(expected, rel=1e-3)
    assert leg["co2e_kg"] > leg["co2_kg"]


def test_business_class_costs_about_three_economy_seats_on_long_haul():
    economy = leg_emissions(8000, cabin="Economy")["co2e_kg"]
    business = leg_emissions(8000, cabin="Business")["co2e_kg"]
    assert business == pytest.approx(economy * CABIN_CLASSES["Business"]["multiplier"], rel=1e-3)


def test_premium_cabins_are_capped_on_short_haul_aircraft():
    assert cabin_multiplier("First", 600) == SHORT_HAUL_CABIN_CAP
    assert cabin_multiplier("Business", 600) == SHORT_HAUL_CABIN_CAP
    assert cabin_multiplier("Economy", 600) == 1.0


def test_unknown_cabin_falls_back_to_economy():
    assert cabin_multiplier("Sleeper Pod", 8000) == CABIN_CLASSES[DEFAULT_CABIN]["multiplier"]
    assert leg_emissions(8000, cabin="Sleeper Pod")["cabin"] == DEFAULT_CABIN


def test_an_emptier_aircraft_raises_the_per_passenger_footprint():
    full = leg_emissions(5000, load_factor=1.0)["co2e_kg"]
    typical = leg_emissions(5000, load_factor=DEFAULT_LOAD_FACTOR)["co2e_kg"]
    half_empty = leg_emissions(5000, load_factor=0.5)["co2e_kg"]
    assert full < typical < half_empty


def test_load_factor_is_clamped_to_a_sane_range():
    absurd = leg_emissions(5000, load_factor=0.01)["load_factor"]
    assert absurd == 0.3
    assert leg_emissions(5000, load_factor=5)["load_factor"] == 1.0


def test_leg_emissions_are_monotonic_in_distance():
    previous = 0.0
    for distance in (200, 800, 1500, 3000, 6000, 12000):
        current = leg_emissions(distance)["co2e_kg"]
        assert current > previous
        previous = current


# ---------------------------------------------------------------------------
# Trips
# ---------------------------------------------------------------------------

def test_a_round_trip_is_exactly_twice_a_one_way():
    one_way = estimate_trip([5000], round_trip=False)
    both_ways = estimate_trip([5000], round_trip=True)
    assert both_ways["co2e_kg"] == pytest.approx(one_way["co2e_kg"] * 2, rel=1e-6)
    assert both_ways["leg_count"] == one_way["leg_count"] * 2


def test_two_passengers_double_the_trip_footprint():
    solo = estimate_trip([3000], passengers=1)
    pair = estimate_trip([3000], passengers=2)
    assert pair["co2e_kg"] == pytest.approx(solo["co2e_kg"] * 2, rel=1e-6)


def test_turning_radiative_forcing_off_reports_co2_only():
    with_rf = estimate_trip([6000], include_radiative_forcing=True)
    without_rf = estimate_trip([6000], include_radiative_forcing=False)
    assert without_rf["non_co2_kg"] == 0.0
    assert without_rf["co2e_kg"] == without_rf["co2_kg"]
    assert with_rf["co2e_kg"] > without_rf["co2e_kg"]


def test_a_connection_costs_more_than_one_leg_of_the_same_total_distance():
    direct = estimate_trip([4000], round_trip=False)
    connecting = estimate_trip([2000, 2000], round_trip=False)
    assert connecting["co2e_kg"] > direct["co2e_kg"]
    assert connecting["distance_km"] == direct["distance_km"]


def test_empty_and_junk_legs_are_dropped():
    trip = estimate_trip([0, None, "banana", 1200])
    assert trip["leg_count"] == 2  # the single valid leg, there and back
    assert trip["co2e_kg"] > 0


def test_a_trip_with_no_valid_legs_is_zero_not_an_error():
    trip = estimate_trip([])
    assert trip["co2e_kg"] == 0.0
    assert trip["leg_count"] == 0


def test_estimate_route_uses_the_airport_table():
    trip = estimate_route("LHR", "JFK", round_trip=True)
    assert trip["route"] == ["LHR", "JFK"]
    assert trip["distance_km"] == pytest.approx(route_distance_km("LHR", "JFK") * 2, abs=0.5)
    assert trip["co2e_kg"] > 0


def test_estimate_route_rejects_unknown_airports():
    assert estimate_route("LHR", "ZZZ") is None
    assert estimate_route("LHR", "JFK", via=["ZZZ"]) is None


def test_routing_via_a_hub_never_reduces_distance():
    for hub in ("DXB", "FRA", "IST", "AMS"):
        comparison = compare_routings("LHR", "SIN", hub)
        assert comparison["extra_km"] >= 0, hub
        assert comparison["extra_takeoffs"] == 2  # one per direction


def test_a_detour_hub_costs_measurably_more():
    comparison = compare_routings("LHR", "JFK", "DXB")
    assert comparison["extra_km"] > 1000
    assert comparison["extra_kg"] > 0
    assert comparison["extra_pct"] > 0


def test_cabin_comparison_is_ordered_and_economy_anchored():
    options = compare_cabins(9000)
    assert options[0]["cabin"] == "Economy"
    assert options[0]["vs_economy_kg"] == 0.0
    assert [option["co2e_kg"] for option in options] == sorted(
        option["co2e_kg"] for option in options
    )
    assert options[-1]["vs_economy_kg"] > 0


# ---------------------------------------------------------------------------
# Alternatives and budget
# ---------------------------------------------------------------------------

def test_alternatives_are_ranked_lowest_carbon_first():
    trip = estimate_trip([600], round_trip=False)
    rows = compare_to_alternatives(600, trip["co2e_kg"])
    assert [row["co2e_kg"] for row in rows] == sorted(row["co2e_kg"] for row in rows)
    assert rows[0]["mode"] == "Video call instead"
    assert rows[0]["co2e_kg"] == VIDEO_CALL_KG


def test_rail_beats_flying_over_a_short_distance():
    trip = estimate_trip([500], round_trip=False)
    rows = {row["mode"]: row for row in compare_to_alternatives(500, trip["co2e_kg"])}
    assert rows["High-speed rail"]["saving_pct"] > 80
    assert rows["High-speed rail"]["plausible_at_this_distance"] is True


def test_surface_options_are_flagged_implausible_across_an_ocean():
    trip = estimate_trip([9000], round_trip=False)
    rows = {row["mode"]: row for row in compare_to_alternatives(9000, trip["co2e_kg"])}
    assert rows["Intercity rail"]["plausible_at_this_distance"] is False
    assert rows["Long-distance coach"]["plausible_at_this_distance"] is False
    assert rows["Video call instead"]["plausible_at_this_distance"] is True


def test_savings_never_go_negative():
    rows = compare_to_alternatives(9000, 10.0)
    assert all(row["saving_kg"] >= 0 for row in rows)


def test_shared_surface_transport_beats_flying_per_kilometre():
    per_km = leg_emissions(800)["kg_per_km"]
    shared = ["High-speed rail", "Intercity rail", "Long-distance coach", "Car (3 occupants)"]
    assert all(SURFACE_ALTERNATIVES[mode] < per_km for mode in shared)


def test_driving_alone_is_not_automatically_better_than_flying():
    # Worth keeping honest: a solo car over a medium distance can beat a full
    # aircraft on paper, so the model must not pretend driving always wins.
    per_km = leg_emissions(800)["kg_per_km"]
    assert SURFACE_ALTERNATIVES["Car (1 occupant)"] > per_km


def test_budget_share_is_a_percentage_of_the_annual_allowance():
    assert budget_share(PERSONAL_ANNUAL_BUDGET_KG) == 100.0
    assert budget_share(PERSONAL_ANNUAL_BUDGET_KG / 2) == 50.0
    assert budget_share(0) == 0.0
    assert budget_share(-100) == 0.0


def test_trips_within_budget_counts_whole_journeys():
    assert trips_within_budget(PERSONAL_ANNUAL_BUDGET_KG) == 1.0
    assert trips_within_budget(PERSONAL_ANNUAL_BUDGET_KG / 4) == 4.0
    assert trips_within_budget(0) is None


def test_annual_summary_adds_up_and_finds_the_worst_trip():
    trips = [
        {"label": "Weekend break", "co2_kg": 200.0, "co2e_kg": 300.0, "distance_km": 1800},
        {"label": "Long haul holiday", "co2_kg": 1200.0, "co2e_kg": 1900.0, "distance_km": 17000},
        {"label": "Work trip", "co2_kg": 150.0, "co2e_kg": 220.0, "distance_km": 1400},
    ]
    summary = annual_summary(trips)
    assert summary["trip_count"] == 3
    assert summary["co2e_kg"] == 2420.0
    assert summary["co2_kg"] == 1550.0
    assert summary["non_co2_kg"] == 870.0
    assert summary["biggest_trip"]["label"] == "Long haul holiday"
    assert summary["biggest_trip_share_pct"] > 50
    assert summary["over_budget"] is True


def test_annual_summary_of_nothing_is_empty_not_broken():
    summary = annual_summary([])
    assert summary["trip_count"] == 0
    assert summary["co2e_kg"] == 0.0
    assert summary["biggest_trip"] is None
    assert summary["average_kg_per_trip"] == 0.0
    assert summary["over_budget"] is False


# ---------------------------------------------------------------------------
# Tips
# ---------------------------------------------------------------------------

def test_tips_prompt_for_input_when_there_is_nothing_to_advise_on():
    tips = get_reduction_tips(annual_summary([]))
    assert len(tips) == 1
    assert "Add a flight" in tips[0]


def test_tips_call_out_a_dominant_trip():
    trips = [
        {"label": "Sydney holiday", "co2_kg": 2000.0, "co2e_kg": 3400.0},
        {"label": "Short hop", "co2_kg": 60.0, "co2e_kg": 80.0},
    ]
    tips = " ".join(get_reduction_tips(annual_summary(trips)))
    assert "Sydney holiday" in tips


def test_tips_respect_the_limit():
    summary = annual_summary([{"label": "Trip", "co2_kg": 100.0, "co2e_kg": 150.0}])
    assert len(get_reduction_tips(summary, limit=3)) == 3
    assert get_reduction_tips(summary, limit=0) == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_saved_trips_come_back_intact():
    estimate = estimate_route("LHR", "JFK", cabin="Business")
    trip_id = save_trip(1, "New York work trip", estimate)
    assert trip_id is not None

    trips = get_trips(1)
    assert len(trips) == 1
    assert trips[0]["label"] == "New York work trip"
    assert trips[0]["route"] == "LHR - JFK"
    assert trips[0]["cabin"] == "Business"
    assert trips[0]["round_trip"] is True
    assert trips[0]["co2e_kg"] == estimate["co2e_kg"]
    assert trips[0]["detail"]["legs"]


def test_trips_are_scoped_to_their_owner():
    save_trip(1, "Mine", estimate_route("LHR", "CDG"))
    save_trip(2, "Theirs", estimate_route("LHR", "MAD"))
    assert len(get_trips(1)) == 1
    assert len(get_trips(2)) == 1
    assert get_trips(1)[0]["label"] == "Mine"


def test_an_unnamed_trip_still_saves():
    trip_id = save_trip(1, "   ", estimate_route("DEL", "BOM"))
    assert trip_id is not None
    assert get_trips(1)[0]["label"] == "Flight"


def test_a_manual_distance_trip_saves_without_a_route():
    trip_id = save_trip(1, "Charter", estimate_trip([1200]))
    assert trip_id is not None
    assert get_trips(1)[0]["route"] == "Manual distance"


def test_deleting_a_trip_removes_only_that_trip():
    first = save_trip(1, "Keep", estimate_route("LHR", "AMS"))
    second = save_trip(1, "Remove", estimate_route("LHR", "ZRH"))
    assert delete_trip(second) is True
    remaining = get_trips(1)
    assert len(remaining) == 1
    assert remaining[0]["id"] == first


def test_deleting_a_missing_trip_reports_failure():
    assert delete_trip(9999) is False


def test_the_trip_limit_is_honoured():
    for index in range(6):
        save_trip(1, f"Trip {index}", estimate_route("LHR", "BCN"))
    assert len(get_trips(1, limit=4)) == 4
