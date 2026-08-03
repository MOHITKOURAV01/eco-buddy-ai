"""Tests for the Avoidable Food Waste tracker."""
import os
import tempfile

import pytest

os.environ["ECO_BUDDY_DB"] = ":memory:"

import food_waste
from food_waste import (
    DEFAULT_DISPOSAL,
    DEFAULT_STORAGE,
    DISPOSAL_ROUTES,
    FOOD_ITEMS,
    MAX_SPOILAGE_RISK,
    MEAL_KG,
    STORAGE_LOCATIONS,
    WEEKS_PER_YEAR,
    at_risk_items,
    avoidable_split,
    best_storage,
    compare_disposal_routes,
    delete_waste_entry,
    disposal_factor,
    get_food_item,
    get_waste_log,
    get_waste_tips,
    list_categories,
    list_disposal_routes,
    list_food_items,
    log_waste,
    over_purchase_diagnosis,
    shelf_life_days,
    spoilage_risk,
    summarise_log,
    undercount_vs_disposal_only,
    waste_footprint,
)


@pytest.fixture(autouse=True)
def temp_db():
    """Point the module at a throwaway SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    original = food_waste.DB_NAME
    food_waste.DB_NAME = db_path
    yield db_path
    food_waste.DB_NAME = original
    try:
        os.unlink(db_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def test_every_food_item_is_fully_specified():
    for name, item in FOOD_ITEMS.items():
        assert item["co2_per_kg"] > 0, name
        assert item["water_per_kg"] > 0, name
        assert item["price_per_kg"] > 0, name
        assert item["pantry_days"] > 0, name
        assert 0 <= item["unavoidable"] < 1, name
        assert item["category"], name
        assert isinstance(item["freezable"], bool), name


def test_items_are_listed_worst_first():
    footprints = [item["co2_per_kg"] for item in list_food_items()]
    assert footprints == sorted(footprints, reverse=True)
    assert list_food_items()[0]["name"] == "Beef"


def test_items_can_be_filtered_by_category():
    meat = list_food_items("Meat & fish")
    assert meat
    assert all(item["category"] == "Meat & fish" for item in meat)


def test_categories_cover_the_catalogue():
    categories = list_categories()
    assert categories == sorted(categories)
    assert set(categories) == {item["category"] for item in FOOD_ITEMS.values()}


def test_unknown_food_returns_none():
    assert get_food_item("Ambrosia") is None


def test_disposal_routes_are_listed_best_first():
    factors = [route["co2_per_kg"] for route in list_disposal_routes()]
    assert factors == sorted(factors)
    assert list_disposal_routes()[-1]["name"] == "Landfill"


def test_landfill_is_the_worst_disposal_route():
    assert DISPOSAL_ROUTES["Landfill"]["co2_per_kg"] == max(
        route["co2_per_kg"] for route in DISPOSAL_ROUTES.values()
    )


def test_an_unknown_disposal_route_assumes_landfill():
    assert disposal_factor("Thrown in a hedge") == DISPOSAL_ROUTES[DEFAULT_DISPOSAL]["co2_per_kg"]


# ---------------------------------------------------------------------------
# Shelf life and spoilage
# ---------------------------------------------------------------------------

def test_the_fridge_beats_the_counter_and_the_freezer_beats_both():
    counter = shelf_life_days("Chicken", "Counter / pantry")
    fridge = shelf_life_days("Chicken", "Fridge")
    freezer = shelf_life_days("Chicken", "Freezer")
    assert counter < fridge < freezer


def test_freezing_is_not_offered_for_food_it_would_ruin():
    # A frozen salad is not a salad, so it gets fridge life, not freezer life.
    assert shelf_life_days("Leafy salad", "Freezer") == shelf_life_days("Leafy salad", "Fridge")
    assert get_food_item("Leafy salad")["freezable"] is False


def test_shelf_life_of_unknown_food_is_zero_not_a_guess():
    assert shelf_life_days("Ambrosia") == 0.0
    assert best_storage("Ambrosia") is None


def test_best_storage_reports_the_gain_over_the_counter():
    result = best_storage("Bread")
    assert result["best"] == "Freezer"
    assert result["gain_days"] > 0
    assert len(result["options"]) == len(STORAGE_LOCATIONS)


def test_spoilage_risk_rises_with_time():
    risks = [spoilage_risk("Milk", days, "Fridge") for days in (0, 1, 2, 3, 5)]
    assert risks == sorted(risks)
    assert risks[0] == 0.0


def test_spoilage_risk_is_bounded():
    for days in (0, 1, 10, 100, 5000):
        risk = spoilage_risk("Berries", days, "Fridge")
        assert 0.0 <= risk <= MAX_SPOILAGE_RISK


def test_fresh_food_is_not_reported_as_half_spoiled():
    # Nothing is half rotten on day one of seven.
    life = shelf_life_days("Apples", "Fridge")
    assert spoilage_risk("Apples", life * 0.25, "Fridge") < 0.05


def test_food_well_past_its_life_is_nearly_certain_to_be_spoiled():
    life = shelf_life_days("Milk", "Fridge")
    assert spoilage_risk("Milk", life * 3, "Fridge") >= 0.9


def test_freezing_keeps_risk_low_far_longer():
    assert spoilage_risk("Bread", 20, "Freezer") < spoilage_risk("Bread", 20, "Counter / pantry")


def test_spoilage_risk_of_unknown_food_is_zero():
    assert spoilage_risk("Ambrosia", 5) == 0.0


def test_at_risk_items_are_ranked_by_carbon_at_stake():
    inventory = [
        {"item": "Beef", "kg": 0.5, "days_held": 4, "storage": "Fridge"},
        {"item": "Apples", "kg": 1.0, "days_held": 4, "storage": "Fridge"},
        {"item": "Ambrosia", "kg": 5.0, "days_held": 4},
    ]
    ranked = at_risk_items(inventory)
    assert [entry["item"] for entry in ranked] == ["Beef", "Apples"]
    assert ranked[0]["co2_at_risk_kg"] > ranked[1]["co2_at_risk_kg"]
    assert ranked[0]["value_at_risk"] > 0


def test_at_risk_items_respect_a_limit():
    inventory = [
        {"item": "Beef", "kg": 0.5, "days_held": 4},
        {"item": "Apples", "kg": 1.0, "days_held": 4},
    ]
    assert len(at_risk_items(inventory, limit=1)) == 1
    assert at_risk_items([]) == []


# ---------------------------------------------------------------------------
# Avoidable vs unavoidable
# ---------------------------------------------------------------------------

def test_peel_and_bones_are_separated_from_edible_waste():
    split = avoidable_split("Bananas", 1.0)
    assert split["unavoidable_kg"] == pytest.approx(FOOD_ITEMS["Bananas"]["unavoidable"], abs=1e-3)
    assert split["avoidable_kg"] + split["unavoidable_kg"] == pytest.approx(1.0, abs=1e-3)


def test_food_with_no_inedible_part_is_entirely_avoidable():
    split = avoidable_split("Milk", 2.0)
    assert split["unavoidable_kg"] == 0.0
    assert split["avoidable_kg"] == 2.0


def test_avoidable_split_needs_a_known_food():
    assert avoidable_split("Ambrosia", 1.0) is None


# ---------------------------------------------------------------------------
# The footprint
# ---------------------------------------------------------------------------

def test_production_dominates_disposal_for_meat():
    result = waste_footprint("Beef", 1.0, "Landfill")
    assert result["production_co2_kg"] > result["disposal_co2_kg"] * 20
    assert result["production_share_pct"] > 90


def test_the_footprint_is_production_plus_disposal():
    result = waste_footprint("Cheese", 0.5, "Home compost")
    assert result["co2_kg"] == pytest.approx(
        result["production_co2_kg"] + result["disposal_co2_kg"], abs=1e-3
    )


def test_only_the_edible_share_is_charged_to_production_by_default():
    result = waste_footprint("Bananas", 1.0)
    item = get_food_item("Bananas")
    expected = (1.0 - item["unavoidable"]) * item["co2_per_kg"]
    assert result["production_co2_kg"] == pytest.approx(expected, abs=1e-3)


def test_disposal_emissions_apply_to_the_peel_too():
    result = waste_footprint("Bananas", 1.0, "Landfill")
    assert result["disposal_co2_kg"] == pytest.approx(
        1.0 * DISPOSAL_ROUTES["Landfill"]["co2_per_kg"], abs=1e-3
    )


def test_counting_the_inedible_share_raises_the_figure():
    edible_only = waste_footprint("Chicken", 1.0)
    everything = waste_footprint("Chicken", 1.0, count_unavoidable=True)
    assert everything["production_co2_kg"] > edible_only["production_co2_kg"]


def test_composting_beats_landfill_but_changes_less_than_expected():
    landfill = waste_footprint("Beef", 1.0, "Landfill")
    compost = waste_footprint("Beef", 1.0, "Home compost")
    assert compost["co2_kg"] < landfill["co2_kg"]
    # The saving is real but small next to the production footprint.
    assert (landfill["co2_kg"] - compost["co2_kg"]) < landfill["co2_kg"] * 0.1


def test_water_and_money_track_the_edible_share():
    result = waste_footprint("Berries", 0.5)
    item = get_food_item("Berries")
    edible = 0.5 * (1 - item["unavoidable"])
    assert result["water_litres"] == pytest.approx(edible * item["water_per_kg"], abs=1)
    assert result["money"] == pytest.approx(edible * item["price_per_kg"], abs=0.05)


def test_waste_is_expressed_in_meals():
    result = waste_footprint("Cooked leftovers", MEAL_KG * 4)
    assert result["meals_equivalent"] == pytest.approx(4.0, abs=0.05)


def test_zero_waste_costs_nothing():
    result = waste_footprint("Beef", 0)
    assert result["co2_kg"] == 0.0
    assert result["money"] == 0.0


def test_junk_quantities_do_not_crash_the_model():
    assert waste_footprint("Beef", None)["co2_kg"] == 0.0
    assert waste_footprint("Beef", "a lot")["co2_kg"] == 0.0
    assert waste_footprint("Beef", -5)["co2_kg"] == 0.0


def test_footprint_needs_a_known_food():
    assert waste_footprint("Ambrosia", 1.0) is None


def test_disposal_comparison_is_ranked_and_complete():
    rows = compare_disposal_routes("Bread", 1.0)
    assert len(rows) == len(DISPOSAL_ROUTES)
    assert [row["total_co2_kg"] for row in rows] == sorted(row["total_co2_kg"] for row in rows)
    assert rows[-1]["route"] == "Landfill"


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def test_a_log_summary_adds_up():
    entries = [
        waste_footprint("Beef", 0.4, "Landfill"),
        waste_footprint("Bread", 0.6, "Home compost"),
        waste_footprint("Leafy salad", 0.2, "Kerbside food collection"),
    ]
    summary = summarise_log(entries)
    assert summary["entry_count"] == 3
    assert summary["total_kg"] == pytest.approx(1.2, abs=0.01)
    assert summary["co2_kg"] == pytest.approx(
        sum(entry["co2_kg"] for entry in entries), abs=0.01
    )
    assert summary["worst_item"]["item"] == "Beef"


def test_categories_are_rolled_up():
    entries = [
        waste_footprint("Beef", 0.3),
        waste_footprint("Chicken", 0.3),
        waste_footprint("Bread", 0.3),
    ]
    summary = summarise_log(entries)
    assert set(summary["by_category"]) == {"Meat & fish", "Bakery & grains"}
    assert summary["by_category"]["Meat & fish"]["kg"] == pytest.approx(0.6, abs=0.01)


def test_annual_figures_scale_from_the_period_logged():
    entries = [waste_footprint("Bread", 1.0)]
    one_week = summarise_log(entries, weeks=1)
    four_weeks = summarise_log(entries, weeks=4)
    assert one_week["annual_co2_kg"] == pytest.approx(
        one_week["co2_kg"] * WEEKS_PER_YEAR, abs=0.5
    )
    assert four_weeks["annual_co2_kg"] < one_week["annual_co2_kg"]


def test_an_empty_log_is_empty_not_broken():
    summary = summarise_log([])
    assert summary["entry_count"] == 0
    assert summary["co2_kg"] == 0.0
    assert summary["worst_item"] is None
    assert summary["by_category"] == {}


def test_the_disposal_only_figure_is_shown_to_be_an_undercount():
    entries = [waste_footprint("Beef", 1.0, "Landfill")]
    comparison = undercount_vs_disposal_only(summarise_log(entries))
    assert comparison["multiple"] > 10
    assert comparison["missing_kg"] > 0
    assert comparison["full_kg"] > comparison["disposal_only_kg"]


def test_undercount_needs_something_to_compare():
    assert undercount_vs_disposal_only(summarise_log([])) is None


# ---------------------------------------------------------------------------
# Over-purchasing
# ---------------------------------------------------------------------------

def test_heavy_waste_is_diagnosed_as_a_buying_problem():
    result = over_purchase_diagnosis(bought_kg=20, wasted_kg=6)
    assert result["waste_share_pct"] == 30.0
    assert "buying problem" in result["verdict"]


def test_average_waste_is_named_as_average():
    result = over_purchase_diagnosis(bought_kg=20, wasted_kg=3)
    assert "household average" in result["verdict"]


def test_low_waste_is_acknowledged():
    result = over_purchase_diagnosis(bought_kg=20, wasted_kg=1)
    assert "below the household average" in result["verdict"]


def test_waste_cannot_exceed_what_was_bought():
    result = over_purchase_diagnosis(bought_kg=5, wasted_kg=50)
    assert result["wasted_kg"] == 5
    assert result["eaten_kg"] == 0.0
    assert result["waste_share_pct"] == 100.0


def test_no_shopping_recorded_prompts_for_input():
    assert "Enter what you bought" in over_purchase_diagnosis(0, 0)["verdict"]


# ---------------------------------------------------------------------------
# Tips
# ---------------------------------------------------------------------------

def test_tips_prompt_for_input_when_there_is_nothing_to_advise_on():
    assert "Log something" in get_waste_tips(summarise_log([]))[0]


def test_tips_explain_that_composting_cannot_recover_production():
    summary = summarise_log([waste_footprint("Beef", 1.0, "Landfill")])
    tips = " ".join(get_waste_tips(summary))
    assert "no bin can recover it" in tips


def test_tips_name_the_worst_item():
    summary = summarise_log(
        [waste_footprint("Beef", 1.0), waste_footprint("Apples", 1.0)]
    )
    assert "Beef" in " ".join(get_waste_tips(summary))


def test_tips_surface_items_about_to_spoil():
    summary = summarise_log([waste_footprint("Bread", 0.3)])
    risks = at_risk_items([{"item": "Milk", "kg": 1.0, "days_held": 4, "storage": "Fridge"}])
    tips = " ".join(get_waste_tips(summary, risks))
    assert "Use or freeze" in tips


def test_tips_respect_the_limit():
    summary = summarise_log([waste_footprint("Bread", 0.3)])
    assert len(get_waste_tips(summary, limit=2)) == 2
    assert get_waste_tips(summary, limit=0) == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_logged_waste_comes_back_intact():
    footprint = waste_footprint("Beef", 0.4, "Landfill")
    entry_id = log_waste(1, footprint, reason="Forgot it was in the fridge")
    assert entry_id is not None

    entries = get_waste_log(1)
    assert len(entries) == 1
    assert entries[0]["item"] == "Beef"
    assert entries[0]["category"] == "Meat & fish"
    assert entries[0]["disposal"] == "Landfill"
    assert entries[0]["reason"] == "Forgot it was in the fridge"
    assert entries[0]["co2_kg"] == footprint["co2_kg"]
    assert entries[0]["meals_equivalent"] > 0


def test_a_logged_entry_can_be_summarised_again():
    log_waste(1, waste_footprint("Beef", 0.4))
    log_waste(1, waste_footprint("Bread", 0.5))
    entries = get_waste_log(1)
    summary = summarise_log(entries)
    assert summary["entry_count"] == 2
    assert summary["co2_kg"] > 0
    assert summary["worst_item"]["item"] == "Beef"


def test_a_reason_is_optional():
    assert log_waste(1, waste_footprint("Bread", 0.2), reason="   ") is not None
    assert get_waste_log(1)[0]["reason"] is None


def test_logs_are_scoped_to_their_owner():
    log_waste(1, waste_footprint("Bread", 0.2))
    log_waste(2, waste_footprint("Beef", 0.2))
    assert len(get_waste_log(1)) == 1
    assert get_waste_log(1)[0]["item"] == "Bread"


def test_deleting_an_entry_removes_only_that_entry():
    keep = log_waste(1, waste_footprint("Bread", 0.2))
    remove = log_waste(1, waste_footprint("Beef", 0.2))
    assert delete_waste_entry(remove) is True
    remaining = get_waste_log(1)
    assert len(remaining) == 1
    assert remaining[0]["id"] == keep


def test_deleting_a_missing_entry_reports_failure():
    assert delete_waste_entry(31337) is False


def test_the_log_limit_is_honoured():
    for index in range(6):
        log_waste(1, waste_footprint("Bread", 0.1))
    assert len(get_waste_log(1, limit=3)) == 3
