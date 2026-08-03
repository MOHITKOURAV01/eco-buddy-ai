"""Tests for the Wardrobe & Textile Footprint Tracker."""
import os
import tempfile

import pytest

os.environ["ECO_BUDDY_DB"] = ":memory:"

import wardrobe
from wardrobe import (
    CONDITIONS,
    DEAD_STOCK_WEAR_THRESHOLD,
    DEFAULT_CONDITION,
    DEFAULT_FIBRE,
    DEFAULT_GARMENT,
    DEFAULT_WASH_TEMPERATURE,
    DEFAULT_WEARS_PER_WASH,
    FIBRES,
    GARMENT_TYPES,
    REPLACEMENT_AVOIDED_PER_YEAR,
    WASH_TEMPERATURES,
    care_footprint,
    carbon_per_wear,
    compare_purchase,
    cost_per_wear,
    delete_garment,
    extend_life_saving,
    find_dead_stock,
    garment_footprint,
    get_expected_wears,
    get_garment_mass,
    get_garments,
    get_wardrobe_tips,
    lifetime_footprint,
    list_conditions,
    list_fibres,
    list_garment_types,
    log_wear,
    save_garment,
    utilisation_score,
    wardrobe_summary,
    water_per_wear,
)


@pytest.fixture(autouse=True)
def temp_db():
    """Point the module at a throwaway SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    original = wardrobe.DB_NAME
    wardrobe.DB_NAME = db_path
    yield db_path
    wardrobe.DB_NAME = original
    try:
        os.unlink(db_path)
    except OSError:
        pass


def make_garment(**overrides):
    """A well-worn cotton t-shirt, unless a test says otherwise."""
    garment = {
        "name": "Everyday tee",
        "category": "T-shirt",
        "fibre": "Conventional cotton",
        "condition": "New",
        "price": 15.0,
        "wears": 40,
        "wash_temp": "Warm (40°C)",
        "tumble_dried": False,
        "ironed": False,
    }
    garment.update(overrides)
    return garment


class TestReferenceData:
    def test_every_fibre_has_positive_impacts(self):
        for name, info in FIBRES.items():
            assert info["co2_per_kg"] > 0, name
            assert info["water_per_kg"] > 0, name
            assert info["note"]

    def test_every_garment_has_mass_and_expected_wears(self):
        for name, info in GARMENT_TYPES.items():
            assert info["mass_kg"] > 0, name
            assert info["expected_wears"] > 0, name

    def test_new_carries_the_full_embodied_burden(self):
        assert CONDITIONS["New"]["embodied_share"] == 1.0

    def test_reuse_conditions_are_all_cheaper_than_new(self):
        for name, info in CONDITIONS.items():
            if name != "New":
                assert info["embodied_share"] < 1.0, name

    def test_hemp_is_the_lowest_carbon_natural_fibre(self):
        assert list_fibres()[0]["name"] == "Hemp"

    def test_leather_is_the_highest_impact_material(self):
        assert list_fibres()[-1]["name"] == "Leather"

    def test_garment_types_sorted_heaviest_first(self):
        masses = [item["mass_kg"] for item in list_garment_types()]
        assert masses == sorted(masses, reverse=True)

    def test_conditions_sorted_by_embodied_share(self):
        shares = [item["embodied_share"] for item in list_conditions()]
        assert shares == sorted(shares)

    def test_hotter_washes_always_cost_more(self):
        rates = list(WASH_TEMPERATURES.values())
        assert rates == sorted(rates)

    def test_unknown_lookups_fall_back(self):
        assert get_garment_mass("Spacesuit") == GARMENT_TYPES[DEFAULT_GARMENT]["mass_kg"]
        assert get_expected_wears("Spacesuit") == (
            GARMENT_TYPES[DEFAULT_GARMENT]["expected_wears"]
        )


class TestGarmentFootprint:
    def test_matches_hand_calculation(self):
        result = garment_footprint("T-shirt", "Conventional cotton", "New")
        expected = 0.18 * 19.0
        assert result["embodied_co2_kg"] == pytest.approx(expected, abs=1e-3)

    def test_water_matches_hand_calculation(self):
        result = garment_footprint("T-shirt", "Conventional cotton", "New")
        assert result["embodied_water_l"] == pytest.approx(0.18 * 10500.0, abs=0.1)

    def test_second_hand_is_always_below_new(self):
        for category in GARMENT_TYPES:
            for fibre in FIBRES:
                new = garment_footprint(category, fibre, "New")
                used = garment_footprint(category, fibre, "Second-hand")
                assert used["embodied_co2_kg"] < new["embodied_co2_kg"]

    def test_second_hand_is_exactly_the_documented_share(self):
        new = garment_footprint("Coat", "Wool", "New")
        used = garment_footprint("Coat", "Wool", "Second-hand")
        share = CONDITIONS["Second-hand"]["embodied_share"]
        assert used["embodied_co2_kg"] == pytest.approx(
            new["embodied_co2_kg"] * share, abs=1e-3
        )

    def test_new_co2_is_reported_regardless_of_condition(self):
        used = garment_footprint("Coat", "Wool", "Second-hand")
        new = garment_footprint("Coat", "Wool", "New")
        assert used["new_co2_kg"] == pytest.approx(new["new_co2_kg"])

    def test_unknown_fibre_falls_back_to_the_default(self):
        unknown = garment_footprint("T-shirt", "Unobtainium", "New")
        default = garment_footprint("T-shirt", DEFAULT_FIBRE, "New")
        assert unknown["embodied_co2_kg"] == default["embodied_co2_kg"]

    def test_unknown_condition_is_treated_as_new(self):
        unknown = garment_footprint("T-shirt", "Wool", "Time-travelled")
        assert unknown["embodied_co2_kg"] == pytest.approx(
            garment_footprint("T-shirt", "Wool", DEFAULT_CONDITION)["embodied_co2_kg"]
        )

    def test_explicit_mass_overrides_the_catalogue(self):
        result = garment_footprint("T-shirt", "Wool", "New", mass_kg=2.0)
        assert result["mass_kg"] == 2.0
        assert result["embodied_co2_kg"] == pytest.approx(2.0 * 32.0, abs=1e-3)

    def test_negative_mass_is_floored(self):
        assert garment_footprint("T-shirt", "Wool", "New", mass_kg=-5)["mass_kg"] == 0.0

    def test_heavier_garments_cost_more(self):
        coat = garment_footprint("Coat", "Wool", "New")
        socks = garment_footprint("Socks", "Wool", "New")
        assert coat["embodied_co2_kg"] > socks["embodied_co2_kg"]


class TestCareFootprint:
    def test_zero_wears_means_zero_care(self):
        care = care_footprint(0, 0.5)
        assert care["washes"] == 0
        assert care["care_co2_kg"] == 0.0
        assert care["care_water_l"] == 0.0

    def test_wash_count_follows_wears_per_wash(self):
        assert care_footprint(30, 0.5, wears_per_wash=3)["washes"] == pytest.approx(10.0)

    def test_matches_hand_calculation(self):
        care = care_footprint(30, 0.5, "Warm (40°C)", wears_per_wash=3)
        assert care["care_co2_kg"] == pytest.approx(10 * 0.5 * 0.35, abs=1e-3)

    def test_hotter_washing_costs_more(self):
        cold = care_footprint(60, 0.5, "Cold (20-30°C)")
        hot = care_footprint(60, 0.5, "Very hot (90°C)")
        assert hot["care_co2_kg"] > cold["care_co2_kg"]

    def test_tumble_drying_adds_emissions(self):
        line = care_footprint(60, 0.5, tumble_dried=False)
        tumbled = care_footprint(60, 0.5, tumble_dried=True)
        assert tumbled["care_co2_kg"] > line["care_co2_kg"]

    def test_ironing_adds_emissions(self):
        plain = care_footprint(60, 0.5, ironed=False)
        pressed = care_footprint(60, 0.5, ironed=True)
        assert pressed["care_co2_kg"] > plain["care_co2_kg"]

    def test_care_scales_with_wears(self):
        few = care_footprint(30, 0.5)["care_co2_kg"]
        many = care_footprint(300, 0.5)["care_co2_kg"]
        assert many == pytest.approx(few * 10, abs=1e-3)

    def test_zero_wears_per_wash_does_not_divide_by_zero(self):
        care = care_footprint(30, 0.5, wears_per_wash=0)
        assert care["washes"] == pytest.approx(30 / DEFAULT_WEARS_PER_WASH)

    def test_unknown_temperature_falls_back(self):
        unknown = care_footprint(30, 0.5, "Boiling")
        default = care_footprint(30, 0.5, DEFAULT_WASH_TEMPERATURE)
        assert unknown["care_co2_kg"] == default["care_co2_kg"]


class TestLifetimeAndPerWear:
    def test_total_is_embodied_plus_care(self):
        result = lifetime_footprint(make_garment())
        assert result["total_co2_kg"] == pytest.approx(
            result["embodied_co2_kg"] + result["care_co2_kg"], abs=1e-3
        )

    def test_carbon_per_wear_matches_the_total(self):
        garment = make_garment(wears=40)
        result = lifetime_footprint(garment)
        assert carbon_per_wear(garment) == pytest.approx(
            result["total_co2_kg"] / 40, abs=1e-4
        )

    def test_unworn_garment_has_no_per_wear_figure(self):
        assert carbon_per_wear(make_garment(wears=0)) is None
        assert cost_per_wear(50.0, 0) is None
        assert water_per_wear(make_garment(wears=0)) is None

    def test_more_wears_always_lowers_carbon_per_wear(self):
        rarely = carbon_per_wear(make_garment(wears=5))
        often = carbon_per_wear(make_garment(wears=200))
        assert often < rarely

    def test_cost_per_wear_matches_hand_calculation(self):
        assert cost_per_wear(120.0, 40) == pytest.approx(3.0)

    def test_expensive_but_worn_beats_cheap_and_idle(self):
        assert cost_per_wear(300.0, 400) < cost_per_wear(20.0, 4)

    def test_second_hand_has_lower_carbon_per_wear(self):
        new = carbon_per_wear(make_garment(condition="New"))
        used = carbon_per_wear(make_garment(condition="Second-hand"))
        assert used < new

    def test_missing_fields_still_produce_a_result(self):
        result = lifetime_footprint({})
        assert result["category"] == DEFAULT_GARMENT
        assert result["total_co2_kg"] >= 0

    def test_garbage_wear_count_is_treated_as_zero(self):
        assert lifetime_footprint(make_garment(wears="lots"))["wears"] == 0

    def test_name_falls_back_to_the_category(self):
        garment = make_garment()
        garment.pop("name")
        assert lifetime_footprint(garment)["name"] == "T-shirt"


class TestWardrobeSummary:
    def test_totals_equal_the_sum_of_the_items(self):
        garments = [make_garment(), make_garment(category="Jeans", fibre="Wool")]
        summary = wardrobe_summary(garments)
        assert summary["item_count"] == 2
        assert summary["total_co2_kg"] == pytest.approx(
            sum(item["total_co2_kg"] for item in summary["items"]), abs=0.01
        )

    def test_empty_wardrobe_is_all_zeroes(self):
        summary = wardrobe_summary([])
        assert summary["item_count"] == 0
        assert summary["total_co2_kg"] == 0
        assert summary["average_wears"] == 0.0
        assert summary["carbon_per_wear_kg"] is None
        assert summary["utilisation_score"] == 0.0

    def test_none_is_handled(self):
        assert wardrobe_summary(None)["item_count"] == 0

    def test_condition_split_counts_items(self):
        garments = [
            make_garment(condition="New"),
            make_garment(condition="New"),
            make_garment(condition="Second-hand"),
        ]
        split = wardrobe_summary(garments)["condition_split"]
        assert split["New"] == 2
        assert split["Second-hand"] == 1

    def test_fibre_split_sums_to_the_total(self):
        garments = [make_garment(), make_garment(fibre="Wool", category="Coat")]
        summary = wardrobe_summary(garments)
        assert sum(summary["fibre_split"].values()) == pytest.approx(
            summary["total_co2_kg"], abs=0.05
        )

    def test_all_unworn_wardrobe_has_no_carbon_per_wear(self):
        summary = wardrobe_summary([make_garment(wears=0), make_garment(wears=0)])
        assert summary["carbon_per_wear_kg"] is None


class TestUtilisationScore:
    def test_empty_wardrobe_scores_zero(self):
        assert utilisation_score([]) == 0.0

    def test_score_is_bounded(self):
        for wears in (0, 1, 50, 5000):
            assert 0.0 <= utilisation_score([make_garment(wears=wears)]) <= 100.0

    def test_fully_worn_wardrobe_scores_one_hundred(self):
        garment = make_garment(wears=get_expected_wears("T-shirt"))
        assert utilisation_score([garment]) == 100.0

    def test_overworn_items_do_not_exceed_one_hundred(self):
        assert utilisation_score([make_garment(wears=100000)]) == 100.0

    def test_idle_wardrobe_scores_near_zero(self):
        assert utilisation_score([make_garment(wears=0)]) == 0.0

    def test_thirty_worn_items_beat_a_hundred_idle_ones(self):
        worn = [make_garment(wears=50) for _ in range(30)]
        idle = [make_garment(wears=1) for _ in range(100)]
        assert utilisation_score(worn) > utilisation_score(idle)

    def test_score_uses_the_garment_type_expectation(self):
        # 50 wears is a full life for a t-shirt but a third of a coat's.
        shirt = utilisation_score([make_garment(category="T-shirt", wears=50)])
        coat = utilisation_score([make_garment(category="Coat", wears=50)])
        assert shirt > coat


class TestDeadStock:
    def test_finds_barely_worn_items(self):
        garments = [make_garment(wears=100), make_garment(name="Idle", wears=1)]
        dead = find_dead_stock(garments)
        assert len(dead) == 1
        assert dead[0]["name"] == "Idle"

    def test_ranked_by_wasted_embodied_carbon(self):
        garments = [
            make_garment(name="Tee", category="T-shirt", wears=0),
            make_garment(name="Coat", category="Coat", fibre="Wool", wears=0),
        ]
        dead = find_dead_stock(garments)
        assert dead[0]["name"] == "Coat"
        carbon = [item["embodied_co2_kg"] for item in dead]
        assert carbon == sorted(carbon, reverse=True)

    def test_threshold_is_inclusive(self):
        garments = [make_garment(wears=DEAD_STOCK_WEAR_THRESHOLD)]
        assert len(find_dead_stock(garments)) == 1

    def test_custom_threshold_is_respected(self):
        garments = [make_garment(wears=20)]
        assert find_dead_stock(garments, threshold=5) == []
        assert len(find_dead_stock(garments, threshold=25)) == 1

    def test_empty_wardrobe_has_no_dead_stock(self):
        assert find_dead_stock([]) == []
        assert find_dead_stock(None) == []


class TestExtendLife:
    def test_saving_scales_with_years(self):
        garments = [make_garment()]
        one = extend_life_saving(garments, 1)["co2_saved_kg"]
        three = extend_life_saving(garments, 3)["co2_saved_kg"]
        assert three == pytest.approx(one * 3, abs=0.01)

    def test_zero_years_saves_nothing(self):
        assert extend_life_saving([make_garment()], 0)["co2_saved_kg"] == 0.0

    def test_matches_the_documented_model(self):
        garments = [make_garment()]
        result = extend_life_saving(garments, 1)
        new_co2 = lifetime_footprint(garments[0])["new_co2_kg"]
        assert result["co2_saved_kg"] == pytest.approx(
            new_co2 * REPLACEMENT_AVOIDED_PER_YEAR, abs=0.01
        )

    def test_empty_wardrobe_saves_nothing(self):
        result = extend_life_saving([], 5)
        assert result["co2_saved_kg"] == 0.0
        assert result["items_covered"] == 0

    def test_saving_is_never_negative(self):
        assert extend_life_saving([make_garment()], -3)["co2_saved_kg"] == 0.0

    def test_bigger_wardrobes_save_more(self):
        one = extend_life_saving([make_garment()], 1)["co2_saved_kg"]
        five = extend_life_saving([make_garment() for _ in range(5)], 1)["co2_saved_kg"]
        assert five > one


class TestComparePurchase:
    def test_not_buying_always_wins(self):
        options = compare_purchase("Coat", "Wool")
        assert options[0]["condition"] == "Do not buy"
        assert options[0]["total_co2_kg"] == 0.0

    def test_new_is_always_the_worst_option(self):
        options = compare_purchase("Jeans", "Conventional cotton")
        assert options[-1]["condition"] == "New"

    def test_options_are_sorted_cleanest_first(self):
        totals = [item["total_co2_kg"] for item in compare_purchase("Coat", "Wool")]
        assert totals == sorted(totals)

    def test_every_option_carries_an_explanation(self):
        assert all(item["note"] for item in compare_purchase("T-shirt", "Linen"))

    def test_expected_wears_lowers_carbon_per_wear(self):
        rarely = compare_purchase("Dress", "Polyester", expected_wears=5)
        often = compare_purchase("Dress", "Polyester", expected_wears=200)
        new_rarely = next(o for o in rarely if o["condition"] == "New")
        new_often = next(o for o in often if o["condition"] == "New")
        assert new_often["carbon_per_wear_kg"] < new_rarely["carbon_per_wear_kg"]

    def test_defaults_to_the_type_expectation(self):
        options = compare_purchase("Jeans", "Conventional cotton")
        new = next(o for o in options if o["condition"] == "New")
        assert new["carbon_per_wear_kg"] is not None


class TestTips:
    def test_empty_wardrobe_gets_a_prompt(self):
        tips = get_wardrobe_tips(wardrobe_summary([]))
        assert len(tips) == 1
        assert "add" in tips[0].lower()

    def test_idle_wardrobe_is_called_out(self):
        summary = wardrobe_summary([make_garment(wears=0) for _ in range(5)])
        tips = get_wardrobe_tips(summary)
        assert any("barely worn" in tip.lower() for tip in tips)

    def test_well_used_wardrobe_is_praised(self):
        summary = wardrobe_summary([make_garment(category="T-shirt", wears=50)])
        assert any("genuinely good" in tip.lower() for tip in get_wardrobe_tips(summary))

    def test_all_new_wardrobe_is_nudged_to_second_hand(self):
        summary = wardrobe_summary([make_garment(condition="New") for _ in range(5)])
        assert any("second-hand" in tip.lower() for tip in get_wardrobe_tips(summary))

    def test_limit_is_respected(self):
        summary = wardrobe_summary([make_garment(wears=0) for _ in range(5)])
        assert len(get_wardrobe_tips(summary, limit=2)) <= 2

    def test_zero_limit_returns_nothing(self):
        summary = wardrobe_summary([make_garment()])
        assert get_wardrobe_tips(summary, limit=0) == []


class TestPersistence:
    def test_save_and_load_round_trip(self):
        garment_id = save_garment(1, make_garment())
        assert garment_id is not None

        saved = get_garments(1)
        assert len(saved) == 1
        assert saved[0]["name"] == "Everyday tee"
        assert saved[0]["category"] == "T-shirt"
        assert saved[0]["wears"] == 40
        assert saved[0]["total_co2_kg"] > 0

    def test_saved_garment_can_be_remodelled(self):
        save_garment(1, make_garment(tumble_dried=True))
        loaded = get_garments(1)[0]
        assert loaded["tumble_dried"] is True
        # The reloaded record feeds straight back into the model.
        assert lifetime_footprint(loaded)["total_co2_kg"] > 0

    def test_blank_name_falls_back_to_the_category(self):
        save_garment(2, make_garment(name="   "))
        assert get_garments(2)[0]["name"] == "T-shirt"

    def test_wardrobes_are_scoped_per_user(self):
        save_garment(10, make_garment(name="Mine"))
        save_garment(11, make_garment(name="Theirs"))
        assert len(get_garments(10)) == 1
        assert get_garments(10)[0]["name"] == "Mine"

    def test_log_wear_increments_the_count(self):
        garment_id = save_garment(3, make_garment(wears=10))
        assert log_wear(garment_id) is True
        assert get_garments(3)[0]["wears"] == 11

    def test_log_wear_accepts_multiple_wears(self):
        garment_id = save_garment(3, make_garment(wears=10))
        log_wear(garment_id, times=5)
        assert get_garments(3)[0]["wears"] == 15

    def test_log_wear_refreshes_the_lifetime_total(self):
        garment_id = save_garment(3, make_garment(wears=10))
        before = get_garments(3)[0]["total_co2_kg"]
        log_wear(garment_id, times=200)
        assert get_garments(3)[0]["total_co2_kg"] > before

    def test_log_wear_on_a_missing_row_returns_false(self):
        assert log_wear(999999) is False

    def test_limit_is_applied(self):
        for index in range(5):
            save_garment(4, make_garment(name=f"Item {index}"))
        assert len(get_garments(4, limit=2)) == 2

    def test_delete_removes_only_the_target(self):
        first = save_garment(5, make_garment(name="One"))
        save_garment(5, make_garment(name="Two"))
        assert delete_garment(first) is True
        remaining = get_garments(5)
        assert len(remaining) == 1
        assert remaining[0]["name"] == "Two"

    def test_deleting_a_missing_row_returns_false(self):
        assert delete_garment(999999) is False

    def test_no_garments_for_a_new_user(self):
        assert get_garments(12345) == []
