"""Tests for the Last-Mile Delivery & Packaging Footprint."""
import os
import tempfile

import pytest

os.environ["ECO_BUDDY_DB"] = ":memory:"

import delivery_footprint
from delivery_footprint import (
    AIR_FREIGHT_PER_PARCEL_KM,
    DEFAULT_ITEM_EMBODIED_CO2,
    DEFAULT_LAST_MILE_KM,
    DEFAULT_PACKAGING_MATERIAL,
    DEFAULT_PARCEL_SIZE,
    DEFAULT_RESALE_PROBABILITY,
    DEFAULT_SPEED,
    DEFAULT_VEHICLE,
    FULFILMENT_CO2_PER_PARCEL,
    LAST_MILE_VEHICLES,
    PACKAGING_MATERIALS,
    PARCEL_SIZES,
    RETURN_HANDLING_MULTIPLIER,
    SHIPPING_SPEEDS,
    annual_footprint,
    click_and_collect,
    compare_scenarios,
    consolidation_saving,
    delete_delivery_profile,
    failed_attempt_cost,
    get_delivery_profiles,
    get_delivery_tips,
    list_packaging_materials,
    list_speeds,
    list_vehicles,
    optimise_orders,
    packaging_footprint,
    parcel_footprint,
    return_footprint,
    save_delivery_profile,
    transport_footprint,
)


@pytest.fixture(autouse=True)
def temp_db():
    """Point the module at a throwaway SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    original = delivery_footprint.DB_NAME
    delivery_footprint.DB_NAME = db_path
    yield db_path
    delivery_footprint.DB_NAME = original
    try:
        os.unlink(db_path)
    except OSError:
        pass


def make_profile(**overrides):
    """A fairly typical online shopper, unless a test says otherwise."""
    profile = {
        "orders_per_year": 30,
        "distance_km": 12.0,
        "vehicle": "Diesel van",
        "speed": "Standard (3-5 days)",
        "parcel_size": "Medium box",
        "materials": ["Corrugated cardboard"],
        "attempts": 1,
        "return_rate": 0.1,
        "resale_probability": 0.75,
        "item_embodied_co2": 8.0,
    }
    profile.update(overrides)
    return profile


class TestReferenceData:
    def test_every_vehicle_has_a_positive_rate(self):
        for name, info in LAST_MILE_VEHICLES.items():
            assert info["co2_per_parcel_km"] > 0, name
            assert info["note"]

    def test_cargo_bike_is_the_cleanest_option(self):
        assert list_vehicles()[0]["name"] == "Cargo bike"

    def test_a_private_car_is_the_worst_per_parcel(self):
        assert list_vehicles()[-1]["name"] == "Private car (collection)"

    def test_electric_van_beats_diesel(self):
        assert (
            LAST_MILE_VEHICLES["Electric van"]["co2_per_parcel_km"]
            < LAST_MILE_VEHICLES["Diesel van"]["co2_per_parcel_km"]
        )

    def test_standard_shipping_is_the_baseline(self):
        assert SHIPPING_SPEEDS[DEFAULT_SPEED]["multiplier"] == 1.0
        assert SHIPPING_SPEEDS[DEFAULT_SPEED]["air_share"] == 0.0

    def test_faster_shipping_always_costs_more(self):
        multipliers = [item["multiplier"] for item in list_speeds()]
        assert multipliers == sorted(multipliers)
        air_shares = [item["air_share"] for item in list_speeds()]
        assert air_shares == sorted(air_shares)

    def test_bigger_parcels_use_more_packaging(self):
        masses = [info["packaging_kg"] for info in PARCEL_SIZES.values()]
        assert min(masses) > 0
        assert PARCEL_SIZES["Oversized"]["packaging_kg"] > PARCEL_SIZES["Letter / small"]["packaging_kg"]

    def test_void_fill_is_the_most_intensive_material(self):
        assert list_packaging_materials()[-1]["name"] == "Bubble wrap"

    def test_recycled_cardboard_beats_virgin(self):
        assert (
            PACKAGING_MATERIALS["Recycled cardboard"]["co2_per_kg"]
            < PACKAGING_MATERIALS["Corrugated cardboard"]["co2_per_kg"]
        )

    def test_packaging_materials_sorted_by_carbon(self):
        rates = [item["co2_per_kg"] for item in list_packaging_materials()]
        assert rates == sorted(rates)


class TestTransportFootprint:
    def test_matches_hand_calculation(self):
        result = transport_footprint(10.0, "Diesel van", "Standard (3-5 days)")
        assert result["road_co2_kg"] == pytest.approx(10.0 * 0.0195, abs=1e-4)
        assert result["air_co2_kg"] == 0.0

    def test_air_freight_matches_hand_calculation(self):
        result = transport_footprint(10.0, "Diesel van", "Next-day")
        expected_air = 10.0 * 0.25 * AIR_FREIGHT_PER_PARCEL_KM
        assert result["air_co2_kg"] == pytest.approx(expected_air, abs=1e-4)

    def test_express_is_always_worse_than_standard(self):
        for vehicle in LAST_MILE_VEHICLES:
            standard = transport_footprint(12.0, vehicle, "Standard (3-5 days)")
            express = transport_footprint(12.0, vehicle, "Next-day")
            assert express["co2_kg"] > standard["co2_kg"], vehicle

    def test_same_day_is_the_worst_speed(self):
        totals = [
            transport_footprint(12.0, "Diesel van", speed)["co2_kg"]
            for speed in ("Standard (3-5 days)", "Two-day", "Next-day", "Same-day")
        ]
        assert totals == sorted(totals)

    def test_standard_shipping_never_flies(self):
        assert transport_footprint(500.0, "Diesel van", "Standard (3-5 days)")["air_co2_kg"] == 0.0

    def test_longer_distances_cost_more(self):
        near = transport_footprint(5.0)["co2_kg"]
        far = transport_footprint(50.0)["co2_kg"]
        assert far > near

    def test_zero_distance_means_zero_transport(self):
        assert transport_footprint(0.0)["co2_kg"] == 0.0

    def test_negative_distance_is_floored(self):
        assert transport_footprint(-40.0)["co2_kg"] == 0.0

    def test_garbage_distance_falls_back_to_the_default(self):
        assert transport_footprint("far")["distance_km"] == DEFAULT_LAST_MILE_KM

    def test_unknown_vehicle_and_speed_fall_back(self):
        unknown = transport_footprint(12.0, "Teleporter", "Instant")
        default = transport_footprint(12.0, DEFAULT_VEHICLE, DEFAULT_SPEED)
        assert unknown["co2_kg"] == default["co2_kg"]


class TestPackagingFootprint:
    def test_matches_hand_calculation(self):
        result = packaging_footprint("Medium box", ["Corrugated cardboard"])
        assert result["co2_kg"] == pytest.approx(0.22 * 0.94, abs=1e-4)

    def test_bigger_parcels_cost_more(self):
        small = packaging_footprint("Letter / small")["co2_kg"]
        oversized = packaging_footprint("Oversized")["co2_kg"]
        assert oversized > small

    def test_multiple_materials_share_the_mass(self):
        single = packaging_footprint("Medium box", ["Corrugated cardboard"])
        mixed = packaging_footprint("Medium box", ["Corrugated cardboard", "Bubble wrap"])
        assert mixed["co2_kg"] > single["co2_kg"]
        assert mixed["packaging_kg"] == single["packaging_kg"]

    def test_moulded_pulp_beats_bubble_wrap(self):
        pulp = packaging_footprint("Medium box", ["Moulded pulp"])["co2_kg"]
        bubble = packaging_footprint("Medium box", ["Bubble wrap"])["co2_kg"]
        assert pulp < bubble

    def test_unknown_materials_fall_back_to_the_default(self):
        unknown = packaging_footprint("Medium box", ["Unobtainium"])
        default = packaging_footprint("Medium box", [DEFAULT_PACKAGING_MATERIAL])
        assert unknown["co2_kg"] == default["co2_kg"]
        assert unknown["materials"] == [DEFAULT_PACKAGING_MATERIAL]

    def test_empty_material_list_falls_back(self):
        assert packaging_footprint("Medium box", [])["materials"] == [
            DEFAULT_PACKAGING_MATERIAL
        ]

    def test_unknown_size_falls_back(self):
        assert packaging_footprint("Enormous")["packaging_kg"] == (
            PARCEL_SIZES[DEFAULT_PARCEL_SIZE]["packaging_kg"]
        )

    def test_void_fill_share_is_clamped(self):
        assert packaging_footprint("Medium box", void_fill_share=5.0)["void_fill_share"] == 1.0
        assert packaging_footprint("Medium box", void_fill_share=-2.0)["void_fill_share"] == 0.0


class TestParcelFootprint:
    def test_total_is_the_sum_of_its_parts(self):
        result = parcel_footprint()
        assert result["co2_kg"] == pytest.approx(
            result["transport_co2_kg"]
            + result["packaging_co2_kg"]
            + result["fulfilment_co2_kg"],
            abs=1e-4,
        )

    def test_fulfilment_is_always_charged_once(self):
        assert parcel_footprint(attempts=3)["fulfilment_co2_kg"] == FULFILMENT_CO2_PER_PARCEL

    def test_repeated_attempts_only_repeat_transport(self):
        once = parcel_footprint(attempts=1)
        twice = parcel_footprint(attempts=2)
        assert twice["transport_co2_kg"] == pytest.approx(once["transport_co2_kg"] * 2, abs=1e-4)
        assert twice["packaging_co2_kg"] == once["packaging_co2_kg"]

    def test_attempts_are_at_least_one(self):
        assert parcel_footprint(attempts=0)["attempts"] == 1
        assert parcel_footprint(attempts=-5)["attempts"] == 1

    def test_express_parcel_beats_no_standard_parcel(self):
        standard = parcel_footprint(speed="Standard (3-5 days)")["co2_kg"]
        express = parcel_footprint(speed="Same-day")["co2_kg"]
        assert express > standard


class TestFailedAttempts:
    def test_one_attempt_costs_nothing_extra(self):
        base = parcel_footprint(attempts=1)
        assert failed_attempt_cost(base, 1)["extra_co2_kg"] == 0.0

    def test_extra_attempts_scale_linearly(self):
        base = parcel_footprint(attempts=3)
        two_extra = failed_attempt_cost(base, 3)
        assert two_extra["failed_attempts"] == 2
        per_attempt = base["transport_co2_kg"] / 3
        assert two_extra["extra_co2_kg"] == pytest.approx(per_attempt * 2, abs=1e-4)

    def test_zero_attempts_is_never_negative(self):
        base = parcel_footprint(attempts=1)
        assert failed_attempt_cost(base, 0)["extra_co2_kg"] == 0.0

    def test_missing_keys_do_not_divide_by_zero(self):
        assert failed_attempt_cost({}, 3)["extra_co2_kg"] == 0.0


class TestConsolidation:
    def test_saving_matches_hand_calculation(self):
        result = consolidation_saving(items=4, orders=4, per_parcel_footprint=1.0)
        assert result["current_co2_kg"] == pytest.approx(4.0)
        assert result["consolidated_co2_kg"] == pytest.approx(1.0)
        assert result["saving_kg"] == pytest.approx(3.0)

    def test_already_consolidated_saves_nothing(self):
        result = consolidation_saving(items=4, orders=1, per_parcel_footprint=1.0)
        assert result["saving_kg"] == 0.0
        assert result["saving_pct"] == 0.0

    def test_saving_is_never_negative(self):
        for orders in (0, 1, 2, 50):
            result = consolidation_saving(items=4, orders=orders, per_parcel_footprint=1.0)
            assert result["saving_kg"] >= 0

    def test_orders_cannot_exceed_items(self):
        # Ten shipments of four items is not physically meaningful.
        assert consolidation_saving(items=4, orders=10, per_parcel_footprint=1.0)["orders"] == 4

    def test_more_orders_means_more_saving_available(self):
        few = consolidation_saving(10, 2, 1.0)["saving_kg"]
        many = consolidation_saving(10, 10, 1.0)["saving_kg"]
        assert many > few

    def test_saving_percentage_is_bounded(self):
        result = consolidation_saving(items=100, orders=100, per_parcel_footprint=1.0)
        assert 0.0 <= result["saving_pct"] <= 100.0

    def test_zero_footprint_does_not_divide_by_zero(self):
        assert consolidation_saving(4, 4, 0.0)["saving_pct"] == 0.0


class TestReturns:
    def test_no_returns_costs_nothing(self):
        assert return_footprint(2.0, return_rate=0.0)["co2_kg"] == 0.0

    def test_journey_matches_hand_calculation(self):
        result = return_footprint(2.0, return_rate=0.5, resale_probability=1.0)
        assert result["return_journey_co2_kg"] == pytest.approx(
            2.0 * 0.5 * RETURN_HANDLING_MULTIPLIER, abs=1e-4
        )

    def test_fully_resold_returns_have_no_write_off(self):
        assert return_footprint(2.0, 0.5, resale_probability=1.0)["unsold_write_off_co2_kg"] == 0.0

    def test_unsold_returns_write_off_the_item(self):
        result = return_footprint(2.0, return_rate=1.0, resale_probability=0.0, item_embodied_co2=10.0)
        assert result["unsold_write_off_co2_kg"] == pytest.approx(10.0, abs=1e-4)

    def test_write_off_usually_dominates_the_journey(self):
        result = return_footprint(2.0, 0.3, 0.5, item_embodied_co2=40.0)
        assert result["unsold_write_off_co2_kg"] > result["return_journey_co2_kg"]

    def test_higher_return_rates_always_cost_more(self):
        low = return_footprint(2.0, 0.1)["co2_kg"]
        high = return_footprint(2.0, 0.6)["co2_kg"]
        assert high > low

    def test_rates_are_clamped(self):
        assert return_footprint(2.0, return_rate=5.0)["return_rate"] == 1.0
        assert return_footprint(2.0, return_rate=-3.0)["return_rate"] == 0.0

    def test_resale_probability_is_clamped(self):
        assert return_footprint(2.0, 0.5, resale_probability=9.0)["resale_probability"] == 1.0

    def test_total_is_the_sum_of_its_parts(self):
        result = return_footprint(2.0, 0.4, 0.6, 15.0)
        assert result["co2_kg"] == pytest.approx(
            result["return_journey_co2_kg"] + result["unsold_write_off_co2_kg"], abs=1e-4
        )


class TestClickAndCollect:
    def test_a_dedicated_car_trip_is_worse_than_a_van(self):
        result = click_and_collect(10.0, dedicated_trip=True)
        assert result["better_than_delivery"] is False
        assert result["difference_kg"] > 0

    def test_a_trip_you_were_making_anyway_is_free(self):
        result = click_and_collect(10.0, dedicated_trip=False)
        assert result["collection_co2_kg"] == 0.0
        assert result["better_than_delivery"] is True

    def test_a_very_short_dedicated_trip_can_still_win(self):
        assert click_and_collect(0.2, dedicated_trip=True)["better_than_delivery"] is True

    def test_the_trip_is_counted_both_ways(self):
        near = click_and_collect(5.0)["collection_co2_kg"]
        far = click_and_collect(10.0)["collection_co2_kg"]
        assert far == pytest.approx(near * 2, abs=1e-4)

    def test_walking_or_cycling_to_collect_wins(self):
        result = click_and_collect(3.0, vehicle="Cargo bike", dedicated_trip=True)
        assert result["better_than_delivery"] is True

    def test_negative_distance_is_floored(self):
        assert click_and_collect(-5.0)["collection_co2_kg"] == 0.0


class TestAnnualFootprint:
    def test_breakdown_sums_to_the_total(self):
        result = annual_footprint(make_profile())
        assert sum(result["breakdown"].values()) == pytest.approx(
            result["total_co2_kg"], abs=0.05
        )

    def test_scales_with_order_count(self):
        few = annual_footprint(make_profile(orders_per_year=10))["total_co2_kg"]
        many = annual_footprint(make_profile(orders_per_year=100))["total_co2_kg"]
        assert many > few

    def test_no_orders_means_no_footprint(self):
        result = annual_footprint(make_profile(orders_per_year=0))
        assert result["total_co2_kg"] == 0.0
        assert result["returns_share_pct"] == 0.0

    def test_empty_profile_still_produces_a_result(self):
        result = annual_footprint({})
        assert result["total_co2_kg"] > 0
        assert result["speed"] == DEFAULT_SPEED

    def test_express_shipping_raises_the_total(self):
        standard = annual_footprint(make_profile(speed="Standard (3-5 days)"))
        express = annual_footprint(make_profile(speed="Same-day"))
        assert express["total_co2_kg"] > standard["total_co2_kg"]

    def test_failed_attempts_are_reported_separately_not_double_counted(self):
        result = annual_footprint(make_profile(attempts=3))
        assert result["breakdown"]["Failed attempts"] > 0
        assert sum(result["breakdown"].values()) == pytest.approx(
            result["total_co2_kg"], abs=0.05
        )

    def test_no_failed_attempts_means_a_zero_term(self):
        result = annual_footprint(make_profile(attempts=1))
        assert result["breakdown"]["Failed attempts"] == 0.0

    def test_returns_share_is_bounded(self):
        for rate in (0.0, 0.25, 1.0):
            result = annual_footprint(make_profile(return_rate=rate))
            assert 0.0 <= result["returns_share_pct"] <= 100.0

    def test_standard_shipping_has_no_air_freight(self):
        assert annual_footprint(make_profile(speed="Standard (3-5 days)"))["air_co2_kg"] == 0.0

    def test_express_shipping_puts_freight_in_the_air(self):
        assert annual_footprint(make_profile(speed="Next-day"))["air_co2_kg"] > 0


class TestOptimiseOrders:
    def test_options_are_ranked_by_saving(self):
        savings = [item["saving_kg"] for item in optimise_orders(make_profile(speed="Same-day"))]
        assert savings == sorted(savings, reverse=True)

    def test_savings_are_never_negative(self):
        options = optimise_orders(make_profile(speed="Next-day", attempts=2, return_rate=0.3))
        assert all(item["saving_kg"] > 0 for item in options)

    def test_express_shipping_produces_a_speed_option(self):
        actions = [item["action"].lower() for item in optimise_orders(make_profile(speed="Same-day"), limit=99)]
        assert any("standard shipping" in action for action in actions)

    def test_standard_shipping_produces_no_speed_option(self):
        actions = [
            item["action"].lower()
            for item in optimise_orders(make_profile(speed="Standard (3-5 days)"), limit=99)
        ]
        assert not any("standard shipping" in action for action in actions)

    def test_frequent_ordering_produces_a_batching_option(self):
        actions = [item["action"].lower() for item in optimise_orders(make_profile(orders_per_year=48), limit=99)]
        assert any("batch" in action for action in actions)

    def test_infrequent_ordering_produces_no_batching_option(self):
        actions = [item["action"].lower() for item in optimise_orders(make_profile(orders_per_year=6), limit=99)]
        assert not any("batch" in action for action in actions)

    def test_failed_attempts_produce_a_delivery_slot_option(self):
        actions = [item["action"].lower() for item in optimise_orders(make_profile(attempts=3), limit=99)]
        assert any("home for" in action for action in actions)

    def test_plastic_packaging_produces_a_packaging_option(self):
        profile = make_profile(materials=["Bubble wrap"])
        actions = [item["action"].lower() for item in optimise_orders(profile, limit=99)]
        assert any("packaging" in action for action in actions)

    def test_limit_is_respected(self):
        assert len(optimise_orders(make_profile(speed="Same-day"), limit=2)) <= 2

    def test_zero_limit_returns_nothing(self):
        assert optimise_orders(make_profile(speed="Same-day"), limit=0) == []

    def test_every_option_carries_an_explanation(self):
        assert all(item["note"] for item in optimise_orders(make_profile(speed="Same-day"), limit=99))


class TestCompareScenarios:
    def test_detects_an_improvement(self):
        before = make_profile(speed="Same-day", orders_per_year=60)
        after = make_profile(speed="Standard (3-5 days)", orders_per_year=20)
        result = compare_scenarios(before, after)
        assert result["improved"] is True
        assert result["difference_kg"] > 0

    def test_detects_a_regression(self):
        before = make_profile(speed="Standard (3-5 days)")
        after = make_profile(speed="Same-day")
        result = compare_scenarios(before, after)
        assert result["improved"] is False
        assert result["difference_kg"] < 0

    def test_identical_profiles_show_no_change(self):
        result = compare_scenarios(make_profile(), make_profile())
        assert result["difference_kg"] == 0.0
        assert result["change_pct"] == 0.0

    def test_zero_baseline_does_not_divide_by_zero(self):
        result = compare_scenarios(make_profile(orders_per_year=0), make_profile())
        assert result["change_pct"] == 0.0


class TestTips:
    def test_zero_footprint_gets_a_prompt(self):
        tips = get_delivery_tips(annual_footprint(make_profile(orders_per_year=0)))
        assert len(tips) == 1
        assert "enter your ordering" in tips[0].lower()

    def test_biggest_source_is_named_first(self):
        tips = get_delivery_tips(annual_footprint(make_profile()))
        assert "largest source" in tips[0].lower()

    def test_high_return_rate_is_called_out(self):
        result = annual_footprint(make_profile(return_rate=0.6, resale_probability=0.3))
        assert any("returns" in tip.lower() for tip in get_delivery_tips(result))

    def test_air_freight_is_called_out(self):
        result = annual_footprint(make_profile(speed="Next-day"))
        assert any("flying" in tip.lower() for tip in get_delivery_tips(result))

    def test_failed_attempts_are_called_out(self):
        result = annual_footprint(make_profile(attempts=3))
        assert any("failed deliveries" in tip.lower() for tip in get_delivery_tips(result))

    def test_limit_is_respected(self):
        result = annual_footprint(make_profile(speed="Next-day", attempts=3, return_rate=0.5))
        assert len(get_delivery_tips(result, limit=2)) <= 2

    def test_zero_limit_returns_nothing(self):
        assert get_delivery_tips(annual_footprint(make_profile()), limit=0) == []


class TestPersistence:
    def test_save_and_load_round_trip(self):
        profile_id = save_delivery_profile(1, "My ordering", make_profile())
        assert profile_id is not None

        saved = get_delivery_profiles(1)
        assert len(saved) == 1
        assert saved[0]["profile_name"] == "My ordering"
        assert saved[0]["orders_per_year"] == 30
        assert saved[0]["total_co2_kg"] > 0

    def test_saved_profile_can_be_remodelled(self):
        save_delivery_profile(1, "Mine", make_profile(speed="Next-day"))
        loaded = get_delivery_profiles(1)[0]
        assert loaded["profile"]["speed"] == "Next-day"
        assert annual_footprint(loaded["profile"])["total_co2_kg"] == pytest.approx(
            loaded["total_co2_kg"], abs=0.05
        )

    def test_blank_name_gets_a_default(self):
        save_delivery_profile(2, "  ", make_profile())
        assert get_delivery_profiles(2)[0]["profile_name"] == "My ordering"

    def test_profiles_are_scoped_per_user(self):
        save_delivery_profile(10, "Mine", make_profile())
        save_delivery_profile(11, "Theirs", make_profile())
        assert len(get_delivery_profiles(10)) == 1
        assert get_delivery_profiles(10)[0]["profile_name"] == "Mine"

    def test_limit_is_applied(self):
        for index in range(5):
            save_delivery_profile(3, f"Profile {index}", make_profile())
        assert len(get_delivery_profiles(3, limit=2)) == 2

    def test_delete_removes_only_the_target(self):
        first = save_delivery_profile(4, "One", make_profile())
        save_delivery_profile(4, "Two", make_profile())
        assert delete_delivery_profile(first) is True
        remaining = get_delivery_profiles(4)
        assert len(remaining) == 1
        assert remaining[0]["profile_name"] == "Two"

    def test_deleting_a_missing_row_returns_false(self):
        assert delete_delivery_profile(999999) is False

    def test_no_profiles_for_a_new_user(self):
        assert get_delivery_profiles(12345) == []
