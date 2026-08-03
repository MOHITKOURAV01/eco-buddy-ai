"""Tests for the Pet Carbon Pawprint Calculator."""
import os
import tempfile

import pytest

os.environ["ECO_BUDDY_DB"] = ":memory:"

import pet_footprint
from pet_footprint import (
    ACCESSORY_CO2_EACH,
    CARNIVORE_UNSAFE_FOODS,
    DAYS_PER_YEAR,
    DEFAULT_BAG_TYPE,
    DEFAULT_FOOD_TYPE,
    DEFAULT_HUMAN_BASELINE,
    DEFAULT_LITTER,
    DEFAULT_SPECIES,
    FOOD_TYPES,
    HUMAN_DIET_BASELINES,
    LITTER_TYPES,
    OBLIGATE_CARNIVORES,
    PORTION_TOLERANCE,
    SPECIES_PROFILES,
    TOY_CO2_EACH,
    VET_VISIT_CO2,
    WASTE_BAG_TYPES,
    compare_to_human_diet,
    consumables_footprint,
    delete_pet,
    food_footprint,
    get_pet_tips,
    get_pets,
    get_species_profile,
    household_pawprint,
    is_obligate_carnivore,
    list_food_types,
    list_litter_types,
    list_species,
    litter_footprint,
    portion_check,
    reduction_options,
    save_pet,
    total_pawprint,
    vet_footprint,
)


@pytest.fixture(autouse=True)
def temp_db():
    """Point the module at a throwaway SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    original = pet_footprint.DB_NAME
    pet_footprint.DB_NAME = db_path
    yield db_path
    pet_footprint.DB_NAME = original
    try:
        os.unlink(db_path)
    except OSError:
        pass


def make_pet(**overrides):
    """A medium dog on standard kibble, unless a test says otherwise."""
    pet = {
        "name": "Rex",
        "species": "Dog (medium, 10-25kg)",
        "food_type": "Standard dry kibble (mixed)",
        "daily_grams": 280,
        "bags_per_week": 7,
        "bag_type": "Standard plastic",
        "bedding_kg_per_year": 1.0,
        "toys_per_year": 4,
        "accessories_per_year": 1,
        "vet_visits": 2,
        "grooming_visits": 0,
        "litter_type": DEFAULT_LITTER,
        "litter_kg_per_month": 0.0,
    }
    pet.update(overrides)
    return pet


def make_cat(**overrides):
    """A cat, which is the interesting case for the carnivore rules."""
    pet = {
        "name": "Mog",
        "species": "Cat",
        "food_type": "Premium beef / lamb (human-grade cuts)",
        "daily_grams": 190,
        "litter_type": "Clay / bentonite (clumping)",
        "litter_kg_per_month": 8.0,
        "bags_per_week": 0,
        "vet_visits": 1,
    }
    pet.update(overrides)
    return pet


class TestReferenceData:
    def test_every_species_has_a_positive_intake(self):
        for name, info in SPECIES_PROFILES.items():
            assert info["daily_grams"] > 0, name
            assert isinstance(info["litter"], bool)
            assert info["note"]

    def test_every_food_type_has_a_positive_intensity(self):
        for name, info in FOOD_TYPES.items():
            assert info["co2_per_kg"] > 0, name
            assert info["note"]

    def test_by_products_are_cheaper_than_human_grade_cuts(self):
        # The single most important distinction in the whole module.
        assert (
            FOOD_TYPES["By-product based (dry)"]["co2_per_kg"]
            < FOOD_TYPES["Premium beef / lamb (human-grade cuts)"]["co2_per_kg"]
        )

    def test_red_meat_is_the_heaviest_option(self):
        assert list_food_types()[-1]["name"] == "Premium beef / lamb (human-grade cuts)"

    def test_species_sorted_by_intake(self):
        intakes = [item["daily_grams"] for item in list_species()]
        assert intakes == sorted(intakes, reverse=True)

    def test_litter_sorted_by_carbon(self):
        rates = [item["co2_per_kg"] for item in list_litter_types()]
        assert rates == sorted(rates)

    def test_clay_litter_is_worse_than_wood(self):
        assert (
            LITTER_TYPES["Clay / bentonite (clumping)"]["co2_per_kg"]
            > LITTER_TYPES["Wood pellet"]["co2_per_kg"]
        )

    def test_flushing_waste_has_no_bag_cost(self):
        assert WASTE_BAG_TYPES["None / flushed"] == 0.0

    def test_unknown_species_falls_back(self):
        assert get_species_profile("Dragon") == SPECIES_PROFILES[DEFAULT_SPECIES]

    def test_species_profile_is_a_copy(self):
        profile = get_species_profile("Cat")
        profile["daily_grams"] = 99999
        assert SPECIES_PROFILES["Cat"]["daily_grams"] != 99999

    def test_cats_are_the_obligate_carnivores(self):
        assert is_obligate_carnivore("Cat") is True
        assert is_obligate_carnivore("Dog (medium, 10-25kg)") is False
        assert OBLIGATE_CARNIVORES == ("Cat",)


class TestFoodFootprint:
    def test_matches_hand_calculation(self):
        result = food_footprint("Dog (medium, 10-25kg)", "Standard dry kibble (mixed)")
        annual_food = 280 / 1000 * DAYS_PER_YEAR
        assert result["annual_food_kg"] == pytest.approx(annual_food, abs=0.01)
        assert result["co2_kg"] == pytest.approx(annual_food * 3.4, abs=0.05)

    def test_uses_the_species_profile_by_default(self):
        assert food_footprint("Cat")["daily_grams"] == SPECIES_PROFILES["Cat"]["daily_grams"]

    def test_explicit_portion_overrides_the_profile(self):
        assert food_footprint("Cat", daily_grams=400)["daily_grams"] == 400

    def test_bigger_dogs_eat_more(self):
        giant = food_footprint("Dog (giant, 45kg+)")["co2_kg"]
        small = food_footprint("Dog (small, under 10kg)")["co2_kg"]
        assert giant > small

    def test_premium_red_meat_costs_far_more_than_by_products(self):
        premium = food_footprint("Cat", "Premium beef / lamb (human-grade cuts)")["co2_kg"]
        byproduct = food_footprint("Cat", "By-product based (dry)")["co2_kg"]
        assert premium > byproduct * 5

    def test_zero_portion_means_zero_emissions(self):
        assert food_footprint("Cat", daily_grams=0)["co2_kg"] == 0.0

    def test_negative_portion_is_floored(self):
        assert food_footprint("Cat", daily_grams=-100)["co2_kg"] == 0.0

    def test_garbage_portion_is_treated_as_zero(self):
        assert food_footprint("Cat", daily_grams="lots")["co2_kg"] == 0.0

    def test_unknown_food_type_falls_back(self):
        unknown = food_footprint("Cat", "Caviar")["co2_kg"]
        assert unknown == food_footprint("Cat", DEFAULT_FOOD_TYPE)["co2_kg"]


class TestLitterAndConsumables:
    def test_litter_matches_hand_calculation(self):
        result = litter_footprint("Clay / bentonite (clumping)", 8.0)
        assert result["annual_litter_kg"] == pytest.approx(96.0)
        assert result["co2_kg"] == pytest.approx(96.0 * 0.65, abs=0.05)

    def test_no_litter_means_no_emissions(self):
        assert litter_footprint("Clay / bentonite (clumping)", 0)["co2_kg"] == 0.0

    def test_switching_to_wood_pellet_cuts_the_litter_term(self):
        clay = litter_footprint("Clay / bentonite (clumping)", 8.0)["co2_kg"]
        wood = litter_footprint("Wood pellet", 8.0)["co2_kg"]
        assert wood < clay

    def test_unknown_litter_falls_back(self):
        assert litter_footprint("Moon dust", 5.0)["co2_kg"] == (
            litter_footprint(DEFAULT_LITTER, 5.0)["co2_kg"]
        )

    def test_bags_scale_to_a_full_year(self):
        assert consumables_footprint(bags_per_week=7)["bags_per_year"] == 364

    def test_consumables_total_is_the_sum_of_its_parts(self):
        result = consumables_footprint(7, "Standard plastic", 2.0, 5, 2)
        assert result["co2_kg"] == pytest.approx(
            result["bags_co2_kg"]
            + result["bedding_co2_kg"]
            + result["toys_co2_kg"]
            + result["accessories_co2_kg"],
            abs=0.05,
        )

    def test_toys_match_hand_calculation(self):
        assert consumables_footprint(toys_per_year=10)["toys_co2_kg"] == pytest.approx(
            10 * TOY_CO2_EACH, abs=0.05
        )

    def test_accessories_match_hand_calculation(self):
        assert consumables_footprint(accessories_per_year=3)["accessories_co2_kg"] == (
            pytest.approx(3 * ACCESSORY_CO2_EACH, abs=0.05)
        )

    def test_nothing_bought_means_nothing_emitted(self):
        assert consumables_footprint()["co2_kg"] == 0.0

    def test_flushing_removes_the_bag_term(self):
        assert consumables_footprint(7, "None / flushed")["bags_co2_kg"] == 0.0


class TestVetFootprint:
    def test_matches_hand_calculation(self):
        assert vet_footprint(2, 0)["co2_kg"] == pytest.approx(2 * VET_VISIT_CO2, abs=0.05)

    def test_grooming_adds_to_the_total(self):
        assert vet_footprint(1, 4)["co2_kg"] > vet_footprint(1, 0)["co2_kg"]

    def test_no_visits_means_no_emissions(self):
        assert vet_footprint(0, 0)["co2_kg"] == 0.0

    def test_negative_visits_are_floored(self):
        assert vet_footprint(-3, -2)["co2_kg"] == 0.0


class TestTotalPawprint:
    def test_total_is_the_sum_of_the_categories(self):
        result = total_pawprint(make_pet())
        assert result["total_co2_kg"] == pytest.approx(
            result["food_co2_kg"]
            + result["litter_co2_kg"]
            + result["consumables_co2_kg"]
            + result["vet_co2_kg"],
            abs=0.05,
        )

    def test_breakdown_matches_the_total(self):
        result = total_pawprint(make_pet())
        assert sum(result["breakdown"].values()) == pytest.approx(
            result["total_co2_kg"], abs=0.05
        )

    def test_food_dominates_for_a_dog(self):
        assert total_pawprint(make_pet())["food_share_pct"] > 50

    def test_empty_pet_still_produces_a_result(self):
        result = total_pawprint({})
        assert result["species"] == DEFAULT_SPECIES
        assert result["total_co2_kg"] > 0

    def test_name_falls_back_to_the_species(self):
        pet = make_pet()
        pet.pop("name")
        assert total_pawprint(pet)["name"] == "Dog (medium, 10-25kg)"

    def test_litter_species_is_flagged(self):
        assert total_pawprint(make_cat())["uses_litter"] is True
        assert total_pawprint(make_pet())["uses_litter"] is False

    def test_zero_everything_does_not_divide_by_zero(self):
        empty = {
            "species": "Fish tank",
            "daily_grams": 0,
            "vet_visits": 0,
            "grooming_visits": 0,
        }
        result = total_pawprint(empty)
        assert result["total_co2_kg"] == 0.0
        assert result["food_share_pct"] == 0.0

    def test_giant_dog_costs_more_than_a_hamster(self):
        giant = total_pawprint(make_pet(species="Dog (giant, 45kg+)", daily_grams=None))
        rodent = total_pawprint(make_pet(species="Small rodent", daily_grams=None))
        assert giant["total_co2_kg"] > rodent["total_co2_kg"]


class TestHouseholdPawprint:
    def test_total_equals_the_sum_of_the_pets(self):
        household = household_pawprint([make_pet(), make_cat()])
        assert household["pet_count"] == 2
        assert household["total_co2_kg"] == pytest.approx(
            sum(pet["total_co2_kg"] for pet in household["pets"]), abs=0.05
        )

    def test_breakdown_sums_to_the_total(self):
        household = household_pawprint([make_pet(), make_cat()])
        assert sum(household["breakdown"].values()) == pytest.approx(
            household["total_co2_kg"], abs=0.1
        )

    def test_pets_are_ranked_by_footprint(self):
        # daily_grams=None makes each animal use its own profile intake, so the
        # ranking reflects a real difference rather than identical inputs.
        household = household_pawprint(
            [
                make_pet(species="Small rodent", daily_grams=None),
                make_pet(species="Dog (giant, 45kg+)", daily_grams=None),
            ]
        )
        totals = [pet["total_co2_kg"] for pet in household["pets"]]
        assert totals == sorted(totals, reverse=True)
        assert household["pets"][0]["species"] == "Dog (giant, 45kg+)"

    def test_empty_household_is_all_zeroes(self):
        household = household_pawprint([])
        assert household["pet_count"] == 0
        assert household["total_co2_kg"] == 0.0
        assert household["average_per_pet_kg"] == 0.0
        assert household["food_share_pct"] == 0.0

    def test_none_is_handled(self):
        assert household_pawprint(None)["pet_count"] == 0

    def test_average_matches_the_total(self):
        household = household_pawprint([make_pet(), make_pet()])
        assert household["average_per_pet_kg"] == pytest.approx(
            household["total_co2_kg"] / 2, abs=0.05
        )


class TestHumanDietComparison:
    def test_share_matches_hand_calculation(self):
        result = compare_to_human_diet(700.0, "Average omnivore")
        assert result["share_of_human_diet_pct"] == pytest.approx(50.0, abs=0.1)

    def test_equivalent_matches_the_share(self):
        result = compare_to_human_diet(1400.0, "Average omnivore")
        assert result["human_diet_equivalent"] == pytest.approx(1.0, abs=0.01)

    def test_stricter_baselines_make_a_pet_look_bigger(self):
        omnivore = compare_to_human_diet(700.0, "Average omnivore")
        vegan = compare_to_human_diet(700.0, "Vegan")
        assert vegan["share_of_human_diet_pct"] > omnivore["share_of_human_diet_pct"]

    def test_zero_footprint_is_zero_share(self):
        assert compare_to_human_diet(0.0)["share_of_human_diet_pct"] == 0.0

    def test_unknown_baseline_falls_back(self):
        unknown = compare_to_human_diet(700.0, "Breatharian")
        default = compare_to_human_diet(700.0, DEFAULT_HUMAN_BASELINE)
        assert unknown["share_of_human_diet_pct"] == default["share_of_human_diet_pct"]

    def test_every_baseline_is_positive(self):
        assert all(value > 0 for value in HUMAN_DIET_BASELINES.values())


class TestPortionCheck:
    def test_profile_portion_is_fine(self):
        assert portion_check("Cat", SPECIES_PROFILES["Cat"]["daily_grams"])["status"] == "ok"

    def test_generous_overfeeding_is_flagged(self):
        assert portion_check("Cat", 400)["status"] == "over"

    def test_underfeeding_is_flagged(self):
        assert portion_check("Cat", 50)["status"] == "under"

    def test_tolerance_boundaries_are_respected(self):
        expected = SPECIES_PROFILES["Cat"]["daily_grams"]
        just_inside = expected * (1 + PORTION_TOLERANCE - 0.01)
        just_outside = expected * (1 + PORTION_TOLERANCE + 0.01)
        assert portion_check("Cat", just_inside)["status"] == "ok"
        assert portion_check("Cat", just_outside)["status"] == "over"

    def test_under_tolerance_boundary(self):
        expected = SPECIES_PROFILES["Cat"]["daily_grams"]
        assert portion_check("Cat", expected * (1 - PORTION_TOLERANCE + 0.01))["status"] == "ok"
        assert portion_check("Cat", expected * (1 - PORTION_TOLERANCE - 0.01))["status"] == "under"

    def test_excess_food_is_quantified(self):
        result = portion_check("Cat", 290)
        expected_excess = (290 - SPECIES_PROFILES["Cat"]["daily_grams"]) / 1000 * DAYS_PER_YEAR
        assert result["excess_food_kg_per_year"] == pytest.approx(expected_excess, abs=0.05)

    def test_underfeeding_reports_no_excess(self):
        assert portion_check("Cat", 50)["excess_food_kg_per_year"] == 0.0

    def test_zero_portion_is_under(self):
        assert portion_check("Cat", 0)["status"] == "under"


class TestReductionOptions:
    def test_options_are_ranked_by_saving(self):
        savings = [item["saving_kg"] for item in reduction_options(make_cat())]
        assert savings == sorted(savings, reverse=True)

    def test_savings_are_never_negative(self):
        for pet in (make_pet(), make_cat()):
            assert all(item["saving_kg"] > 0 for item in reduction_options(pet))

    def test_plant_based_is_never_offered_to_a_cat(self):
        actions = [item["action"].lower() for item in reduction_options(make_cat(), limit=99)]
        for unsafe in CARNIVORE_UNSAFE_FOODS:
            assert not any(unsafe.lower() in action for action in actions)

    def test_plant_based_is_offered_to_a_dog(self):
        actions = [item["action"].lower() for item in reduction_options(make_pet(), limit=99)]
        assert any("plant-based" in action for action in actions)

    def test_food_swap_leads_for_a_premium_fed_cat(self):
        assert reduction_options(make_cat())[0]["category"] == "Food"

    def test_overfeeding_produces_a_portion_option(self):
        options = reduction_options(make_pet(daily_grams=500), limit=99)
        assert any(item["category"] == "Portion" for item in options)

    def test_correct_portion_produces_no_portion_option(self):
        options = reduction_options(make_pet(daily_grams=280), limit=99)
        assert not any(item["category"] == "Portion" for item in options)

    def test_litter_options_only_appear_for_litter_users(self):
        cat_options = reduction_options(make_cat(), limit=99)
        dog_options = reduction_options(make_pet(), limit=99)
        assert any(item["category"] == "Litter" for item in cat_options)
        assert not any(item["category"] == "Litter" for item in dog_options)

    def test_already_optimal_food_is_not_offered_again(self):
        pet = make_pet(food_type="Plant-based / vegetarian")
        actions = [item["action"].lower() for item in reduction_options(pet, limit=99)]
        assert not any("switch food to plant-based" in action for action in actions)

    def test_limit_is_respected(self):
        assert len(reduction_options(make_cat(), limit=2)) <= 2

    def test_zero_limit_returns_nothing(self):
        assert reduction_options(make_cat(), limit=0) == []

    def test_every_option_carries_an_explanation(self):
        assert all(item["note"] for item in reduction_options(make_cat(), limit=99))


class TestTips:
    def test_empty_household_gets_a_prompt(self):
        tips = get_pet_tips(household_pawprint([]))
        assert len(tips) == 1
        assert "add a pet" in tips[0].lower()

    def test_food_share_is_always_mentioned_first(self):
        tips = get_pet_tips(household_pawprint([make_pet()]))
        assert "food" in tips[0].lower()

    def test_cats_get_the_carnivore_caveat(self):
        tips = get_pet_tips(household_pawprint([make_cat()]))
        assert any("obligate carnivore" in tip.lower() for tip in tips)

    def test_dogs_alone_do_not_get_the_carnivore_caveat(self):
        tips = get_pet_tips(household_pawprint([make_pet()]))
        assert not any("obligate carnivore" in tip.lower() for tip in tips)

    def test_premium_feeding_is_called_out(self):
        tips = get_pet_tips(household_pawprint([make_cat()]))
        assert any("human-grade" in tip.lower() for tip in tips)

    def test_limit_is_respected(self):
        assert len(get_pet_tips(household_pawprint([make_cat()]), limit=2)) <= 2

    def test_zero_limit_returns_nothing(self):
        assert get_pet_tips(household_pawprint([make_pet()]), limit=0) == []


class TestPersistence:
    def test_save_and_load_round_trip(self):
        pet_id = save_pet(1, make_pet())
        assert pet_id is not None

        saved = get_pets(1)
        assert len(saved) == 1
        assert saved[0]["name"] == "Rex"
        assert saved[0]["species"] == "Dog (medium, 10-25kg)"
        assert saved[0]["total_co2_kg"] > 0

    def test_saved_pet_can_be_remodelled(self):
        save_pet(1, make_cat())
        loaded = get_pets(1)[0]
        assert loaded["litter_kg_per_month"] == 8.0
        assert total_pawprint(loaded)["total_co2_kg"] == pytest.approx(
            loaded["total_co2_kg"], abs=0.05
        )

    def test_blank_name_falls_back_to_the_species(self):
        save_pet(2, make_pet(name="  "))
        assert get_pets(2)[0]["name"] == "Dog (medium, 10-25kg)"

    def test_pets_are_scoped_per_user(self):
        save_pet(10, make_pet(name="Mine"))
        save_pet(11, make_pet(name="Theirs"))
        assert len(get_pets(10)) == 1
        assert get_pets(10)[0]["name"] == "Mine"

    def test_limit_is_applied(self):
        for index in range(5):
            save_pet(3, make_pet(name=f"Pet {index}"))
        assert len(get_pets(3, limit=2)) == 2

    def test_delete_removes_only_the_target(self):
        first = save_pet(4, make_pet(name="One"))
        save_pet(4, make_pet(name="Two"))
        assert delete_pet(first) is True
        remaining = get_pets(4)
        assert len(remaining) == 1
        assert remaining[0]["name"] == "Two"

    def test_deleting_a_missing_row_returns_false(self):
        assert delete_pet(999999) is False

    def test_no_pets_for_a_new_user(self):
        assert get_pets(12345) == []

    def test_saved_household_aggregates_correctly(self):
        save_pet(7, make_pet())
        save_pet(7, make_cat())
        household = household_pawprint(get_pets(7))
        assert household["pet_count"] == 2
        assert household["total_co2_kg"] > 0
