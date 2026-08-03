"""Annual carbon footprint of the animals in a household.

Pets are entirely absent from the rest of the model. ``emissions.py`` and the
diet inputs count only what the *human* eats, so a household with a large dog
is under-reported by a figure that can rival a short-haul flight. That is not
a rounding error - it is a whole emitting member of the household the app
pretends does not exist.

The module deliberately keeps the arithmetic transparent and the framing
neutral. It does not argue about whether to own a pet. It reports what the
animal costs and which levers actually move that number, and in practice the
answer is almost always the same one: food type dominates everything else
combined, portion accuracy comes next, and litter and disposables are a
distant third.

One distinction matters enough to be worth stating up front. Pet food made
from human-grade cuts carries roughly the impact of the meat itself; food made
from by-products - the parts of the animal humans do not eat - carries far
less, because that material was produced anyway. Treating all "meat-based" pet
food as equivalent would make the feature badly misleading, so the factors
separate them.

The module is self-contained: its SQLite table is created lazily and no shared
files are modified.
"""

import os
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

# Typical daily feed intake in grams, by species and size band. A 40 kg dog
# and a hamster are not remotely the same problem, so a bare "number of pets"
# input would be meaningless.
SPECIES_PROFILES = {
    "Dog (giant, 45kg+)": {
        "daily_grams": 700, "litter": False,
        "note": "The largest household footprint of any common pet.",
    },
    "Dog (large, 25-45kg)": {
        "daily_grams": 480, "litter": False,
        "note": "Feed volume scales roughly with body mass.",
    },
    "Dog (medium, 10-25kg)": {
        "daily_grams": 280, "litter": False,
        "note": "The most common size band in most households.",
    },
    "Dog (small, under 10kg)": {
        "daily_grams": 150, "litter": False,
        "note": "A fraction of a large dog's intake.",
    },
    "Cat": {
        "daily_grams": 190, "litter": True,
        "note": "Obligate carnivore, so food type is hard to change much.",
    },
    "Rabbit": {
        "daily_grams": 180, "litter": True,
        "note": "Largely hay and greens, so the food term is small.",
    },
    "Small rodent": {
        "daily_grams": 25, "litter": True,
        "note": "Hamsters, gerbils and mice - a negligible footprint.",
    },
    "Bird": {
        "daily_grams": 30, "litter": True,
        "note": "Seed and pellet based, very low impact.",
    },
    "Reptile": {
        "daily_grams": 45, "litter": False,
        "note": "Feed is modest, but heat lamps run around the clock.",
    },
    "Fish tank": {
        "daily_grams": 8, "litter": False,
        "note": "Feed is trivial; the filter and heater are the real cost.",
    },
}

DEFAULT_SPECIES = "Dog (medium, 10-25kg)"

# kg CO2e per kg of feed as fed. The critical split is between food made from
# human-grade cuts and food made from by-products: the latter uses material
# that was produced regardless of whether any pet ate it.
FOOD_TYPES = {
    "Premium beef / lamb (human-grade cuts)": {
        "co2_per_kg": 18.5,
        "note": "Prime cuts diverted to pet food carry the full livestock impact.",
    },
    "Premium chicken / turkey (human-grade)": {
        "co2_per_kg": 6.2,
        "note": "Poultry is far lighter than red meat, cut for cut.",
    },
    "Fish-based": {
        "co2_per_kg": 5.4,
        "note": "Varies enormously with the fishery, but generally moderate.",
    },
    "Standard dry kibble (mixed)": {
        "co2_per_kg": 3.4,
        "note": "Mostly grain with by-product meal - the common default.",
    },
    "By-product based (dry)": {
        "co2_per_kg": 1.9,
        "note": "Uses offal and trimmings humans do not eat. Much lower impact.",
    },
    "Wet food (tinned, mixed)": {
        "co2_per_kg": 2.6,
        "note": "Looks low per kg only because most of the tin is water.",
    },
    "Insect protein": {
        "co2_per_kg": 1.4,
        "note": "The lowest-impact complete protein currently sold.",
    },
    "Plant-based / vegetarian": {
        "co2_per_kg": 1.1,
        "note": "Viable for dogs and rabbits; not appropriate for cats.",
    },
    "Hay and fresh greens": {
        "co2_per_kg": 0.6,
        "note": "Herbivore feed - the lightest option on this list.",
    },
    "Seed and pellet mix": {
        "co2_per_kg": 1.3,
        "note": "Standard for birds and small rodents.",
    },
}

DEFAULT_FOOD_TYPE = "Standard dry kibble (mixed)"

# Food types that are not nutritionally appropriate for obligate carnivores.
# The reduction engine refuses to recommend these for cats.
CARNIVORE_UNSAFE_FOODS = ("Plant-based / vegetarian", "Hay and fresh greens")

OBLIGATE_CARNIVORES = ("Cat",)

# kg CO2e per kg of litter. Clay is strip-mined and heavy, which is why it
# dominates the litter term despite looking like an afterthought.
LITTER_TYPES = {
    "Clay / bentonite (clumping)": {
        "co2_per_kg": 0.65,
        "note": "Strip-mined, heavy to ship, and landfilled after use.",
    },
    "Silica crystal": {
        "co2_per_kg": 0.52,
        "note": "Energy-intensive to produce, but lasts far longer per fill.",
    },
    "Recycled paper": {
        "co2_per_kg": 0.18,
        "note": "Made from waste already in the system.",
    },
    "Wood pellet": {
        "co2_per_kg": 0.14,
        "note": "Sawmill residue - about the lowest option available.",
    },
    "Corn / wheat": {
        "co2_per_kg": 0.31,
        "note": "Renewable, though it competes with food crops.",
    },
    "Walnut shell": {
        "co2_per_kg": 0.22,
        "note": "An agricultural by-product, and compostable.",
    },
}

DEFAULT_LITTER = "Clay / bentonite (clumping)"

# kg CO2e per waste bag, by material.
WASTE_BAG_TYPES = {
    "Standard plastic": 0.0075,
    "Recycled plastic": 0.0042,
    "Compostable starch": 0.0055,
    "Paper": 0.0031,
    "None / flushed": 0.0,
}

DEFAULT_BAG_TYPE = "Standard plastic"

BEDDING_CO2_PER_KG = 2.1      # Synthetic beds and blankets, per kg replaced.
TOY_CO2_EACH = 1.4            # Average mixed-material toy.
ACCESSORY_CO2_EACH = 3.2      # Collars, leads, bowls, carriers.
VET_VISIT_CO2 = 6.5           # Facility energy, consumables and travel.
GROOMING_CO2 = 3.8            # Salon energy, water and products.

DAYS_PER_YEAR = 365

# Annual dietary footprint of a human, for scale. Pet totals mean very little
# in the abstract, so they are reported against a number people already met
# elsewhere in the app.
HUMAN_DIET_BASELINES = {
    "Average omnivore": 1400.0,
    "Low-meat": 950.0,
    "Vegetarian": 700.0,
    "Vegan": 550.0,
}

DEFAULT_HUMAN_BASELINE = "Average omnivore"

# Tolerance before the portion check complains, as a share of the profile.
PORTION_TOLERANCE = 0.20


def list_species():
    """Return the species catalogue, largest daily intake first."""
    return sorted(
        ({"name": name, **info} for name, info in SPECIES_PROFILES.items()),
        key=lambda item: item["daily_grams"],
        reverse=True,
    )


def list_food_types():
    """Return pet food options, lowest carbon first."""
    return sorted(
        ({"name": name, **info} for name, info in FOOD_TYPES.items()),
        key=lambda item: item["co2_per_kg"],
    )


def list_litter_types():
    """Return litter options, lowest carbon first."""
    return sorted(
        ({"name": name, **info} for name, info in LITTER_TYPES.items()),
        key=lambda item: item["co2_per_kg"],
    )


def get_species_profile(species):
    """Return the reference profile for a species, falling back sensibly."""
    return dict(SPECIES_PROFILES.get(species, SPECIES_PROFILES[DEFAULT_SPECIES]))


def _clean_number(value, maximum, default=0.0):
    """Coerce a user-supplied number into a sane, non-negative range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return max(0.0, min(number, maximum))


def _clean_count(value, maximum=10000):
    """Whole, non-negative counts."""
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(count, maximum))


def is_obligate_carnivore(species):
    """Whether a species cannot safely be moved onto plant-based feed."""
    return species in OBLIGATE_CARNIVORES


def food_footprint(species, food_type=DEFAULT_FOOD_TYPE, daily_grams=None):
    """Annual kg CO2e from feeding one animal.

    This is the dominant term for every species larger than a hamster, which
    is why it is the first thing the reduction engine looks at.
    """
    profile = get_species_profile(species)
    grams = (
        profile["daily_grams"] if daily_grams is None else _clean_number(daily_grams, 10000.0)
    )
    intensity = FOOD_TYPES.get(food_type, FOOD_TYPES[DEFAULT_FOOD_TYPE])["co2_per_kg"]

    annual_kg_food = grams / 1000.0 * DAYS_PER_YEAR

    return {
        "daily_grams": round(grams, 1),
        "annual_food_kg": round(annual_kg_food, 2),
        "co2_kg": round(annual_kg_food * intensity, 2),
    }


def litter_footprint(litter_type=DEFAULT_LITTER, kg_per_month=0.0):
    """Annual kg CO2e from cat or small-animal litter."""
    monthly = _clean_number(kg_per_month, 500.0)
    intensity = LITTER_TYPES.get(litter_type, LITTER_TYPES[DEFAULT_LITTER])["co2_per_kg"]
    annual_kg = monthly * 12

    return {
        "annual_litter_kg": round(annual_kg, 2),
        "co2_kg": round(annual_kg * intensity, 2),
    }


def consumables_footprint(
    bags_per_week=0,
    bag_type=DEFAULT_BAG_TYPE,
    bedding_kg_per_year=0.0,
    toys_per_year=0,
    accessories_per_year=0,
):
    """Annual kg CO2e from the long tail of pet ownership."""
    bag_rate = WASTE_BAG_TYPES.get(bag_type, WASTE_BAG_TYPES[DEFAULT_BAG_TYPE])
    bags = _clean_count(bags_per_week, 200) * 52

    bags_co2 = bags * bag_rate
    bedding_co2 = _clean_number(bedding_kg_per_year, 500.0) * BEDDING_CO2_PER_KG
    toys_co2 = _clean_count(toys_per_year, 1000) * TOY_CO2_EACH
    accessories_co2 = _clean_count(accessories_per_year, 1000) * ACCESSORY_CO2_EACH

    return {
        "bags_per_year": bags,
        "bags_co2_kg": round(bags_co2, 2),
        "bedding_co2_kg": round(bedding_co2, 2),
        "toys_co2_kg": round(toys_co2, 2),
        "accessories_co2_kg": round(accessories_co2, 2),
        "co2_kg": round(bags_co2 + bedding_co2 + toys_co2 + accessories_co2, 2),
    }


def vet_footprint(visits_per_year=1, grooming_per_year=0):
    """Annual kg CO2e from veterinary and grooming services."""
    visits = _clean_count(visits_per_year, 500)
    grooms = _clean_count(grooming_per_year, 500)
    return {
        "visits": visits,
        "grooming": grooms,
        "co2_kg": round(visits * VET_VISIT_CO2 + grooms * GROOMING_CO2, 2),
    }


def total_pawprint(pet):
    """The full annual footprint of one animal, with its breakdown.

    ``pet`` is a plain dict; every field falls back to a sensible default so a
    half-filled form still produces a usable answer.
    """
    species = pet.get("species", DEFAULT_SPECIES)
    profile = get_species_profile(species)

    food = food_footprint(
        species, pet.get("food_type", DEFAULT_FOOD_TYPE), pet.get("daily_grams")
    )
    litter = litter_footprint(
        pet.get("litter_type", DEFAULT_LITTER), pet.get("litter_kg_per_month", 0.0)
    )
    consumables = consumables_footprint(
        pet.get("bags_per_week", 0),
        pet.get("bag_type", DEFAULT_BAG_TYPE),
        pet.get("bedding_kg_per_year", 0.0),
        pet.get("toys_per_year", 0),
        pet.get("accessories_per_year", 0),
    )
    vet = vet_footprint(pet.get("vet_visits", 1), pet.get("grooming_visits", 0))

    total = food["co2_kg"] + litter["co2_kg"] + consumables["co2_kg"] + vet["co2_kg"]

    return {
        "name": pet.get("name") or species,
        "species": species,
        "food_type": pet.get("food_type", DEFAULT_FOOD_TYPE),
        "uses_litter": profile["litter"],
        "food_co2_kg": food["co2_kg"],
        "litter_co2_kg": litter["co2_kg"],
        "consumables_co2_kg": consumables["co2_kg"],
        "vet_co2_kg": vet["co2_kg"],
        "total_co2_kg": round(total, 2),
        "food_share_pct": round(food["co2_kg"] / total * 100, 1) if total else 0.0,
        "annual_food_kg": food["annual_food_kg"],
        "daily_grams": food["daily_grams"],
        "breakdown": {
            "Food": food["co2_kg"],
            "Litter": litter["co2_kg"],
            "Consumables": consumables["co2_kg"],
            "Vet and grooming": vet["co2_kg"],
        },
    }


def household_pawprint(pets):
    """Aggregate every animal in the household."""
    results = [total_pawprint(pet) for pet in (pets or [])]

    total = sum(item["total_co2_kg"] for item in results)
    breakdown = {"Food": 0.0, "Litter": 0.0, "Consumables": 0.0, "Vet and grooming": 0.0}
    for item in results:
        for category, value in item["breakdown"].items():
            breakdown[category] = round(breakdown[category] + value, 2)

    return {
        "pet_count": len(results),
        "total_co2_kg": round(total, 2),
        "average_per_pet_kg": round(total / len(results), 2) if results else 0.0,
        "breakdown": breakdown,
        "food_share_pct": round(breakdown["Food"] / total * 100, 1) if total else 0.0,
        "pets": sorted(results, key=lambda item: item["total_co2_kg"], reverse=True),
    }


def compare_to_human_diet(total_co2_kg, baseline=DEFAULT_HUMAN_BASELINE):
    """Express a pawprint against a human diet, so the number has a scale."""
    reference = HUMAN_DIET_BASELINES.get(baseline, HUMAN_DIET_BASELINES[DEFAULT_HUMAN_BASELINE])
    total = _clean_number(total_co2_kg, 10 ** 7)

    return {
        "baseline": baseline,
        "baseline_kg": reference,
        "share_of_human_diet_pct": round(total / reference * 100, 1) if reference else 0.0,
        "human_diet_equivalent": round(total / reference, 2) if reference else 0.0,
    }


def portion_check(species, daily_grams):
    """Flag over- and under-feeding against the species profile.

    Overfeeding is both a welfare problem and pure wasted carbon, and it is
    common enough to be worth checking before any other advice is offered.
    """
    profile = get_species_profile(species)
    expected = profile["daily_grams"]
    actual = _clean_number(daily_grams, 10000.0)

    if expected <= 0:
        return {"status": "unknown", "expected_grams": expected, "actual_grams": actual}

    ratio = actual / expected
    if ratio > 1 + PORTION_TOLERANCE:
        status = "over"
    elif ratio < 1 - PORTION_TOLERANCE:
        status = "under"
    else:
        status = "ok"

    excess_kg = max(0.0, (actual - expected)) / 1000.0 * DAYS_PER_YEAR

    return {
        "status": status,
        "expected_grams": expected,
        "actual_grams": round(actual, 1),
        "ratio": round(ratio, 2),
        "excess_food_kg_per_year": round(excess_kg, 2),
    }


def reduction_options(pet, limit=6):
    """Ranked, quantified swaps for one animal.

    Every option reports the kg CO2e it actually saves, and options that would
    be nutritionally inappropriate for the species are never offered.
    """
    species = pet.get("species", DEFAULT_SPECIES)
    current = total_pawprint(pet)
    options = []

    # Food type is almost always the dominant lever, so it is tried first.
    current_food = pet.get("food_type", DEFAULT_FOOD_TYPE)
    for candidate in FOOD_TYPES:
        if candidate == current_food:
            continue
        if is_obligate_carnivore(species) and candidate in CARNIVORE_UNSAFE_FOODS:
            continue
        swapped = dict(pet)
        swapped["food_type"] = candidate
        saving = current["total_co2_kg"] - total_pawprint(swapped)["total_co2_kg"]
        if saving > 0:
            options.append(
                {
                    "action": f"Switch food to {candidate.lower()}",
                    "category": "Food",
                    "saving_kg": round(saving, 2),
                    "note": FOOD_TYPES[candidate]["note"],
                }
            )

    # Correcting an over-portion, where one exists.
    portion = portion_check(species, current["daily_grams"])
    if portion["status"] == "over":
        corrected = dict(pet)
        corrected["daily_grams"] = portion["expected_grams"]
        saving = current["total_co2_kg"] - total_pawprint(corrected)["total_co2_kg"]
        if saving > 0:
            options.append(
                {
                    "action": (
                        f"Feed the profile portion of {portion['expected_grams']}g a day "
                        f"instead of {portion['actual_grams']:.0f}g"
                    ),
                    "category": "Portion",
                    "saving_kg": round(saving, 2),
                    "note": "Better for the animal's health as well as its footprint.",
                }
            )

    # Litter, where the species uses any.
    current_litter = pet.get("litter_type", DEFAULT_LITTER)
    if pet.get("litter_kg_per_month", 0):
        for candidate in LITTER_TYPES:
            if candidate == current_litter:
                continue
            swapped = dict(pet)
            swapped["litter_type"] = candidate
            saving = current["total_co2_kg"] - total_pawprint(swapped)["total_co2_kg"]
            if saving > 0:
                options.append(
                    {
                        "action": f"Switch litter to {candidate.lower()}",
                        "category": "Litter",
                        "saving_kg": round(saving, 2),
                        "note": LITTER_TYPES[candidate]["note"],
                    }
                )

    # Waste bags.
    current_bag = pet.get("bag_type", DEFAULT_BAG_TYPE)
    if pet.get("bags_per_week", 0) and current_bag != "Recycled plastic":
        swapped = dict(pet)
        swapped["bag_type"] = "Recycled plastic"
        saving = current["total_co2_kg"] - total_pawprint(swapped)["total_co2_kg"]
        if saving > 0:
            options.append(
                {
                    "action": "Use recycled-content waste bags",
                    "category": "Consumables",
                    "saving_kg": round(saving, 2),
                    "note": "A small term, but it costs nothing to change.",
                }
            )

    options.sort(key=lambda item: item["saving_kg"], reverse=True)
    return options[: max(0, int(limit))]


def get_pet_tips(household, limit=6):
    """Advice ranked by the household's own breakdown."""
    tips = []
    pets = household.get("pets", [])

    if not pets:
        return ["Add a pet above to see what the animals in your home actually cost."]

    food_share = household.get("food_share_pct", 0.0)
    tips.append(
        f"Food is {food_share:.0f}% of your pets' total footprint. Everything else "
        "on this page is a rounding error by comparison, so start there."
    )

    biggest = pets[0]
    tips.append(
        f"{biggest['name']} accounts for {biggest['total_co2_kg']:.0f} kg CO₂e a year, "
        f"the largest of your animals. Its food type alone drives "
        f"{biggest['food_share_pct']:.0f}% of that."
    )

    carnivores = [pet for pet in pets if is_obligate_carnivore(pet["species"])]
    if carnivores:
        tips.append(
            "Cats are obligate carnivores, so plant-based feed is not an option. "
            "By-product-based food is the realistic lever - it uses material that "
            "was produced anyway."
        )

    premium = [pet for pet in pets if "human-grade" in pet.get("food_type", "").lower()]
    if premium:
        tips.append(
            "Human-grade cuts in pet food carry the full livestock impact. "
            "By-product-based food uses the parts people do not eat, at a fraction "
            "of the carbon and with no loss of nutritional completeness."
        )

    litter_total = household.get("breakdown", {}).get("Litter", 0.0)
    if litter_total > 20:
        tips.append(
            "Clay litter is strip-mined and heavy to ship. Wood pellet or recycled "
            "paper cuts that term by around three quarters."
        )

    tips.append(
        "Weigh the food rather than eyeballing a scoop. Overfeeding is common, and "
        "it is carbon spent on something that harms the animal."
    )

    return tips[: max(0, int(limit))]


def _get_conn():
    return sqlite3.connect(DB_NAME)


def init_pet_db():
    """Create the pet profile table if it does not exist yet."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pet_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                species TEXT NOT NULL,
                food_type TEXT NOT NULL,
                daily_grams REAL NOT NULL,
                total_co2_kg REAL NOT NULL,
                food_co2_kg REAL NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Pet footprint init error: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def save_pet(user_id, pet):
    """Persist one animal. Returns the new row id or None."""
    init_pet_db()
    conn = None
    try:
        conn = _get_conn()
        result = total_pawprint(pet)
        cursor = conn.execute(
            """
            INSERT INTO pet_profiles (
                user_id, name, species, food_type, daily_grams,
                total_co2_kg, food_co2_kg, detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                (pet.get("name") or result["species"]).strip() or result["species"],
                result["species"],
                result["food_type"],
                result["daily_grams"],
                result["total_co2_kg"],
                result["food_co2_kg"],
                json.dumps(
                    {
                        "litter_type": pet.get("litter_type", DEFAULT_LITTER),
                        "litter_kg_per_month": pet.get("litter_kg_per_month", 0.0),
                        "bags_per_week": pet.get("bags_per_week", 0),
                        "bag_type": pet.get("bag_type", DEFAULT_BAG_TYPE),
                        "bedding_kg_per_year": pet.get("bedding_kg_per_year", 0.0),
                        "toys_per_year": pet.get("toys_per_year", 0),
                        "accessories_per_year": pet.get("accessories_per_year", 0),
                        "vet_visits": pet.get("vet_visits", 1),
                        "grooming_visits": pet.get("grooming_visits", 0),
                    }
                ),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("Unable to save pet: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


def get_pets(user_id, limit=50):
    """Return a user's pets, newest first, ready to feed back into the model."""
    init_pet_db()
    conn = None
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, name, species, food_type, daily_grams, total_co2_kg,
                   food_co2_kg, detail_json, created_at
            FROM pet_profiles
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

        pets = []
        for row in rows:
            record = dict(row)
            try:
                record.update(json.loads(record.pop("detail_json")))
            except (TypeError, ValueError):
                pass
            pets.append(record)
        return pets
    except sqlite3.Error as exc:
        logger.error("Unable to load pets: %s", exc)
        return []
    finally:
        if conn:
            conn.close()


def delete_pet(pet_id):
    """Remove a pet profile."""
    init_pet_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM pet_profiles WHERE id = ?", (pet_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Unable to delete pet: %s", exc)
        return False
    finally:
        if conn:
            conn.close()
