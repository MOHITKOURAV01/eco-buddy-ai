"""Avoidable food waste: the upstream footprint of food nobody eats.

``waste.py`` has a category called "Food Scraps" with a single factor of
2.5 kg CO2 per kg. That number describes what happens to food *after* it is
thrown out - the methane from a landfill, or the lack of it from a compost
heap. It is the small half of the problem.

The large half is that the food was grown, watered, fertilised, processed,
refrigerated, packaged and driven across the country before it was thrown
away, and every one of those emissions still happened. Beef discarded uneaten
carries about 60 kg CO2e per kilogram of production footprint against roughly
1 kg from its disposal. Treating both as "2.5 kg CO2 per kg of food scraps"
undercounts the real figure by more than an order of magnitude.

This module counts the whole thing::

    waste footprint = production footprint of what was thrown out
                    + emissions of the disposal route it went to

It also separates *avoidable* waste from *unavoidable* waste. Banana skins,
bones and eggshells were never going to be eaten and are part of the cost of
the meal. A loaf that went stale in the cupboard is a different thing
entirely, and it is the only part anyone can actually act on.

The module is self-contained: it imports no Streamlit, its SQLite table is
created lazily and no shared files are modified.
"""

import os
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

# Storage locations, and how they multiply an item's baseline shelf life.
STORAGE_LOCATIONS = {
    "Counter / pantry": 1.0,
    "Fridge": 2.6,
    "Freezer": 24.0,
}

DEFAULT_STORAGE = "Fridge"

# Food catalogue.
#
#   co2_per_kg       cradle-to-shop production footprint (Poore & Nemecek, 2018)
#   water_per_kg     litres of virtual water (Mekonnen & Hoekstra, 2011)
#   price_per_kg     representative retail price
#   pantry_days      baseline shelf life on the counter
#   unavoidable      share of the purchased mass nobody was ever going to eat
FOOD_ITEMS = {
    "Beef": {
        "category": "Meat & fish", "co2_per_kg": 60.0, "water_per_kg": 15400,
        "price_per_kg": 12.0, "pantry_days": 0.5, "unavoidable": 0.05,
        "freezable": True,
    },
    "Lamb": {
        "category": "Meat & fish", "co2_per_kg": 24.0, "water_per_kg": 10400,
        "price_per_kg": 13.0, "pantry_days": 0.5, "unavoidable": 0.12,
        "freezable": True,
    },
    "Pork": {
        "category": "Meat & fish", "co2_per_kg": 7.2, "water_per_kg": 6000,
        "price_per_kg": 7.0, "pantry_days": 0.5, "unavoidable": 0.08,
        "freezable": True,
    },
    "Chicken": {
        "category": "Meat & fish", "co2_per_kg": 6.1, "water_per_kg": 4300,
        "price_per_kg": 6.5, "pantry_days": 0.5, "unavoidable": 0.15,
        "freezable": True,
    },
    "Farmed fish": {
        "category": "Meat & fish", "co2_per_kg": 5.1, "water_per_kg": 3700,
        "price_per_kg": 11.0, "pantry_days": 0.5, "unavoidable": 0.20,
        "freezable": True,
    },
    "Cheese": {
        "category": "Dairy & eggs", "co2_per_kg": 21.0, "water_per_kg": 5000,
        "price_per_kg": 9.0, "pantry_days": 2.0, "unavoidable": 0.02,
        "freezable": True,
    },
    "Butter": {
        "category": "Dairy & eggs", "co2_per_kg": 12.0, "water_per_kg": 5550,
        "price_per_kg": 8.0, "pantry_days": 3.0, "unavoidable": 0.0,
        "freezable": True,
    },
    "Milk": {
        "category": "Dairy & eggs", "co2_per_kg": 3.2, "water_per_kg": 1020,
        "price_per_kg": 1.1, "pantry_days": 0.5, "unavoidable": 0.0,
        "freezable": True,
    },
    "Yoghurt": {
        "category": "Dairy & eggs", "co2_per_kg": 2.2, "water_per_kg": 900,
        "price_per_kg": 2.5, "pantry_days": 1.0, "unavoidable": 0.0,
        "freezable": False,
    },
    "Eggs": {
        "category": "Dairy & eggs", "co2_per_kg": 4.7, "water_per_kg": 3300,
        "price_per_kg": 4.0, "pantry_days": 7.0, "unavoidable": 0.12,
        "freezable": False,
    },
    "Bread": {
        "category": "Bakery & grains", "co2_per_kg": 1.3, "water_per_kg": 1600,
        "price_per_kg": 2.6, "pantry_days": 4.0, "unavoidable": 0.02,
        "freezable": True,
    },
    "Rice": {
        "category": "Bakery & grains", "co2_per_kg": 4.5, "water_per_kg": 2500,
        "price_per_kg": 2.0, "pantry_days": 365.0, "unavoidable": 0.0,
        "freezable": True,
    },
    "Pasta": {
        "category": "Bakery & grains", "co2_per_kg": 1.9, "water_per_kg": 1850,
        "price_per_kg": 1.8, "pantry_days": 365.0, "unavoidable": 0.0,
        "freezable": True,
    },
    "Potatoes": {
        "category": "Vegetables", "co2_per_kg": 0.5, "water_per_kg": 290,
        "price_per_kg": 1.2, "pantry_days": 21.0, "unavoidable": 0.20,
        "freezable": False,
    },
    "Leafy salad": {
        "category": "Vegetables", "co2_per_kg": 2.0, "water_per_kg": 240,
        "price_per_kg": 6.0, "pantry_days": 1.5, "unavoidable": 0.05,
        "freezable": False,
    },
    "Tomatoes": {
        "category": "Vegetables", "co2_per_kg": 2.1, "water_per_kg": 210,
        "price_per_kg": 3.2, "pantry_days": 4.0, "unavoidable": 0.03,
        "freezable": False,
    },
    "Root vegetables": {
        "category": "Vegetables", "co2_per_kg": 0.4, "water_per_kg": 280,
        "price_per_kg": 1.5, "pantry_days": 14.0, "unavoidable": 0.15,
        "freezable": True,
    },
    "Bananas": {
        "category": "Fruit", "co2_per_kg": 0.9, "water_per_kg": 790,
        "price_per_kg": 1.4, "pantry_days": 5.0, "unavoidable": 0.35,
        "freezable": True,
    },
    "Apples": {
        "category": "Fruit", "co2_per_kg": 0.4, "water_per_kg": 820,
        "price_per_kg": 2.2, "pantry_days": 14.0, "unavoidable": 0.10,
        "freezable": False,
    },
    "Berries": {
        "category": "Fruit", "co2_per_kg": 1.5, "water_per_kg": 700,
        "price_per_kg": 12.0, "pantry_days": 2.0, "unavoidable": 0.03,
        "freezable": True,
    },
    "Cooked leftovers": {
        "category": "Prepared food", "co2_per_kg": 4.0, "water_per_kg": 1800,
        "price_per_kg": 5.0, "pantry_days": 0.5, "unavoidable": 0.0,
        "freezable": True,
    },
}

# Disposal routes, in kg CO2e per kg of food waste. Landfill is the worst by
# far because food buried without oxygen produces methane.
DISPOSAL_ROUTES = {
    "Landfill": {
        "co2_per_kg": 2.53,
        "note": "Buried food rots without oxygen and produces methane.",
    },
    "General waste (incinerated)": {
        "co2_per_kg": 0.72,
        "note": "Burning wet food recovers little energy, but no methane forms.",
    },
    "Kerbside food collection": {
        "co2_per_kg": 0.18,
        "note": "Usually anaerobic digestion - captures the methane as fuel.",
    },
    "Home compost": {
        "co2_per_kg": 0.09,
        "note": "Aerobic and local. Almost nothing beyond the CO2 the food held.",
    },
    "Fed to animals": {
        "co2_per_kg": 0.03,
        "note": "Displaces feed that would otherwise be grown.",
    },
}

DEFAULT_DISPOSAL = "Landfill"

# A typical cooked meal, used to translate kilograms into something people can
# picture.
MEAL_KG = 0.45

# Spoilage risk model. Risk stays near zero for most of an item's life and
# then climbs steeply as it passes its shelf life.
SPOILAGE_STEEPNESS = 4.0
MAX_SPOILAGE_RISK = 0.99

DAYS_PER_YEAR = 365.0
WEEKS_PER_YEAR = 52.0


def _clean_positive(value, maximum, default=0.0):
    """Coerce a user-supplied number into a sane, non-negative range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return max(0.0, min(number, maximum))


# ---------------------------------------------------------------------------
# Reference data helpers
# ---------------------------------------------------------------------------

def list_food_items(category=None):
    """Return the catalogue, highest production footprint first."""
    items = [
        {"name": name, **details}
        for name, details in FOOD_ITEMS.items()
        if category is None or details["category"] == category
    ]
    return sorted(items, key=lambda item: item["co2_per_kg"], reverse=True)


def list_categories():
    """Return the distinct food categories, alphabetically."""
    return sorted({details["category"] for details in FOOD_ITEMS.values()})


def get_food_item(name):
    """Look up a food item. None if unknown."""
    details = FOOD_ITEMS.get(name)
    if not details:
        return None
    return {"name": name, **details}


def list_disposal_routes():
    """Return disposal routes, lowest emissions first."""
    return sorted(
        ({"name": name, **info} for name, info in DISPOSAL_ROUTES.items()),
        key=lambda route: route["co2_per_kg"],
    )


def disposal_factor(route):
    """Emissions per kg for a disposal route. Unknown routes assume landfill."""
    info = DISPOSAL_ROUTES.get(route)
    if not info:
        return DISPOSAL_ROUTES[DEFAULT_DISPOSAL]["co2_per_kg"]
    return info["co2_per_kg"]


# ---------------------------------------------------------------------------
# Shelf life and spoilage
# ---------------------------------------------------------------------------

def shelf_life_days(item_name, storage=DEFAULT_STORAGE):
    """How long an item lasts in a given place, in days.

    Freezing is not offered for items it ruins - a frozen leafy salad is not
    a salad, and pretending otherwise would be bad advice.
    """
    item = get_food_item(item_name)
    if not item:
        return 0.0

    multiplier = STORAGE_LOCATIONS.get(storage, 1.0)
    if storage == "Freezer" and not item["freezable"]:
        multiplier = STORAGE_LOCATIONS["Fridge"]

    return round(item["pantry_days"] * multiplier, 1)


def best_storage(item_name):
    """The storage location that keeps an item longest, and by how much."""
    item = get_food_item(item_name)
    if not item:
        return None

    options = [
        {"storage": storage, "days": shelf_life_days(item_name, storage)}
        for storage in STORAGE_LOCATIONS
    ]
    options.sort(key=lambda option: option["days"], reverse=True)
    baseline = shelf_life_days(item_name, "Counter / pantry")

    return {
        "item": item_name,
        "best": options[0]["storage"],
        "days": options[0]["days"],
        "gain_days": round(options[0]["days"] - baseline, 1),
        "options": options,
        "freezable": item["freezable"],
    }


def spoilage_risk(item_name, days_held, storage=DEFAULT_STORAGE):
    """Probability-like score, 0-1, that an item has gone bad.

    Risk stays low for most of the item's life and then climbs steeply once it
    passes its shelf life, which is how food actually behaves - nothing is
    half-spoiled on day three of seven.
    """
    life = shelf_life_days(item_name, storage)
    if life <= 0:
        return 0.0

    held = _clean_positive(days_held, 3650.0)
    ratio = held / life
    if ratio <= 0:
        return 0.0

    risk = ratio ** SPOILAGE_STEEPNESS
    return round(min(MAX_SPOILAGE_RISK, risk), 3)


def at_risk_items(inventory, storage_key="storage", limit=None):
    """Rank an inventory by how likely each item is to be thrown away.

    ``inventory`` entries need ``item``, ``kg`` and ``days_held``; ``storage``
    is optional and defaults to the fridge.
    """
    ranked = []
    for entry in inventory or []:
        item_name = entry.get("item")
        if not get_food_item(item_name):
            continue

        storage = entry.get(storage_key, DEFAULT_STORAGE)
        risk = spoilage_risk(item_name, entry.get("days_held", 0), storage)
        kg = _clean_positive(entry.get("kg", 0.0), 1000.0)
        item = get_food_item(item_name)

        ranked.append(
            {
                "item": item_name,
                "kg": round(kg, 3),
                "storage": storage,
                "days_held": _clean_positive(entry.get("days_held", 0), 3650.0),
                "shelf_life_days": shelf_life_days(item_name, storage),
                "risk": risk,
                "co2_at_risk_kg": round(kg * item["co2_per_kg"] * risk, 3),
                "value_at_risk": round(kg * item["price_per_kg"] * risk, 2),
            }
        )

    ranked.sort(key=lambda entry: entry["co2_at_risk_kg"], reverse=True)
    return ranked[:limit] if limit else ranked


# ---------------------------------------------------------------------------
# The footprint of thrown-away food
# ---------------------------------------------------------------------------

def avoidable_split(item_name, kg):
    """Split a discarded mass into the part that could have been eaten and the
    part that never could.

    Peel, bones and shells were always going to be discarded; they are the cost
    of eating the food, not a failure. Only the avoidable share is actionable.
    """
    item = get_food_item(item_name)
    if not item:
        return None

    total = _clean_positive(kg, 1000.0)
    unavoidable = total * min(1.0, max(0.0, item["unavoidable"]))

    return {
        "item": item_name,
        "total_kg": round(total, 3),
        "avoidable_kg": round(total - unavoidable, 3),
        "unavoidable_kg": round(unavoidable, 3),
        "avoidable_share": round(1 - item["unavoidable"], 3),
    }


def waste_footprint(item_name, kg, disposal=DEFAULT_DISPOSAL, count_unavoidable=False):
    """The full footprint of throwing food away.

    By default only the avoidable share is charged to waste, because the peel
    was always going to be discarded. ``count_unavoidable`` reports the whole
    discarded mass instead, for anyone comparing against a bin-weight figure.
    """
    item = get_food_item(item_name)
    if not item:
        return None

    split = avoidable_split(item_name, kg)
    charged_kg = split["total_kg"] if count_unavoidable else split["avoidable_kg"]

    production_co2 = charged_kg * item["co2_per_kg"]
    # Disposal emissions apply to everything that goes in the bin, peel included.
    disposal_co2 = split["total_kg"] * disposal_factor(disposal)
    total_co2 = production_co2 + disposal_co2

    return {
        "item": item_name,
        "category": item["category"],
        "disposal": disposal,
        "total_kg": split["total_kg"],
        "avoidable_kg": split["avoidable_kg"],
        "unavoidable_kg": split["unavoidable_kg"],
        "production_co2_kg": round(production_co2, 3),
        "disposal_co2_kg": round(disposal_co2, 3),
        "co2_kg": round(total_co2, 3),
        "water_litres": round(charged_kg * item["water_per_kg"], 1),
        "money": round(charged_kg * item["price_per_kg"], 2),
        "meals_equivalent": round(charged_kg / MEAL_KG, 1) if MEAL_KG > 0 else 0.0,
        "production_share_pct": (
            round(production_co2 / total_co2 * 100, 1) if total_co2 > 0 else 0.0
        ),
    }


def compare_disposal_routes(item_name, kg):
    """The same discarded food down every disposal route, best first.

    The point of this table is usually the opposite of what people expect:
    composting helps, but it changes far less than not buying the food would.
    """
    rows = []
    for route in DISPOSAL_ROUTES:
        result = waste_footprint(item_name, kg, route)
        if result:
            rows.append(
                {
                    "route": route,
                    "note": DISPOSAL_ROUTES[route]["note"],
                    "disposal_co2_kg": result["disposal_co2_kg"],
                    "total_co2_kg": result["co2_kg"],
                    "production_share_pct": result["production_share_pct"],
                }
            )
    rows.sort(key=lambda row: row["total_co2_kg"])
    return rows


def summarise_log(entries, weeks=1):
    """Aggregate a waste log into totals, offenders and an annual figure."""
    records = [entry for entry in (entries or []) if entry]
    period_weeks = max(1e-6, _clean_positive(weeks, 520.0, 1.0) or 1.0)

    if not records:
        return {
            "entry_count": 0, "total_kg": 0.0, "avoidable_kg": 0.0,
            "co2_kg": 0.0, "production_co2_kg": 0.0, "disposal_co2_kg": 0.0,
            "water_litres": 0.0, "money": 0.0, "meals_equivalent": 0.0,
            "annual_co2_kg": 0.0, "annual_money": 0.0,
            "worst_item": None, "by_category": {}, "production_share_pct": 0.0,
        }

    total_kg = sum(entry["total_kg"] for entry in records)
    avoidable_kg = sum(entry["avoidable_kg"] for entry in records)
    production = sum(entry["production_co2_kg"] for entry in records)
    disposal = sum(entry["disposal_co2_kg"] for entry in records)
    co2 = production + disposal
    money = sum(entry["money"] for entry in records)

    by_category = {}
    for entry in records:
        bucket = by_category.setdefault(
            entry["category"], {"kg": 0.0, "co2_kg": 0.0, "money": 0.0}
        )
        bucket["kg"] = round(bucket["kg"] + entry["total_kg"], 3)
        bucket["co2_kg"] = round(bucket["co2_kg"] + entry["co2_kg"], 3)
        bucket["money"] = round(bucket["money"] + entry["money"], 2)

    worst = max(records, key=lambda entry: entry["co2_kg"])

    return {
        "entry_count": len(records),
        "total_kg": round(total_kg, 3),
        "avoidable_kg": round(avoidable_kg, 3),
        "unavoidable_kg": round(total_kg - avoidable_kg, 3),
        "co2_kg": round(co2, 3),
        "production_co2_kg": round(production, 3),
        "disposal_co2_kg": round(disposal, 3),
        "production_share_pct": round(production / co2 * 100, 1) if co2 > 0 else 0.0,
        "water_litres": round(sum(entry["water_litres"] for entry in records), 1),
        "money": round(money, 2),
        "meals_equivalent": round(sum(entry["meals_equivalent"] for entry in records), 1),
        "annual_co2_kg": round(co2 / period_weeks * WEEKS_PER_YEAR, 1),
        "annual_money": round(money / period_weeks * WEEKS_PER_YEAR, 2),
        "worst_item": worst,
        "by_category": by_category,
    }


def undercount_vs_disposal_only(summary):
    """How far a disposal-only figure understates the real footprint.

    This is the comparison against the existing flat "food scraps" factor, and
    it is the reason the module exists.
    """
    if not summary or not summary.get("co2_kg"):
        return None

    disposal_only = summary["disposal_co2_kg"]
    full = summary["co2_kg"]

    return {
        "disposal_only_kg": round(disposal_only, 3),
        "full_kg": round(full, 3),
        "missing_kg": round(full - disposal_only, 3),
        "multiple": round(full / disposal_only, 1) if disposal_only > 0 else None,
    }


def over_purchase_diagnosis(bought_kg, wasted_kg):
    """What share of the shopping never got eaten."""
    bought = _clean_positive(bought_kg, 10000.0)
    wasted = min(bought, _clean_positive(wasted_kg, 10000.0))

    share = wasted / bought * 100 if bought > 0 else 0.0

    if bought <= 0:
        verdict = "Enter what you bought to see how much of it was wasted."
    elif share >= 25:
        verdict = (
            f"A quarter or more of your shopping is being thrown away ({share:.0f}%). "
            f"This is a buying problem, not a storage problem."
        )
    elif share >= 12:
        verdict = (
            f"{share:.0f}% of your shopping is wasted - around the household average. "
            f"Meal planning and a smaller weekly shop move this the fastest."
        )
    elif share > 0:
        verdict = f"Only {share:.0f}% wasted, which is well below the household average."
    else:
        verdict = "Nothing recorded as wasted."

    return {
        "bought_kg": round(bought, 3),
        "wasted_kg": round(wasted, 3),
        "eaten_kg": round(bought - wasted, 3),
        "waste_share_pct": round(share, 1),
        "verdict": verdict,
    }


def get_waste_tips(summary, inventory_risks=None, limit=6):
    """Advice ranked by what this particular household throws away."""
    if not summary or not summary.get("entry_count"):
        return ["Log something you threw away to see what it actually cost."]

    tips = []

    undercount = undercount_vs_disposal_only(summary)
    if undercount and undercount["multiple"] and undercount["multiple"] > 2:
        tips.append(
            f"Composting changes {undercount['disposal_only_kg']:.1f} kg of this. "
            f"The other {undercount['missing_kg']:.1f} kg was spent growing and "
            f"transporting food nobody ate, and no bin can recover it."
        )

    worst = summary.get("worst_item")
    if worst:
        tips.append(
            f"{worst['item']} is your single worst line: {worst['co2_kg']:.1f} kg CO₂e "
            f"from {worst['total_kg']:.2f} kg thrown out. Buying less of it beats "
            f"managing all the rest better."
        )

    if summary["avoidable_kg"] > 0 and summary["total_kg"] > 0:
        avoidable_share = summary["avoidable_kg"] / summary["total_kg"] * 100
        tips.append(
            f"{avoidable_share:.0f}% of what you threw out was edible. Peel and bones "
            f"are the cost of eating; the rest is the part worth attacking."
        )

    if inventory_risks:
        urgent = [entry for entry in inventory_risks if entry["risk"] >= 0.5]
        if urgent:
            names = ", ".join(entry["item"] for entry in urgent[:3])
            tips.append(f"Use or freeze these before they go: {names}.")

    tips.append(
        f"At this rate you throw away about {summary['annual_co2_kg']:,.0f} kg CO₂e "
        f"and {summary['annual_money']:,.0f} of food a year."
    )
    tips.append(
        "Freeze bread, cooked leftovers and meat on the day you buy them, not on the "
        "day they turn - freezing extends shelf life roughly twenty-fold."
    )
    tips.append(
        "Shop twice for half as much. Most household waste comes from buying for the "
        "week you planned rather than the week you had."
    )

    return tips[: max(0, int(limit))]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _get_conn():
    return sqlite3.connect(DB_NAME)


def init_food_waste_db():
    """Create the food waste log table if it does not exist yet."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS food_waste_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item TEXT NOT NULL,
                category TEXT NOT NULL,
                disposal TEXT NOT NULL,
                total_kg REAL NOT NULL,
                avoidable_kg REAL NOT NULL,
                co2_kg REAL NOT NULL,
                production_co2_kg REAL NOT NULL,
                disposal_co2_kg REAL NOT NULL,
                water_litres REAL NOT NULL,
                money REAL NOT NULL,
                reason TEXT,
                detail_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Food waste init error: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def log_waste(user_id, footprint, reason=None):
    """Persist one waste entry. Returns the new row id or None."""
    init_food_waste_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute(
            """
            INSERT INTO food_waste_log (
                user_id, item, category, disposal, total_kg, avoidable_kg,
                co2_kg, production_co2_kg, disposal_co2_kg, water_litres,
                money, reason, detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                footprint.get("item", ""),
                footprint.get("category", ""),
                footprint.get("disposal", DEFAULT_DISPOSAL),
                footprint.get("total_kg", 0.0),
                footprint.get("avoidable_kg", 0.0),
                footprint.get("co2_kg", 0.0),
                footprint.get("production_co2_kg", 0.0),
                footprint.get("disposal_co2_kg", 0.0),
                footprint.get("water_litres", 0.0),
                footprint.get("money", 0.0),
                (reason or "").strip() or None,
                json.dumps(
                    {
                        "unavoidable_kg": footprint.get("unavoidable_kg", 0.0),
                        "meals_equivalent": footprint.get("meals_equivalent", 0.0),
                        "production_share_pct": footprint.get("production_share_pct", 0.0),
                    }
                ),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("Unable to log food waste: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


def get_waste_log(user_id, limit=100):
    """Return a user's waste log, newest first."""
    init_food_waste_db()
    conn = None
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, item, category, disposal, total_kg, avoidable_kg, co2_kg,
                   production_co2_kg, disposal_co2_kg, water_litres, money,
                   reason, detail_json, created_at
            FROM food_waste_log
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

        entries = []
        for row in rows:
            record = dict(row)
            try:
                detail = json.loads(record.pop("detail_json"))
            except (TypeError, ValueError):
                detail = {}
            record["unavoidable_kg"] = detail.get("unavoidable_kg", 0.0)
            record["meals_equivalent"] = detail.get("meals_equivalent", 0.0)
            record["production_share_pct"] = detail.get("production_share_pct", 0.0)
            entries.append(record)
        return entries
    except sqlite3.Error as exc:
        logger.error("Unable to load food waste log: %s", exc)
        return []
    finally:
        if conn:
            conn.close()


def delete_waste_entry(entry_id):
    """Delete one logged waste entry."""
    init_food_waste_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM food_waste_log WHERE id = ?", (entry_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Unable to delete food waste entry: %s", exc)
        return False
    finally:
        if conn:
            conn.close()
