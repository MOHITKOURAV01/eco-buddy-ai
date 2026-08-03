"""Wardrobe and textile footprint: embodied carbon, water and carbon-per-wear.

Clothing appears in this app in exactly one place - as a disposal stream in
``waste.py``. That is the last couple of percent of a garment's impact. The
overwhelming majority is embodied: fibre production, spinning, dyeing,
finishing and freight, all of it committed at the moment of purchase, long
before anything is thrown away.

This module models the garment itself. The core idea is that the interesting
number is not what a garment cost the planet, but what it costs *per wear*::

    carbon per wear = (embodied carbon + accumulated care emissions) / wears

That single reframing is what makes the maths useful. A wool coat with a large
embodied footprint worn four hundred times is a far better object than a cheap
polyester top worn eight times, and no total-impact figure can show that.

Two consequences follow, and the module is built around them:

* Buying second-hand carries only a fraction of the new embodied burden,
  because that production has already happened regardless.
* Wearing what you already own for longer is the single largest lever, and it
  is modelled explicitly as avoided replacement purchases.

The module is self-contained: its SQLite table is created lazily and no shared
files are modified.
"""

import os
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

# Fibre impacts per kilogram of finished fabric. Carbon covers cultivation or
# polymerisation, spinning, dyeing and finishing; water is the full blue and
# green water footprint, which is why natural fibres can look worse on water
# while looking better on carbon.
FIBRES = {
    "Organic cotton": {
        "co2_per_kg": 15.0, "water_per_kg": 6500.0,
        "note": "Lower inputs than conventional cotton, but still thirsty.",
    },
    "Conventional cotton": {
        "co2_per_kg": 19.0, "water_per_kg": 10500.0,
        "note": "The classic water-intensive crop - irrigation dominates.",
    },
    "Recycled polyester": {
        "co2_per_kg": 12.0, "water_per_kg": 200.0,
        "note": "Roughly half the carbon of virgin polyester, and barely any water.",
    },
    "Polyester": {
        "co2_per_kg": 22.0, "water_per_kg": 350.0,
        "note": "Petrochemical, low water, but sheds microfibres in every wash.",
    },
    "Nylon": {
        "co2_per_kg": 29.0, "water_per_kg": 450.0,
        "note": "The most carbon-intensive common synthetic.",
    },
    "Acrylic": {
        "co2_per_kg": 26.0, "water_per_kg": 400.0,
        "note": "Energy-hungry to produce and prone to pilling, so it ages badly.",
    },
    "Wool": {
        "co2_per_kg": 32.0, "water_per_kg": 5800.0,
        "note": "High at production, but wool garments are worn for years.",
    },
    "Linen": {
        "co2_per_kg": 10.0, "water_per_kg": 2600.0,
        "note": "Flax needs little irrigation and almost no pesticide.",
    },
    "Hemp": {
        "co2_per_kg": 8.5, "water_per_kg": 2300.0,
        "note": "The lowest-impact common natural fibre by both measures.",
    },
    "Viscose": {
        "co2_per_kg": 17.0, "water_per_kg": 3300.0,
        "note": "Wood pulp based; impact hinges on where the forest came from.",
    },
    "Leather": {
        "co2_per_kg": 48.0, "water_per_kg": 17000.0,
        "note": "The highest of any common material, driven by the cattle upstream.",
    },
    "Cotton / polyester blend": {
        "co2_per_kg": 20.5, "water_per_kg": 5400.0,
        "note": "Averages the two, and is much harder to recycle than either.",
    },
}

DEFAULT_FIBRE = "Conventional cotton"

# Typical finished mass of a garment in kilograms, so users enter items rather
# than weighing their clothes.
GARMENT_TYPES = {
    "T-shirt": {"mass_kg": 0.18, "expected_wears": 50},
    "Shirt / blouse": {"mass_kg": 0.25, "expected_wears": 60},
    "Jumper / sweater": {"mass_kg": 0.50, "expected_wears": 80},
    "Jeans": {"mass_kg": 0.70, "expected_wears": 150},
    "Trousers": {"mass_kg": 0.55, "expected_wears": 120},
    "Dress": {"mass_kg": 0.40, "expected_wears": 40},
    "Skirt": {"mass_kg": 0.30, "expected_wears": 50},
    "Jacket": {"mass_kg": 0.90, "expected_wears": 150},
    "Coat": {"mass_kg": 1.60, "expected_wears": 200},
    "Shoes": {"mass_kg": 0.85, "expected_wears": 250},
    "Underwear": {"mass_kg": 0.06, "expected_wears": 80},
    "Socks": {"mass_kg": 0.05, "expected_wears": 80},
    "Activewear top": {"mass_kg": 0.20, "expected_wears": 70},
    "Scarf / accessory": {"mass_kg": 0.15, "expected_wears": 60},
}

DEFAULT_GARMENT = "T-shirt"

# How much of the new embodied impact a purchase actually carries. Second-hand
# and rented items reuse production that has already happened, so only the
# handling and transport of that particular transaction is attributed.
CONDITIONS = {
    "New": {
        "embodied_share": 1.00,
        "note": "Carries the full production burden.",
    },
    "Second-hand": {
        "embodied_share": 0.10,
        "note": "Production already happened - only resale handling counts.",
    },
    "Rented": {
        "embodied_share": 0.15,
        "note": "Shared across many users, but cleaning and shipping add up.",
    },
    "Inherited / gifted": {
        "embodied_share": 0.05,
        "note": "Essentially free in carbon terms - the best way to acquire clothes.",
    },
    "Repaired / altered": {
        "embodied_share": 0.08,
        "note": "A fraction of a replacement, which is the whole point of mending.",
    },
}

DEFAULT_CONDITION = "New"

# Emissions per wash cycle, per kilogram of garment, by wash temperature.
WASH_TEMPERATURES = {
    "Cold (20-30°C)": 0.18,
    "Warm (40°C)": 0.35,
    "Hot (60°C)": 0.62,
    "Very hot (90°C)": 0.95,
}

DEFAULT_WASH_TEMPERATURE = "Warm (40°C)"

TUMBLE_DRY_PER_KG = 1.05  # kg CO2e per kg of garment per drying cycle.
IRON_PER_KG = 0.09        # kg CO2e per kg of garment per pressing.

# Wears between washes. Washing after every single wear is both unusual and a
# meaningful share of a garment's lifetime footprint.
DEFAULT_WEARS_PER_WASH = 3

# Water used per wash cycle per kg, in litres.
WASH_WATER_PER_KG = 18.0

# A garment worn fewer times than this is treated as dead stock: the embodied
# carbon is spent and almost nothing has been got back for it.
DEAD_STOCK_WEAR_THRESHOLD = 5

# Extending a garment's life avoids a replacement purchase. This is the share
# of a replacement avoided per additional year of use, based on typical
# garment replacement cycles.
REPLACEMENT_AVOIDED_PER_YEAR = 0.45


def list_fibres():
    """Return the fibre catalogue, lowest carbon first."""
    return sorted(
        ({"name": name, **info} for name, info in FIBRES.items()),
        key=lambda item: item["co2_per_kg"],
    )


def list_garment_types():
    """Return the garment catalogue, heaviest first."""
    return sorted(
        ({"name": name, **info} for name, info in GARMENT_TYPES.items()),
        key=lambda item: item["mass_kg"],
        reverse=True,
    )


def list_conditions():
    """Return purchase conditions, lowest embodied share first."""
    return sorted(
        ({"name": name, **info} for name, info in CONDITIONS.items()),
        key=lambda item: item["embodied_share"],
    )


def _clean_number(value, maximum, default=0.0):
    """Coerce a user-supplied number into a sane, non-negative range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return max(0.0, min(number, maximum))


def _clean_wears(wears):
    """Wear counts are whole numbers and never negative."""
    try:
        count = int(float(wears))
    except (TypeError, ValueError):
        return 0
    return max(0, min(count, 100000))


def get_garment_mass(category):
    """Return the typical finished mass of a garment type, in kg."""
    return GARMENT_TYPES.get(category, GARMENT_TYPES[DEFAULT_GARMENT])["mass_kg"]


def get_expected_wears(category):
    """Return how many wears a garment type can reasonably be expected to give."""
    return GARMENT_TYPES.get(category, GARMENT_TYPES[DEFAULT_GARMENT])["expected_wears"]


def garment_footprint(category, fibre, condition=DEFAULT_CONDITION, mass_kg=None):
    """Embodied carbon and water in one item, before it is ever worn.

    This is the number committed at the till. Everything the owner does
    afterwards - washing it, or wearing it for a decade - only changes how
    that fixed burden gets divided up.
    """
    fibre_info = FIBRES.get(fibre, FIBRES[DEFAULT_FIBRE])
    share = CONDITIONS.get(condition, CONDITIONS[DEFAULT_CONDITION])["embodied_share"]

    mass = get_garment_mass(category) if mass_kg is None else _clean_number(mass_kg, 50.0)

    return {
        "category": category,
        "fibre": fibre,
        "condition": condition,
        "mass_kg": round(mass, 3),
        "embodied_co2_kg": round(mass * fibre_info["co2_per_kg"] * share, 3),
        "embodied_water_l": round(mass * fibre_info["water_per_kg"] * share, 1),
        "new_co2_kg": round(mass * fibre_info["co2_per_kg"], 3),
    }


def care_footprint(
    wears,
    mass_kg,
    wash_temp=DEFAULT_WASH_TEMPERATURE,
    tumble_dried=False,
    ironed=False,
    wears_per_wash=DEFAULT_WEARS_PER_WASH,
):
    """Emissions and water accumulated by washing, drying and pressing.

    Small per cycle, but it compounds: a garment worn two hundred times and
    tumble dried every time can accumulate care emissions rivalling what it
    took to make.
    """
    worn = _clean_wears(wears)
    mass = _clean_number(mass_kg, 50.0)
    per_wash = max(1, _clean_wears(wears_per_wash) or DEFAULT_WEARS_PER_WASH)

    washes = worn / per_wash if worn else 0.0
    wash_rate = WASH_TEMPERATURES.get(wash_temp, WASH_TEMPERATURES[DEFAULT_WASH_TEMPERATURE])

    co2 = washes * mass * wash_rate
    if tumble_dried:
        co2 += washes * mass * TUMBLE_DRY_PER_KG
    if ironed:
        co2 += worn * mass * IRON_PER_KG

    return {
        "washes": round(washes, 2),
        "care_co2_kg": round(co2, 3),
        "care_water_l": round(washes * mass * WASH_WATER_PER_KG, 1),
    }


def lifetime_footprint(garment):
    """Total impact of one garment across the wears it has actually had.

    ``garment`` is a plain dict with at least ``category`` and ``fibre``; the
    remaining keys fall back to sensible defaults so a half-filled form still
    produces a usable answer.
    """
    category = garment.get("category", DEFAULT_GARMENT)
    fibre = garment.get("fibre", DEFAULT_FIBRE)
    condition = garment.get("condition", DEFAULT_CONDITION)

    embodied = garment_footprint(category, fibre, condition, garment.get("mass_kg"))
    care = care_footprint(
        garment.get("wears", 0),
        embodied["mass_kg"],
        garment.get("wash_temp", DEFAULT_WASH_TEMPERATURE),
        garment.get("tumble_dried", False),
        garment.get("ironed", False),
        garment.get("wears_per_wash", DEFAULT_WEARS_PER_WASH),
    )

    total_co2 = embodied["embodied_co2_kg"] + care["care_co2_kg"]
    total_water = embodied["embodied_water_l"] + care["care_water_l"]

    return {
        **embodied,
        **care,
        "name": garment.get("name", category),
        "wears": _clean_wears(garment.get("wears", 0)),
        "price": _clean_number(garment.get("price", 0.0), 1000000.0),
        "total_co2_kg": round(total_co2, 3),
        "total_water_l": round(total_water, 1),
    }


def carbon_per_wear(garment):
    """Kilograms of CO2e attributable to each time the item was worn.

    Returns ``None`` for an unworn garment rather than dividing by zero. That
    is the honest answer: an item worn zero times has no per-wear figure, it
    simply has a debt.
    """
    result = lifetime_footprint(garment)
    if result["wears"] <= 0:
        return None
    return round(result["total_co2_kg"] / result["wears"], 4)


def cost_per_wear(price, wears):
    """The familiar money version of the same idea."""
    worn = _clean_wears(wears)
    if worn <= 0:
        return None
    return round(_clean_number(price, 1000000.0) / worn, 2)


def water_per_wear(garment):
    """Litres of water attributable to each wear."""
    result = lifetime_footprint(garment)
    if result["wears"] <= 0:
        return None
    return round(result["total_water_l"] / result["wears"], 1)


def wardrobe_summary(garments):
    """Aggregate a whole wardrobe and describe how hard it is working."""
    items = [lifetime_footprint(item) for item in (garments or [])]

    total_co2 = sum(item["total_co2_kg"] for item in items)
    total_water = sum(item["total_water_l"] for item in items)
    total_wears = sum(item["wears"] for item in items)
    total_spend = sum(item["price"] for item in items)

    condition_split = {}
    fibre_split = {}
    for item in items:
        condition_split[item["condition"]] = condition_split.get(item["condition"], 0) + 1
        fibre_split[item["fibre"]] = round(
            fibre_split.get(item["fibre"], 0.0) + item["total_co2_kg"], 3
        )

    return {
        "item_count": len(items),
        "total_co2_kg": round(total_co2, 2),
        "total_water_l": round(total_water, 1),
        "total_spend": round(total_spend, 2),
        "total_wears": total_wears,
        "average_wears": round(total_wears / len(items), 1) if items else 0.0,
        "average_co2_per_item_kg": round(total_co2 / len(items), 3) if items else 0.0,
        "carbon_per_wear_kg": round(total_co2 / total_wears, 4) if total_wears else None,
        "condition_split": condition_split,
        "fibre_split": fibre_split,
        "utilisation_score": utilisation_score(garments),
        "items": items,
    }


def utilisation_score(garments):
    """Score 0-100 for how hard a wardrobe works, not how small it is.

    Thirty well-worn items beat a hundred idle ones, so the score compares
    each garment's actual wears against what its type could reasonably give,
    then averages. A wardrobe of items worn to their full expected life scores
    100; one full of things worn twice scores near zero.
    """
    items = list(garments or [])
    if not items:
        return 0.0

    ratios = []
    for item in items:
        category = item.get("category", DEFAULT_GARMENT)
        expected = max(1, get_expected_wears(category))
        ratios.append(min(1.0, _clean_wears(item.get("wears", 0)) / expected))

    return round(sum(ratios) / len(ratios) * 100, 1)


def find_dead_stock(garments, threshold=DEAD_STOCK_WEAR_THRESHOLD):
    """Barely-worn items, ranked by how much embodied carbon is going to waste.

    These are the wardrobe's real problem: the carbon is already spent and
    nothing has been got back for it. Wearing one of these is free.
    """
    limit = _clean_wears(threshold)
    dead = [
        lifetime_footprint(item)
        for item in (garments or [])
        if _clean_wears(item.get("wears", 0)) <= limit
    ]
    return sorted(dead, key=lambda item: item["embodied_co2_kg"], reverse=True)


def extend_life_saving(garments, extra_years=1.0):
    """Carbon avoided by keeping what you own instead of replacing it.

    Modelled as avoided replacement purchases: each additional year of use
    displaces a share of a new garment, at that garment's own new-item
    footprint rather than a generic average.
    """
    years = _clean_number(extra_years, 50.0)
    items = [lifetime_footprint(item) for item in (garments or [])]

    avoided = sum(
        item["new_co2_kg"] * REPLACEMENT_AVOIDED_PER_YEAR * years for item in items
    )
    avoided_water = sum(
        item["embodied_water_l"] * REPLACEMENT_AVOIDED_PER_YEAR * years for item in items
    )

    return {
        "extra_years": round(years, 2),
        "items_covered": len(items),
        "co2_saved_kg": round(avoided, 2),
        "water_saved_l": round(avoided_water, 1),
        "replacements_avoided": round(len(items) * REPLACEMENT_AVOIDED_PER_YEAR * years, 1),
    }


def compare_purchase(category, fibre, expected_wears=None, price=0.0):
    """Answer "should I buy this?" across new, second-hand and not at all.

    Returns the options sorted cleanest first. Not buying is always included,
    because it is always an option and it is always the winner.
    """
    wears = _clean_wears(expected_wears) or get_expected_wears(category)

    options = []
    for condition in ("New", "Second-hand", "Rented", "Inherited / gifted"):
        garment = {
            "category": category,
            "fibre": fibre,
            "condition": condition,
            "wears": wears,
            "price": price,
        }
        result = lifetime_footprint(garment)
        options.append(
            {
                "condition": condition,
                "total_co2_kg": result["total_co2_kg"],
                "embodied_co2_kg": result["embodied_co2_kg"],
                "carbon_per_wear_kg": carbon_per_wear(garment),
                "note": CONDITIONS[condition]["note"],
            }
        )

    options.append(
        {
            "condition": "Do not buy",
            "total_co2_kg": 0.0,
            "embodied_co2_kg": 0.0,
            "carbon_per_wear_kg": 0.0,
            "note": "Wearing something you already own costs nothing extra.",
        }
    )

    return sorted(options, key=lambda item: item["total_co2_kg"])


def get_wardrobe_tips(summary, limit=6):
    """Advice ranked by the user's own wardrobe rather than a generic list."""
    tips = []
    items = summary.get("items", [])

    if not items:
        return ["Add a few garments to see where your wardrobe's carbon actually sits."]

    score = summary.get("utilisation_score", 0.0)
    if score < 30:
        tips.append(
            f"Your utilisation score is {score:.0f}/100 — most of these clothes are "
            "barely worn. The carbon is already spent, so wearing what is hanging "
            "there is the single cheapest reduction available to you."
        )
    elif score > 70:
        tips.append(
            f"A utilisation score of {score:.0f}/100 is genuinely good: you wear "
            "what you own, which matters far more than what it is made of."
        )

    dead = find_dead_stock(items)
    if dead:
        worst = dead[0]
        tips.append(
            f"{len(dead)} item(s) have been worn {DEAD_STOCK_WEAR_THRESHOLD} times or "
            f"fewer. The worst is the {worst['name'].lower()} at "
            f"{worst['embodied_co2_kg']:.1f} kg CO₂e with almost nothing to show for it."
        )

    worn = [item for item in items if item["wears"] > 0]
    if worn:
        priciest = max(worn, key=lambda item: item["total_co2_kg"] / item["wears"])
        tips.append(
            f"Highest carbon-per-wear: the {priciest['name'].lower()} at "
            f"{priciest['total_co2_kg'] / priciest['wears']:.3f} kg per wear. "
            "Wearing it more is the only thing that improves that number."
        )

    hot_wash = [item for item in items if item.get("washes", 0) > 5]
    if hot_wash:
        tips.append(
            "Washing cold and skipping the tumble dryer cuts care emissions by "
            "well over half, and clothes last longer for it."
        )

    new_items = summary.get("condition_split", {}).get("New", 0)
    if new_items and new_items / len(items) > 0.8:
        tips.append(
            f"{new_items} of {len(items)} items were bought new. Second-hand carries "
            "roughly a tenth of the embodied burden — the production already happened."
        )

    tips.append(
        "Keeping clothes in use for one more year is worth more than any fibre "
        "choice you can make at the till."
    )

    return tips[: max(0, int(limit))]


def _get_conn():
    return sqlite3.connect(DB_NAME)


def init_wardrobe_db():
    """Create the wardrobe table if it does not exist yet."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wardrobe_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                fibre TEXT NOT NULL,
                condition TEXT NOT NULL,
                price REAL DEFAULT 0,
                wears INTEGER DEFAULT 0,
                embodied_co2_kg REAL NOT NULL,
                total_co2_kg REAL NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Wardrobe init error: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def save_garment(user_id, garment):
    """Persist one garment. Returns the new row id or None."""
    init_wardrobe_db()
    conn = None
    try:
        conn = _get_conn()
        result = lifetime_footprint(garment)
        cursor = conn.execute(
            """
            INSERT INTO wardrobe_items (
                user_id, name, category, fibre, condition, price, wears,
                embodied_co2_kg, total_co2_kg, detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                (garment.get("name") or result["category"]).strip() or result["category"],
                result["category"],
                result["fibre"],
                result["condition"],
                result["price"],
                result["wears"],
                result["embodied_co2_kg"],
                result["total_co2_kg"],
                json.dumps(
                    {
                        "wash_temp": garment.get("wash_temp", DEFAULT_WASH_TEMPERATURE),
                        "tumble_dried": bool(garment.get("tumble_dried", False)),
                        "ironed": bool(garment.get("ironed", False)),
                        "wears_per_wash": garment.get(
                            "wears_per_wash", DEFAULT_WEARS_PER_WASH
                        ),
                        "mass_kg": result["mass_kg"],
                    }
                ),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("Unable to save garment: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


def get_garments(user_id, limit=200):
    """Return a user's wardrobe, newest first, ready to feed back into the model."""
    init_wardrobe_db()
    conn = None
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, name, category, fibre, condition, price, wears,
                   embodied_co2_kg, total_co2_kg, detail_json, created_at
            FROM wardrobe_items
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

        garments = []
        for row in rows:
            record = dict(row)
            try:
                detail = json.loads(record.pop("detail_json"))
            except (TypeError, ValueError):
                detail = {}
            record.update(detail)
            garments.append(record)
        return garments
    except sqlite3.Error as exc:
        logger.error("Unable to load wardrobe: %s", exc)
        return []
    finally:
        if conn:
            conn.close()


def log_wear(garment_id, times=1):
    """Record that an item was worn, and refresh its stored lifetime total."""
    init_wardrobe_db()
    conn = None
    try:
        added = max(1, _clean_wears(times) or 1)
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT category, fibre, condition, wears, price, detail_json
            FROM wardrobe_items WHERE id = ?
            """,
            (garment_id,),
        ).fetchone()
        if row is None:
            return False

        record = dict(row)
        try:
            detail = json.loads(record.get("detail_json") or "{}")
        except (TypeError, ValueError):
            detail = {}

        new_wears = _clean_wears(record["wears"]) + added
        refreshed = lifetime_footprint(
            {
                "category": record["category"],
                "fibre": record["fibre"],
                "condition": record["condition"],
                "price": record["price"],
                "wears": new_wears,
                **detail,
            }
        )

        conn.execute(
            "UPDATE wardrobe_items SET wears = ?, total_co2_kg = ? WHERE id = ?",
            (new_wears, refreshed["total_co2_kg"], garment_id),
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Unable to log wear: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def delete_garment(garment_id):
    """Remove a garment from the wardrobe."""
    init_wardrobe_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM wardrobe_items WHERE id = ?", (garment_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Unable to delete garment: %s", exc)
        return False
    finally:
        if conn:
            conn.close()
