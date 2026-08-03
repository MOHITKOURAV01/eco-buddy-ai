"""Last-mile delivery and packaging footprint for online shopping.

Online retail is invisible in this app. ``emissions.py`` covers the user's own
transport - the trips they take - but not the trips taken *for* them. A
household that has stopped driving to shops has not eliminated those journeys,
it has outsourced them, and the app currently records that as a pure win.

It usually is a win. A full van serving forty doorsteps beats forty cars. But
that advantage is not automatic, and it collapses in four specific ways this
module models explicitly:

* **Speed.** Express and same-day shipping break consolidation. A parcel that
  must move now travels on a partly empty vehicle, and at the fastest tiers it
  travels by air.
* **Failed attempts.** Nobody home means the entire last mile happens again.
* **Returns.** The worst case: the journey is paid twice, and a returned item
  is not always resold.
* **Collection trips.** Driving specially to a pickup point can be worse than
  letting the van come.

Every lever the module surfaces is free - order less often, wait a day, be
home, return less - which is why it is worth modelling at all.

The module is self-contained: its SQLite table is created lazily and no shared
files are modified.
"""

import os
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

# kg CO2e per parcel-kilometre on the final leg. These are per-parcel figures,
# so the van's efficiency is already amortised over the drops it makes.
LAST_MILE_VEHICLES = {
    "Diesel van": {
        "co2_per_parcel_km": 0.0195,
        "note": "The industry default, and the baseline everything else beats.",
    },
    "Electric van": {
        "co2_per_parcel_km": 0.0072,
        "note": "Roughly a third of diesel, depending on the grid behind it.",
    },
    "Cargo bike": {
        "co2_per_parcel_km": 0.0008,
        "note": "Near zero, and increasingly common for dense urban routes.",
    },
    "Motorbike / scooter": {
        "co2_per_parcel_km": 0.0410,
        "note": "Fast, but it carries one parcel at a time - the worst per drop.",
    },
    "Private car (collection)": {
        "co2_per_parcel_km": 0.1710,
        "note": "A whole car for one parcel. Only sensible on a trip you were making anyway.",
    },
}

DEFAULT_VEHICLE = "Diesel van"

# How shipping speed affects the per-parcel figure. The multiplier reflects
# collapsing load factors: a parcel that has to move now travels with fewer
# companions, on a less optimal route, and at the top tier partly by air.
SHIPPING_SPEEDS = {
    "Standard (3-5 days)": {
        "multiplier": 1.00, "air_share": 0.0,
        "note": "Fully consolidated - the van is packed and the route is planned.",
    },
    "Two-day": {
        "multiplier": 1.35, "air_share": 0.05,
        "note": "Some consolidation lost, and a little freight moves by air.",
    },
    "Next-day": {
        "multiplier": 2.10, "air_share": 0.25,
        "note": "Load factors drop sharply and a quarter of it flies.",
    },
    "Same-day": {
        "multiplier": 3.40, "air_share": 0.35,
        "note": "Often a dedicated vehicle for a single parcel.",
    },
}

DEFAULT_SPEED = "Standard (3-5 days)"

# Extra kg CO2e per parcel-km for the share that moves by air.
AIR_FREIGHT_PER_PARCEL_KM = 0.58

# Packaging mass in kg by parcel size, and the materials that make it up.
PARCEL_SIZES = {
    "Letter / small": {"packaging_kg": 0.03, "note": "An envelope or padded mailer."},
    "Medium box": {"packaging_kg": 0.22, "note": "The standard shoebox-sized parcel."},
    "Large box": {"packaging_kg": 0.55, "note": "Bulky goods, and a lot of void fill."},
    "Oversized": {"packaging_kg": 1.80, "note": "Furniture and appliances."},
}

DEFAULT_PARCEL_SIZE = "Medium box"

# kg CO2e per kg of packaging material.
PACKAGING_MATERIALS = {
    "Corrugated cardboard": {"co2_per_kg": 0.94, "note": "Widely recycled, and the default."},
    "Recycled cardboard": {"co2_per_kg": 0.62, "note": "Same box, lower input burden."},
    "Paper mailer": {"co2_per_kg": 1.10, "note": "Light, though the paper itself is intensive."},
    "Plastic mailer": {"co2_per_kg": 2.35, "note": "Very light per unit, but rarely recycled."},
    "Bubble wrap": {"co2_per_kg": 3.10, "note": "Void fill is the worst part of most parcels."},
    "Air pillows": {"co2_per_kg": 2.05, "note": "Mostly air, which is the point."},
    "Moulded pulp": {"co2_per_kg": 0.78, "note": "Made from waste paper and compostable."},
}

DEFAULT_PACKAGING_MATERIAL = "Corrugated cardboard"

# Warehouse picking, packing and sortation, per parcel handled.
FULFILMENT_CO2_PER_PARCEL = 0.38

# Typical distance from the final sortation depot to the doorstep.
DEFAULT_LAST_MILE_KM = 12.0

# A returned item travels back and is re-handled. Not every return is resold:
# some is liquidated or destroyed, and that writes off the item's own embodied
# carbon, which is why returns are the most expensive habit on this page.
RETURN_HANDLING_MULTIPLIER = 1.15
DEFAULT_RESALE_PROBABILITY = 0.75
DEFAULT_ITEM_EMBODIED_CO2 = 8.0

# A failed delivery repeats the last mile, though not the packaging or the
# warehouse handling - the parcel already exists.
DEFAULT_DELIVERY_ATTEMPTS = 1


def list_vehicles():
    """Return last-mile vehicles, cleanest per parcel first."""
    return sorted(
        ({"name": name, **info} for name, info in LAST_MILE_VEHICLES.items()),
        key=lambda item: item["co2_per_parcel_km"],
    )


def list_speeds():
    """Return shipping speeds, most consolidated first."""
    return sorted(
        ({"name": name, **info} for name, info in SHIPPING_SPEEDS.items()),
        key=lambda item: item["multiplier"],
    )


def list_packaging_materials():
    """Return packaging materials, lowest carbon first."""
    return sorted(
        ({"name": name, **info} for name, info in PACKAGING_MATERIALS.items()),
        key=lambda item: item["co2_per_kg"],
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


def _clean_count(value, maximum=100000):
    """Whole, non-negative counts."""
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(count, maximum))


def _clean_probability(value, default=0.0):
    """Clamp a share into 0-1."""
    return max(0.0, min(1.0, _clean_number(value, 1.0, default)))


def transport_footprint(
    distance_km=DEFAULT_LAST_MILE_KM,
    vehicle=DEFAULT_VEHICLE,
    speed=DEFAULT_SPEED,
):
    """kg CO2e for moving one parcel down the last mile.

    Speed enters twice: once as a multiplier on the road leg, because faster
    parcels travel on emptier vehicles, and once as an air-freight share,
    because at the top tiers part of the journey simply flies.
    """
    distance = _clean_number(distance_km, 20000.0, DEFAULT_LAST_MILE_KM)
    vehicle_info = LAST_MILE_VEHICLES.get(vehicle, LAST_MILE_VEHICLES[DEFAULT_VEHICLE])
    speed_info = SHIPPING_SPEEDS.get(speed, SHIPPING_SPEEDS[DEFAULT_SPEED])

    road = distance * vehicle_info["co2_per_parcel_km"] * speed_info["multiplier"]
    air = distance * speed_info["air_share"] * AIR_FREIGHT_PER_PARCEL_KM

    return {
        "distance_km": round(distance, 2),
        "road_co2_kg": round(road, 4),
        "air_co2_kg": round(air, 4),
        "co2_kg": round(road + air, 4),
    }


def packaging_footprint(parcel_size=DEFAULT_PARCEL_SIZE, materials=None, void_fill_share=0.25):
    """kg CO2e for the box and its void fill.

    ``materials`` is a list of material names sharing the packaging mass. Void
    fill is broken out because it is usually the worst part of a parcel by
    carbon intensity while looking like nothing at all.
    """
    size_info = PARCEL_SIZES.get(parcel_size, PARCEL_SIZES[DEFAULT_PARCEL_SIZE])
    mass = size_info["packaging_kg"]
    chosen = [
        name for name in (materials or [DEFAULT_PACKAGING_MATERIAL]) if name in PACKAGING_MATERIALS
    ]
    if not chosen:
        chosen = [DEFAULT_PACKAGING_MATERIAL]

    share = _clean_probability(void_fill_share)
    per_material_mass = mass / len(chosen)

    total = sum(
        per_material_mass * PACKAGING_MATERIALS[name]["co2_per_kg"] for name in chosen
    )

    return {
        "packaging_kg": round(mass, 3),
        "materials": chosen,
        "void_fill_share": round(share, 2),
        "co2_kg": round(total, 4),
    }


def parcel_footprint(
    distance_km=DEFAULT_LAST_MILE_KM,
    vehicle=DEFAULT_VEHICLE,
    speed=DEFAULT_SPEED,
    parcel_size=DEFAULT_PARCEL_SIZE,
    materials=None,
    attempts=DEFAULT_DELIVERY_ATTEMPTS,
):
    """The complete footprint of one parcel arriving at a doorstep."""
    transport = transport_footprint(distance_km, vehicle, speed)
    packaging = packaging_footprint(parcel_size, materials)
    delivery_attempts = max(1, _clean_count(attempts, 20))

    # Repeated attempts redo the last mile only - the box and the warehouse
    # handling already happened the first time.
    transport_total = transport["co2_kg"] * delivery_attempts

    total = transport_total + packaging["co2_kg"] + FULFILMENT_CO2_PER_PARCEL

    return {
        "attempts": delivery_attempts,
        "transport_co2_kg": round(transport_total, 4),
        "packaging_co2_kg": packaging["co2_kg"],
        "fulfilment_co2_kg": FULFILMENT_CO2_PER_PARCEL,
        "air_co2_kg": round(transport["air_co2_kg"] * delivery_attempts, 4),
        "co2_kg": round(total, 4),
        "speed": speed,
        "vehicle": vehicle,
        "parcel_size": parcel_size,
    }


def failed_attempt_cost(base_footprint, attempts):
    """The extra emitted by repeating the last mile when nobody is home."""
    extra = max(0, _clean_count(attempts, 20) - 1)
    transport = base_footprint.get("transport_co2_kg", 0.0) / max(
        1, base_footprint.get("attempts", 1)
    )
    return {
        "failed_attempts": extra,
        "extra_co2_kg": round(transport * extra, 4),
    }


def consolidation_saving(items, orders, per_parcel_footprint):
    """What combining the same items into fewer shipments would save.

    This is the single biggest free lever on the page: the goods are
    identical, only the number of vans changes. Saving is never negative -
    splitting an order into more parcels is a cost, not a saving, and is
    reported as zero here rather than as a negative number.
    """
    item_count = max(1, _clean_count(items, 10000))
    order_count = max(1, min(_clean_count(orders, 10000) or 1, item_count))
    per_parcel = _clean_number(per_parcel_footprint, 10000.0)

    current = order_count * per_parcel
    consolidated = per_parcel  # Everything in one shipment.

    return {
        "items": item_count,
        "orders": order_count,
        "current_co2_kg": round(current, 4),
        "consolidated_co2_kg": round(consolidated, 4),
        "saving_kg": round(max(0.0, current - consolidated), 4),
        "saving_pct": round(
            max(0.0, (current - consolidated) / current * 100) if current else 0.0, 1
        ),
    }


def return_footprint(
    base_footprint,
    return_rate=0.0,
    resale_probability=DEFAULT_RESALE_PROBABILITY,
    item_embodied_co2=DEFAULT_ITEM_EMBODIED_CO2,
):
    """The cost of sending things back.

    A return pays the journey twice and adds re-handling. On top of that,
    whatever share is not resold writes off the item's own embodied carbon,
    which is usually the larger of the two terms and the reason returns are
    the most expensive habit modelled here.
    """
    rate = _clean_probability(return_rate)
    resale = _clean_probability(resale_probability, DEFAULT_RESALE_PROBABILITY)
    embodied = _clean_number(item_embodied_co2, 100000.0, DEFAULT_ITEM_EMBODIED_CO2)
    base = _clean_number(base_footprint, 10000.0)

    journey = base * rate * RETURN_HANDLING_MULTIPLIER
    write_off = embodied * rate * (1.0 - resale)

    return {
        "return_rate": round(rate, 3),
        "resale_probability": round(resale, 3),
        "return_journey_co2_kg": round(journey, 4),
        "unsold_write_off_co2_kg": round(write_off, 4),
        "co2_kg": round(journey + write_off, 4),
    }


def click_and_collect(distance_km, vehicle="Private car (collection)", dedicated_trip=True):
    """The honest comparison for collecting a parcel yourself.

    Collection is only a win if the trip was happening anyway. Driving
    specially to a pickup point puts a whole car on the road for one parcel,
    which is worse than letting the van come - and the function says so rather
    than assuming collection is always greener.
    """
    distance = _clean_number(distance_km, 1000.0)
    vehicle_info = LAST_MILE_VEHICLES.get(vehicle, LAST_MILE_VEHICLES["Private car (collection)"])

    # A round trip, and only attributed at all if it was made specially.
    co2 = distance * 2 * vehicle_info["co2_per_parcel_km"] if dedicated_trip else 0.0
    home_delivery = transport_footprint(DEFAULT_LAST_MILE_KM, DEFAULT_VEHICLE, DEFAULT_SPEED)

    return {
        "distance_km": round(distance, 2),
        "dedicated_trip": bool(dedicated_trip),
        "collection_co2_kg": round(co2, 4),
        "home_delivery_co2_kg": home_delivery["co2_kg"],
        "better_than_delivery": co2 <= home_delivery["co2_kg"],
        "difference_kg": round(co2 - home_delivery["co2_kg"], 4),
    }


def annual_footprint(profile):
    """The yearly total from an ordering profile, split by source.

    ``profile`` is a plain dict; every field falls back to a sensible default
    so a half-filled form still produces a usable answer.
    """
    orders = _clean_count(profile.get("orders_per_year", 24), 10000)
    parcel = parcel_footprint(
        profile.get("distance_km", DEFAULT_LAST_MILE_KM),
        profile.get("vehicle", DEFAULT_VEHICLE),
        profile.get("speed", DEFAULT_SPEED),
        profile.get("parcel_size", DEFAULT_PARCEL_SIZE),
        profile.get("materials"),
        profile.get("attempts", DEFAULT_DELIVERY_ATTEMPTS),
    )

    returns = return_footprint(
        parcel["co2_kg"],
        profile.get("return_rate", 0.0),
        profile.get("resale_probability", DEFAULT_RESALE_PROBABILITY),
        profile.get("item_embodied_co2", DEFAULT_ITEM_EMBODIED_CO2),
    )
    failed = failed_attempt_cost(parcel, profile.get("attempts", DEFAULT_DELIVERY_ATTEMPTS))

    # The parcel total already includes repeated attempts, so the failed-attempt
    # figure is reported separately for visibility rather than added again.
    transport_only = parcel["transport_co2_kg"] - failed["extra_co2_kg"]

    breakdown = {
        "Transport": round(transport_only * orders, 3),
        "Failed attempts": round(failed["extra_co2_kg"] * orders, 3),
        "Packaging": round(parcel["packaging_co2_kg"] * orders, 3),
        "Fulfilment": round(parcel["fulfilment_co2_kg"] * orders, 3),
        "Returns": round(returns["co2_kg"] * orders, 3),
    }

    total = sum(breakdown.values())

    return {
        "orders_per_year": orders,
        "per_parcel_co2_kg": parcel["co2_kg"],
        "total_co2_kg": round(total, 2),
        "breakdown": breakdown,
        "air_co2_kg": round(parcel["air_co2_kg"] * orders, 3),
        "returns_share_pct": round(breakdown["Returns"] / total * 100, 1) if total else 0.0,
        "speed": parcel["speed"],
        "vehicle": parcel["vehicle"],
    }


def optimise_orders(profile, limit=6):
    """Ranked, quantified changes to an ordering habit.

    Each option is evaluated by re-running the whole annual model with that
    one change applied, so the saving reported is the saving actually
    delivered rather than an estimate of one term in isolation.
    """
    current = annual_footprint(profile)
    options = []

    def _consider(action, changes, note):
        candidate = dict(profile)
        candidate.update(changes)
        saving = current["total_co2_kg"] - annual_footprint(candidate)["total_co2_kg"]
        if saving > 0:
            options.append(
                {"action": action, "saving_kg": round(saving, 2), "note": note}
            )

    current_speed = profile.get("speed", DEFAULT_SPEED)
    if current_speed != DEFAULT_SPEED:
        _consider(
            "Choose standard shipping instead of expedited",
            {"speed": DEFAULT_SPEED},
            "Waiting a couple of days lets your parcel travel on a full van.",
        )

    orders = _clean_count(profile.get("orders_per_year", 24), 10000)
    if orders > 12:
        _consider(
            f"Batch to {max(12, orders // 2)} orders a year instead of {orders}",
            {"orders_per_year": max(12, orders // 2)},
            "Same goods, half the vans. The cheapest change on this list.",
        )

    if _clean_probability(profile.get("return_rate", 0.0)) > 0.05:
        halved = _clean_probability(profile.get("return_rate", 0.0)) / 2
        _consider(
            "Halve your return rate by checking sizes before ordering",
            {"return_rate": halved},
            "Returns pay the journey twice, and not everything sent back is resold.",
        )

    if profile.get("attempts", DEFAULT_DELIVERY_ATTEMPTS) > 1:
        _consider(
            "Use a delivery slot you will actually be home for",
            {"attempts": 1},
            "A failed attempt repeats the entire last mile.",
        )

    if profile.get("vehicle", DEFAULT_VEHICLE) == "Diesel van":
        _consider(
            "Pick a retailer using electric vans or cargo bikes",
            {"vehicle": "Electric van"},
            "Not always in your control, but worth choosing where it is.",
        )

    materials = profile.get("materials") or [DEFAULT_PACKAGING_MATERIAL]
    if any(name in ("Bubble wrap", "Plastic mailer", "Air pillows") for name in materials):
        _consider(
            "Ask for paper or moulded-pulp packaging",
            {"materials": ["Moulded pulp"]},
            "Void fill is the worst part of most parcels by carbon intensity.",
        )

    options.sort(key=lambda item: item["saving_kg"], reverse=True)
    return options[: max(0, int(limit))]


def compare_scenarios(profile_a, profile_b):
    """Before-and-after for two ordering habits."""
    first = annual_footprint(profile_a)
    second = annual_footprint(profile_b)
    difference = first["total_co2_kg"] - second["total_co2_kg"]

    return {
        "before_co2_kg": first["total_co2_kg"],
        "after_co2_kg": second["total_co2_kg"],
        "difference_kg": round(difference, 2),
        "improved": difference > 0,
        "change_pct": round(
            (difference / first["total_co2_kg"] * 100) if first["total_co2_kg"] else 0.0, 1
        ),
    }


def get_delivery_tips(result, limit=6):
    """Advice ranked by the user's own breakdown."""
    tips = []
    breakdown = result.get("breakdown", {})
    total = result.get("total_co2_kg", 0.0)

    if total <= 0:
        return ["Enter your ordering habits above to see where the carbon goes."]

    ranked = sorted(breakdown.items(), key=lambda item: item[1], reverse=True)
    biggest, biggest_value = ranked[0]
    tips.append(
        f"{biggest} is your largest source at {biggest_value:.0f} kg CO₂e a year, "
        f"{biggest_value / total * 100:.0f}% of the total. Start there."
    )

    if result.get("returns_share_pct", 0.0) > 20:
        tips.append(
            "Returns are more than a fifth of your footprint. They pay the journey "
            "twice, and whatever is not resold writes off the item entirely — "
            "checking sizes before ordering is worth more than any packaging choice."
        )

    if result.get("air_co2_kg", 0.0) > 1:
        tips.append(
            "Part of your freight is flying. Expedited shipping is the only reason "
            "that happens, and standard delivery removes it completely."
        )

    if breakdown.get("Failed attempts", 0.0) > 0:
        tips.append(
            "Failed deliveries are repeating the entire last mile. A pickup point "
            "or a slot you will be home for eliminates that term."
        )

    if result.get("orders_per_year", 0) > 24:
        tips.append(
            f"{result['orders_per_year']} orders a year is a lot of separate vans. "
            "Batching the same goods into fewer shipments changes nothing about "
            "what you buy and cuts the transport term proportionally."
        )

    tips.append(
        "Delivery usually beats driving to the shops — but not at same-day speed, "
        "not with a high return rate, and not if you drive specially to collect."
    )

    return tips[: max(0, int(limit))]


def _get_conn():
    return sqlite3.connect(DB_NAME)


def init_delivery_db():
    """Create the delivery profile table if it does not exist yet."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                profile_name TEXT NOT NULL,
                orders_per_year INTEGER NOT NULL,
                speed TEXT NOT NULL,
                vehicle TEXT NOT NULL,
                total_co2_kg REAL NOT NULL,
                returns_share_pct REAL,
                profile_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Delivery footprint init error: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def save_delivery_profile(user_id, profile_name, profile):
    """Persist an ordering profile. Returns the new row id or None."""
    init_delivery_db()
    conn = None
    try:
        conn = _get_conn()
        result = annual_footprint(profile)
        cursor = conn.execute(
            """
            INSERT INTO delivery_profiles (
                user_id, profile_name, orders_per_year, speed, vehicle,
                total_co2_kg, returns_share_pct, profile_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                (profile_name or "My ordering").strip() or "My ordering",
                result["orders_per_year"],
                result["speed"],
                result["vehicle"],
                result["total_co2_kg"],
                result["returns_share_pct"],
                json.dumps(profile),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("Unable to save delivery profile: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


def get_delivery_profiles(user_id, limit=25):
    """Return a user's saved ordering profiles, newest first."""
    init_delivery_db()
    conn = None
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, profile_name, orders_per_year, speed, vehicle,
                   total_co2_kg, returns_share_pct, profile_json, created_at
            FROM delivery_profiles
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

        profiles = []
        for row in rows:
            record = dict(row)
            try:
                record["profile"] = json.loads(record.pop("profile_json"))
            except (TypeError, ValueError):
                record["profile"] = {}
            profiles.append(record)
        return profiles
    except sqlite3.Error as exc:
        logger.error("Unable to load delivery profiles: %s", exc)
        return []
    finally:
        if conn:
            conn.close()


def delete_delivery_profile(profile_id):
    """Delete a saved ordering profile."""
    init_delivery_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute(
            "DELETE FROM delivery_profiles WHERE id = ?", (profile_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Unable to delete delivery profile: %s", exc)
        return False
    finally:
        if conn:
            conn.close()
