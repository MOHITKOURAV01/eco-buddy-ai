"""Air travel footprint with distance bands, cabin class and radiative forcing.

The main assessment in ``app.py`` asks for a single number - "annual flights" -
and multiplies it by one flat factor. That is the least accurate line in the
whole calculator, because two flights can differ by a factor of forty:

* a 400 km hop and a 12,000 km haul are both "one flight";
* an economy seat and a first-class suite on the same aircraft differ by ~4x,
  because a premium seat occupies the floor space of several economy seats;
* half of aviation's warming does not come from CO2 at all. Contrails, soot and
  nitrogen oxides released at cruise altitude roughly double the effect, and
  they only happen on the cruise portion of a flight.

This module models each of those explicitly.

Method
------
Fuel burn per passenger-kilometre is not constant with distance. Take-off is
enormously fuel-hungry and is amortised over the whole trip, so a short flight
burns far more per kilometre than a long one. That is modelled as a fixed
landing-and-take-off (LTO) charge per leg plus a cruise rate per kilometre::

    co2_kg = lto_kg_per_leg * legs + cruise_kg_per_km * cruise_km

which reproduces the familiar shape of published distance-band factors while
also making the cost of a connection visible: two legs mean two take-offs.

Radiative forcing is applied to the *cruise* portion only, because contrails
form at altitude and a short hop that never reaches cruise level produces
almost none. The multiplier is reported separately from the CO2 so a user can
see both the regulatory number and the honest one.

The module is self-contained: it imports no Streamlit, its SQLite table is
created lazily and no shared files are modified.
"""

import os
import json
import math
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

EARTH_RADIUS_KM = 6371.0088

# Real routes are not great circles: air traffic control, holding patterns and
# waypoint routing add distance. DEFRA adds 95 km to every great-circle leg.
ROUTING_UPLIFT_KM = 95.0

# Fixed CO2 charge per passenger for one take-off and landing cycle, in kg.
# This is what makes a connection expensive and short hops inefficient.
LTO_KG_PER_LEG = 12.5

# Kilometres of a leg treated as climb/descent rather than cruise. Below this
# a flight barely reaches contrail-forming altitude at all.
CLIMB_DESCENT_KM = 300.0

# Cruise CO2 in kg per passenger-kilometre, by aircraft duty. Longer flights
# use larger, more efficient aircraft but carry more fuel to carry the fuel.
CRUISE_KG_PER_KM = {
    "short": 0.121,   # regional and narrow-body, under 1,500 km
    "medium": 0.097,  # narrow-body at its efficient range
    "long": 0.088,    # wide-body, over 3,700 km
}

SHORT_HAUL_MAX_KM = 1500.0
MEDIUM_HAUL_MAX_KM = 3700.0

# Seat-space multipliers. A premium seat is charged for the floor area it
# occupies, which is why business class is roughly three economy seats.
CABIN_CLASSES = {
    "Economy": {
        "multiplier": 1.0,
        "note": "The baseline: the aircraft's emissions divided by a full economy cabin.",
    },
    "Premium Economy": {
        "multiplier": 1.6,
        "note": "About 60% more floor space per passenger than economy.",
    },
    "Business": {
        "multiplier": 2.9,
        "note": "A lie-flat seat occupies the space of roughly three economy seats.",
    },
    "First": {
        "multiplier": 4.0,
        "note": "A suite can take the floor area of four or more economy seats.",
    },
}

DEFAULT_CABIN = "Economy"

# Premium cabins barely exist on short regional aircraft, so the seat-space
# penalty is capped there - a "business" seat on a 1-hour hop is an economy
# seat with the middle one blocked.
SHORT_HAUL_CABIN_CAP = 1.3

# Non-CO2 effects (contrail cirrus, NOx, soot) as a multiplier on cruise CO2.
# 1.9 sits inside the range in Lee et al. (2021); the uncertainty is real and
# is surfaced to the user rather than hidden.
RADIATIVE_FORCING_MULTIPLIER = 1.9
RF_RANGE = (1.4, 2.7)

# Typical passenger load factor. An emptier aircraft spreads the same fuel
# across fewer people.
DEFAULT_LOAD_FACTOR = 0.82

# kg CO2e per passenger-kilometre for the surface alternatives, for the
# "could I have taken the train?" comparison.
SURFACE_ALTERNATIVES = {
    "High-speed rail": 0.006,
    "Intercity rail": 0.035,
    "Long-distance coach": 0.027,
    "Car (1 occupant)": 0.171,
    "Car (3 occupants)": 0.057,
}

# A day of video calls instead of the journey, in kg CO2e. It does not scale
# with distance - that is the whole point of it.
VIDEO_CALL_KG = 0.5

# Rail and coach stop being plausible substitutes well before the aircraft does.
RAIL_PLAUSIBLE_MAX_KM = 1000.0
COACH_PLAUSIBLE_MAX_KM = 700.0

# A widely used per-person annual allowance consistent with a 1.5C pathway.
PERSONAL_ANNUAL_BUDGET_KG = 2300.0

# A small offline airport table so the feature works with no network access.
# Coordinates are in decimal degrees.
AIRPORTS = {
    "AMS": {"name": "Amsterdam Schiphol", "city": "Amsterdam", "country": "Netherlands", "lat": 52.3105, "lon": 4.7683},
    "ATL": {"name": "Hartsfield-Jackson", "city": "Atlanta", "country": "United States", "lat": 33.6407, "lon": -84.4277},
    "BCN": {"name": "Barcelona El Prat", "city": "Barcelona", "country": "Spain", "lat": 41.2974, "lon": 2.0833},
    "BLR": {"name": "Kempegowda", "city": "Bengaluru", "country": "India", "lat": 13.1986, "lon": 77.7066},
    "BOM": {"name": "Chhatrapati Shivaji", "city": "Mumbai", "country": "India", "lat": 19.0896, "lon": 72.8656},
    "CDG": {"name": "Charles de Gaulle", "city": "Paris", "country": "France", "lat": 49.0097, "lon": 2.5479},
    "CPT": {"name": "Cape Town", "city": "Cape Town", "country": "South Africa", "lat": -33.9715, "lon": 18.6021},
    "DEL": {"name": "Indira Gandhi", "city": "Delhi", "country": "India", "lat": 28.5562, "lon": 77.1000},
    "DXB": {"name": "Dubai", "city": "Dubai", "country": "United Arab Emirates", "lat": 25.2532, "lon": 55.3657},
    "FRA": {"name": "Frankfurt", "city": "Frankfurt", "country": "Germany", "lat": 50.0379, "lon": 8.5622},
    "GRU": {"name": "Guarulhos", "city": "Sao Paulo", "country": "Brazil", "lat": -23.4356, "lon": -46.4731},
    "HKG": {"name": "Hong Kong", "city": "Hong Kong", "country": "Hong Kong", "lat": 22.3080, "lon": 113.9185},
    "HND": {"name": "Haneda", "city": "Tokyo", "country": "Japan", "lat": 35.5494, "lon": 139.7798},
    "IST": {"name": "Istanbul", "city": "Istanbul", "country": "Turkey", "lat": 41.2753, "lon": 28.7519},
    "JFK": {"name": "John F. Kennedy", "city": "New York", "country": "United States", "lat": 40.6413, "lon": -73.7781},
    "LAX": {"name": "Los Angeles", "city": "Los Angeles", "country": "United States", "lat": 33.9416, "lon": -118.4085},
    "LHR": {"name": "Heathrow", "city": "London", "country": "United Kingdom", "lat": 51.4700, "lon": -0.4543},
    "MAA": {"name": "Chennai", "city": "Chennai", "country": "India", "lat": 12.9941, "lon": 80.1709},
    "MAD": {"name": "Adolfo Suarez Barajas", "city": "Madrid", "country": "Spain", "lat": 40.4983, "lon": -3.5676},
    "MEX": {"name": "Benito Juarez", "city": "Mexico City", "country": "Mexico", "lat": 19.4361, "lon": -99.0719},
    "NBO": {"name": "Jomo Kenyatta", "city": "Nairobi", "country": "Kenya", "lat": -1.3192, "lon": 36.9278},
    "NRT": {"name": "Narita", "city": "Tokyo", "country": "Japan", "lat": 35.7720, "lon": 140.3929},
    "ORD": {"name": "O'Hare", "city": "Chicago", "country": "United States", "lat": 41.9742, "lon": -87.9073},
    "PEK": {"name": "Beijing Capital", "city": "Beijing", "country": "China", "lat": 40.0799, "lon": 116.6031},
    "SFO": {"name": "San Francisco", "city": "San Francisco", "country": "United States", "lat": 37.6213, "lon": -122.3790},
    "SIN": {"name": "Changi", "city": "Singapore", "country": "Singapore", "lat": 1.3644, "lon": 103.9915},
    "SYD": {"name": "Kingsford Smith", "city": "Sydney", "country": "Australia", "lat": -33.9399, "lon": 151.1753},
    "YYZ": {"name": "Toronto Pearson", "city": "Toronto", "country": "Canada", "lat": 43.6777, "lon": -79.6248},
    "ZRH": {"name": "Zurich", "city": "Zurich", "country": "Switzerland", "lat": 47.4647, "lon": 8.5492},
}


# ---------------------------------------------------------------------------
# Reference data helpers
# ---------------------------------------------------------------------------

def list_airports():
    """Return the airport table as a sorted list of records."""
    return [
        {"code": code, **details}
        for code, details in sorted(AIRPORTS.items(), key=lambda item: item[0])
    ]


def get_airport(code):
    """Look up an airport by IATA code, case-insensitively. None if unknown."""
    if not code:
        return None
    details = AIRPORTS.get(str(code).strip().upper())
    if not details:
        return None
    return {"code": str(code).strip().upper(), **details}


def list_cabin_classes():
    """Return cabin classes from least to most seat-space intensive."""
    return sorted(
        ({"name": name, **info} for name, info in CABIN_CLASSES.items()),
        key=lambda item: item["multiplier"],
    )


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
# Distance
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in kilometres."""
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return round(2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a))), 1)


def route_distance_km(origin, destination, apply_uplift=True):
    """Flown distance between two airport codes, in kilometres.

    Returns ``None`` when either code is unknown, so the caller can ask the
    user for a manual distance rather than silently reporting zero.
    """
    start = get_airport(origin)
    end = get_airport(destination)
    if not start or not end:
        return None

    distance = haversine_km(start["lat"], start["lon"], end["lat"], end["lon"])
    if apply_uplift and distance > 0:
        distance += ROUTING_UPLIFT_KM
    return round(distance, 1)


def haul_type(distance_km):
    """Classify a leg as short, medium or long haul."""
    distance = _clean_positive(distance_km, 30000.0)
    if distance <= SHORT_HAUL_MAX_KM:
        return "short"
    if distance <= MEDIUM_HAUL_MAX_KM:
        return "medium"
    return "long"


def cabin_multiplier(cabin, distance_km):
    """Seat-space multiplier for a cabin, capped on short-haul aircraft."""
    info = CABIN_CLASSES.get(cabin) or CABIN_CLASSES[DEFAULT_CABIN]
    multiplier = info["multiplier"]
    if haul_type(distance_km) == "short":
        return min(multiplier, SHORT_HAUL_CABIN_CAP)
    return multiplier


# ---------------------------------------------------------------------------
# Core emissions model
# ---------------------------------------------------------------------------

def leg_emissions(distance_km, cabin=DEFAULT_CABIN, load_factor=DEFAULT_LOAD_FACTOR):
    """CO2 and radiative-forcing figures for a single flown leg.

    The return value separates the three pieces a user should be able to see:
    the take-off charge, the cruise CO2, and the non-CO2 warming that only the
    cruise portion produces.
    """
    distance = _clean_positive(distance_km, 30000.0)
    if distance <= 0:
        return {
            "distance_km": 0.0,
            "haul": "short",
            "cabin": cabin if cabin in CABIN_CLASSES else DEFAULT_CABIN,
            "cabin_multiplier": 1.0,
            "lto_kg": 0.0,
            "cruise_km": 0.0,
            "cruise_kg": 0.0,
            "co2_kg": 0.0,
            "non_co2_kg": 0.0,
            "co2e_kg": 0.0,
            "kg_per_km": 0.0,
        }

    haul = haul_type(distance)
    multiplier = cabin_multiplier(cabin, distance)

    # Load factor scales the whole leg: the same fuel over fewer passengers.
    occupancy = _clean_positive(load_factor, 1.0, DEFAULT_LOAD_FACTOR) or DEFAULT_LOAD_FACTOR
    occupancy = max(0.3, min(1.0, occupancy))
    occupancy_scale = DEFAULT_LOAD_FACTOR / occupancy

    cruise_km = max(0.0, distance - CLIMB_DESCENT_KM)
    cruise_kg = cruise_km * CRUISE_KG_PER_KM[haul]
    # The climb and descent kilometres still burn fuel - at the cruise rate,
    # on top of the fixed LTO charge - they simply do not form contrails.
    climb_kg = min(distance, CLIMB_DESCENT_KM) * CRUISE_KG_PER_KM[haul]

    lto_kg = LTO_KG_PER_LEG
    co2_kg = (lto_kg + cruise_kg + climb_kg) * multiplier * occupancy_scale
    # Non-CO2 forcing applies only to the cruise share.
    non_co2_kg = (
        cruise_kg * multiplier * occupancy_scale * (RADIATIVE_FORCING_MULTIPLIER - 1.0)
    )

    return {
        "distance_km": round(distance, 1),
        "haul": haul,
        "cabin": cabin if cabin in CABIN_CLASSES else DEFAULT_CABIN,
        "cabin_multiplier": round(multiplier, 2),
        "load_factor": round(occupancy, 2),
        "lto_kg": round(lto_kg * multiplier * occupancy_scale, 2),
        "cruise_km": round(cruise_km, 1),
        "cruise_kg": round(cruise_kg * multiplier * occupancy_scale, 2),
        "co2_kg": round(co2_kg, 2),
        "non_co2_kg": round(non_co2_kg, 2),
        "co2e_kg": round(co2_kg + non_co2_kg, 2),
        "kg_per_km": round(co2_kg / distance, 4),
    }


def estimate_trip(
    legs,
    cabin=DEFAULT_CABIN,
    round_trip=True,
    passengers=1,
    include_radiative_forcing=True,
    load_factor=DEFAULT_LOAD_FACTOR,
):
    """Estimate a whole trip made of one or more legs.

    ``legs`` is a list of distances in kilometres - one entry per flown
    segment, so a connection is two entries. Setting ``round_trip`` doubles
    the itinerary, which is what a user almost always means by "a flight".
    """
    distances = [
        _clean_positive(distance, 30000.0)
        for distance in (legs or [])
        if _clean_positive(distance, 30000.0) > 0
    ]
    people = max(1, int(_clean_positive(passengers, 20.0, 1.0) or 1))

    outbound = [leg_emissions(distance, cabin, load_factor) for distance in distances]
    itinerary = outbound + ([dict(leg) for leg in outbound] if round_trip else [])

    co2_kg = sum(leg["co2_kg"] for leg in itinerary) * people
    non_co2_kg = sum(leg["non_co2_kg"] for leg in itinerary) * people
    total_km = sum(leg["distance_km"] for leg in itinerary) * people

    if not include_radiative_forcing:
        non_co2_kg = 0.0

    return {
        "legs": itinerary,
        "leg_count": len(itinerary),
        "passengers": people,
        "round_trip": bool(round_trip),
        "cabin": cabin if cabin in CABIN_CLASSES else DEFAULT_CABIN,
        "distance_km": round(total_km, 1),
        "co2_kg": round(co2_kg, 2),
        "non_co2_kg": round(non_co2_kg, 2),
        "co2e_kg": round(co2_kg + non_co2_kg, 2),
        "radiative_forcing_applied": bool(include_radiative_forcing),
        "budget_share_pct": budget_share(co2_kg + non_co2_kg),
    }


def estimate_route(
    origin,
    destination,
    via=None,
    cabin=DEFAULT_CABIN,
    round_trip=True,
    passengers=1,
    include_radiative_forcing=True,
    load_factor=DEFAULT_LOAD_FACTOR,
):
    """Estimate a trip described by airport codes, optionally via a hub.

    Returns ``None`` when any airport code is unknown.
    """
    stops = [origin] + [code for code in (via or []) if code] + [destination]
    distances = []
    for start, end in zip(stops, stops[1:]):
        distance = route_distance_km(start, end)
        if distance is None:
            return None
        distances.append(distance)

    estimate = estimate_trip(
        distances,
        cabin=cabin,
        round_trip=round_trip,
        passengers=passengers,
        include_radiative_forcing=include_radiative_forcing,
        load_factor=load_factor,
    )
    estimate["route"] = [str(code).strip().upper() for code in stops]
    estimate["direct_distance_km"] = route_distance_km(origin, destination)
    return estimate


def compare_routings(origin, destination, via, cabin=DEFAULT_CABIN, round_trip=True):
    """Compare a direct flight with the same journey through a hub.

    Connections cost twice: extra kilometres, and a second take-off cycle.
    """
    direct = estimate_route(
        origin, destination, cabin=cabin, round_trip=round_trip
    )
    connecting = estimate_route(
        origin, destination, via=[via], cabin=cabin, round_trip=round_trip
    )
    if not direct or not connecting:
        return None

    extra_kg = round(connecting["co2e_kg"] - direct["co2e_kg"], 2)
    return {
        "direct": direct,
        "connecting": connecting,
        "extra_km": round(connecting["distance_km"] - direct["distance_km"], 1),
        "extra_kg": extra_kg,
        "extra_pct": (
            round(extra_kg / direct["co2e_kg"] * 100, 1) if direct["co2e_kg"] > 0 else 0.0
        ),
        "extra_takeoffs": connecting["leg_count"] - direct["leg_count"],
    }


def compare_cabins(distance_km, round_trip=True):
    """The same journey priced in every cabin, cheapest first."""
    options = []
    for name in CABIN_CLASSES:
        estimate = estimate_trip([distance_km], cabin=name, round_trip=round_trip)
        options.append(
            {
                "cabin": name,
                "co2e_kg": estimate["co2e_kg"],
                "multiplier": cabin_multiplier(name, distance_km),
            }
        )
    options.sort(key=lambda option: option["co2e_kg"])

    baseline = options[0]["co2e_kg"] if options else 0.0
    for option in options:
        option["vs_economy_kg"] = round(option["co2e_kg"] - baseline, 2)
    return options


def compare_to_alternatives(distance_km, co2e_kg):
    """Compare the flight with surface options over the same distance.

    Options that are not realistic at the distance are still returned but
    flagged, so the comparison stays honest instead of suggesting a coach to
    another continent.
    """
    distance = _clean_positive(distance_km, 30000.0)
    flight_kg = max(0.0, float(co2e_kg or 0.0))
    rows = []

    options = [(mode, distance * factor) for mode, factor in SURFACE_ALTERNATIVES.items()]
    options.append(("Video call instead", VIDEO_CALL_KG))

    for mode, raw_kg in options:
        mode_kg = round(raw_kg, 2)
        if mode.endswith("rail"):
            plausible = distance <= RAIL_PLAUSIBLE_MAX_KM
        elif "coach" in mode.lower():
            plausible = distance <= COACH_PLAUSIBLE_MAX_KM
        elif mode.startswith("Car"):
            plausible = distance <= RAIL_PLAUSIBLE_MAX_KM
        else:
            plausible = True

        rows.append(
            {
                "mode": mode,
                "co2e_kg": mode_kg,
                "saving_kg": round(max(0.0, flight_kg - mode_kg), 2),
                "saving_pct": (
                    round(max(0.0, flight_kg - mode_kg) / flight_kg * 100, 1)
                    if flight_kg > 0
                    else 0.0
                ),
                "plausible_at_this_distance": plausible,
            }
        )

    rows.sort(key=lambda row: row["co2e_kg"])
    return rows


def budget_share(co2e_kg, budget_kg=PERSONAL_ANNUAL_BUDGET_KG):
    """What share of a 1.5C-consistent annual allowance a trip consumes."""
    budget = max(1.0, float(budget_kg or PERSONAL_ANNUAL_BUDGET_KG))
    return round(max(0.0, float(co2e_kg or 0.0)) / budget * 100, 1)


def annual_summary(trips, budget_kg=PERSONAL_ANNUAL_BUDGET_KG):
    """Aggregate a year of trips into one picture.

    Each trip is a dict with at least ``co2e_kg``; ``distance_km``, ``label``
    and ``cabin`` are used when present.
    """
    records = [trip for trip in (trips or []) if trip]
    total_co2 = sum(max(0.0, float(trip.get("co2_kg", 0.0) or 0.0)) for trip in records)
    total_co2e = sum(max(0.0, float(trip.get("co2e_kg", 0.0) or 0.0)) for trip in records)
    total_km = sum(max(0.0, float(trip.get("distance_km", 0.0) or 0.0)) for trip in records)

    ranked = sorted(
        records,
        key=lambda trip: float(trip.get("co2e_kg", 0.0) or 0.0),
        reverse=True,
    )
    biggest = ranked[0] if ranked else None

    return {
        "trip_count": len(records),
        "distance_km": round(total_km, 1),
        "co2_kg": round(total_co2, 2),
        "non_co2_kg": round(total_co2e - total_co2, 2),
        "co2e_kg": round(total_co2e, 2),
        "budget_share_pct": budget_share(total_co2e, budget_kg),
        "over_budget": total_co2e > max(1.0, float(budget_kg or PERSONAL_ANNUAL_BUDGET_KG)),
        "biggest_trip": biggest,
        "biggest_trip_share_pct": (
            round(float(biggest.get("co2e_kg", 0.0) or 0.0) / total_co2e * 100, 1)
            if biggest and total_co2e > 0
            else 0.0
        ),
        "average_kg_per_trip": round(total_co2e / len(records), 2) if records else 0.0,
    }


def trips_within_budget(trip_co2e_kg, budget_kg=PERSONAL_ANNUAL_BUDGET_KG):
    """How many trips of a given size fit inside an annual allowance."""
    per_trip = max(0.0, float(trip_co2e_kg or 0.0))
    if per_trip <= 0:
        return None
    return round(max(1.0, float(budget_kg or PERSONAL_ANNUAL_BUDGET_KG)) / per_trip, 2)


def get_reduction_tips(summary, limit=6):
    """Advice ranked by what this particular flyer's numbers look like."""
    if not summary or not summary.get("trip_count"):
        return ["Add a flight above to see where your air travel actually goes."]

    tips = []

    if summary.get("over_budget"):
        tips.append(
            f"Your flying alone is {summary['budget_share_pct']}% of a "
            f"1.5C-consistent annual allowance for everything - housing, food and "
            f"transport included."
        )

    share = summary.get("biggest_trip_share_pct", 0.0)
    biggest = summary.get("biggest_trip") or {}
    if share >= 50 and biggest.get("label"):
        tips.append(
            f"One trip - {biggest['label']} - is {share}% of your flying. "
            f"Skipping or combining that single journey beats optimising all the others."
        )

    non_co2 = summary.get("non_co2_kg", 0.0)
    if non_co2 > 0:
        tips.append(
            f"{non_co2:.0f} kg of your total is non-CO2 warming from contrails and "
            f"NOx at cruise altitude. Airline calculators usually leave this out."
        )

    tips.append(
        "Fly economy. A business seat on the same aircraft is roughly three "
        "times the footprint because it occupies three times the floor space."
    )
    tips.append(
        "Prefer direct flights: every extra take-off adds a fixed fuel charge "
        "before the aircraft has covered any useful distance."
    )
    tips.append(
        "Stay longer, fly less often. One two-week trip costs far less than two "
        "one-week trips to the same place."
    )
    tips.append(
        "Under about 1,000 km, rail is usually faster door to door and around "
        "80-95% lower carbon."
    )

    return tips[: max(0, int(limit))]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _get_conn():
    return sqlite3.connect(DB_NAME)


def init_flight_db():
    """Create the flight trip table if it does not exist yet."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flight_trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                route TEXT NOT NULL,
                cabin TEXT NOT NULL,
                round_trip INTEGER NOT NULL DEFAULT 1,
                passengers INTEGER NOT NULL DEFAULT 1,
                distance_km REAL NOT NULL,
                co2_kg REAL NOT NULL,
                non_co2_kg REAL NOT NULL,
                co2e_kg REAL NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Flight footprint init error: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def save_trip(user_id, label, estimate):
    """Persist an estimated trip. Returns the new row id or None."""
    init_flight_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute(
            """
            INSERT INTO flight_trips (
                user_id, label, route, cabin, round_trip, passengers,
                distance_km, co2_kg, non_co2_kg, co2e_kg, detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                (label or "Flight").strip() or "Flight",
                " - ".join(estimate.get("route", [])) or "Manual distance",
                estimate.get("cabin", DEFAULT_CABIN),
                1 if estimate.get("round_trip") else 0,
                int(estimate.get("passengers", 1) or 1),
                estimate.get("distance_km", 0.0),
                estimate.get("co2_kg", 0.0),
                estimate.get("non_co2_kg", 0.0),
                estimate.get("co2e_kg", 0.0),
                json.dumps(
                    {
                        "legs": estimate.get("legs", []),
                        "radiative_forcing_applied": estimate.get(
                            "radiative_forcing_applied", True
                        ),
                    }
                ),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("Unable to save flight trip: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


def get_trips(user_id, limit=50):
    """Return a user's saved trips, newest first."""
    init_flight_db()
    conn = None
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, label, route, cabin, round_trip, passengers, distance_km,
                   co2_kg, non_co2_kg, co2e_kg, detail_json, created_at
            FROM flight_trips
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

        trips = []
        for row in rows:
            record = dict(row)
            record["round_trip"] = bool(record["round_trip"])
            try:
                record["detail"] = json.loads(record.pop("detail_json"))
            except (TypeError, ValueError):
                record["detail"] = {}
            trips.append(record)
        return trips
    except sqlite3.Error as exc:
        logger.error("Unable to load flight trips: %s", exc)
        return []
    finally:
        if conn:
            conn.close()


def delete_trip(trip_id):
    """Delete a saved trip."""
    init_flight_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM flight_trips WHERE id = ?", (trip_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Unable to delete flight trip: %s", exc)
        return False
    finally:
        if conn:
            conn.close()
