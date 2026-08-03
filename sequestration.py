"""Tree and garden carbon sequestration, with growth curves and honest limits.

"Plant a tree" is the most common piece of advice in this whole application
and the only one with no arithmetic behind it. ``recommendations.py`` and the
marketplace features gesture at offsetting, but nothing can currently answer
how much carbon a real planting actually removes.

The naive version of that advice is wrong in two specific ways, and this
module is built to correct both.

**Sequestration is not linear.** A sapling absorbs almost nothing in its first
years. Uptake follows an S-curve: slow establishment, a steep middle period,
then a plateau as the tree matures. Applying a flat "22 kg a year" multiplier
overstates the first decade badly and hands the user a number that never
arrives. Here, uptake in year *n* comes from a logistic curve::

    rate(n) = mature_rate / (1 + e^(-k * (n - midpoint)))

**Sequestration is not unlimited.** A garden is a fixed area, spacing is a
physical constraint, and for most gardens the honest answer to "how long until
this offsets my footprint?" is "it does not, within any sensible horizon".
``years_to_offset`` is allowed to return ``None`` and the UI says so plainly
rather than inventing a comforting year.

Everything reported is net: survival losses and maintenance emissions are
subtracted, because a plan that ignores the mower is not a carbon plan.

The module is self-contained: its SQLite table is created lazily and no shared
files are modified.
"""

import os
import json
import math
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

# Planting types with their mature annual uptake in kg CO2 per plant (or per
# square metre for ground cover), how long they take to reach it, the ground
# each one needs, and the maintenance they demand.
PLANTING_TYPES = {
    "Large broadleaf (oak, beech)": {
        "mature_rate_kg": 28.0,
        "years_to_maturity": 25,
        "spacing_m2": 60.0,
        "maintenance_kg": 1.2,
        "per_area": False,
        "native_habitat": True,
        "note": "The highest ceiling of anything here, and the longest wait.",
    },
    "Medium broadleaf (birch, rowan)": {
        "mature_rate_kg": 16.0,
        "years_to_maturity": 18,
        "spacing_m2": 30.0,
        "maintenance_kg": 0.9,
        "per_area": False,
        "native_habitat": True,
        "note": "A good compromise between speed and capacity for most gardens.",
    },
    "Small broadleaf (hawthorn, hazel)": {
        "mature_rate_kg": 8.5,
        "years_to_maturity": 12,
        "spacing_m2": 12.0,
        "maintenance_kg": 0.6,
        "per_area": False,
        "native_habitat": True,
        "note": "Fits a small garden and establishes quickly.",
    },
    "Conifer (pine, spruce)": {
        "mature_rate_kg": 22.0,
        "years_to_maturity": 20,
        "spacing_m2": 25.0,
        "maintenance_kg": 0.7,
        "per_area": False,
        "native_habitat": False,
        "note": "Fast and dense, though poorer habitat than a native broadleaf.",
    },
    "Fruit tree": {
        "mature_rate_kg": 11.0,
        "years_to_maturity": 10,
        "spacing_m2": 20.0,
        "maintenance_kg": 1.8,
        "per_area": False,
        "native_habitat": False,
        "note": "Modest uptake, but it also displaces food you would have bought.",
    },
    "Hedgerow (per metre)": {
        "mature_rate_kg": 4.2,
        "years_to_maturity": 8,
        "spacing_m2": 2.0,
        "maintenance_kg": 0.4,
        "per_area": False,
        "native_habitat": True,
        "note": "Excellent habitat per square metre and quick to establish.",
    },
    "Shrub": {
        "mature_rate_kg": 2.6,
        "years_to_maturity": 6,
        "spacing_m2": 4.0,
        "maintenance_kg": 0.2,
        "per_area": False,
        "native_habitat": True,
        "note": "Low capacity, but it fills space a tree cannot.",
    },
    "Bamboo": {
        "mature_rate_kg": 9.0,
        "years_to_maturity": 5,
        "spacing_m2": 6.0,
        "maintenance_kg": 0.5,
        "per_area": False,
        "native_habitat": False,
        "note": "The fastest establishment here; invasive if not contained.",
    },
    "Wildflower meadow (per m²)": {
        "mature_rate_kg": 0.32,
        "years_to_maturity": 4,
        "spacing_m2": 1.0,
        "maintenance_kg": 0.02,
        "per_area": True,
        "native_habitat": True,
        "note": "Soil carbon rather than wood, and the best habitat per m².",
    },
    "Lawn to no-mow (per m²)": {
        "mature_rate_kg": 0.18,
        "years_to_maturity": 5,
        "spacing_m2": 1.0,
        "maintenance_kg": 0.0,
        "per_area": True,
        "native_habitat": False,
        "note": "Slow, but it also removes the mower's own emissions.",
    },
}

DEFAULT_PLANTING_TYPE = "Medium broadleaf (birch, rowan)"

# Steepness of the logistic growth curve. Higher means a sharper transition
# from establishment to full uptake.
GROWTH_STEEPNESS = 0.45

# The curve's midpoint sits at this share of the time to maturity, which puts
# the steep phase where real growth data puts it.
GROWTH_MIDPOINT_SHARE = 0.55

# Share of plantings that survive to maturity. Losses are front-loaded, but a
# flat rate is honest enough at this precision and keeps the maths inspectable.
DEFAULT_SURVIVAL_RATE = 0.85

# How far ahead the planner will look. Beyond this, projections are fiction.
DEFAULT_HORIZON_YEARS = 40
MAX_HORIZON_YEARS = 100

# A plan whose entire mature uptake cannot cover a footprint will never offset
# it, no matter how long the horizon.
NEVER_OFFSETS = None

# Biodiversity scoring. A monoculture of the single highest-uptake species is
# a poor garden, so species mix and native habitat are both rewarded.
BIODIVERSITY_SPECIES_WEIGHT = 60.0
BIODIVERSITY_NATIVE_WEIGHT = 40.0


def list_planting_types():
    """Return the planting catalogue, highest mature uptake first."""
    return sorted(
        ({"name": name, **info} for name, info in PLANTING_TYPES.items()),
        key=lambda item: item["mature_rate_kg"],
        reverse=True,
    )


def get_planting_type(name):
    """Return one planting type's reference data, falling back sensibly."""
    return dict(PLANTING_TYPES.get(name, PLANTING_TYPES[DEFAULT_PLANTING_TYPE]))


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


def _clean_horizon(years):
    """Clamp a projection horizon to something defensible."""
    horizon = _clean_count(years, MAX_HORIZON_YEARS)
    return max(1, horizon)


def capacity_for_area(area_m2, planting_type):
    """How many of a thing physically fit in a space.

    Spacing is a hard constraint, not a guideline: crowded trees compete and
    neither reaches its mature rate. Enforcing it here keeps plans real rather
    than letting a user put forty oaks in a courtyard.
    """
    area = _clean_number(area_m2, 10 ** 7)
    info = get_planting_type(planting_type)
    spacing = info["spacing_m2"] or 1.0
    return int(area // spacing)


def growth_factor(planting_type, year):
    """The share of mature uptake reached in a given year, 0 to just under 1.

    This is the S-curve that makes the whole module honest. Year one is a
    fraction of the mature rate, not equal to it.
    """
    info = get_planting_type(planting_type)
    maturity = max(1, info["years_to_maturity"])
    elapsed = _clean_number(year, MAX_HORIZON_YEARS)

    midpoint = maturity * GROWTH_MIDPOINT_SHARE
    try:
        return 1.0 / (1.0 + math.exp(-GROWTH_STEEPNESS * (elapsed - midpoint)))
    except OverflowError:
        return 0.0 if elapsed < midpoint else 1.0


def annual_sequestration(planting_type, count, year):
    """Uptake in one specific year, on the growth curve rather than at maturity."""
    info = get_planting_type(planting_type)
    units = _clean_number(count, 10 ** 6)
    return round(info["mature_rate_kg"] * units * growth_factor(planting_type, year), 3)


def _plan_entries(plan):
    """Normalise a plan into (type, count) pairs, dropping empty entries."""
    entries = []
    for entry in plan or []:
        planting_type = entry.get("planting_type", DEFAULT_PLANTING_TYPE)
        count = _clean_number(entry.get("count", 0), 10 ** 6)
        if count > 0:
            entries.append((planting_type, count))
    return entries


def sequestration_curve(plan, years=DEFAULT_HORIZON_YEARS):
    """Year-by-year uptake for a whole plan, for charting."""
    horizon = _clean_horizon(years)
    entries = _plan_entries(plan)

    return [
        round(
            sum(annual_sequestration(planting_type, count, year) for planting_type, count in entries),
            3,
        )
        for year in range(1, horizon + 1)
    ]


def cumulative_sequestration(plan, years=DEFAULT_HORIZON_YEARS):
    """Total gross uptake across a plan's lifetime."""
    return round(sum(sequestration_curve(plan, years)), 2)


def cumulative_curve(plan, years=DEFAULT_HORIZON_YEARS):
    """The running total year by year, which is what users actually want to see."""
    running = 0.0
    curve = []
    for value in sequestration_curve(plan, years):
        running += value
        curve.append(round(running, 2))
    return curve


def maintenance_emissions(plan, years=DEFAULT_HORIZON_YEARS):
    """Watering, pruning and mowing, subtracted so the figure is net."""
    horizon = _clean_horizon(years)
    total = 0.0
    for planting_type, count in _plan_entries(plan):
        info = get_planting_type(planting_type)
        total += info["maintenance_kg"] * count * horizon
    return round(total, 2)


def net_sequestration(plan, years=DEFAULT_HORIZON_YEARS, survival_rate=DEFAULT_SURVIVAL_RATE):
    """Gross uptake less survival losses and maintenance emissions.

    Never exceeds the gross figure, and is allowed to be negative for a plan
    whose maintenance outweighs its uptake - a heavily-watered fruit orchard
    over a short horizon can genuinely fall on the wrong side of zero, and the
    module reports that rather than clamping it away.
    """
    horizon = _clean_horizon(years)
    survival = max(0.0, min(1.0, _clean_number(survival_rate, 1.0, DEFAULT_SURVIVAL_RATE)))

    gross = cumulative_sequestration(plan, horizon)
    surviving = gross * survival
    maintenance = maintenance_emissions(plan, horizon) * survival

    return {
        "years": horizon,
        "survival_rate": round(survival, 3),
        "gross_co2_kg": gross,
        "surviving_co2_kg": round(surviving, 2),
        "maintenance_co2_kg": round(maintenance, 2),
        "net_co2_kg": round(surviving - maintenance, 2),
    }


def mature_annual_rate(plan, survival_rate=DEFAULT_SURVIVAL_RATE):
    """The plan's ceiling: net uptake per year once everything is grown."""
    survival = max(0.0, min(1.0, _clean_number(survival_rate, 1.0, DEFAULT_SURVIVAL_RATE)))
    total = 0.0
    for planting_type, count in _plan_entries(plan):
        info = get_planting_type(planting_type)
        total += (info["mature_rate_kg"] - info["maintenance_kg"]) * count * survival
    return round(total, 2)


def years_to_offset(
    plan,
    annual_footprint_kg,
    horizon=DEFAULT_HORIZON_YEARS,
    survival_rate=DEFAULT_SURVIVAL_RATE,
):
    """How long until the planting cancels one year of the user's emissions.

    Returns ``None`` when it never does within the horizon. For most gardens
    that is the truthful answer, and saying so is the point of this module.
    """
    target = _clean_number(annual_footprint_kg, 10 ** 7)
    if target <= 0:
        return 0

    limit = _clean_horizon(horizon)
    survival = max(0.0, min(1.0, _clean_number(survival_rate, 1.0, DEFAULT_SURVIVAL_RATE)))

    running = 0.0
    for year in range(1, limit + 1):
        gross = sum(
            annual_sequestration(planting_type, count, year)
            for planting_type, count in _plan_entries(plan)
        )
        maintenance = sum(
            get_planting_type(planting_type)["maintenance_kg"] * count
            for planting_type, count in _plan_entries(plan)
        )
        running += (gross - maintenance) * survival
        if running >= target:
            return year

    return NEVER_OFFSETS


def offset_share(
    plan,
    annual_footprint_kg,
    year,
    survival_rate=DEFAULT_SURVIVAL_RATE,
):
    """The percentage of one year's footprint the planting covers in a given year."""
    target = _clean_number(annual_footprint_kg, 10 ** 7)
    if target <= 0:
        return 0.0

    survival = max(0.0, min(1.0, _clean_number(survival_rate, 1.0, DEFAULT_SURVIVAL_RATE)))
    entries = _plan_entries(plan)

    gross = sum(annual_sequestration(planting_type, count, year) for planting_type, count in entries)
    maintenance = sum(
        get_planting_type(planting_type)["maintenance_kg"] * count for planting_type, count in entries
    )
    net = (gross - maintenance) * survival

    return round(max(0.0, min(100.0, net / target * 100)), 1)


def design_plan(area_m2, goal="balanced"):
    """Suggest a planting mix for the space available.

    ``goal`` trades early uptake against mature capacity: "fast" favours
    quick-establishing types, "capacity" favours the highest ceiling, and
    "balanced" mixes them. Every suggestion respects spacing, so the plan that
    comes back actually fits.
    """
    area = _clean_number(area_m2, 10 ** 7)
    if area <= 0:
        return []

    mixes = {
        "fast": [
            ("Bamboo", 0.25),
            ("Small broadleaf (hawthorn, hazel)", 0.25),
            ("Hedgerow (per metre)", 0.20),
            ("Wildflower meadow (per m²)", 0.30),
        ],
        "capacity": [
            ("Large broadleaf (oak, beech)", 0.45),
            ("Conifer (pine, spruce)", 0.25),
            ("Medium broadleaf (birch, rowan)", 0.20),
            ("Wildflower meadow (per m²)", 0.10),
        ],
        "balanced": [
            ("Medium broadleaf (birch, rowan)", 0.30),
            ("Small broadleaf (hawthorn, hazel)", 0.20),
            ("Hedgerow (per metre)", 0.20),
            ("Shrub", 0.10),
            ("Wildflower meadow (per m²)", 0.20),
        ],
    }

    chosen = mixes.get(goal, mixes["balanced"])
    plan = []
    for planting_type, share in chosen:
        count = capacity_for_area(area * share, planting_type)
        if count > 0:
            plan.append({"planting_type": planting_type, "count": count})

    # A garden too small for any tree still gets something it can actually do.
    if not plan:
        meadow = capacity_for_area(area, "Wildflower meadow (per m²)")
        if meadow > 0:
            plan.append({"planting_type": "Wildflower meadow (per m²)", "count": meadow})

    return plan


def plan_area_used(plan):
    """Ground the plan occupies, so an oversized plan can be caught."""
    total = 0.0
    for planting_type, count in _plan_entries(plan):
        total += get_planting_type(planting_type)["spacing_m2"] * count
    return round(total, 2)


def plan_fits(plan, area_m2):
    """Whether a plan physically fits the space available."""
    return plan_area_used(plan) <= _clean_number(area_m2, 10 ** 7) + 1e-9


def biodiversity_score(plan):
    """Score 0-100 rewarding species mix and native habitat.

    A monoculture of the single highest-uptake species is a poor garden, so
    optimising carbon alone is the wrong objective and the score exists to say
    so. Mix counts for more than nativeness, but both count.
    """
    entries = _plan_entries(plan)
    if not entries:
        return 0.0

    # Species diversity: five or more distinct types is treated as full marks.
    distinct = len({planting_type for planting_type, _ in entries})
    diversity = min(1.0, distinct / 5.0)

    # Native habitat share, weighted by how much of the plan it represents.
    total_units = sum(count for _, count in entries)
    native_units = sum(
        count for planting_type, count in entries if get_planting_type(planting_type)["native_habitat"]
    )
    native_share = native_units / total_units if total_units else 0.0

    score = diversity * BIODIVERSITY_SPECIES_WEIGHT + native_share * BIODIVERSITY_NATIVE_WEIGHT
    return round(min(100.0, score), 1)


def build_plan_summary(
    plan,
    area_m2=0.0,
    annual_footprint_kg=0.0,
    years=DEFAULT_HORIZON_YEARS,
    survival_rate=DEFAULT_SURVIVAL_RATE,
):
    """Everything the UI needs about a plan, in one call."""
    horizon = _clean_horizon(years)
    net = net_sequestration(plan, horizon, survival_rate)

    return {
        "plan": [
            {"planting_type": planting_type, "count": count}
            for planting_type, count in _plan_entries(plan)
        ],
        "years": horizon,
        "area_used_m2": plan_area_used(plan),
        "fits": plan_fits(plan, area_m2) if area_m2 else True,
        "curve": sequestration_curve(plan, horizon),
        "cumulative_curve": cumulative_curve(plan, horizon),
        "gross_co2_kg": net["gross_co2_kg"],
        "net_co2_kg": net["net_co2_kg"],
        "maintenance_co2_kg": net["maintenance_co2_kg"],
        "mature_annual_kg": mature_annual_rate(plan, survival_rate),
        "years_to_offset": years_to_offset(plan, annual_footprint_kg, horizon, survival_rate),
        "offset_share_at_horizon_pct": offset_share(
            plan, annual_footprint_kg, horizon, survival_rate
        ),
        "biodiversity_score": biodiversity_score(plan),
    }


def get_planting_tips(summary, annual_footprint_kg=0.0, limit=6):
    """Advice ranked by the user's own plan."""
    tips = []
    if not summary.get("plan"):
        return ["Add some plantings above to see what your space could actually absorb."]

    horizon = summary.get("years", DEFAULT_HORIZON_YEARS)
    offset_year = summary.get("years_to_offset")

    if offset_year is None and annual_footprint_kg > 0:
        mature = summary.get("mature_annual_kg", 0.0)
        tips.append(
            f"This planting will not offset your {annual_footprint_kg:,.0f} kg annual "
            f"footprint within {horizon} years. At full maturity it absorbs about "
            f"{mature:,.0f} kg a year. That is not a reason to skip it — it is a "
            "reason to treat planting as a supplement to reductions, not a substitute."
        )
    elif offset_year:
        tips.append(
            f"This planting cancels one year of your emissions in year **{offset_year}**. "
            "Note the wait: that is the part 'plant a tree' advice always leaves out."
        )

    curve = summary.get("curve", [])
    if curve:
        tips.append(
            f"Year one absorbs {curve[0]:,.1f} kg; year {horizon} absorbs "
            f"{curve[-1]:,.1f} kg. Sequestration is an S-curve, not a flat rate — "
            "almost nothing happens early, which is exactly why planting sooner matters."
        )

    score = summary.get("biodiversity_score", 0.0)
    if score < 40:
        tips.append(
            f"Biodiversity score {score:.0f}/100. Mixing species and choosing natives "
            "costs you very little carbon and makes an enormously better garden."
        )
    elif score > 70:
        tips.append(
            f"Biodiversity score {score:.0f}/100 — a good mix. Optimising carbon alone "
            "would have given you a monoculture, which is a worse garden."
        )

    if summary.get("maintenance_co2_kg", 0.0) > summary.get("gross_co2_kg", 0.0) * 0.2:
        tips.append(
            "Maintenance is eating a large share of the uptake. Less mowing, less "
            "watering once established, and no powered tools where you can avoid them."
        )

    if not summary.get("fits", True):
        tips.append(
            f"This plan needs {summary['area_used_m2']:,.0f} m², more than you have. "
            "Crowded trees compete and none of them reach their mature rate."
        )

    tips.append(
        "Water hard for the first two summers. Establishment failure is the main "
        "reason plantings never reach the curve above."
    )

    return tips[: max(0, int(limit))]


def _get_conn():
    return sqlite3.connect(DB_NAME)


def init_sequestration_db():
    """Create the planting plan table if it does not exist yet."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS planting_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plan_name TEXT NOT NULL,
                area_m2 REAL NOT NULL,
                horizon_years INTEGER NOT NULL,
                net_co2_kg REAL NOT NULL,
                mature_annual_kg REAL NOT NULL,
                years_to_offset INTEGER,
                biodiversity_score REAL,
                plan_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Sequestration init error: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def save_planting_plan(user_id, plan_name, summary, area_m2=0.0):
    """Persist a planting plan. Returns the new row id or None."""
    init_sequestration_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute(
            """
            INSERT INTO planting_plans (
                user_id, plan_name, area_m2, horizon_years, net_co2_kg,
                mature_annual_kg, years_to_offset, biodiversity_score, plan_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                (plan_name or "My garden").strip() or "My garden",
                _clean_number(area_m2, 10 ** 7),
                summary.get("years", DEFAULT_HORIZON_YEARS),
                summary.get("net_co2_kg", 0.0),
                summary.get("mature_annual_kg", 0.0),
                summary.get("years_to_offset"),
                summary.get("biodiversity_score", 0.0),
                json.dumps(summary.get("plan", [])),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("Unable to save planting plan: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


def get_planting_plans(user_id, limit=25):
    """Return a user's saved planting plans, newest first."""
    init_sequestration_db()
    conn = None
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, plan_name, area_m2, horizon_years, net_co2_kg,
                   mature_annual_kg, years_to_offset, biodiversity_score,
                   plan_json, created_at
            FROM planting_plans
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

        plans = []
        for row in rows:
            record = dict(row)
            try:
                record["plan"] = json.loads(record.pop("plan_json"))
            except (TypeError, ValueError):
                record["plan"] = []
            plans.append(record)
        return plans
    except sqlite3.Error as exc:
        logger.error("Unable to load planting plans: %s", exc)
        return []
    finally:
        if conn:
            conn.close()


def delete_planting_plan(plan_id):
    """Delete a saved planting plan."""
    init_sequestration_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM planting_plans WHERE id = ?", (plan_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Unable to delete planting plan: %s", exc)
        return False
    finally:
        if conn:
            conn.close()
