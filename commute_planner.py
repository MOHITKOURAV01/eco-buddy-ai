"""Hybrid-working commute planner: travel, office energy and home energy.

``plugins/route_emissions.py`` prices a single journey. That is the right model
for a trip you might or might not take, but it is the wrong model for a
commute, because a commute is a *pattern* - the same journey repeated, or not
made at all, on a schedule someone actually chooses.

It also answers only half the question. Working from home is routinely treated
as a zero-emission day. It is not:

* your home is heated and lit all day instead of being empty;
* the office is heated and lit anyway, whether you are in it or not.

That second point is the one most hybrid-work advice gets wrong. An office's
energy is mostly a *baseline* - the building is conditioned, lit and running
whether five people or five hundred walk in. So one person staying home saves
almost none of it. The saving only becomes real when a whole team consolidates
onto the same days and the building can genuinely be shut on the others.

This module models all three pieces together::

    weekly CO2 = commute travel + office days + home-working days

and is explicit about the fact that the answer is sometimes "go in more, on
fewer days".

Cold starts
-----------
Short car trips are much worse per kilometre than long ones. A cold engine
runs rich and its catalyst does not work until it is warm, which is a fixed
penalty per trip rather than a rate per kilometre - so it hurts a 3 km commute
far more than a 30 km one. Electric cars have the mirror-image problem: cabin
heating comes out of the battery, which is worst on short winter trips.

The module is self-contained: it imports no Streamlit, its SQLite table is
created lazily and no shared files are modified.
"""

import os
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

WORKING_DAYS_PER_WEEK = 5
WORKING_WEEKS_PER_YEAR = 46  # 52 minus leave and public holidays

# Travel modes. ``kg_per_km`` is per vehicle-kilometre for private modes and
# per passenger-kilometre for public transport, which is why occupancy only
# applies to the private ones.
#
#   cold_start_kg   fixed penalty per trip while the engine and catalyst warm
#   winter_uplift   extra energy in cold weather, as a fraction
#   speed_kmh       door-to-door average, used for the time comparison
TRAVEL_MODES = {
    "Petrol car": {
        "kg_per_km": 0.171, "per_vehicle": True, "cold_start_kg": 0.260,
        "winter_uplift": 0.10, "speed_kmh": 30.0, "max_km": None,
        "note": "Worst per kilometre on short trips, because of the cold start.",
    },
    "Diesel car": {
        "kg_per_km": 0.164, "per_vehicle": True, "cold_start_kg": 0.290,
        "winter_uplift": 0.12, "speed_kmh": 30.0, "max_km": None,
        "note": "Efficient on long runs; a short urban commute is its worst case.",
    },
    "Hybrid car": {
        "kg_per_km": 0.120, "per_vehicle": True, "cold_start_kg": 0.140,
        "winter_uplift": 0.10, "speed_kmh": 30.0, "max_km": None,
        "note": "The electric assist covers much, though not all, of the cold start.",
    },
    "Electric car": {
        "kg_per_km": 0.053, "per_vehicle": True, "cold_start_kg": 0.020,
        "winter_uplift": 0.25, "speed_kmh": 30.0, "max_km": None,
        "note": "No cold-start penalty, but cabin heating comes out of the battery.",
    },
    "Motorbike": {
        "kg_per_km": 0.113, "per_vehicle": True, "cold_start_kg": 0.120,
        "winter_uplift": 0.05, "speed_kmh": 35.0, "max_km": None,
        "note": "Lower carbon than a car, though not by as much as people expect.",
    },
    "Bus": {
        "kg_per_km": 0.102, "per_vehicle": False, "cold_start_kg": 0.0,
        "winter_uplift": 0.03, "speed_kmh": 18.0, "max_km": None,
        "note": "Already running whether you board it or not.",
    },
    "Tram or metro": {
        "kg_per_km": 0.029, "per_vehicle": False, "cold_start_kg": 0.0,
        "winter_uplift": 0.03, "speed_kmh": 25.0, "max_km": None,
        "note": "Electric, high occupancy, and unaffected by traffic.",
    },
    "Suburban train": {
        "kg_per_km": 0.035, "per_vehicle": False, "cold_start_kg": 0.0,
        "winter_uplift": 0.03, "speed_kmh": 45.0, "max_km": None,
        "note": "The only mode that gets *better* the further you commute.",
    },
    "E-bike": {
        "kg_per_km": 0.005, "per_vehicle": True, "cold_start_kg": 0.0,
        "winter_uplift": 0.15, "speed_kmh": 18.0, "max_km": 20.0,
        "note": "Flattens hills and headwinds - the reason most people stop cycling.",
    },
    "Bicycle": {
        "kg_per_km": 0.0, "per_vehicle": True, "cold_start_kg": 0.0,
        "winter_uplift": 0.0, "speed_kmh": 15.0, "max_km": 15.0,
        "note": "Zero at the tailpipe and usually fastest across a congested city.",
    },
    "Walking": {
        "kg_per_km": 0.0, "per_vehicle": True, "cold_start_kg": 0.0,
        "winter_uplift": 0.0, "speed_kmh": 5.0, "max_km": 5.0,
        "note": "Only realistic for a genuinely short commute.",
    },
}

DEFAULT_MODE = "Petrol car"

# Modes where sharing the vehicle divides the emissions.
SHAREABLE_MODES = {"Petrol car", "Diesel car", "Hybrid car", "Electric car"}

# Office energy per person per working day, in kWh.
#
# The split is the point: the fixed share is spent conditioning and lighting
# the building whether anyone comes in or not, and one person's absence does
# not recover it. Only the marginal share follows attendance.
OFFICE_FIXED_KWH_PER_DESK_DAY = 5.6
OFFICE_MARGINAL_KWH_PER_PERSON_DAY = 2.4

# Extra household energy on a day worked at home: heating or cooling a home
# that would otherwise be empty, plus lighting and equipment.
HOME_WORKING_KWH = {
    "Winter": 6.8,
    "Shoulder": 3.1,
    "Summer": 1.9,
}

SEASONS = list(HOME_WORKING_KWH)
DEFAULT_SEASON = "Shoulder"

# Season weights for an annual average, in working weeks.
SEASON_WEEKS = {"Winter": 16, "Shoulder": 20, "Summer": 10}

# Grid intensity in kg CO2e per kWh, for both office and home electricity.
DEFAULT_GRID_INTENSITY = 0.207

# A cold start is a fixed cost per journey, not a rate per kilometre - which
# is why it hurts a short commute so much more. On a trip this short the engine
# never reaches operating temperature at all, so the penalty is worse still.
COLD_START_SHORT_TRIP_KM = 2.0
COLD_START_SHORT_TRIP_MULTIPLIER = 1.5

# Winter uplift only applies in the cold half of the year.
WINTER_SEASONS = {"Winter"}


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

def list_modes(distance_km=None):
    """Return travel modes, lowest carbon first, flagged for feasibility."""
    modes = []
    for name, details in TRAVEL_MODES.items():
        feasible = True
        if distance_km is not None and details["max_km"] is not None:
            feasible = _clean_positive(distance_km, 1000.0) <= details["max_km"]
        modes.append({"name": name, "feasible": feasible, **details})
    return sorted(modes, key=lambda mode: mode["kg_per_km"])


def get_mode(name):
    """Look up a travel mode, falling back to the default."""
    details = TRAVEL_MODES.get(name)
    if not details:
        return {"name": DEFAULT_MODE, **TRAVEL_MODES[DEFAULT_MODE]}
    return {"name": name, **details}


def is_shareable(mode_name):
    """Whether occupancy divides the emissions for this mode."""
    return mode_name in SHAREABLE_MODES


# ---------------------------------------------------------------------------
# One commute
# ---------------------------------------------------------------------------

def cold_start_penalty(mode_name, distance_km):
    """Fixed extra emissions per trip while the engine and catalyst warm up.

    This is charged per *start*, not per kilometre, which is exactly why a 3 km
    car commute is so much worse per kilometre than a 30 km one - the same
    penalty is spread over a tenth of the distance. On a very short trip the
    engine never reaches operating temperature at all, so it is worse again.
    """
    mode = get_mode(mode_name)
    penalty = mode["cold_start_kg"]
    if penalty <= 0:
        return 0.0

    distance = _clean_positive(distance_km, 1000.0)
    if distance <= 0:
        return 0.0
    if distance < COLD_START_SHORT_TRIP_KM:
        penalty *= COLD_START_SHORT_TRIP_MULTIPLIER
    return round(penalty, 4)


def trip_emissions(mode_name, distance_km, occupants=1, season=DEFAULT_SEASON):
    """Emissions for one one-way commute leg, in kg CO2e per person."""
    mode = get_mode(mode_name)
    distance = _clean_positive(distance_km, 1000.0)
    if distance <= 0:
        return {
            "mode": mode["name"], "distance_km": 0.0, "running_kg": 0.0,
            "cold_start_kg": 0.0, "total_kg": 0.0, "kg_per_km": 0.0,
            "occupants": 1, "minutes": 0.0,
        }

    uplift = 1.0 + (mode["winter_uplift"] if season in WINTER_SEASONS else 0.0)
    running = distance * mode["kg_per_km"] * uplift
    cold_start = cold_start_penalty(mode["name"], distance) * uplift

    people = max(1, int(_clean_positive(occupants, 8.0, 1.0) or 1))
    if not is_shareable(mode["name"]):
        people = 1

    total = (running + cold_start) / people

    return {
        "mode": mode["name"],
        "distance_km": round(distance, 2),
        "running_kg": round(running / people, 4),
        "cold_start_kg": round(cold_start / people, 4),
        "total_kg": round(total, 4),
        "kg_per_km": round(total / distance, 4),
        "occupants": people,
        "minutes": round(distance / mode["speed_kmh"] * 60, 1) if mode["speed_kmh"] > 0 else 0.0,
        "season": season,
    }


def compare_modes(distance_km, occupants=1, season=DEFAULT_SEASON, current_mode=DEFAULT_MODE):
    """The same commute in every mode, lowest carbon first.

    Modes that are not realistic at the distance are still returned, but
    flagged, so a 40 km walk is never presented as a suggestion.
    """
    baseline = trip_emissions(current_mode, distance_km, occupants, season)
    rows = []

    for mode in list_modes(distance_km):
        result = trip_emissions(mode["name"], distance_km, occupants, season)
        rows.append(
            {
                "mode": mode["name"],
                "feasible": mode["feasible"],
                "note": mode["note"],
                "total_kg": result["total_kg"],
                "kg_per_km": result["kg_per_km"],
                "minutes": result["minutes"],
                "saving_kg": round(max(0.0, baseline["total_kg"] - result["total_kg"]), 4),
                "saving_pct": (
                    round(
                        max(0.0, baseline["total_kg"] - result["total_kg"])
                        / baseline["total_kg"] * 100,
                        1,
                    )
                    if baseline["total_kg"] > 0
                    else 0.0
                ),
            }
        )

    rows.sort(key=lambda row: (not row["feasible"], row["total_kg"]))
    return rows


# ---------------------------------------------------------------------------
# Office and home days
# ---------------------------------------------------------------------------

def office_day_emissions(days_in_office, team_size=1, office_open_days=None,
                         grid_intensity=DEFAULT_GRID_INTENSITY):
    """Emissions from the office building for one person's week.

    The fixed share is charged for every day the building is *open*, not every
    day the person attends, because that energy is spent regardless. Only when
    the office can close for a day does anyone stop paying for it.
    """
    attend = min(WORKING_DAYS_PER_WEEK, max(0, int(_clean_positive(days_in_office, 7.0))))
    open_days = (
        WORKING_DAYS_PER_WEEK
        if office_open_days is None
        else min(WORKING_DAYS_PER_WEEK, max(0, int(_clean_positive(office_open_days, 7.0))))
    )
    open_days = max(open_days, attend)
    intensity = _clean_positive(grid_intensity, 2.0, DEFAULT_GRID_INTENSITY)

    fixed_kwh = OFFICE_FIXED_KWH_PER_DESK_DAY * open_days
    marginal_kwh = OFFICE_MARGINAL_KWH_PER_PERSON_DAY * attend

    return {
        "days_attended": attend,
        "office_open_days": open_days,
        "fixed_kwh": round(fixed_kwh, 2),
        "marginal_kwh": round(marginal_kwh, 2),
        "total_kwh": round(fixed_kwh + marginal_kwh, 2),
        "co2_kg": round((fixed_kwh + marginal_kwh) * intensity, 3),
    }


def home_day_emissions(days_at_home, season=DEFAULT_SEASON,
                       grid_intensity=DEFAULT_GRID_INTENSITY, home_kwh_override=None):
    """Emissions from working at home, which are not zero.

    A home worked in is heated, lit and powered for a day it would otherwise
    have sat empty. In winter that is the single largest term in the whole
    comparison.
    """
    days = min(WORKING_DAYS_PER_WEEK, max(0, int(_clean_positive(days_at_home, 7.0))))
    per_day = (
        _clean_positive(home_kwh_override, 50.0)
        if home_kwh_override is not None
        else HOME_WORKING_KWH.get(season, HOME_WORKING_KWH[DEFAULT_SEASON])
    )
    intensity = _clean_positive(grid_intensity, 2.0, DEFAULT_GRID_INTENSITY)

    return {
        "days_at_home": days,
        "kwh_per_day": round(per_day, 2),
        "total_kwh": round(per_day * days, 2),
        "co2_kg": round(per_day * days * intensity, 3),
    }


def weekly_plan(days_in_office, distance_km, mode=DEFAULT_MODE, occupants=1,
                season=DEFAULT_SEASON, office_open_days=None,
                grid_intensity=DEFAULT_GRID_INTENSITY, home_kwh_override=None):
    """A full week: commuting, the office building and the home office."""
    attend = min(WORKING_DAYS_PER_WEEK, max(0, int(_clean_positive(days_in_office, 7.0))))
    at_home = WORKING_DAYS_PER_WEEK - attend

    leg = trip_emissions(mode, distance_km, occupants, season)
    commute_kg = leg["total_kg"] * 2 * attend  # there and back

    office = office_day_emissions(attend, office_open_days=office_open_days,
                                  grid_intensity=grid_intensity)
    home = home_day_emissions(at_home, season, grid_intensity, home_kwh_override)

    total = commute_kg + office["co2_kg"] + home["co2_kg"]

    return {
        "days_in_office": attend,
        "days_at_home": at_home,
        "mode": leg["mode"],
        "season": season,
        "leg": leg,
        "commute_kg": round(commute_kg, 3),
        "office": office,
        "home": home,
        "total_kg": round(total, 3),
        "annual_kg": round(total * WORKING_WEEKS_PER_YEAR, 1),
        "commute_share_pct": round(commute_kg / total * 100, 1) if total > 0 else 0.0,
        "minutes_travelled": round(leg["minutes"] * 2 * attend, 1),
    }


def compare_schedules(distance_km, mode=DEFAULT_MODE, occupants=1, season=DEFAULT_SEASON,
                      office_open_days=None, grid_intensity=DEFAULT_GRID_INTENSITY,
                      home_kwh_override=None):
    """Every attendance pattern from zero to five days in the office."""
    return [
        weekly_plan(days, distance_km, mode, occupants, season, office_open_days,
                    grid_intensity, home_kwh_override)
        for days in range(WORKING_DAYS_PER_WEEK + 1)
    ]


def best_schedule(distance_km, mode=DEFAULT_MODE, occupants=1, season=DEFAULT_SEASON,
                  office_open_days=None, grid_intensity=DEFAULT_GRID_INTENSITY,
                  home_kwh_override=None):
    """The lowest-emission attendance pattern, and by how much it wins.

    For a long car commute this is usually "stay home". For a short walk in a
    cold climate it is genuinely "go in", because heating an extra home all
    winter costs more than the journey does.
    """
    schedules = compare_schedules(distance_km, mode, occupants, season, office_open_days,
                                  grid_intensity, home_kwh_override)
    best = min(schedules, key=lambda plan: plan["total_kg"])
    worst = max(schedules, key=lambda plan: plan["total_kg"])

    return {
        "best": best,
        "worst": worst,
        "schedules": schedules,
        "spread_kg": round(worst["total_kg"] - best["total_kg"], 3),
        "annual_spread_kg": round(
            (worst["total_kg"] - best["total_kg"]) * WORKING_WEEKS_PER_YEAR, 1
        ),
        "home_is_better": best["days_in_office"] < worst["days_in_office"],
    }


def consolidation_benefit(days_in_office, distance_km, mode=DEFAULT_MODE, occupants=1,
                          season=DEFAULT_SEASON, grid_intensity=DEFAULT_GRID_INTENSITY,
                          home_kwh_override=None):
    """What a team gains by attending on the *same* days rather than scattered.

    Attendance spread across the week keeps the building open five days for the
    same number of desk-days. Anchor days let it close on the rest, and the
    fixed energy is the part that actually stops being spent.
    """
    attend = min(WORKING_DAYS_PER_WEEK, max(0, int(_clean_positive(days_in_office, 7.0))))

    scattered = weekly_plan(attend, distance_km, mode, occupants, season,
                            office_open_days=WORKING_DAYS_PER_WEEK,
                            grid_intensity=grid_intensity,
                            home_kwh_override=home_kwh_override)
    consolidated = weekly_plan(attend, distance_km, mode, occupants, season,
                               office_open_days=attend,
                               grid_intensity=grid_intensity,
                               home_kwh_override=home_kwh_override)

    saving = round(scattered["total_kg"] - consolidated["total_kg"], 3)

    return {
        "days_in_office": attend,
        "scattered_kg": scattered["total_kg"],
        "consolidated_kg": consolidated["total_kg"],
        "saving_kg": saving,
        "annual_saving_kg": round(saving * WORKING_WEEKS_PER_YEAR, 1),
        "closed_days": max(0, WORKING_DAYS_PER_WEEK - attend),
        "worth_doing": saving > 0,
    }


def annual_summary(days_in_office, distance_km, mode=DEFAULT_MODE, occupants=1,
                   office_open_days=None, grid_intensity=DEFAULT_GRID_INTENSITY):
    """A year of this pattern, weighted across the seasons.

    Averaging seasons matters because the home-working penalty is three times
    larger in winter than in summer, and a single-season answer is misleading.
    """
    seasons = []
    total = 0.0
    for season, weeks in SEASON_WEEKS.items():
        plan = weekly_plan(days_in_office, distance_km, mode, occupants, season,
                           office_open_days, grid_intensity)
        seasons.append(
            {
                "season": season,
                "weeks": weeks,
                "weekly_kg": plan["total_kg"],
                "season_kg": round(plan["total_kg"] * weeks, 1),
            }
        )
        total += plan["total_kg"] * weeks

    return {
        "days_in_office": min(WORKING_DAYS_PER_WEEK, max(0, int(_clean_positive(days_in_office, 7.0)))),
        "seasons": seasons,
        "annual_kg": round(total, 1),
        "weeks_counted": sum(SEASON_WEEKS.values()),
        "worst_season": max(seasons, key=lambda row: row["weekly_kg"])["season"],
    }


def get_commute_advice(plan, comparison=None, consolidation=None, limit=6):
    """Advice ranked by what this particular commute looks like."""
    if not plan or plan.get("total_kg", 0) <= 0:
        return ["Enter your commute above to see where the week's emissions go."]

    advice = []

    if comparison and comparison["best"]["days_in_office"] != plan["days_in_office"]:
        best = comparison["best"]
        advice.append(
            f"{best['days_in_office']} day(s) in the office would be lowest for you - "
            f"{comparison['annual_spread_kg']:,.0f} kg a year separates the best "
            f"pattern from the worst."
        )

    if plan["commute_share_pct"] < 40 and plan["days_in_office"] > 0:
        advice.append(
            f"Only {plan['commute_share_pct']}% of this week's emissions are the "
            f"journey itself. The rest is buildings - the office is heated whether "
            f"you attend or not, and your home is heated on the days you stay in it."
        )

    leg = plan.get("leg", {})
    if leg.get("cold_start_kg", 0) > leg.get("running_kg", 0) * 0.25:
        advice.append(
            "A large share of each trip is the cold start. On a commute this short "
            "the engine barely warms up, which is exactly where cycling or walking "
            "wins by the largest margin."
        )

    if consolidation and consolidation["worth_doing"]:
        advice.append(
            f"If your team attends on the same days, the office can close on "
            f"{consolidation['closed_days']} - worth about "
            f"{consolidation['annual_saving_kg']:,.0f} kg a year, and none of it "
            f"requires anyone to travel differently."
        )

    if plan["season"] == "Winter" and plan["days_at_home"] > 0:
        advice.append(
            "Working from home in winter is not free: heating a home that would "
            "otherwise be empty is the largest single term in this comparison."
        )

    advice.append(
        "Batch your office days. Two days in one week and three the next is worse "
        "than a steady pattern the building can be scheduled around."
    )
    advice.append(
        "If you drive, sharing the car with one colleague halves the journey - the "
        "single largest change available without changing mode at all."
    )

    return advice[: max(0, int(limit))]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _get_conn():
    return sqlite3.connect(DB_NAME)


def init_commute_db():
    """Create the commute plan table if it does not exist yet."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS commute_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plan_name TEXT NOT NULL,
                mode TEXT NOT NULL,
                distance_km REAL NOT NULL,
                days_in_office INTEGER NOT NULL,
                occupants INTEGER NOT NULL DEFAULT 1,
                season TEXT NOT NULL,
                weekly_kg REAL NOT NULL,
                annual_kg REAL NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Commute planner init error: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def save_commute_plan(user_id, plan_name, plan, distance_km=0.0):
    """Persist a weekly plan. Returns the new row id or None."""
    init_commute_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute(
            """
            INSERT INTO commute_plans (
                user_id, plan_name, mode, distance_km, days_in_office,
                occupants, season, weekly_kg, annual_kg, detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                (plan_name or "My commute").strip() or "My commute",
                plan.get("mode", DEFAULT_MODE),
                _clean_positive(distance_km or plan.get("leg", {}).get("distance_km", 0.0), 1000.0),
                plan.get("days_in_office", 0),
                plan.get("leg", {}).get("occupants", 1),
                plan.get("season", DEFAULT_SEASON),
                plan.get("total_kg", 0.0),
                plan.get("annual_kg", 0.0),
                json.dumps(
                    {
                        "leg": plan.get("leg", {}),
                        "office": plan.get("office", {}),
                        "home": plan.get("home", {}),
                        "commute_kg": plan.get("commute_kg", 0.0),
                    }
                ),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("Unable to save commute plan: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


def get_commute_plans(user_id, limit=25):
    """Return a user's saved commute plans, newest first."""
    init_commute_db()
    conn = None
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, plan_name, mode, distance_km, days_in_office, occupants,
                   season, weekly_kg, annual_kg, detail_json, created_at
            FROM commute_plans
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
                record["detail"] = json.loads(record.pop("detail_json"))
            except (TypeError, ValueError):
                record["detail"] = {}
            plans.append(record)
        return plans
    except sqlite3.Error as exc:
        logger.error("Unable to load commute plans: %s", exc)
        return []
    finally:
        if conn:
            conn.close()


def delete_commute_plan(plan_id):
    """Delete a saved commute plan."""
    init_commute_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM commute_plans WHERE id = ?", (plan_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Unable to delete commute plan: %s", exc)
        return False
    finally:
        if conn:
            conn.close()
