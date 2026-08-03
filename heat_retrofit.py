"""Home heat loss, retrofit sequencing and heat pump sizing.

``energy_audit.py`` measures *appliances*: plug loads, standby draw, hours of
use. In most temperate homes that is the smaller half of the energy bill. The
larger half - space heating - is missing entirely, and with it the single
biggest decision a household will ever make about its emissions: what to do
about the heating system, and in what order.

This module models the building rather than the appliances.

Method
------
Heat loss is a property of the fabric, expressed as a heat loss coefficient in
watts per kelvin::

    HLC = sum(U_element * area_element) + 0.33 * air_changes_per_hour * volume

Annual heat demand then follows the standard degree-day method::

    kWh = HLC * heating_degree_days * 24 / 1000

Heating degree days come from monthly mean outdoor temperatures against a base
temperature that already allows for internal and solar gains, so a warm
climate produces a small demand without any extra fudge factor.

Why the sequencing matters
--------------------------
A heat pump is not a like-for-like boiler swap. Its efficiency collapses as the
water temperature it must produce rises, and a leaky house needs hotter water
to stay warm through the same radiators. So the same heat pump in the same
house can deliver a seasonal COP of 4.2 or 2.2 depending entirely on whether
the fabric was fixed first. This module makes that chain explicit:

    fabric -> peak heat loss -> required flow temperature -> COP -> running cost

which is why "fabric first" is advice and not a slogan.

The module is self-contained: it imports no Streamlit, its SQLite table is
created lazily and no shared files are modified.
"""

import os
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Base temperature for degree days. It sits below room temperature because
# bodies, cooking, appliances and sunlight already supply part of the heat.
BASE_TEMPERATURE_C = 15.5

# Comfort temperature and the cold-snap temperature a system must still cope
# with, used for sizing rather than for annual energy.
DESIGN_INDOOR_C = 21.0

# Ventilation heat loss constant: the volumetric heat capacity of air,
# 0.33 watt-hours per cubic metre per kelvin.
AIR_HEAT_CAPACITY = 0.33

# Monthly mean outdoor temperatures in degrees Celsius, plus the design
# outdoor temperature used for sizing.
CLIMATE_ZONES = {
    "Cold continental": {
        "monthly_c": [-8, -6, 0, 8, 15, 20, 22, 21, 15, 7, -1, -6],
        "design_c": -15.0,
    },
    "Temperate maritime": {
        "monthly_c": [5, 5, 7, 9, 13, 16, 18, 18, 15, 12, 8, 6],
        "design_c": -2.0,
    },
    "Temperate continental": {
        "monthly_c": [0, 2, 6, 11, 16, 19, 21, 20, 16, 11, 5, 1],
        "design_c": -10.0,
    },
    "Mediterranean": {
        "monthly_c": [10, 11, 13, 16, 20, 24, 27, 27, 23, 19, 14, 11],
        "design_c": 2.0,
    },
    "Humid subtropical": {
        "monthly_c": [12, 14, 17, 21, 25, 28, 29, 29, 26, 21, 16, 13],
        "design_c": 4.0,
    },
    "Highland / alpine": {
        "monthly_c": [-4, -3, 1, 5, 10, 14, 16, 15, 11, 6, 1, -3],
        "design_c": -14.0,
    },
}

DEFAULT_CLIMATE_ZONE = "Temperate maritime"

# U-values in W/m2K. Lower is better: it is the rate heat escapes per square
# metre for each degree of difference between inside and outside.
WALL_TYPES = {
    "Solid brick, uninsulated": 2.10,
    "Cavity, uninsulated": 1.60,
    "Cavity, insulated": 0.55,
    "Solid wall, internally insulated": 0.45,
    "Modern insulated (current regulations)": 0.28,
    "Passive-house standard": 0.15,
}

ROOF_TYPES = {
    "No loft insulation": 2.30,
    "100 mm loft insulation": 0.40,
    "270 mm loft insulation": 0.16,
    "Warm roof / passive standard": 0.11,
}

FLOOR_TYPES = {
    "Suspended timber, uninsulated": 0.70,
    "Solid concrete, uninsulated": 0.75,
    "Insulated floor": 0.22,
    "Passive-house standard": 0.12,
}

GLAZING_TYPES = {
    "Single glazed": 4.80,
    "Older double glazed": 3.10,
    "Modern double glazed": 1.40,
    "Triple glazed": 0.80,
}

DOOR_TYPES = {
    "Uninsulated timber or single-glazed": 3.00,
    "Insulated composite": 1.40,
}

# Air changes per hour from infiltration - draughts, not deliberate ventilation.
AIRTIGHTNESS_LEVELS = {
    "Draughty (older, unsealed)": 1.00,
    "Average": 0.60,
    "Well sealed": 0.35,
    "Airtight with heat recovery": 0.15,
}

DEFAULT_STOREY_HEIGHT_M = 2.5

# Fraction of external wall area that is window or door, by dwelling type.
GLAZING_FRACTION = 0.18
DOOR_AREA_M2 = 3.6

# Heating systems. ``efficiency`` is kWh of heat per kWh of fuel; heat pumps
# get theirs from the COP model instead, so they are marked as variable.
HEATING_SYSTEMS = {
    "Gas boiler (condensing)": {
        "fuel": "Natural gas", "efficiency": 0.88, "variable_efficiency": False,
    },
    "Gas boiler (older, non-condensing)": {
        "fuel": "Natural gas", "efficiency": 0.72, "variable_efficiency": False,
    },
    "Oil boiler": {
        "fuel": "Heating oil", "efficiency": 0.83, "variable_efficiency": False,
    },
    "LPG boiler": {
        "fuel": "LPG", "efficiency": 0.85, "variable_efficiency": False,
    },
    "Electric resistance heating": {
        "fuel": "Electricity", "efficiency": 1.00, "variable_efficiency": False,
    },
    "Air source heat pump": {
        "fuel": "Electricity", "efficiency": None, "variable_efficiency": True,
        "carnot_quality": 0.42, "source": "air",
    },
    "Ground source heat pump": {
        "fuel": "Electricity", "efficiency": None, "variable_efficiency": True,
        "carnot_quality": 0.48, "source": "ground",
    },
    "Biomass (wood pellet)": {
        "fuel": "Wood pellets", "efficiency": 0.80, "variable_efficiency": False,
    },
}

# kg CO2e per kWh of delivered fuel, and a representative unit price.
FUELS = {
    "Natural gas": {"kg_co2_per_kwh": 0.203, "price_per_kwh": 0.07},
    "Heating oil": {"kg_co2_per_kwh": 0.267, "price_per_kwh": 0.09},
    "LPG": {"kg_co2_per_kwh": 0.214, "price_per_kwh": 0.11},
    "Electricity": {"kg_co2_per_kwh": 0.207, "price_per_kwh": 0.25},
    "Wood pellets": {"kg_co2_per_kwh": 0.039, "price_per_kwh": 0.08},
}

# Emitters, expressed as the water temperature they need to deliver design
# output. This is the number that decides whether a heat pump works well.
EMITTER_TYPES = {
    "Existing radiators (unchanged)": {"design_flow_c": 65.0, "oversize_factor": 1.0},
    "Existing radiators, one size up": {"design_flow_c": 55.0, "oversize_factor": 1.4},
    "Oversized low-temperature radiators": {"design_flow_c": 45.0, "oversize_factor": 2.0},
    "Underfloor heating": {"design_flow_c": 35.0, "oversize_factor": 3.0},
}

DEFAULT_EMITTER = "Existing radiators (unchanged)"

# Radiators in an existing house were sized to heat it at this flow
# temperature, which is the reference the emitter model works back from.
REFERENCE_FLOW_C = 65.0

# Practical limits: no wet system usefully runs below 30 °C, and nothing in a
# domestic retrofit runs above 75 °C.
MIN_FLOW_C = 30.0
MAX_FLOW_C = 75.0

# Ground loops sit at a stable temperature all year rather than tracking air.
GROUND_SOURCE_TEMPERATURE_C = 8.0

COP_FLOOR = 1.4
COP_CEILING = 6.0

# Retrofit measures: the element they improve, the type they upgrade it to,
# and an installed cost. Costs are indicative and fully editable in the UI.
RETROFIT_MEASURES = {
    "Loft insulation to 270 mm": {
        "element": "roof", "upgrade_to": "270 mm loft insulation",
        "cost_per_m2": 22.0, "fixed_cost": 150.0,
        "note": "Cheapest measure in almost every house. Do it first.",
    },
    "Cavity wall insulation": {
        "element": "wall", "upgrade_to": "Cavity, insulated",
        "cost_per_m2": 28.0, "fixed_cost": 300.0,
        "requires_from": ["Cavity, uninsulated"],
        "note": "Only possible if the walls actually have a cavity.",
    },
    "Internal solid wall insulation": {
        "element": "wall", "upgrade_to": "Solid wall, internally insulated",
        "cost_per_m2": 120.0, "fixed_cost": 900.0,
        "requires_from": ["Solid brick, uninsulated"],
        "note": "Expensive and disruptive, but solid walls leak enormously.",
    },
    "Floor insulation": {
        "element": "floor", "upgrade_to": "Insulated floor",
        "cost_per_m2": 55.0, "fixed_cost": 400.0,
        "note": "Best done while floors are up for another reason.",
    },
    "Replace single glazing with double": {
        "element": "glazing", "upgrade_to": "Modern double glazed",
        "cost_per_m2": 520.0, "fixed_cost": 0.0,
        "requires_from": ["Single glazed", "Older double glazed"],
        "note": "Rarely pays back on energy alone - it buys comfort and quiet.",
    },
    "Upgrade to triple glazing": {
        "element": "glazing", "upgrade_to": "Triple glazed",
        "cost_per_m2": 700.0, "fixed_cost": 0.0,
        "note": "Only worth it in a cold climate or a very efficient house.",
    },
    "Draught proofing": {
        "element": "airtightness", "upgrade_to": "Well sealed",
        "cost_per_m2": 0.0, "fixed_cost": 450.0,
        "note": "Tiny cost, immediate comfort gain. Ventilate deliberately afterwards.",
    },
    "Insulated front and back doors": {
        "element": "door", "upgrade_to": "Insulated composite",
        "cost_per_m2": 0.0, "fixed_cost": 1400.0,
        "note": "Small energy effect; mostly a draught and comfort measure.",
    },
}


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

def list_climate_zones():
    """Return climate zones, coldest first."""
    return sorted(
        (
            {"name": name, "mean_c": round(sum(data["monthly_c"]) / 12.0, 1), **data}
            for name, data in CLIMATE_ZONES.items()
        ),
        key=lambda zone: zone["mean_c"],
    )


def get_climate(zone):
    """Return the climate record for a zone, falling back to the default."""
    data = CLIMATE_ZONES.get(zone) or CLIMATE_ZONES[DEFAULT_CLIMATE_ZONE]
    return {"monthly_c": list(data["monthly_c"]), "design_c": data["design_c"]}


def u_value(element, construction):
    """U-value for a named construction, or the worst case if unrecognised."""
    tables = {
        "wall": WALL_TYPES,
        "roof": ROOF_TYPES,
        "floor": FLOOR_TYPES,
        "glazing": GLAZING_TYPES,
        "door": DOOR_TYPES,
    }
    table = tables.get(element)
    if not table:
        return 0.0
    if construction in table:
        return table[construction]
    # Unknown constructions are assumed to be the worst in their class rather
    # than the best, so the model never flatters a house it does not know.
    return max(table.values())


def air_changes(level):
    """Infiltration rate for an airtightness level."""
    return AIRTIGHTNESS_LEVELS.get(level, max(AIRTIGHTNESS_LEVELS.values()))


# ---------------------------------------------------------------------------
# Building geometry and fabric
# ---------------------------------------------------------------------------

def estimate_envelope(floor_area_m2, storeys=2, attached_walls=0, storey_height_m=DEFAULT_STOREY_HEIGHT_M):
    """Estimate envelope areas from the numbers a householder actually knows.

    ``attached_walls`` is how many of the four walls are shared with a
    neighbour - a mid-terrace house loses heat through two walls, not four,
    which matters more than almost any construction detail.
    """
    total_floor = _clean_positive(floor_area_m2, 2000.0)
    levels = max(1, int(_clean_positive(storeys, 6.0, 1.0) or 1))
    height = _clean_positive(storey_height_m, 5.0, DEFAULT_STOREY_HEIGHT_M) or DEFAULT_STOREY_HEIGHT_M
    shared = max(0, min(3, int(_clean_positive(attached_walls, 3.0))))

    footprint = total_floor / levels
    side = footprint ** 0.5 if footprint > 0 else 0.0
    perimeter = 4 * side
    exposed_perimeter = perimeter * (4 - shared) / 4.0

    gross_wall = exposed_perimeter * height * levels
    glazing_area = gross_wall * GLAZING_FRACTION
    door_area = min(DOOR_AREA_M2, max(0.0, gross_wall - glazing_area))
    net_wall = max(0.0, gross_wall - glazing_area - door_area)

    return {
        "floor_area_m2": round(total_floor, 1),
        "storeys": levels,
        "footprint_m2": round(footprint, 1),
        "wall_m2": round(net_wall, 1),
        "glazing_m2": round(glazing_area, 1),
        "door_m2": round(door_area, 1),
        "roof_m2": round(footprint, 1),
        "ground_floor_m2": round(footprint, 1),
        "volume_m3": round(total_floor * height, 1),
        "attached_walls": shared,
    }


def heat_loss_coefficient(envelope, fabric):
    """Whole-house heat loss coefficient in watts per kelvin.

    Returns the total together with its per-element breakdown, because the
    breakdown is what tells a user which measure to do first.
    """
    elements = {
        "wall": (envelope.get("wall_m2", 0.0), fabric.get("wall")),
        "roof": (envelope.get("roof_m2", 0.0), fabric.get("roof")),
        "floor": (envelope.get("ground_floor_m2", 0.0), fabric.get("floor")),
        "glazing": (envelope.get("glazing_m2", 0.0), fabric.get("glazing")),
        "door": (envelope.get("door_m2", 0.0), fabric.get("door")),
    }

    breakdown = {}
    fabric_total = 0.0
    for element, (area, construction) in elements.items():
        area = _clean_positive(area, 5000.0)
        loss = area * u_value(element, construction)
        breakdown[element] = round(loss, 2)
        fabric_total += loss

    volume = _clean_positive(envelope.get("volume_m3", 0.0), 20000.0)
    ventilation = AIR_HEAT_CAPACITY * air_changes(fabric.get("airtightness")) * volume
    breakdown["ventilation"] = round(ventilation, 2)

    total = fabric_total + ventilation
    return {
        "total_w_per_k": round(total, 2),
        "fabric_w_per_k": round(fabric_total, 2),
        "ventilation_w_per_k": round(ventilation, 2),
        "breakdown": breakdown,
        "worst_element": (
            max(breakdown, key=breakdown.get) if any(breakdown.values()) else None
        ),
    }


# ---------------------------------------------------------------------------
# Degree days and demand
# ---------------------------------------------------------------------------

def monthly_degree_days(climate_zone, base_temperature_c=BASE_TEMPERATURE_C):
    """Heating degree days for each month, in kelvin-days."""
    climate = get_climate(climate_zone)
    base = _clean_positive(base_temperature_c, 30.0, BASE_TEMPERATURE_C)
    return [
        round(max(0.0, base - temperature) * days, 1)
        for temperature, days in zip(climate["monthly_c"], DAYS_IN_MONTH)
    ]


def annual_degree_days(climate_zone, base_temperature_c=BASE_TEMPERATURE_C):
    """Total heating degree days in a year."""
    return round(sum(monthly_degree_days(climate_zone, base_temperature_c)), 1)


def heat_demand_kwh(hlc_w_per_k, climate_zone, base_temperature_c=BASE_TEMPERATURE_C):
    """Annual space heating demand at the emitter, in kWh."""
    hlc = _clean_positive(hlc_w_per_k, 20000.0)
    degree_days = annual_degree_days(climate_zone, base_temperature_c)
    return round(hlc * degree_days * 24 / 1000.0, 1)


def monthly_heat_demand_kwh(hlc_w_per_k, climate_zone, base_temperature_c=BASE_TEMPERATURE_C):
    """Month-by-month heating demand, in kWh."""
    hlc = _clean_positive(hlc_w_per_k, 20000.0)
    return [
        round(hlc * degree_days * 24 / 1000.0, 1)
        for degree_days in monthly_degree_days(climate_zone, base_temperature_c)
    ]


def peak_heat_load_w(hlc_w_per_k, climate_zone):
    """Heat output needed on the coldest design day, in watts."""
    hlc = _clean_positive(hlc_w_per_k, 20000.0)
    design_outdoor = get_climate(climate_zone)["design_c"]
    return round(hlc * (DESIGN_INDOOR_C - design_outdoor), 0)


# ---------------------------------------------------------------------------
# Heat pump behaviour
# ---------------------------------------------------------------------------

def emitter_capacity_w(baseline_peak_w, emitter_type):
    """Heat the emitters can deliver at the reference 65 °C flow temperature.

    Existing radiators were sized to heat the house *as it was*, so the
    baseline peak load is their capacity. Fitting bigger radiators or
    underfloor pipes multiplies that capacity.
    """
    emitter = EMITTER_TYPES.get(emitter_type) or EMITTER_TYPES[DEFAULT_EMITTER]
    return round(_clean_positive(baseline_peak_w, 200000.0) * emitter["oversize_factor"], 0)


def required_flow_temperature(peak_load_w, emitter_type, baseline_peak_w=None):
    """Water temperature the emitters need to meet the peak load.

    Radiator output scales with the temperature difference to the room raised
    to about 1.3, so::

        needed_delta = reference_delta * (load / capacity) ** (1 / 1.3)

    A house that has been insulated asks less of the same radiators, which
    lets them run cooler - and cooler water is exactly what a heat pump wants.
    That single relationship is the whole argument for doing fabric first.
    """
    peak = _clean_positive(peak_load_w, 200000.0)
    if peak <= 0:
        return round(MIN_FLOW_C, 1)

    # With no baseline given, assume the emitters were sized for this load.
    capacity = emitter_capacity_w(
        baseline_peak_w if baseline_peak_w is not None else peak, emitter_type
    )
    if capacity <= 0:
        return round(MAX_FLOW_C, 1)

    reference_delta = REFERENCE_FLOW_C - DESIGN_INDOOR_C
    needed_delta = reference_delta * (peak / capacity) ** (1 / 1.3)
    return round(min(MAX_FLOW_C, max(MIN_FLOW_C, DESIGN_INDOOR_C + needed_delta)), 1)


def seasonal_cop(system_name, climate_zone, flow_temperature_c):
    """Seasonal coefficient of performance for a heat pump.

    A Carnot-fraction model: the theoretical limit between the source and the
    water temperature, multiplied by how close real machines get to it.
    """
    system = HEATING_SYSTEMS.get(system_name)
    if not system or not system.get("variable_efficiency"):
        return None

    climate = get_climate(climate_zone)
    heating_months = [
        temperature
        for temperature in climate["monthly_c"]
        if temperature < BASE_TEMPERATURE_C
    ]
    mean_source_c = (
        sum(heating_months) / len(heating_months)
        if heating_months
        else sum(climate["monthly_c"]) / 12.0
    )
    if system.get("source") == "ground":
        mean_source_c = GROUND_SOURCE_TEMPERATURE_C

    flow_c = max(20.0, min(75.0, float(flow_temperature_c or 55.0)))
    flow_k = flow_c + 273.15
    lift_k = max(5.0, flow_c - mean_source_c)

    cop = system["carnot_quality"] * flow_k / lift_k
    return round(max(COP_FLOOR, min(COP_CEILING, cop)), 2)


def system_efficiency(system_name, climate_zone=DEFAULT_CLIMATE_ZONE, flow_temperature_c=None):
    """Delivered heat per unit of fuel for any system, heat pumps included."""
    system = HEATING_SYSTEMS.get(system_name)
    if not system:
        return 1.0
    if system.get("variable_efficiency"):
        return seasonal_cop(system_name, climate_zone, flow_temperature_c or 55.0)
    return system["efficiency"]


def running_cost_and_emissions(demand_kwh, system_name, climate_zone=DEFAULT_CLIMATE_ZONE,
                               flow_temperature_c=None, fuel_overrides=None):
    """Fuel use, cost and emissions to meet a heating demand."""
    system = HEATING_SYSTEMS.get(system_name)
    if not system:
        return None

    demand = _clean_positive(demand_kwh, 500000.0)
    efficiency = system_efficiency(system_name, climate_zone, flow_temperature_c) or 1.0
    fuel_name = system["fuel"]
    fuel = dict(FUELS[fuel_name])
    if fuel_overrides and fuel_name in fuel_overrides:
        fuel.update(fuel_overrides[fuel_name])

    fuel_kwh = demand / efficiency if efficiency > 0 else 0.0

    return {
        "system": system_name,
        "fuel": fuel_name,
        "efficiency": round(efficiency, 2),
        "demand_kwh": round(demand, 1),
        "fuel_kwh": round(fuel_kwh, 1),
        "cost": round(fuel_kwh * fuel["price_per_kwh"], 2),
        "co2_kg": round(fuel_kwh * fuel["kg_co2_per_kwh"], 1),
    }


def compare_systems(demand_kwh, climate_zone=DEFAULT_CLIMATE_ZONE,
                    flow_temperature_c=None, fuel_overrides=None):
    """Every heating system against the same demand, lowest carbon first."""
    rows = []
    for name in HEATING_SYSTEMS:
        result = running_cost_and_emissions(
            demand_kwh, name, climate_zone, flow_temperature_c, fuel_overrides
        )
        if result:
            rows.append(result)
    rows.sort(key=lambda row: row["co2_kg"])
    return rows


# ---------------------------------------------------------------------------
# Retrofit measures
# ---------------------------------------------------------------------------

def measure_applies(measure_name, fabric):
    """Whether a measure is possible and would actually improve the house."""
    measure = RETROFIT_MEASURES.get(measure_name)
    if not measure:
        return False

    element = measure["element"]
    current = fabric.get(element)

    allowed = measure.get("requires_from")
    if allowed and current not in allowed:
        return False

    if element == "airtightness":
        return air_changes(current) > AIRTIGHTNESS_LEVELS[measure["upgrade_to"]]

    return u_value(element, current) > u_value(element, measure["upgrade_to"])


def measure_cost(measure_name, envelope):
    """Installed cost of a measure for this particular house."""
    measure = RETROFIT_MEASURES.get(measure_name)
    if not measure:
        return 0.0

    area_keys = {
        "wall": "wall_m2",
        "roof": "roof_m2",
        "floor": "ground_floor_m2",
        "glazing": "glazing_m2",
        "door": "door_m2",
        "airtightness": None,
    }
    key = area_keys.get(measure["element"])
    area = _clean_positive(envelope.get(key, 0.0), 5000.0) if key else 0.0
    return round(area * measure["cost_per_m2"] + measure["fixed_cost"], 2)


def apply_measure(fabric, measure_name):
    """Return a copy of the fabric with one measure applied."""
    measure = RETROFIT_MEASURES.get(measure_name)
    updated = dict(fabric)
    if measure:
        updated[measure["element"]] = measure["upgrade_to"]
    return updated


def rank_measures(envelope, fabric, climate_zone=DEFAULT_CLIMATE_ZONE,
                  system_name="Gas boiler (condensing)", fuel_overrides=None):
    """Rank every applicable measure by cost per kWh saved.

    Each measure is evaluated against the *current* house, so the ranking
    answers "what should I do next", not "what looks good in a brochure".
    """
    emitter = fabric.get("emitter", DEFAULT_EMITTER)
    baseline_hlc = heat_loss_coefficient(envelope, fabric)["total_w_per_k"]
    baseline_demand = heat_demand_kwh(baseline_hlc, climate_zone)
    baseline_peak = peak_heat_load_w(baseline_hlc, climate_zone)
    baseline = running_cost_and_emissions(
        baseline_demand, system_name, climate_zone,
        required_flow_temperature(baseline_peak, emitter, baseline_peak),
        fuel_overrides,
    )

    ranked = []
    for name in RETROFIT_MEASURES:
        if not measure_applies(name, fabric):
            continue

        improved_fabric = apply_measure(fabric, name)
        improved_hlc = heat_loss_coefficient(envelope, improved_fabric)["total_w_per_k"]
        improved_demand = heat_demand_kwh(improved_hlc, climate_zone)
        improved = running_cost_and_emissions(
            improved_demand, system_name, climate_zone,
            required_flow_temperature(
                peak_heat_load_w(improved_hlc, climate_zone), emitter, baseline_peak
            ),
            fuel_overrides,
        )

        saved_kwh = round(baseline_demand - improved_demand, 1)
        saved_cost = round(baseline["cost"] - improved["cost"], 2)
        saved_co2 = round(baseline["co2_kg"] - improved["co2_kg"], 1)
        cost = measure_cost(name, envelope)

        ranked.append(
            {
                "measure": name,
                "note": RETROFIT_MEASURES[name]["note"],
                "element": RETROFIT_MEASURES[name]["element"],
                "cost": cost,
                "saved_kwh": saved_kwh,
                "saved_cost": saved_cost,
                "saved_co2_kg": saved_co2,
                "cost_per_kwh_saved": (
                    round(cost / saved_kwh, 2) if saved_kwh > 0 else None
                ),
                "payback_years": (
                    round(cost / saved_cost, 1) if saved_cost > 0 else None
                ),
                "hlc_after": improved_hlc,
            }
        )

    ranked.sort(
        key=lambda row: (
            row["cost_per_kwh_saved"] if row["cost_per_kwh_saved"] is not None else 1e9
        )
    )
    return {"baseline_demand_kwh": baseline_demand, "baseline": baseline, "measures": ranked}


def build_retrofit_plan(envelope, fabric, measures, climate_zone=DEFAULT_CLIMATE_ZONE,
                        system_name="Gas boiler (condensing)", emitter=DEFAULT_EMITTER,
                        fuel_overrides=None):
    """Apply a sequence of measures and report the house after each one.

    Savings are computed against the running total, not against the original
    house, so overlapping measures cannot each claim the same kilowatt-hour.
    """
    current_fabric = dict(fabric)
    current_fabric.setdefault("emitter", emitter)

    baseline_hlc = heat_loss_coefficient(envelope, current_fabric)["total_w_per_k"]
    baseline_demand = heat_demand_kwh(baseline_hlc, climate_zone)
    baseline_peak = peak_heat_load_w(baseline_hlc, climate_zone)
    baseline_flow = required_flow_temperature(baseline_peak, emitter, baseline_peak)
    baseline_result = running_cost_and_emissions(
        baseline_demand, system_name, climate_zone, baseline_flow, fuel_overrides
    )

    steps = []
    total_cost = 0.0
    previous_demand = baseline_demand
    previous_cost = baseline_result["cost"]
    previous_co2 = baseline_result["co2_kg"]

    for name in measures or []:
        if not measure_applies(name, current_fabric):
            continue

        cost = measure_cost(name, envelope)
        current_fabric = apply_measure(current_fabric, name)
        hlc = heat_loss_coefficient(envelope, current_fabric)["total_w_per_k"]
        demand = heat_demand_kwh(hlc, climate_zone)
        peak = peak_heat_load_w(hlc, climate_zone)
        flow = required_flow_temperature(peak, emitter, baseline_peak)
        result = running_cost_and_emissions(
            demand, system_name, climate_zone, flow, fuel_overrides
        )

        total_cost += cost
        steps.append(
            {
                "measure": name,
                "cost": cost,
                "cumulative_cost": round(total_cost, 2),
                "hlc_w_per_k": hlc,
                "demand_kwh": demand,
                "peak_load_w": peak,
                "flow_temperature_c": flow,
                "efficiency": result["efficiency"],
                "cost_per_year": result["cost"],
                "co2_kg": result["co2_kg"],
                "step_saved_kwh": round(previous_demand - demand, 1),
                "step_saved_cost": round(previous_cost - result["cost"], 2),
                "step_saved_co2_kg": round(previous_co2 - result["co2_kg"], 1),
            }
        )
        previous_demand = demand
        previous_cost = result["cost"]
        previous_co2 = result["co2_kg"]

    final_hlc = heat_loss_coefficient(envelope, current_fabric)["total_w_per_k"]
    final_demand = heat_demand_kwh(final_hlc, climate_zone)
    final_peak = peak_heat_load_w(final_hlc, climate_zone)
    final_flow = required_flow_temperature(final_peak, emitter, baseline_peak)
    final_result = running_cost_and_emissions(
        final_demand, system_name, climate_zone, final_flow, fuel_overrides
    )

    annual_saving = round(baseline_result["cost"] - final_result["cost"], 2)

    return {
        "climate_zone": climate_zone,
        "system": system_name,
        "emitter": emitter,
        "envelope": dict(envelope),
        "fabric_before": dict(fabric),
        "fabric_after": current_fabric,
        "baseline": {
            "hlc_w_per_k": baseline_hlc,
            "demand_kwh": baseline_demand,
            "peak_load_w": baseline_peak,
            "flow_temperature_c": baseline_flow,
            "efficiency": baseline_result["efficiency"],
            "cost_per_year": baseline_result["cost"],
            "co2_kg": baseline_result["co2_kg"],
        },
        "after": {
            "hlc_w_per_k": final_hlc,
            "demand_kwh": final_demand,
            "peak_load_w": final_peak,
            "flow_temperature_c": final_flow,
            "efficiency": final_result["efficiency"],
            "cost_per_year": final_result["cost"],
            "co2_kg": final_result["co2_kg"],
        },
        "steps": steps,
        "total_cost": round(total_cost, 2),
        "demand_saved_kwh": round(baseline_demand - final_demand, 1),
        "demand_saved_pct": (
            round((baseline_demand - final_demand) / baseline_demand * 100, 1)
            if baseline_demand > 0
            else 0.0
        ),
        "co2_saved_kg": round(baseline_result["co2_kg"] - final_result["co2_kg"], 1),
        "annual_saving": annual_saving,
        "payback_years": (
            round(total_cost / annual_saving, 1) if annual_saving > 0 and total_cost > 0 else None
        ),
    }


def fabric_first_check(envelope, fabric, climate_zone=DEFAULT_CLIMATE_ZONE,
                       emitter=DEFAULT_EMITTER, measures=None):
    """Compare fitting a heat pump now with fitting it after the fabric work.

    This is the question the whole module exists to answer, because the two
    orderings give the same house very different heat pump performance.
    """
    plan = build_retrofit_plan(
        envelope, fabric, measures or [], climate_zone,
        system_name="Air source heat pump", emitter=emitter,
    )

    before = plan["baseline"]
    after = plan["after"]

    return {
        "cop_now": before["efficiency"],
        "cop_after_fabric": after["efficiency"],
        "flow_now_c": before["flow_temperature_c"],
        "flow_after_c": after["flow_temperature_c"],
        "peak_now_w": before["peak_load_w"],
        "peak_after_w": after["peak_load_w"],
        "electricity_now_kwh": round(
            before["demand_kwh"] / before["efficiency"], 1
        ) if before["efficiency"] else 0.0,
        "electricity_after_kwh": round(
            after["demand_kwh"] / after["efficiency"], 1
        ) if after["efficiency"] else 0.0,
        "cop_gain": round((after["efficiency"] or 0) - (before["efficiency"] or 0), 2),
        "smaller_unit_kw": round(max(0.0, before["peak_load_w"] - after["peak_load_w"]) / 1000.0, 1),
        "verdict": (
            "Fabric first: the same heat pump runs measurably better after the "
            "insulation work, and you can buy a smaller unit."
            if (after["efficiency"] or 0) > (before["efficiency"] or 0) + 0.15
            else "This house is already good enough that a heat pump would run "
            "well today - the fabric work is worth doing but is not blocking it."
        ),
    }


def get_retrofit_advice(plan, limit=6):
    """Advice ranked by what this particular house looks like."""
    if not plan or not plan.get("baseline", {}).get("demand_kwh"):
        return ["Describe your home above to see where its heat is going."]

    advice = []
    hlc = heat_loss_coefficient(plan.get("envelope", {}), plan.get("fabric_before", {})) \
        if plan.get("envelope") else None

    if hlc and hlc.get("worst_element"):
        worst = hlc["worst_element"]
        readable = {
            "wall": "the walls",
            "roof": "the roof",
            "floor": "the ground floor",
            "glazing": "the windows",
            "door": "the doors",
            "ventilation": "draughts and air leakage",
        }.get(worst, worst)
        advice.append(f"Most of this house's heat escapes through {readable} - start there.")

    if plan["demand_saved_pct"] > 0:
        advice.append(
            f"The measures you selected cut heat demand by {plan['demand_saved_pct']}% "
            f"({plan['demand_saved_kwh']:,.0f} kWh a year)."
        )

    if plan.get("payback_years"):
        advice.append(
            f"At current fuel prices the work pays back in about "
            f"{plan['payback_years']} years, before counting comfort."
        )
    elif plan.get("total_cost"):
        advice.append(
            "These measures do not pay back on fuel savings alone at current "
            "prices - they buy comfort, quiet and a warmer house in a cold snap."
        )

    before_flow = plan["baseline"]["flow_temperature_c"]
    after_flow = plan["after"]["flow_temperature_c"]
    if after_flow < before_flow - 2:
        advice.append(
            f"Flow temperature needed drops from {before_flow} °C to {after_flow} °C. "
            f"That is what makes a heat pump viable here."
        )

    advice.append(
        "Draught proofing is the cheapest measure in almost every home, but "
        "ventilate deliberately afterwards - sealed and unventilated causes damp."
    )
    advice.append(
        "Size the heating system after the insulation, not before. An oversized "
        "unit cycles, runs badly and costs more to buy."
    )

    return advice[: max(0, int(limit))]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _get_conn():
    return sqlite3.connect(DB_NAME)


def init_retrofit_db():
    """Create the retrofit plan table if it does not exist yet."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrofit_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plan_name TEXT NOT NULL,
                climate_zone TEXT NOT NULL,
                system TEXT NOT NULL,
                floor_area_m2 REAL NOT NULL,
                baseline_demand_kwh REAL NOT NULL,
                final_demand_kwh REAL NOT NULL,
                co2_saved_kg REAL NOT NULL,
                total_cost REAL NOT NULL,
                payback_years REAL,
                plan_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Retrofit init error: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def save_retrofit_plan(user_id, plan_name, plan, floor_area_m2=0.0):
    """Persist a retrofit plan. Returns the new row id or None."""
    init_retrofit_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute(
            """
            INSERT INTO retrofit_plans (
                user_id, plan_name, climate_zone, system, floor_area_m2,
                baseline_demand_kwh, final_demand_kwh, co2_saved_kg,
                total_cost, payback_years, plan_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                (plan_name or "My home").strip() or "My home",
                plan.get("climate_zone", DEFAULT_CLIMATE_ZONE),
                plan.get("system", "Gas boiler (condensing)"),
                _clean_positive(floor_area_m2, 2000.0),
                plan.get("baseline", {}).get("demand_kwh", 0.0),
                plan.get("after", {}).get("demand_kwh", 0.0),
                plan.get("co2_saved_kg", 0.0),
                plan.get("total_cost", 0.0),
                plan.get("payback_years"),
                json.dumps(
                    {
                        "steps": plan.get("steps", []),
                        "fabric_before": plan.get("fabric_before", {}),
                        "fabric_after": plan.get("fabric_after", {}),
                    }
                ),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("Unable to save retrofit plan: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


def get_retrofit_plans(user_id, limit=25):
    """Return a user's saved retrofit plans, newest first."""
    init_retrofit_db()
    conn = None
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, plan_name, climate_zone, system, floor_area_m2,
                   baseline_demand_kwh, final_demand_kwh, co2_saved_kg,
                   total_cost, payback_years, plan_json, created_at
            FROM retrofit_plans
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
                record["detail"] = json.loads(record.pop("plan_json"))
            except (TypeError, ValueError):
                record["detail"] = {}
            plans.append(record)
        return plans
    except sqlite3.Error as exc:
        logger.error("Unable to load retrofit plans: %s", exc)
        return []
    finally:
        if conn:
            conn.close()


def delete_retrofit_plan(plan_id):
    """Delete a saved retrofit plan."""
    init_retrofit_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM retrofit_plans WHERE id = ?", (plan_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Unable to delete retrofit plan: %s", exc)
        return False
    finally:
        if conn:
            conn.close()
