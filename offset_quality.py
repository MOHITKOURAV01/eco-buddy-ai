"""Carbon offset quality scoring and the mitigation hierarchy.

The app can tell a user their footprint. The next thing many of them do is buy
offsets - and that is the point at which the product currently goes silent.
It is also the point where the most money gets wasted, because "one tonne" of
carbon credit is not one thing:

* a tonne *avoided* somewhere else is not a tonne *removed* from the air;
* a tonne stored in a forest that can burn down is not a tonne stored in rock;
* a project that would have happened anyway sells credits for nothing at all;
* and a credit priced at a dollar cannot pay for a tonne of real removal.

Investigations into voluntary carbon markets have repeatedly found that a large
share of issued credits do not represent the reductions claimed. A footprint
app that stays quiet about that is not neutral - it is quietly endorsing
whatever the user buys next.

What this module does
---------------------
It scores a credit on the dimensions that decide whether it is real, converts
a nominal tonne into an honest *effective* tonne, and refuses to let offsetting
be presented as a substitute for reducing.

    effective tonnes = nominal
                     x additionality confidence
                     x durability discount
                     x (1 - leakage)
                     x (1 - buffer contribution)
                     x measurement confidence

Every factor is between 0 and 1, so the effective figure can never exceed the
nominal one. That single property is the honesty guarantee of the whole model.

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

# Removal takes carbon out of the air. Avoidance stops it going in. Both can be
# useful; only one of them is what "offsetting an emission" literally means.
REMOVAL = "removal"
AVOIDANCE = "avoidance"

# Project archetypes with the properties that actually decide credit quality.
#
#   additionality      confidence the reduction would not have happened anyway
#   permanence_years   how long the carbon stays put
#   leakage            fraction of the benefit displaced elsewhere
#   measurement        confidence the tonnage is measured rather than modelled
#   co_benefits        social and ecological value beyond the carbon
#   floor_price        below this, the claimed tonne cannot be paid for
PROJECT_TYPES = {
    "Afforestation / reforestation": {
        "kind": REMOVAL, "additionality": 0.75, "permanence_years": 60,
        "leakage": 0.15, "measurement": 0.70, "co_benefits": 0.80,
        "floor_price": 15.0, "typical_price": 30.0,
        "note": "Real removal, but trees are reversible - fire, drought, felling.",
    },
    "Avoided deforestation (REDD+)": {
        "kind": AVOIDANCE, "additionality": 0.35, "permanence_years": 40,
        "leakage": 0.35, "measurement": 0.45, "co_benefits": 0.85,
        "floor_price": 8.0, "typical_price": 12.0,
        "note": "Depends entirely on a counterfactual baseline nobody can observe.",
    },
    "Improved forest management": {
        "kind": AVOIDANCE, "additionality": 0.45, "permanence_years": 45,
        "leakage": 0.30, "measurement": 0.55, "co_benefits": 0.65,
        "floor_price": 10.0, "typical_price": 18.0,
        "note": "Often pays landowners for harvest deferral they had planned anyway.",
    },
    "Clean cookstoves": {
        "kind": AVOIDANCE, "additionality": 0.50, "permanence_years": 100,
        "leakage": 0.20, "measurement": 0.35, "co_benefits": 0.95,
        "floor_price": 6.0, "typical_price": 12.0,
        "note": "Strong health co-benefits; tonnage rests on self-reported usage.",
    },
    "Grid-connected wind or solar": {
        "kind": AVOIDANCE, "additionality": 0.15, "permanence_years": 100,
        "leakage": 0.05, "measurement": 0.90, "co_benefits": 0.55,
        "floor_price": 2.0, "typical_price": 4.0,
        "note": "Cheapest generation on most grids - it was getting built regardless.",
    },
    "Landfill gas capture": {
        "kind": AVOIDANCE, "additionality": 0.55, "permanence_years": 100,
        "leakage": 0.05, "measurement": 0.85, "co_benefits": 0.45,
        "floor_price": 5.0, "typical_price": 10.0,
        "note": "Well measured, though often already required by regulation.",
    },
    "Industrial gas destruction": {
        "kind": AVOIDANCE, "additionality": 0.40, "permanence_years": 100,
        "leakage": 0.02, "measurement": 0.90, "co_benefits": 0.20,
        "floor_price": 3.0, "typical_price": 6.0,
        "note": "Cheap tonnes, but has historically created perverse incentives.",
    },
    "Soil carbon sequestration": {
        "kind": REMOVAL, "additionality": 0.45, "permanence_years": 25,
        "leakage": 0.20, "measurement": 0.35, "co_benefits": 0.75,
        "floor_price": 12.0, "typical_price": 25.0,
        "note": "Hard to measure, easy to reverse with one change of practice.",
    },
    "Biochar": {
        "kind": REMOVAL, "additionality": 0.80, "permanence_years": 500,
        "leakage": 0.05, "measurement": 0.75, "co_benefits": 0.60,
        "floor_price": 90.0, "typical_price": 140.0,
        "note": "Durable, measurable and priced like the real thing it is.",
    },
    "Enhanced rock weathering": {
        "kind": REMOVAL, "additionality": 0.85, "permanence_years": 10000,
        "leakage": 0.05, "measurement": 0.50, "co_benefits": 0.45,
        "floor_price": 100.0, "typical_price": 200.0,
        "note": "Geologically durable; the hard part is verifying how much bound.",
    },
    "Direct air capture with storage": {
        "kind": REMOVAL, "additionality": 0.95, "permanence_years": 10000,
        "leakage": 0.02, "measurement": 0.95, "co_benefits": 0.15,
        "floor_price": 250.0, "typical_price": 600.0,
        "note": "The most verifiable and durable option, and by far the priciest.",
    },
    "Bioenergy with carbon capture": {
        "kind": REMOVAL, "additionality": 0.70, "permanence_years": 10000,
        "leakage": 0.25, "measurement": 0.80, "co_benefits": 0.25,
        "floor_price": 80.0, "typical_price": 160.0,
        "note": "Durable storage, but the biomass supply chain carries the risk.",
    },
}

# Standards and registries, weighted by how much scrutiny a credit has had.
# The weight multiplies additionality and measurement confidence.
REGISTRIES = {
    "Gold Standard": {"weight": 1.00, "note": "Strong methodology review and co-benefit requirements."},
    "Verra (VCS)": {"weight": 0.92, "note": "The largest registry; methodology quality varies by protocol."},
    "Puro.earth": {"weight": 0.95, "note": "Focused on durable removals with physical measurement."},
    "American Carbon Registry": {"weight": 0.88, "note": "Established registry, mostly North American projects."},
    "Climate Action Reserve": {"weight": 0.88, "note": "Established registry with public protocols."},
    "Clean Development Mechanism": {"weight": 0.70, "note": "Older UN mechanism; many legacy credits are weak."},
    "Unverified / self-declared": {"weight": 0.30, "note": "Nobody independent has checked this. Treat with suspicion."},
}

DEFAULT_REGISTRY = "Unverified / self-declared"

# Durability. A tonne released again in 30 years has not been offset, it has
# been rented. The reference is geological storage.
DURABILITY_REFERENCE_YEARS = 1000.0
MIN_DURABILITY_FACTOR = 0.15

# Registries hold back a share of credits to cover reversals. A project with no
# buffer at all is carrying its reversal risk on the buyer's behalf.
TYPICAL_BUFFER_SHARE = 0.15

# Credits older than this have usually been superseded by better methodologies
# and weaker baselines than would be accepted today.
VINTAGE_STALE_AFTER_YEARS = 8
VINTAGE_PENALTY_PER_YEAR = 0.03
MAX_VINTAGE_PENALTY = 0.35

# Grade boundaries for the 0-100 quality score.
GRADE_BANDS = [
    (80, "A", "High-integrity: durable, additional and independently measured."),
    (65, "B", "Solid, with one or two weaknesses worth knowing about."),
    (50, "C", "Mixed. Usable, but do not treat these tonnes as equivalent to reductions."),
    (35, "D", "Weak. The claimed tonnage is unlikely to be fully real."),
    (0, "F", "Do not buy. This is very unlikely to represent a real tonne."),
]

# Scoring weights. They sum to 1.0 and are asserted to in the tests.
SCORE_WEIGHTS = {
    "additionality": 0.30,
    "durability": 0.25,
    "measurement": 0.20,
    "leakage": 0.10,
    "registry": 0.10,
    "co_benefits": 0.05,
}

# The mitigation hierarchy: offsetting is the last step, not the first.
HEALTHY_REDUCTION_SHARE = 0.5


def _clean_positive(value, maximum, default=0.0):
    """Coerce a user-supplied number into a sane, non-negative range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return max(0.0, min(number, maximum))


def _clean_fraction(value, default=0.0):
    """Coerce a value into the 0-1 range."""
    return min(1.0, _clean_positive(value, 1.0, default))


# ---------------------------------------------------------------------------
# Reference data helpers
# ---------------------------------------------------------------------------

def list_project_types(kind=None):
    """Return project archetypes, most durable first, optionally filtered."""
    projects = [
        {"name": name, **details}
        for name, details in PROJECT_TYPES.items()
        if kind is None or details["kind"] == kind
    ]
    return sorted(projects, key=lambda project: project["permanence_years"], reverse=True)


def get_project_type(name):
    """Look up a project archetype. None if unknown."""
    details = PROJECT_TYPES.get(name)
    if not details:
        return None
    return {"name": name, **details}


def list_registries():
    """Return registries, most scrutinised first."""
    return sorted(
        ({"name": name, **info} for name, info in REGISTRIES.items()),
        key=lambda registry: registry["weight"],
        reverse=True,
    )


def registry_weight(registry):
    """Credibility weight for a registry. Unknown names get the lowest."""
    info = REGISTRIES.get(registry)
    if not info:
        return REGISTRIES[DEFAULT_REGISTRY]["weight"]
    return info["weight"]


# ---------------------------------------------------------------------------
# The individual quality factors
# ---------------------------------------------------------------------------

def durability_discount(permanence_years):
    """How much of a tonne survives, given how long it stays stored.

    Storage for 30 years is not equivalent to storage for a thousand. The
    discount is logarithmic rather than linear, because the difference between
    10 and 100 years matters far more than between 5,000 and 10,000.
    """
    import math

    years = _clean_positive(permanence_years, 1_000_000.0)
    if years <= 0:
        return 0.0
    if years >= DURABILITY_REFERENCE_YEARS:
        return 1.0
    factor = math.log10(1 + years) / math.log10(1 + DURABILITY_REFERENCE_YEARS)
    return round(max(MIN_DURABILITY_FACTOR, min(1.0, factor)), 3)


def vintage_penalty(vintage_year, current_year):
    """Discount applied to credits issued long ago under older methodologies."""
    try:
        age = int(current_year) - int(vintage_year)
    except (TypeError, ValueError):
        return 0.0
    if age <= VINTAGE_STALE_AFTER_YEARS:
        return 0.0
    penalty = (age - VINTAGE_STALE_AFTER_YEARS) * VINTAGE_PENALTY_PER_YEAR
    return round(min(MAX_VINTAGE_PENALTY, penalty), 3)


def price_credibility(price_per_tonne, project_type):
    """Whether the price could plausibly pay for the tonne being claimed.

    A removal that costs 200 to perform cannot be sold for 3 and still have
    happened. Cheapness is the most visible warning sign in this market.
    """
    project = get_project_type(project_type)
    if not project:
        return {"credible": False, "ratio": 0.0, "floor_price": None}

    price = _clean_positive(price_per_tonne, 100000.0)
    floor = project["floor_price"]
    ratio = price / floor if floor > 0 else 1.0

    return {
        "credible": price >= floor,
        "ratio": round(ratio, 2),
        "floor_price": floor,
        "typical_price": project["typical_price"],
        "suspiciously_cheap": price < floor * 0.5,
    }


def default_buffer_share(project_type):
    """Buffer pool a registry would sensibly hold for this project type.

    Reversible storage needs insurance against fire, felling or a change of
    farming practice. Carbon mineralised in rock or injected underground does
    not, so charging it a buffer would penalise the most durable option for a
    risk it does not carry.
    """
    project = get_project_type(project_type)
    if not project:
        return TYPICAL_BUFFER_SHARE
    if project["permanence_years"] >= DURABILITY_REFERENCE_YEARS:
        return 0.0
    return TYPICAL_BUFFER_SHARE


def quality_score(project_type, registry=DEFAULT_REGISTRY, vintage_year=None,
                  current_year=None, buffer_share=None):
    """Score a credit 0-100 across the dimensions that decide integrity."""
    project = get_project_type(project_type)
    if not project:
        return None

    weight = registry_weight(registry)
    if buffer_share is None:
        buffer_share = default_buffer_share(project_type)
    buffer_pool = _clean_fraction(buffer_share, TYPICAL_BUFFER_SHARE)

    additionality = _clean_fraction(project["additionality"]) * weight
    measurement = _clean_fraction(project["measurement"]) * weight
    durability = durability_discount(project["permanence_years"])
    leakage_score = 1.0 - _clean_fraction(project["leakage"])
    co_benefits = _clean_fraction(project["co_benefits"])

    # A buffer pool is a good sign for reversible projects: it means someone is
    # holding stock back to cover failures.
    if project["permanence_years"] < DURABILITY_REFERENCE_YEARS:
        durability = min(1.0, durability + buffer_pool * 0.3)

    components = {
        "additionality": additionality,
        "durability": durability,
        "measurement": measurement,
        "leakage": leakage_score,
        "registry": weight,
        "co_benefits": co_benefits,
    }

    raw = sum(components[key] * SCORE_WEIGHTS[key] for key in SCORE_WEIGHTS)

    penalty = 0.0
    if vintage_year is not None and current_year is not None:
        penalty = vintage_penalty(vintage_year, current_year)

    score = max(0.0, min(100.0, raw * 100 * (1 - penalty)))

    return {
        "score": round(score, 1),
        "components": {key: round(value, 3) for key, value in components.items()},
        "vintage_penalty": penalty,
        "kind": project["kind"],
    }


def grade_for_score(score):
    """Letter grade and plain-language verdict for a 0-100 score."""
    value = max(0.0, min(100.0, float(score or 0.0)))
    for threshold, letter, verdict in GRADE_BANDS:
        if value >= threshold:
            return {"grade": letter, "verdict": verdict}
    return {"grade": "F", "verdict": GRADE_BANDS[-1][2]}


def effective_tonnes(nominal_tonnes, project_type, registry=DEFAULT_REGISTRY,
                     buffer_share=None, vintage_year=None,
                     current_year=None):
    """Convert nominal tonnes into the tonnes that are plausibly real.

    Every factor is bounded to 0-1, so this can never exceed the nominal
    figure. That is the guarantee the whole module rests on.
    """
    project = get_project_type(project_type)
    if not project:
        return None

    nominal = _clean_positive(nominal_tonnes, 1_000_000.0)
    weight = registry_weight(registry)
    if buffer_share is None:
        buffer_share = default_buffer_share(project_type)
    buffer_pool = _clean_fraction(buffer_share, TYPICAL_BUFFER_SHARE)

    additionality = _clean_fraction(project["additionality"]) * weight
    measurement = _clean_fraction(project["measurement"]) * weight
    durability = durability_discount(project["permanence_years"])
    leakage_retained = 1.0 - _clean_fraction(project["leakage"])
    penalty = 1.0 - (
        vintage_penalty(vintage_year, current_year)
        if vintage_year is not None and current_year is not None
        else 0.0
    )

    # The buffer share is held back by the registry, so the buyer does not get
    # to count it - it is insurance, not tonnage.
    delivered = 1.0 - buffer_pool

    effective = (
        nominal * additionality * durability * measurement
        * leakage_retained * delivered * penalty
    )

    return {
        "nominal_tonnes": round(nominal, 3),
        "effective_tonnes": round(effective, 3),
        "shortfall_tonnes": round(nominal - effective, 3),
        "delivery_ratio": round(effective / nominal, 3) if nominal > 0 else 0.0,
        "factors": {
            "additionality": round(additionality, 3),
            "durability": round(durability, 3),
            "measurement": round(measurement, 3),
            "leakage_retained": round(leakage_retained, 3),
            "buffer_delivered": round(delivered, 3),
            "vintage": round(penalty, 3),
        },
    }


def assess_credit(project_type, tonnes, price_per_tonne, registry=DEFAULT_REGISTRY,
                  vintage_year=None, current_year=None, buffer_share=None):
    """Full assessment of one offer: score, grade, real tonnes and warnings."""
    project = get_project_type(project_type)
    if not project:
        return None

    scoring = quality_score(
        project_type, registry, vintage_year, current_year, buffer_share
    )
    delivery = effective_tonnes(
        tonnes, project_type, registry, buffer_share, vintage_year, current_year
    )
    pricing = price_credibility(price_per_tonne, project_type)
    grade = grade_for_score(scoring["score"])

    spend = round(_clean_positive(price_per_tonne, 100000.0) * delivery["nominal_tonnes"], 2)
    real_cost = (
        round(spend / delivery["effective_tonnes"], 2)
        if delivery["effective_tonnes"] > 0
        else None
    )

    assessment = {
        "project_type": project_type,
        "kind": project["kind"],
        "registry": registry,
        "note": project["note"],
        "score": scoring["score"],
        "grade": grade["grade"],
        "verdict": grade["verdict"],
        "components": scoring["components"],
        "vintage_penalty": scoring["vintage_penalty"],
        "nominal_tonnes": delivery["nominal_tonnes"],
        "effective_tonnes": delivery["effective_tonnes"],
        "delivery_ratio": delivery["delivery_ratio"],
        "factors": delivery["factors"],
        "price_per_tonne": round(_clean_positive(price_per_tonne, 100000.0), 2),
        "pricing": pricing,
        "total_spend": spend,
        "cost_per_effective_tonne": real_cost,
        "permanence_years": project["permanence_years"],
    }
    assessment["warnings"] = get_credit_warnings(assessment)
    return assessment


def get_credit_warnings(assessment):
    """Concrete warnings for one credit, worst first."""
    if not assessment:
        return []

    warnings = []

    if assessment["registry"] == DEFAULT_REGISTRY:
        warnings.append(
            "Nobody independent has verified this credit. Unverified offsets are "
            "the single most common way money is wasted in this market."
        )

    pricing = assessment.get("pricing", {})
    if pricing.get("suspiciously_cheap"):
        warnings.append(
            f"At {assessment['price_per_tonne']:.2f} per tonne this is far below the "
            f"{pricing['floor_price']:.0f} it plausibly costs to deliver. Cheap tonnes "
            f"are usually cheap because they are not real."
        )
    elif not pricing.get("credible", True):
        warnings.append(
            f"Priced below the {pricing['floor_price']:.0f} per tonne floor for this "
            f"project type - check what is actually being sold."
        )

    if assessment["kind"] == AVOIDANCE:
        warnings.append(
            "This is an avoidance credit, not a removal. It claims something did "
            "not happen elsewhere; it does not take your emission back out of the air."
        )

    if assessment["permanence_years"] < 100:
        warnings.append(
            f"Storage is credited for about {assessment['permanence_years']} years. "
            f"Carbon released again later has been rented, not offset."
        )

    if assessment["components"].get("additionality", 1.0) < 0.4:
        warnings.append(
            "Weak additionality: this kind of project often proceeds regardless of "
            "credit revenue, in which case the credit bought nothing."
        )

    if assessment["components"].get("measurement", 1.0) < 0.4:
        warnings.append(
            "The tonnage here is modelled or self-reported rather than measured."
        )

    if assessment["vintage_penalty"] > 0:
        warnings.append(
            "This is an old vintage, issued under methodologies that would not be "
            "accepted today."
        )

    if assessment["delivery_ratio"] < 0.5:
        warnings.append(
            f"Only about {assessment['delivery_ratio'] * 100:.0f}% of the tonnes you "
            f"pay for here are likely to be real."
        )

    return warnings


# ---------------------------------------------------------------------------
# Portfolios and the mitigation hierarchy
# ---------------------------------------------------------------------------

def portfolio_summary(assessments):
    """Aggregate a set of assessed credits into one honest picture."""
    credits = [item for item in (assessments or []) if item]
    if not credits:
        return {
            "credit_count": 0,
            "nominal_tonnes": 0.0,
            "effective_tonnes": 0.0,
            "shortfall_tonnes": 0.0,
            "total_spend": 0.0,
            "weighted_score": 0.0,
            "grade": "F",
            "removal_share_pct": 0.0,
            "durable_share_pct": 0.0,
            "worst_credit": None,
        }

    nominal = sum(item["nominal_tonnes"] for item in credits)
    effective = sum(item["effective_tonnes"] for item in credits)
    spend = sum(item["total_spend"] for item in credits)

    weighted_score = (
        sum(item["score"] * item["nominal_tonnes"] for item in credits) / nominal
        if nominal > 0
        else sum(item["score"] for item in credits) / len(credits)
    )
    removal_tonnes = sum(
        item["nominal_tonnes"] for item in credits if item["kind"] == REMOVAL
    )
    durable_tonnes = sum(
        item["nominal_tonnes"] for item in credits if item["permanence_years"] >= 100
    )
    worst = min(credits, key=lambda item: item["score"])

    return {
        "credit_count": len(credits),
        "nominal_tonnes": round(nominal, 3),
        "effective_tonnes": round(effective, 3),
        "shortfall_tonnes": round(nominal - effective, 3),
        "delivery_ratio": round(effective / nominal, 3) if nominal > 0 else 0.0,
        "total_spend": round(spend, 2),
        "cost_per_effective_tonne": round(spend / effective, 2) if effective > 0 else None,
        "weighted_score": round(weighted_score, 1),
        "grade": grade_for_score(weighted_score)["grade"],
        "removal_share_pct": round(removal_tonnes / nominal * 100, 1) if nominal > 0 else 0.0,
        "durable_share_pct": round(durable_tonnes / nominal * 100, 1) if nominal > 0 else 0.0,
        "worst_credit": worst,
    }


def mitigation_hierarchy(footprint_kg, reduced_kg, offset_tonnes):
    """Check that offsetting is not standing in for reducing.

    Offsetting is the last step in the hierarchy - avoid, reduce, substitute,
    then offset the remainder. A user offsetting everything and reducing
    nothing should be told so plainly.
    """
    footprint = _clean_positive(footprint_kg, 1_000_000.0)
    reduced = min(footprint, _clean_positive(reduced_kg, 1_000_000.0))
    offset_kg = _clean_positive(offset_tonnes, 100_000.0) * 1000.0

    residual = max(0.0, footprint - reduced)
    action_total = reduced + offset_kg
    reduction_share = reduced / action_total * 100 if action_total > 0 else 0.0

    if action_total <= 0:
        status = "NOTHING_YET"
        message = "No reductions and no offsets recorded yet."
    elif reduction_share >= HEALTHY_REDUCTION_SHARE * 100:
        status = "REDUCTION_LED"
        message = (
            f"Most of your action is real reduction ({reduction_share:.0f}%), with "
            f"offsets covering what is left. That is the right way round."
        )
    elif reduced <= 0:
        status = "OFFSET_ONLY"
        message = (
            "You are offsetting without reducing anything. Offsets are the last "
            "step in the hierarchy, not a substitute for the first four."
        )
    else:
        status = "OFFSET_HEAVY"
        message = (
            f"Only {reduction_share:.0f}% of your action is actual reduction. "
            f"Cutting the emission is cheaper and more certain than buying a tonne back."
        )

    return {
        "footprint_kg": round(footprint, 1),
        "reduced_kg": round(reduced, 1),
        "offset_kg": round(offset_kg, 1),
        "residual_kg": round(residual, 1),
        "residual_covered_pct": (
            round(min(100.0, offset_kg / residual * 100), 1) if residual > 0 else 100.0
        ),
        "reduction_share_pct": round(reduction_share, 1),
        "over_offset": offset_kg > residual and residual >= 0,
        "status": status,
        "message": message,
    }


def recommend_portfolio(budget, target_tonnes, removal_preference=0.5, current_year=None):
    """Suggest a mix of project types for a budget and a tonnage target.

    Removals are durable but expensive; a budget spent entirely on them buys
    few tonnes. The recommendation blends the two and is explicit about the
    trade-off rather than hiding it behind a single number.
    """
    money = _clean_positive(budget, 10_000_000.0)
    target = _clean_positive(target_tonnes, 100_000.0)
    preference = _clean_fraction(removal_preference, 0.5)

    if money <= 0 or target <= 0:
        return {"allocations": [], "affordable": False, "note": "Set a budget and a target."}

    # Pick the best-scoring project of each kind at a sensible price.
    def best_of(kind):
        candidates = []
        for name, details in PROJECT_TYPES.items():
            if details["kind"] != kind:
                continue
            scoring = quality_score(name, "Gold Standard", current_year=current_year)
            candidates.append((scoring["score"], name, details))
        candidates.sort(reverse=True)
        return candidates[0] if candidates else None

    best_removal = best_of(REMOVAL)
    best_avoidance = best_of(AVOIDANCE)

    allocations = []
    for share, choice in ((preference, best_removal), (1 - preference, best_avoidance)):
        if not choice or share <= 0:
            continue
        _, name, details = choice
        spend = money * share
        price = details["typical_price"]
        tonnes = spend / price if price > 0 else 0.0
        assessment = assess_credit(
            name, tonnes, price, "Gold Standard", current_year=current_year
        )
        allocations.append(
            {
                "project_type": name,
                "kind": details["kind"],
                "share_pct": round(share * 100, 1),
                "spend": round(spend, 2),
                "price_per_tonne": price,
                "nominal_tonnes": round(tonnes, 2),
                "effective_tonnes": assessment["effective_tonnes"],
                "grade": assessment["grade"],
            }
        )

    effective_total = sum(item["effective_tonnes"] for item in allocations)

    return {
        "budget": round(money, 2),
        "target_tonnes": round(target, 2),
        "allocations": allocations,
        "effective_tonnes": round(effective_total, 2),
        "affordable": effective_total >= target,
        "shortfall_tonnes": round(max(0.0, target - effective_total), 2),
        "note": (
            "This mix covers the target on an effective-tonne basis."
            if effective_total >= target
            else "This budget does not cover the target once quality discounts are "
                 "applied. Reducing the emission is the cheaper half of the answer."
        ),
    }


def get_offset_advice(summary, hierarchy=None, limit=6):
    """Advice ranked by what this particular portfolio looks like."""
    if not summary or not summary.get("credit_count"):
        return ["Assess a credit above to see whether it is worth buying."]

    advice = []

    if hierarchy and hierarchy["status"] in ("OFFSET_ONLY", "OFFSET_HEAVY"):
        advice.append(hierarchy["message"])

    if summary["delivery_ratio"] < 0.6:
        advice.append(
            f"You are paying for {summary['nominal_tonnes']:.1f} tonnes but likely "
            f"getting {summary['effective_tonnes']:.1f}. The real price is "
            f"{summary['cost_per_effective_tonne']:.0f} per tonne, not "
            f"{summary['total_spend'] / summary['nominal_tonnes']:.0f}."
        )

    if summary["removal_share_pct"] < 30:
        advice.append(
            f"Only {summary['removal_share_pct']}% of this portfolio is actual removal. "
            f"Avoidance credits claim something did not happen elsewhere - they do not "
            f"take your emission back out of the air."
        )

    if summary["durable_share_pct"] < 50:
        advice.append(
            "Over half of these tonnes are stored for less than a century. Carbon that "
            "comes back later was rented, not offset."
        )

    worst = summary.get("worst_credit")
    if worst and worst["score"] < 50:
        advice.append(
            f"Your weakest holding is {worst['project_type']} ({worst['grade']}, "
            f"{worst['score']}/100). Dropping it would raise the portfolio's integrity most."
        )

    advice.append(
        "Buy fewer, better tonnes. A high-integrity removal at 150 does more than "
        "ten unverified credits at 5."
    )
    advice.append(
        "Offset last. Every tonne you never emit is certain; every tonne you buy back "
        "is a claim about a counterfactual."
    )

    return advice[: max(0, int(limit))]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _get_conn():
    return sqlite3.connect(DB_NAME)


def init_offset_db():
    """Create the offset holdings table if it does not exist yet."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS offset_holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                project_type TEXT NOT NULL,
                registry TEXT NOT NULL,
                kind TEXT NOT NULL,
                nominal_tonnes REAL NOT NULL,
                effective_tonnes REAL NOT NULL,
                price_per_tonne REAL NOT NULL,
                total_spend REAL NOT NULL,
                score REAL NOT NULL,
                grade TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.error("Offset quality init error: %s", exc)
        return False
    finally:
        if conn:
            conn.close()


def save_holding(user_id, label, assessment):
    """Persist an assessed credit. Returns the new row id or None."""
    init_offset_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute(
            """
            INSERT INTO offset_holdings (
                user_id, label, project_type, registry, kind, nominal_tonnes,
                effective_tonnes, price_per_tonne, total_spend, score, grade, detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                (label or "Offset purchase").strip() or "Offset purchase",
                assessment.get("project_type", ""),
                assessment.get("registry", DEFAULT_REGISTRY),
                assessment.get("kind", AVOIDANCE),
                assessment.get("nominal_tonnes", 0.0),
                assessment.get("effective_tonnes", 0.0),
                assessment.get("price_per_tonne", 0.0),
                assessment.get("total_spend", 0.0),
                assessment.get("score", 0.0),
                assessment.get("grade", "F"),
                json.dumps(
                    {
                        "components": assessment.get("components", {}),
                        "factors": assessment.get("factors", {}),
                        "warnings": assessment.get("warnings", []),
                        "permanence_years": assessment.get("permanence_years", 0),
                    }
                ),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("Unable to save offset holding: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


def get_holdings(user_id, limit=50):
    """Return a user's saved holdings, newest first."""
    init_offset_db()
    conn = None
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, label, project_type, registry, kind, nominal_tonnes,
                   effective_tonnes, price_per_tonne, total_spend, score, grade,
                   detail_json, created_at
            FROM offset_holdings
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

        holdings = []
        for row in rows:
            record = dict(row)
            try:
                detail = json.loads(record.pop("detail_json"))
            except (TypeError, ValueError):
                detail = {}
            record["components"] = detail.get("components", {})
            record["factors"] = detail.get("factors", {})
            record["warnings"] = detail.get("warnings", [])
            record["permanence_years"] = detail.get("permanence_years", 0)
            holdings.append(record)
        return holdings
    except sqlite3.Error as exc:
        logger.error("Unable to load offset holdings: %s", exc)
        return []
    finally:
        if conn:
            conn.close()


def delete_holding(holding_id):
    """Delete a saved holding."""
    init_offset_db()
    conn = None
    try:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM offset_holdings WHERE id = ?", (holding_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Unable to delete offset holding: %s", exc)
        return False
    finally:
        if conn:
            conn.close()
