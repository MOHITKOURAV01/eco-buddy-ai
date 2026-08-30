"""Composite indices, and the three choices this app makes without making them.

`src/carbon/confidence_scoring.py` opens with seven numbers that sum to 100, a
linear sum, and three hard bands at 85 and 60. `src/data/data_quality.py` does
the same with `calculate_confidence_score()` and `quality_grade()`.
`src/calculators/eco_score.py` adds three category scores in different units and
calls the total an eco-score.

Every one of those is a composite indicator, and composite indicators have a
literature, a set of well-known failure modes and a standard set of checks. None
of the checks appear anywhere in this repo. The weights are not wrong; the point
is that nobody can say whether they are wrong, because nothing tests what
happens if they are different.

The three choices
-----------------
**Normalisation.** Adding numbers on different scales lets the units do the
weighting. In `eco_score.py` a diet factor runs 1.5-7.0, a transport factor
0.1-8.0, and ``energy_kwh * 0.5`` runs to the hundreds. Energy has an effective
weight near 1.0 and the other two round to zero, and the three
``categories_processed`` in the output imply a balance the arithmetic does not
have. Choosing "none" is still choosing.

**Aggregation.** A linear sum is fully compensatory: a perfect score on one
component buys back a zero on another. That may be the intended policy. It is
not a policy anyone chose — it is what addition does — and for a *confidence*
score it is close to indefensible, because an assessment with unknown emission
factors is not rescued by having every field filled in.

**Weighting.** In a linear sum the influence of a component depends on its
variance as much as its nominal weight. A component with weight 20 that barely
varies contributes almost nothing to the ranking; one with weight 5 and a wide
spread can dominate it. The dictionary says what was intended. Nothing reports
what happened.

Effective weights
-----------------
The standard diagnostic is the Pearson correlation ratio::

    S_i = Var(E[Y | X_i]) / Var(Y)

estimated by binning the component and taking the variance of the conditional
means. On a correlated component set the effective weights and the nominal ones
are routinely far apart, and the gap is the finding.

Refusals
--------
No composite from components on incompatible scales without an explicit
normalisation choice, because the silent version of that choice is the one
`eco_score.py` already makes.

No index where a single component explains almost all the variance. That is not
an index, it is that component with decoration, and it should be reported as
such rather than presented as a balanced score.

No band label where the band probability under a defensible reweighting is
below a stated threshold. 84.9 being "Medium" and 85.1 being "High" is a
categorical difference resting on nothing.

Self-contained by design: nothing here imports the modules it describes, so the
existing scores keep working untouched.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
from typing import Any, Mapping, Sequence

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

ENGINE_VERSION = "1.0.0"

# Requirements --------------------------------------------------------------
MIN_UNITS = 5
MIN_COMPONENTS = 2
MAX_UNITS = 20000
MAX_COMPONENTS = 40

# Diagnostics ---------------------------------------------------------------
# Above this, one component is the index and the rest are decoration.
SINGLE_COMPONENT_CEILING = 0.90
# Above this, two components are measuring the same construct twice.
REDUNDANCY_THRESHOLD = 0.90
# A band label needs at least this much of the reweighting mass behind it.
MIN_BAND_PROBABILITY = 0.60

# Sensitivity ---------------------------------------------------------------
DEFAULT_DRAWS = 500
# Concentration of the Dirichlet the weights are drawn from. Higher means the
# draws stay closer to the stated weights; 50 is a deliberately *generous*
# choice, because a sensitivity analysis that only considers weights nobody
# would defend proves nothing.
DEFAULT_CONCENTRATION = 50.0

# Aggregation ---------------------------------------------------------------
AGGREGATIONS = ("linear", "geometric", "non_compensatory")
NORMALISATIONS = ("minmax", "zscore", "rank", "distance_to_reference")
POLARITIES = ("higher_is_better", "lower_is_better")

# Geometric aggregation needs strictly positive inputs; a normalised zero is
# floored to this so the index is defined, and the floor is small enough that a
# zero still collapses the score, which is the property that makes geometric
# aggregation non-compensatory in the first place.
GEOMETRIC_FLOOR = 1e-4

# The weights currently in src/carbon/confidence_scoring.py. Copied rather than
# imported: this module must not make the scores it diagnoses depend on it.
APP_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "input_completeness": 20.0,
    "estimated_vs_measured": 15.0,
    "data_age": 15.0,
    "factor_provenance": 20.0,
    "unit_conversion": 10.0,
    "category_coverage": 15.0,
    "validation_warnings": 5.0,
}

# The bands in the same module, highest first.
APP_CONFIDENCE_BANDS: tuple[tuple[float, str], ...] = (
    (85.0, "High"),
    (60.0, "Medium"),
    (0.0, "Low"),
)


class CompositeError(ValueError):
    """Raised when an index cannot be built from what was supplied."""


# ---------------------------------------------------------------------------
# Small numerics
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pvariance(values)


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation. Zero where either series is constant."""
    if len(xs) != len(ys):
        raise CompositeError("Series must be the same length.")
    if len(xs) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    sxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(xs)))
    return sxy / math.sqrt(sxx * syy)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def build_component(
    name: str,
    values: Sequence[float],
    weight: float = 1.0,
    polarity: str = "higher_is_better",
) -> dict[str, Any]:
    """One column of the index: a value per unit, a nominal weight, a direction.

    `polarity` exists because half the quantities in this app are better when
    larger and half when smaller, and a linear sum has no way of knowing which.
    Getting it wrong does not produce an error; it produces a ranking in the
    wrong order with no other symptom.
    """
    if polarity not in POLARITIES:
        raise CompositeError(
            "Polarity must be one of %s." % ", ".join(POLARITIES)
        )
    weight_value = _finite(weight)
    if weight_value is None or weight_value < 0:
        raise CompositeError("Component '%s' needs a non-negative weight." % name)

    cleaned: list[float] = []
    for value in values:
        number = _finite(value)
        if number is None:
            raise CompositeError("Component '%s' has a non-numeric value." % name)
        cleaned.append(number)

    if len(cleaned) < MIN_UNITS:
        raise CompositeError(
            "Component '%s' has %d values; at least %d units are needed before "
            "a ranking means anything." % (name, len(cleaned), MIN_UNITS)
        )

    spread = max(cleaned) - min(cleaned)
    return {
        "name": str(name),
        "values": cleaned,
        "weight": weight_value,
        "polarity": polarity,
        "n": len(cleaned),
        "min": min(cleaned),
        "max": max(cleaned),
        "mean": statistics.fmean(cleaned),
        "sd": math.sqrt(_variance(cleaned)),
        "range": spread,
        "constant": spread == 0,
    }


def _validate_components(
    components: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(components) < MIN_COMPONENTS:
        raise CompositeError(
            "An index needs at least %d components." % MIN_COMPONENTS
        )
    if len(components) > MAX_COMPONENTS:
        raise CompositeError("At most %d components are supported." % MAX_COMPONENTS)

    cleaned = []
    names: set[str] = set()
    length = None
    for component in components:
        if not isinstance(component, Mapping) or "values" not in component:
            raise CompositeError("Components must be built with build_component().")
        if component["name"] in names:
            raise CompositeError("Component '%s' appears twice." % component["name"])
        names.add(component["name"])
        if length is None:
            length = component["n"]
        elif component["n"] != length:
            raise CompositeError(
                "Component '%s' has %d values but the first has %d. Every "
                "component must cover the same units."
                % (component["name"], component["n"], length)
            )
        cleaned.append(dict(component))

    if length is not None and length > MAX_UNITS:
        raise CompositeError("At most %d units are supported." % MAX_UNITS)

    if all(component["weight"] == 0 for component in cleaned):
        raise CompositeError("Every weight is zero; there is no index to build.")
    return cleaned


def scale_mismatch(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Are these numbers on scales that can be added?

    This is the `eco_score.py` problem stated as a diagnostic. Adding a
    quantity that ranges over 6 to one that ranges over 800 gives the second an
    effective weight of roughly 1.0 whatever the nominal weights say.
    """
    cleaned = _validate_components(components)
    ranges = {
        component["name"]: component["range"]
        for component in cleaned
        if not component["constant"]
    }
    if not ranges:
        raise CompositeError("Every component is constant; nothing can be ranked.")

    largest = max(ranges.values())
    smallest = min(ranges.values())
    ratio = largest / smallest if smallest > 0 else float("inf")
    dominant = max(ranges, key=lambda name: ranges[name])

    return {
        "ranges": ranges,
        "ratio": ratio,
        "dominant": dominant,
        "mismatched": ratio > 10.0,
        "headline": (
            "The widest component ('%s') spans %.4gx the narrowest. Added "
            "unnormalised, it is very nearly the whole index regardless of the "
            "nominal weights." % (dominant, ratio)
            if ratio > 10.0
            else "Component ranges differ by a factor of %.2f — close enough "
            "that normalisation changes the ordering only slightly." % ratio
        ),
    }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalise(
    component: Mapping[str, Any],
    method: str = "minmax",
    reference: float | None = None,
) -> list[float]:
    """Put a component on a 0-1 scale, by a method the caller has to name.

    Each choice has a consequence and they are not interchangeable:

    ``minmax``    keeps the shape and is at the mercy of a single outlier, which
                  compresses everyone else into the bottom of the range.
    ``zscore``    assumes the spread is meaningful and is unbounded, so an
                  extreme unit can still dominate an aggregate.
    ``rank``      is outlier-proof and discards magnitude entirely; a unit twice
                  as good as the next is one position better.
    ``distance_to_reference``
                  measures against a target rather than against the other
                  units, which is the only one of the four whose values do not
                  change when a new unit joins.
    """
    if method not in NORMALISATIONS:
        raise CompositeError(
            "Normalisation must be one of %s." % ", ".join(NORMALISATIONS)
        )

    values = list(component["values"])
    reverse = component["polarity"] == "lower_is_better"

    if method == "minmax":
        low, high = min(values), max(values)
        if high == low:
            scaled = [0.5] * len(values)
        else:
            scaled = [(value - low) / (high - low) for value in values]

    elif method == "zscore":
        mean = statistics.fmean(values)
        sd = math.sqrt(_variance(values))
        if sd == 0:
            scaled = [0.5] * len(values)
        else:
            # Mapped onto (0, 1) through a logistic so the aggregation below is
            # defined; the ordering is untouched, which is what a z-score is
            # for. Clipping at +/- 3 SD instead would discard information from
            # exactly the units the index is most often asked about.
            scaled = [
                1.0 / (1.0 + math.exp(-(value - mean) / sd)) for value in values
            ]

    elif method == "rank":
        order = sorted(range(len(values)), key=lambda index: values[index])
        ranks = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2.0
            for index in range(position, end + 1):
                ranks[order[index]] = average
            position = end + 1
        divisor = len(values) - 1
        scaled = (
            [rank / divisor for rank in ranks] if divisor > 0 else [0.5] * len(values)
        )

    else:  # distance_to_reference
        if reference is None:
            raise CompositeError(
                "distance_to_reference needs a reference value. Without one it "
                "is min-max with extra steps."
            )
        target = _finite(reference)
        if target is None or target == 0:
            raise CompositeError("The reference must be a non-zero number.")
        ratios = [value / target for value in values]
        low, high = min(ratios), max(ratios)
        if high == low:
            scaled = [0.5] * len(values)
        else:
            scaled = [(ratio - low) / (high - low) for ratio in ratios]

    return [1.0 - value for value in scaled] if reverse else scaled


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(
    normalised: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    method: str = "linear",
) -> list[float]:
    """Combine normalised components, by a rule that encodes a value judgement.

    ``linear``           fully compensatory. A perfect score anywhere buys back
                         a zero anywhere else.
    ``geometric``        partially compensatory. A zero on any component
                         collapses the index, which is usually what a
                         *confidence* score should do.
    ``non_compensatory`` no trade-off at all. Pairwise outranking on the share
                         of weight favouring each unit, scored by net flow, so
                         a large advantage on one component cannot outvote a
                         small disadvantage on many.
    """
    if method not in AGGREGATIONS:
        raise CompositeError(
            "Aggregation must be one of %s." % ", ".join(AGGREGATIONS)
        )
    names = list(normalised)
    if not names:
        raise CompositeError("Nothing to aggregate.")

    total_weight = sum(weights[name] for name in names)
    if total_weight <= 0:
        raise CompositeError("Weights must sum to something positive.")
    share = {name: weights[name] / total_weight for name in names}
    units = len(normalised[names[0]])

    if method == "linear":
        return [
            sum(share[name] * normalised[name][index] for name in names)
            for index in range(units)
        ]

    if method == "geometric":
        scores = []
        for index in range(units):
            total = 0.0
            for name in names:
                value = max(normalised[name][index], GEOMETRIC_FLOOR)
                total += share[name] * math.log(value)
            scores.append(math.exp(total))
        return scores

    # non_compensatory: pairwise outranking, scored by net flow.
    scores = []
    for index in range(units):
        outranks = 0
        outranked = 0
        for other in range(units):
            if other == index:
                continue
            concordance = sum(
                share[name]
                for name in names
                if normalised[name][index] >= normalised[name][other]
            )
            if concordance > 0.5:
                outranks += 1
            elif concordance < 0.5:
                outranked += 1
        scores.append((outranks - outranked) / (units - 1) if units > 1 else 0.0)
    return scores


# ---------------------------------------------------------------------------
# Building an index
# ---------------------------------------------------------------------------


def build_index(
    unit_names: Sequence[str],
    components: Sequence[Mapping[str, Any]],
    normalisation: str = "minmax",
    aggregation: str = "linear",
    reference: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Normalise, weight, aggregate, and report what actually drove the result."""
    cleaned = _validate_components(components)
    units = cleaned[0]["n"]
    if len(unit_names) != units:
        raise CompositeError(
            "%d unit names for %d units." % (len(unit_names), units)
        )

    references = dict(reference or {})
    normalised = {
        component["name"]: normalise(
            component,
            method=normalisation,
            reference=references.get(component["name"]),
        )
        for component in cleaned
    }
    weights = {component["name"]: component["weight"] for component in cleaned}
    scores = aggregate(normalised, weights, method=aggregation)

    order = sorted(range(units), key=lambda index: scores[index], reverse=True)
    ranks = [0] * units
    for position, index in enumerate(order):
        ranks[index] = position + 1

    effective = effective_weights(normalised, scores, weights)
    concentration = max(effective["effective"].values()) if effective["effective"] else 0.0

    return {
        "engine_version": ENGINE_VERSION,
        "units": [str(name) for name in unit_names],
        "components": [component["name"] for component in cleaned],
        "normalisation": normalisation,
        "aggregation": aggregation,
        "weights": weights,
        "normalised": normalised,
        "scores": scores,
        "ranks": ranks,
        "order": [str(unit_names[index]) for index in order],
        "effective_weights": effective,
        "dominated_by_one": concentration > SINGLE_COMPONENT_CEILING,
        "headline": (
            "'%s' accounts for %.0f%% of the variation in this index. That is "
            "not a composite; it is that component with decoration."
            % (
                max(effective["effective"], key=lambda k: effective["effective"][k]),
                concentration * 100.0,
            )
            if concentration > SINGLE_COMPONENT_CEILING
            else "%d units on %d components, %s normalisation, %s aggregation."
            % (units, len(cleaned), normalisation, aggregation)
        ),
    }


# ---------------------------------------------------------------------------
# Effective weights
# ---------------------------------------------------------------------------


def effective_weights(
    normalised: Mapping[str, Sequence[float]],
    scores: Sequence[float],
    weights: Mapping[str, float],
    bins: int = 8,
) -> dict[str, Any]:
    """What each component actually contributed, against what it was assigned.

    Pearson correlation ratio, ``Var(E[Y | X_i]) / Var(Y)``, estimated by
    sorting on the component and taking the variance of the bin means. It is
    the standard first-order sensitivity measure and it needs no assumption
    that the aggregation is linear, which matters because two of the three
    aggregations here are not.

    Shares are normalised to sum to one so they can be read against the nominal
    weights. On a correlated component set they routinely will not agree, and
    the disagreement is the point of computing them.
    """
    names = list(normalised)
    if not names:
        raise CompositeError("Nothing to decompose.")
    total_variance = _variance(list(scores))

    raw: dict[str, float] = {}
    for name in names:
        if total_variance <= 0:
            raw[name] = 0.0
            continue
        values = list(normalised[name])
        order = sorted(range(len(values)), key=lambda index: values[index])
        bucket_count = max(2, min(int(bins), len(values) // 2))
        size = len(order) / bucket_count

        # Bin boundaries are extended to keep tied values together. Splitting a
        # tie across two bins makes the conditional means differ for reasons
        # that have nothing to do with the component, and a *constant*
        # component — every value tied — would otherwise be credited with
        # explaining variance it cannot possibly explain.
        means = []
        counts = []
        start = 0
        for bucket in range(bucket_count):
            if start >= len(order):
                break
            end = int(round((bucket + 1) * size))
            end = max(end, start + 1)
            while end < len(order) and values[order[end]] == values[order[end - 1]]:
                end += 1
            members = order[start:end]
            means.append(statistics.fmean([scores[index] for index in members]))
            counts.append(len(members))
            start = end

        if len(means) < 2:
            # One bin means the component takes a single value across every
            # unit it could distinguish. It explains none of the spread.
            raw[name] = 0.0
            continue
        grand = sum(means[i] * counts[i] for i in range(len(means))) / sum(counts)
        between = sum(
            counts[i] * (means[i] - grand) ** 2 for i in range(len(means))
        ) / sum(counts)
        raw[name] = between / total_variance

    total_raw = sum(raw.values())
    effective = (
        {name: value / total_raw for name, value in raw.items()}
        if total_raw > 0
        else {name: 1.0 / len(names) for name in names}
    )

    total_nominal = sum(weights[name] for name in names)
    nominal = (
        {name: weights[name] / total_nominal for name in names}
        if total_nominal > 0
        else {name: 1.0 / len(names) for name in names}
    )

    gaps = {name: effective[name] - nominal[name] for name in names}
    widest = max(gaps, key=lambda name: abs(gaps[name]))

    # The absolute gap and the ratio pick out different components and both are
    # worth having: a component assigned 5% and delivering 12% is only 7 points
    # adrift, and it is doing more than twice the work it was given.
    ratios = {
        name: (effective[name] / nominal[name]) if nominal[name] > 0 else float("inf")
        for name in names
    }
    most_overworked = max(ratios, key=lambda name: ratios[name])

    return {
        "raw_ratio": raw,
        "effective": effective,
        "nominal": nominal,
        "gaps": gaps,
        "ratios": ratios,
        "largest_gap": widest,
        "largest_gap_size": gaps[widest],
        "most_overworked": most_overworked,
        "most_overworked_ratio": ratios[most_overworked],
        "headline": (
            "'%s' was assigned %.0f%% of the weight and accounts for %.0f%% of "
            "the variation. The nominal weights describe an intention; these "
            "describe the ranking."
            % (widest, nominal[widest] * 100.0, effective[widest] * 100.0)
        ),
    }


def raw_sum_effective_weights(
    components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Effective weights of a weighted sum with **no normalisation at all**.

    This is what `src/calculators/eco_score.py` computes: three category scores
    on three different scales, added. The nominal weights are equal; the
    effective ones are not, because in an unnormalised sum the units do the
    weighting.

    Kept separate from `build_index` on purpose. Normalising first is the fix,
    so a function that normalises can never show the problem — it can only show
    that the problem has been fixed.
    """
    cleaned = _validate_components(components)
    units = cleaned[0]["n"]
    weights = {component["name"]: component["weight"] for component in cleaned}

    raw_values = {}
    for component in cleaned:
        sign = -1.0 if component["polarity"] == "lower_is_better" else 1.0
        raw_values[component["name"]] = [sign * value for value in component["values"]]

    scores = [
        sum(weights[name] * raw_values[name][index] for name in raw_values)
        for index in range(units)
    ]
    decomposition = effective_weights(raw_values, scores, weights)

    dominant = max(
        decomposition["effective"], key=lambda name: decomposition["effective"][name]
    )
    return {
        "scores": scores,
        "effective_weights": decomposition,
        "dominant": dominant,
        "dominant_share": decomposition["effective"][dominant],
        "headline": (
            "Added on their own scales, '%s' accounts for %.0f%% of the "
            "variation while carrying %.0f%% of the nominal weight. The units "
            "did the weighting."
            % (
                dominant,
                decomposition["effective"][dominant] * 100.0,
                decomposition["nominal"][dominant] * 100.0,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------


def component_correlations(
    components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Which components are measuring the same thing twice?

    Two components correlated at 0.95 with 20 points of weight between them buy
    a single underlying construct 20 points of influence, which is not what a
    reader of the weight dictionary would conclude.
    """
    cleaned = _validate_components(components)
    names = [component["name"] for component in cleaned]
    values = {component["name"]: component["values"] for component in cleaned}

    matrix: dict[str, dict[str, float]] = {}
    redundant = []
    for first in names:
        matrix[first] = {}
        for second in names:
            coefficient = (
                1.0
                if first == second
                else correlation(values[first], values[second])
            )
            matrix[first][second] = coefficient
            if first < second and abs(coefficient) >= REDUNDANCY_THRESHOLD:
                redundant.append(
                    {
                        "components": (first, second),
                        "correlation": coefficient,
                        "combined_weight": (
                            next(c["weight"] for c in cleaned if c["name"] == first)
                            + next(c["weight"] for c in cleaned if c["name"] == second)
                        ),
                    }
                )

    return {
        "matrix": matrix,
        "redundant_pairs": redundant,
        "headline": (
            "%d pair(s) correlate above %.2f. Their combined weight buys one "
            "underlying construct that much influence, not two."
            % (len(redundant), REDUNDANCY_THRESHOLD)
            if redundant
            else "No pair correlates above %.2f; each component is carrying "
            "distinct information." % REDUNDANCY_THRESHOLD
        ),
    }


# ---------------------------------------------------------------------------
# Weight sensitivity
# ---------------------------------------------------------------------------


def weight_sensitivity(
    unit_names: Sequence[str],
    components: Sequence[Mapping[str, Any]],
    normalisation: str = "minmax",
    aggregation: str = "linear",
    draws: int = DEFAULT_DRAWS,
    concentration: float = DEFAULT_CONCENTRATION,
    seed: int = 20240722,
) -> dict[str, Any]:
    """Would a defensible alternative weighting change the answer?

    The one question that matters for any composite. Weights are drawn from a
    Dirichlet centred on the stated values, the index is rebuilt, and each unit
    gets a rank interval across the draws.

    A unit whose rank moves ten places under a reweighting nobody would object
    to does not have a rank. If nothing moves, the exact weights did not matter
    and the argument about them can stop.
    """
    import random

    cleaned = _validate_components(components)
    if draws < 20:
        raise CompositeError("At least 20 draws are needed to say anything.")
    if concentration <= 0:
        raise CompositeError("Concentration must be positive.")

    rng = random.Random(seed)
    names = [component["name"] for component in cleaned]
    units = cleaned[0]["n"]

    normalised = {
        component["name"]: normalise(component, method=normalisation)
        for component in cleaned
    }
    base_weights = {component["name"]: component["weight"] for component in cleaned}
    total = sum(base_weights.values())
    centre = {name: base_weights[name] / total for name in names}

    baseline = build_index(
        unit_names, cleaned, normalisation=normalisation, aggregation=aggregation
    )

    rank_draws: dict[str, list[int]] = {str(name): [] for name in unit_names}
    for _ in range(int(draws)):
        gammas = {
            name: rng.gammavariate(max(centre[name] * concentration, 1e-6), 1.0)
            for name in names
        }
        gamma_total = sum(gammas.values())
        if gamma_total <= 0:
            continue
        weights = {name: gammas[name] / gamma_total for name in names}

        scores = aggregate(normalised, weights, method=aggregation)
        order = sorted(range(units), key=lambda index: scores[index], reverse=True)
        for position, index in enumerate(order):
            rank_draws[str(unit_names[index])].append(position + 1)

    intervals = []
    for index, name in enumerate(unit_names):
        ranks = sorted(rank_draws[str(name)])
        if not ranks:
            continue
        low = ranks[int(0.025 * (len(ranks) - 1))]
        high = ranks[int(0.975 * (len(ranks) - 1))]
        intervals.append(
            {
                "unit": str(name),
                "baseline_rank": baseline["ranks"][index],
                "median_rank": statistics.median(ranks),
                "lower": low,
                "upper": high,
                "width": high - low,
                "stable": high - low <= 1,
            }
        )

    widths = [entry["width"] for entry in intervals]
    unstable = [entry for entry in intervals if not entry["stable"]]

    return {
        "draws": int(draws),
        "concentration": concentration,
        "intervals": sorted(intervals, key=lambda entry: entry["baseline_rank"]),
        "mean_width": statistics.fmean(widths) if widths else 0.0,
        "max_width": max(widths) if widths else 0,
        "unstable": len(unstable),
        "stable_share": (
            1.0 - len(unstable) / len(intervals) if intervals else 0.0
        ),
        "robust": not unstable,
        "headline": (
            "Every rank is stable across %d defensible reweightings. The exact "
            "weights do not matter here, which is the useful thing to know "
            "before arguing about them." % draws
            if not unstable
            else "%d of %d units move rank under a defensible reweighting, the "
            "largest by %d positions. Those ranks are statements about the "
            "weights rather than about the units."
            % (len(unstable), len(intervals), max(widths) if widths else 0)
        ),
    }


# ---------------------------------------------------------------------------
# Aggregation disagreement
# ---------------------------------------------------------------------------


def rank_reversals(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Pairs whose order flips between two indices.

    Run across aggregations, the reversals *are* the compensability assumption
    made visible: every flipped pair is a case where one unit's shortfall was
    or was not allowed to be bought back.
    """
    if first["units"] != second["units"]:
        raise CompositeError("Both indices must cover the same units.")

    units = first["units"]
    reversals = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            before = first["ranks"][i] < first["ranks"][j]
            after = second["ranks"][i] < second["ranks"][j]
            if before != after:
                reversals.append(
                    {
                        "units": (units[i], units[j]),
                        "first_order": (first["ranks"][i], first["ranks"][j]),
                        "second_order": (second["ranks"][i], second["ranks"][j]),
                    }
                )

    pairs = len(units) * (len(units) - 1) // 2
    return {
        "reversals": reversals,
        "count": len(reversals),
        "pairs": pairs,
        "share": len(reversals) / pairs if pairs else 0.0,
        "headline": (
            "%d of %d pairs (%.0f%%) reverse between %s and %s aggregation. "
            "Each one is a unit whose weakness was allowed to be compensated "
            "under one rule and not the other."
            % (
                len(reversals),
                pairs,
                (len(reversals) / pairs * 100.0) if pairs else 0.0,
                first["aggregation"],
                second["aggregation"],
            )
        ),
    }


def dominance_violations(index: Mapping[str, Any]) -> dict[str, Any]:
    """Units that are worse on every component and score higher anyway.

    A dominance violation is a defect in the index rather than a property of
    the units, and any aggregation that produces one is doing something the
    person reading the score would not endorse.
    """
    normalised = index["normalised"]
    names = list(normalised)
    units = index["units"]
    scores = index["scores"]

    violations = []
    for i in range(len(units)):
        for j in range(len(units)):
            if i == j:
                continue
            dominated = all(
                normalised[name][i] <= normalised[name][j] for name in names
            ) and any(normalised[name][i] < normalised[name][j] for name in names)
            if dominated and scores[i] > scores[j]:
                violations.append(
                    {
                        "worse_on_everything": units[i],
                        "yet_outranks": units[j],
                        "scores": (scores[i], scores[j]),
                    }
                )

    return {
        "violations": violations,
        "count": len(violations),
        "clean": not violations,
        "headline": (
            "%d unit(s) score higher than a unit they are worse than on every "
            "single component. That is a defect in the index."
            % len(violations)
            if violations
            else "No dominance violations: no unit outranks another it is worse "
            "than on everything."
        ),
    }


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------


def band_probabilities(
    unit_names: Sequence[str],
    components: Sequence[Mapping[str, Any]],
    bands: Sequence[tuple[float, str]] = APP_CONFIDENCE_BANDS,
    scale: float = 100.0,
    normalisation: str = "minmax",
    aggregation: str = "linear",
    draws: int = DEFAULT_DRAWS,
    concentration: float = DEFAULT_CONCENTRATION,
    seed: int = 20240722,
) -> dict[str, Any]:
    """The probability of each band, rather than the band the point estimate lands in.

    84.9 is "Medium" and 85.1 is "High". Nothing about the underlying data
    changes at 85, and the display difference is categorical. Reporting the
    distribution over bands rather than the modal one is what makes that
    visible.
    """
    import random

    cleaned = _validate_components(components)
    ordered_bands = sorted(bands, key=lambda item: item[0], reverse=True)
    if not ordered_bands:
        raise CompositeError("At least one band is needed.")
    if draws < 20:
        raise CompositeError("At least 20 draws are needed to say anything.")

    rng = random.Random(seed)
    names = [component["name"] for component in cleaned]
    units = cleaned[0]["n"]
    normalised = {
        component["name"]: normalise(component, method=normalisation)
        for component in cleaned
    }
    base = {component["name"]: component["weight"] for component in cleaned}
    total = sum(base.values())
    centre = {name: base[name] / total for name in names}

    counts: list[dict[str, int]] = [
        {label: 0 for _, label in ordered_bands} for _ in range(units)
    ]
    for _ in range(int(draws)):
        gammas = {
            name: rng.gammavariate(max(centre[name] * concentration, 1e-6), 1.0)
            for name in names
        }
        gamma_total = sum(gammas.values())
        if gamma_total <= 0:
            continue
        weights = {name: gammas[name] / gamma_total for name in names}
        scores = aggregate(normalised, weights, method=aggregation)
        for index, score in enumerate(scores):
            counts[index][_band_for(score * scale, ordered_bands)] += 1

    rows = []
    for index, name in enumerate(unit_names):
        total_draws = sum(counts[index].values()) or 1
        shares = {
            label: counts[index][label] / total_draws for _, label in ordered_bands
        }
        modal = max(shares, key=lambda label: shares[label])
        rows.append(
            {
                "unit": str(name),
                "probabilities": shares,
                "modal_band": modal,
                "modal_probability": shares[modal],
                "confident": shares[modal] >= MIN_BAND_PROBABILITY,
            }
        )

    borderline = [row for row in rows if not row["confident"]]
    return {
        "bands": [label for _, label in ordered_bands],
        "rows": rows,
        "borderline": len(borderline),
        "threshold": MIN_BAND_PROBABILITY,
        "headline": (
            "%d of %d units sit close enough to a cut that a defensible "
            "reweighting moves them across it. Their band label is a statement "
            "about the weights." % (len(borderline), len(rows))
            if borderline
            else "Every unit's band survives reweighting with at least %.0f%% "
            "of the mass." % (MIN_BAND_PROBABILITY * 100.0)
        ),
    }


def _band_for(score: float, bands: Sequence[tuple[float, str]]) -> str:
    for threshold, label in bands:
        if score >= threshold:
            return label
    return bands[-1][1]


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def analyse(
    unit_names: Sequence[str],
    components: Sequence[Mapping[str, Any]],
    normalisation: str = "minmax",
    draws: int = DEFAULT_DRAWS,
    seed: int = 20240722,
) -> dict[str, Any]:
    """Build the index three ways and report where the three disagree."""
    cleaned = _validate_components(components)

    linear = build_index(
        unit_names, cleaned, normalisation=normalisation, aggregation="linear"
    )
    geometric = build_index(
        unit_names, cleaned, normalisation=normalisation, aggregation="geometric"
    )
    outranking = build_index(
        unit_names, cleaned, normalisation=normalisation, aggregation="non_compensatory"
    )

    return {
        "engine_version": ENGINE_VERSION,
        "units": [str(name) for name in unit_names],
        "normalisation": normalisation,
        "scales": scale_mismatch(cleaned),
        "unnormalised": raw_sum_effective_weights(cleaned),
        "linear": linear,
        "geometric": geometric,
        "non_compensatory": outranking,
        "effective_weights": linear["effective_weights"],
        "correlations": component_correlations(cleaned),
        "sensitivity": weight_sensitivity(
            unit_names,
            cleaned,
            normalisation=normalisation,
            draws=draws,
            seed=seed,
        ),
        "linear_vs_geometric": rank_reversals(linear, geometric),
        "linear_vs_outranking": rank_reversals(linear, outranking),
        "dominance": dominance_violations(linear),
        "dominated_by_one": linear["dominated_by_one"],
        "headline": linear["headline"],
    }


def get_index_notes(result: Mapping[str, Any]) -> list[str]:
    """Plain-language readings of an analysis."""
    notes: list[str] = [result.get("headline", "")]

    scales = result.get("scales")
    if scales and scales["mismatched"]:
        notes.append(scales["headline"])
        unnormalised = result.get("unnormalised")
        if unnormalised:
            notes.append(unnormalised["headline"])

    effective = result.get("effective_weights")
    if effective:
        notes.append(effective["headline"])

    correlations = result.get("correlations")
    if correlations and correlations["redundant_pairs"]:
        notes.append(correlations["headline"])

    sensitivity = result.get("sensitivity")
    if sensitivity:
        notes.append(sensitivity["headline"])

    reversals = result.get("linear_vs_geometric")
    if reversals and reversals["count"]:
        notes.append(
            "%s Under a linear sum a zero anywhere is bought back; under a "
            "geometric mean it is not. Which is right is a value judgement, and "
            "for a confidence score it is probably the second."
            % reversals["headline"]
        )

    dominance = result.get("dominance")
    if dominance and not dominance["clean"]:
        notes.append(dominance["headline"])

    if result.get("dominated_by_one"):
        notes.append(
            "One component carries almost all the variation. Reporting the "
            "other components alongside it implies a balance the arithmetic "
            "does not have."
        )
    return [note for note in notes if note]


def summarise(result: Mapping[str, Any]) -> str:
    """One line for a log or a saved-analysis list."""
    sensitivity = result.get("sensitivity") or {}
    reversals = result.get("linear_vs_geometric") or {}
    return "%d units | %s | %d/%d ranks unstable | %d pair reversals" % (
        len(result.get("units", [])),
        result.get("normalisation", "?"),
        sensitivity.get("unstable", 0),
        len(sensitivity.get("intervals", [])),
        reversals.get("count", 0),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _connect():
    return sqlite3.connect(DB_NAME)


def _ensure_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS composite_index_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            label TEXT NOT NULL,
            units INTEGER NOT NULL,
            normalisation TEXT NOT NULL,
            unstable_ranks INTEGER NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_composite_index_user
        ON composite_index_analyses (user_id)
        """
    )


def save_analysis(user_id: Any, result: Mapping[str, Any], label: str = "") -> int | None:
    """Persist an analysis. None if storage is unavailable."""
    if not user_id or not result.get("engine_version"):
        return None
    try:
        with _connect() as conn:
            _ensure_tables(conn)
            cursor = conn.execute(
                """
                INSERT INTO composite_index_analyses
                    (user_id, label, units, normalisation, unstable_ranks, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_id),
                    str(label or "index"),
                    len(result.get("units", [])),
                    str(result.get("normalisation", "")),
                    int((result.get("sensitivity") or {}).get("unstable", 0)),
                    json.dumps(result, default=str),
                ),
            )
            return cursor.lastrowid
    except sqlite3.Error:
        return None


def get_analyses(user_id: Any, limit: int = 25) -> list[dict[str, Any]]:
    """Most recent analyses for one user."""
    if not user_id:
        return []
    try:
        with _connect() as conn:
            _ensure_tables(conn)
            rows = conn.execute(
                """
                SELECT id, label, units, normalisation, unstable_ranks, payload,
                       created_at
                FROM composite_index_analyses
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(user_id), int(limit)),
            ).fetchall()
    except sqlite3.Error:
        return []

    analyses = []
    for row in rows:
        try:
            payload = json.loads(row[5])
        except (TypeError, ValueError):
            payload = {}
        analyses.append(
            {
                "id": row[0],
                "label": row[1],
                "units": row[2],
                "normalisation": row[3],
                "unstable_ranks": row[4],
                "payload": payload,
                "created_at": row[6],
            }
        )
    return analyses


def delete_analysis(user_id: Any, analysis_id: int) -> bool:
    """Remove one analysis belonging to this user."""
    if not user_id:
        return False
    try:
        with _connect() as conn:
            _ensure_tables(conn)
            cursor = conn.execute(
                "DELETE FROM composite_index_analyses WHERE user_id = ? AND id = ?",
                (str(user_id), int(analysis_id)),
            )
            return cursor.rowcount > 0
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------------------
# Worked examples
# ---------------------------------------------------------------------------


def demo_confidence_index(
    units: int = 40,
    seed: int = 20240722,
) -> tuple[list[str], list[dict[str, Any]]]:
    """The confidence score from `src/carbon/confidence_scoring.py`, as data.

    Uses the real weights. Two of the components — `input_completeness` and
    `category_coverage` — are generated correlated at around 0.95, because in
    that module they are close to the same measurement: both count how much of
    the assessment was filled in. Between them they carry 35 of the 100 points.

    `unit_conversion` is generated with a deliberately narrow spread, because
    almost every user of this app enters metric units. It has 10 nominal points
    and almost no effect on any ranking, which is what the effective-weight
    calculation should report.
    """
    import random

    rng = random.Random(seed)
    names = ["assessment_%02d" % index for index in range(int(units))]

    completeness = [rng.uniform(30.0, 100.0) for _ in range(units)]
    coverage = [
        max(0.0, min(100.0, value + rng.gauss(0.0, 6.0))) for value in completeness
    ]
    generated = {
        "input_completeness": completeness,
        "category_coverage": coverage,
        "estimated_vs_measured": [rng.uniform(0.0, 100.0) for _ in range(units)],
        "data_age": [rng.uniform(20.0, 100.0) for _ in range(units)],
        "factor_provenance": [rng.choice([40.0, 70.0, 100.0]) for _ in range(units)],
        "unit_conversion": [rng.uniform(95.0, 100.0) for _ in range(units)],
        "validation_warnings": [rng.uniform(50.0, 100.0) for _ in range(units)],
    }

    components = [
        build_component(name, generated[name], weight=weight)
        for name, weight in APP_CONFIDENCE_WEIGHTS.items()
    ]
    return names, components


def demo_eco_score(
    units: int = 40,
    seed: int = 20240722,
) -> tuple[list[str], list[dict[str, Any]]]:
    """The scale problem from `src/calculators/eco_score.py`, as data.

    Three components on the scales that module actually adds: a diet factor
    around 1.5-7.0, a transport factor around 0.1-8.0, and ``energy_kwh * 0.5``
    running into the hundreds. All three carry equal nominal weight, and the
    third is the score.
    """
    import random

    rng = random.Random(seed)
    names = ["household_%02d" % index for index in range(int(units))]

    diet = [rng.choice([1.5, 2.5, 4.0, 7.0]) for _ in range(units)]
    transport = [rng.choice([0.1, 1.2, 4.5, 8.0]) for _ in range(units)]
    energy = [rng.uniform(150.0, 900.0) * 0.5 for _ in range(units)]

    components = [
        build_component("diet", diet, weight=1.0, polarity="lower_is_better"),
        build_component("transport", transport, weight=1.0, polarity="lower_is_better"),
        build_component("energy", energy, weight=1.0, polarity="lower_is_better"),
    ]
    return names, components


def demo_compensating_index(
    units: int = 30,
    seed: int = 20240722,
) -> tuple[list[str], list[dict[str, Any]]]:
    """A set built so that compensation actually matters.

    Some units are uniformly mediocre and some are excellent on two components
    and near-zero on a third. A linear sum ranks them together; a geometric
    mean does not, and every reversal between the two is a case where a zero
    was or was not bought back.
    """
    import random

    rng = random.Random(seed)
    names = []
    first, second, third = [], [], []

    for index in range(int(units)):
        names.append("unit_%02d" % index)
        if index % 3 == 0:
            # Spiky: excellent twice, nearly zero once.
            first.append(rng.uniform(85.0, 100.0))
            second.append(rng.uniform(85.0, 100.0))
            third.append(rng.uniform(0.0, 6.0))
        else:
            # Even: mediocre everywhere.
            level = rng.uniform(50.0, 70.0)
            first.append(level + rng.gauss(0.0, 4.0))
            second.append(level + rng.gauss(0.0, 4.0))
            third.append(level + rng.gauss(0.0, 4.0))

    components = [
        build_component("first", first, weight=1.0),
        build_component("second", second, weight=1.0),
        build_component("third", third, weight=1.0),
    ]
    return names, components
