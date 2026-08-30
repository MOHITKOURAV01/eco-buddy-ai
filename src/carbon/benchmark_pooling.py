"""Partial pooling for household benchmarks, so a short history stops
outranking a long one.

`carbon_benchmarking.py` will tell a household with two assessments exactly the
same kind of thing it tells a household with forty. `analyse_trend()` declares
a direction from two points and a fixed 50 kg threshold. `get_leaderboard()`
orders whoever is present. `find_closest_lifestyle()` matches an archetype on a
category breakdown that, for a new user, is one month of half-remembered
numbers.

The estimate from two observations and the estimate from forty are not the same
kind of object, and the app renders them identically. That is not a display
problem. It is an estimation problem with a standard answer: shrink the
small-sample estimate toward the group it belongs to, by an amount determined
by how noisy it is.

The method
----------
One-way random effects, estimated by moments. Observed spread splits into::

    tau^2    between-household variance   real differences between households
    sigma^2  within-household variance    month-to-month noise inside one

Both fall out of the usual ANOVA decomposition on an unbalanced panel. Then for
a household with `n` observations::

    lambda = tau^2 / (tau^2 + sigma^2 / n)
    pooled = lambda * own_mean + (1 - lambda) * grand_mean

`lambda` is the weight on the household's own data. It goes to zero for one
noisy observation and to one for a long clean history, and it is the honest
answer to "how much of this number is you and how much is the group".

Why this is not cosmetic
------------------------
The top of any leaderboard built from unpooled means is populated by whoever
has the fewest, luckiest observations. This is not a subtlety; it is the most
reliable artefact in ranked data, and the leaderboard reproduces it faithfully
every time it renders. Shrinkage is what stops a household with three lucky
months from being reported as the greenest in the neighbourhood.

Refusals
--------
No pooling against a group too small to estimate between-household variance.
No negative variance component clamped silently — that is a signal the model
does not fit, not a number to floor at zero and move on from. No pooled
estimate presented without its reliability, because a pooled estimate with
lambda near zero is a statement about the group and must be labelled as one.

Self-contained by design: the distribution functions below are duplicated in
other engines in this repo rather than shared, because there is no common
numerics module and adding one would mean touching files other work depends on.
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

# Panel requirements --------------------------------------------------------
MIN_HOUSEHOLDS = 4
MIN_OBSERVATIONS = 1
MIN_TREND_POINTS = 3
MAX_HOUSEHOLDS = 5000

# Reliability bands ---------------------------------------------------------
LOW_RELIABILITY = 0.30
HIGH_RELIABILITY = 0.70

# Inference -----------------------------------------------------------------
DEFAULT_CONFIDENCE = 0.95
TREND_ALPHA = 0.05
CONFIDENCE_Z = {0.80: 1.281552, 0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}

# Badges --------------------------------------------------------------------
DEFAULT_BADGES = (
    ("Platinum", 1500.0),
    ("Gold", 2500.0),
    ("Silver", 4000.0),
    ("Bronze", 6000.0),
)


class PoolingError(ValueError):
    """Raised when a panel cannot support the estimate being asked for."""


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


def _z_for(confidence: float) -> float:
    key = round(float(confidence), 2)
    if key not in CONFIDENCE_Z:
        raise PoolingError(
            "Confidence must be one of %s."
            % ", ".join(str(value) for value in sorted(CONFIDENCE_Z))
        )
    return CONFIDENCE_Z[key]


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(value: float, degrees_of_freedom: float) -> float:
    """Student-t cumulative distribution function."""
    if degrees_of_freedom <= 0:
        raise PoolingError("Student-t needs positive degrees of freedom.")
    z = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * betai(degrees_of_freedom / 2.0, 0.5, z)
    return 1.0 - tail if value > 0 else tail


def two_sided_p(statistic: float, degrees_of_freedom: float) -> float:
    return 2.0 * (1.0 - t_cdf(abs(statistic), max(degrees_of_freedom, 1.0)))


# ---------------------------------------------------------------------------
# Households
# ---------------------------------------------------------------------------


def build_household(
    identifier: Any,
    observations: Sequence[float],
    label: str = "",
) -> dict[str, Any]:
    """One household and its assessment history.

    A household with a single observation is allowed. It is the case the module
    exists for, and refusing it would push the problem back to the caller, who
    would go on ranking it unpooled.
    """
    cleaned: list[float] = []
    for value in observations:
        number = _finite(value)
        if number is None:
            raise PoolingError(
                "Household '%s' has a non-numeric observation." % identifier
            )
        cleaned.append(number)
    if len(cleaned) < MIN_OBSERVATIONS:
        raise PoolingError("Household '%s' has no observations." % identifier)

    return {
        "id": str(identifier),
        "label": str(label or identifier),
        "observations": cleaned,
        "n": len(cleaned),
        "mean": statistics.fmean(cleaned),
        "own_variance": statistics.variance(cleaned) if len(cleaned) > 1 else None,
    }


def _validate_panel(households: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(households) < MIN_HOUSEHOLDS:
        raise PoolingError(
            "Need at least %d households to separate between-household variance "
            "from within-household noise. With fewer there is nothing to pool "
            "toward." % MIN_HOUSEHOLDS
        )
    if len(households) > MAX_HOUSEHOLDS:
        raise PoolingError("At most %d households are supported." % MAX_HOUSEHOLDS)

    seen: set[str] = set()
    cleaned = []
    for household in households:
        if not isinstance(household, Mapping) or "observations" not in household:
            raise PoolingError("Households must be built with build_household().")
        if household["id"] in seen:
            raise PoolingError("Household '%s' appears twice." % household["id"])
        seen.add(household["id"])
        cleaned.append(dict(household))
    return cleaned


# ---------------------------------------------------------------------------
# Variance components
# ---------------------------------------------------------------------------


def variance_components(households: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Split observed spread into real differences and month-to-month noise.

    Standard one-way random-effects ANOVA estimator on an unbalanced panel::

        sigma^2 = SSW / (N - k)
        tau^2   = (SSB - (k - 1) sigma^2) / (N - sum(n_i^2) / N)

    Neither number is currently computed anywhere in this repo, and everything
    else in this module follows from the two of them.

    A negative `tau^2` is possible and is reported rather than clamped
    silently. It means the between-household spread is smaller than the
    within-household noise would produce by chance — which is a statement that
    the households are not measurably different from each other, and that is a
    finding about the panel, not a number to floor at zero and move past.
    """
    cleaned = _validate_panel(households)
    total_observations = sum(household["n"] for household in cleaned)
    group_count = len(cleaned)

    if total_observations <= group_count:
        raise PoolingError(
            "Every household has a single observation, so within-household "
            "noise cannot be estimated. Some household must have repeat "
            "measurements or there is no noise model to shrink against."
        )

    grand_mean = (
        sum(household["mean"] * household["n"] for household in cleaned)
        / total_observations
    )

    within_ss = 0.0
    for household in cleaned:
        centre = household["mean"]
        within_ss += sum((value - centre) ** 2 for value in household["observations"])
    within_df = total_observations - group_count
    sigma_squared = within_ss / within_df if within_df > 0 else 0.0

    between_ss = sum(
        household["n"] * (household["mean"] - grand_mean) ** 2 for household in cleaned
    )
    between_df = group_count - 1

    sum_squared_sizes = sum(household["n"] ** 2 for household in cleaned)
    scale = total_observations - sum_squared_sizes / total_observations
    if scale <= 0:
        raise PoolingError("Panel is degenerate; cannot estimate variance components.")

    raw_tau_squared = (between_ss - between_df * sigma_squared) / scale
    negative = raw_tau_squared < 0
    tau_squared = max(0.0, raw_tau_squared)

    return {
        "grand_mean": grand_mean,
        "within_variance": sigma_squared,
        "within_sd": math.sqrt(max(sigma_squared, 0.0)),
        "between_variance": tau_squared,
        "between_sd": math.sqrt(tau_squared),
        "raw_between_variance": raw_tau_squared,
        "negative_component": negative,
        "households": group_count,
        "observations": total_observations,
        "within_df": within_df,
        "between_df": between_df,
        "intraclass_correlation": (
            tau_squared / (tau_squared + sigma_squared)
            if (tau_squared + sigma_squared) > 0
            else 0.0
        ),
        "note": (
            "Between-household variance came out negative (%.1f). The households "
            "in this panel are not measurably different from each other once "
            "month-to-month noise is accounted for, so every pooled estimate "
            "collapses onto the group mean. That is the finding."
            % raw_tau_squared
            if negative
            else "Between-household SD %.0f against within-household SD %.0f; "
            "%.0f%% of observed spread is real difference between households."
            % (
                math.sqrt(tau_squared),
                math.sqrt(max(sigma_squared, 0.0)),
                (tau_squared / (tau_squared + sigma_squared) * 100.0)
                if (tau_squared + sigma_squared) > 0
                else 0.0,
            )
        ),
    }


def reliability(n: int, between_variance: float, within_variance: float) -> float:
    """Weight on a household's own data.

    ``lambda = tau^2 / (tau^2 + sigma^2 / n)``

    Zero for one noisy observation, approaching one for a long clean history.
    Exactly the quantity that is missing everywhere a small-sample mean is
    currently ranked against a large-sample one.
    """
    if n < 1:
        raise PoolingError("Reliability needs at least one observation.")
    if between_variance <= 0:
        return 0.0
    if within_variance <= 0:
        return 1.0
    return between_variance / (between_variance + within_variance / n)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def pool_panel(
    households: Sequence[Mapping[str, Any]],
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Shrink every household toward the group, and say by how much.

    Returns raw and pooled estimates side by side, because the size of the gap
    is the finding. A household whose pooled estimate barely moves had enough
    data; one that moves most of the way to the group mean did not, and was
    being ranked as though it did.
    """
    cleaned = _validate_panel(households)
    components = variance_components(cleaned)
    z = _z_for(confidence)

    grand = components["grand_mean"]
    tau_squared = components["between_variance"]
    sigma_squared = components["within_variance"]

    estimates = []
    for household in cleaned:
        n = household["n"]
        weight = reliability(n, tau_squared, sigma_squared)
        pooled = weight * household["mean"] + (1.0 - weight) * grand
        posterior_variance = weight * sigma_squared / n if n > 0 else tau_squared
        posterior_sd = math.sqrt(max(posterior_variance, 0.0))

        estimates.append(
            {
                "id": household["id"],
                "label": household["label"],
                "n": n,
                "raw_mean": household["mean"],
                "pooled_mean": pooled,
                "shrinkage": household["mean"] - pooled,
                "shrinkage_share": (
                    abs(household["mean"] - pooled) / abs(household["mean"] - grand)
                    if abs(household["mean"] - grand) > 1e-9
                    else 0.0
                ),
                "reliability": weight,
                "posterior_sd": posterior_sd,
                "lower": pooled - z * posterior_sd,
                "upper": pooled + z * posterior_sd,
                "reliability_band": _reliability_band(weight),
                "mostly_group": weight < LOW_RELIABILITY,
            }
        )

    estimates.sort(key=lambda entry: entry["pooled_mean"])
    raw_order = [
        entry["id"] for entry in sorted(estimates, key=lambda item: item["raw_mean"])
    ]
    pooled_order = [entry["id"] for entry in estimates]

    return {
        "engine_version": ENGINE_VERSION,
        "components": components,
        "estimates": estimates,
        "confidence": confidence,
        "raw_order": raw_order,
        "pooled_order": pooled_order,
        "rank_churn": rank_churn(raw_order, pooled_order),
        "ranking": rank_with_uncertainty(estimates),
        "headline": _panel_headline(estimates, components),
    }


def _reliability_band(weight: float) -> str:
    if weight >= HIGH_RELIABILITY:
        return "own_data"
    if weight >= LOW_RELIABILITY:
        return "mixed"
    return "group"


def _panel_headline(
    estimates: Sequence[Mapping[str, Any]],
    components: Mapping[str, Any],
) -> str:
    thin = [entry for entry in estimates if entry["mostly_group"]]
    return (
        "%d households, %d observations. %d of them have a reliability below "
        "%.2f, meaning their reported figure is mostly the group mean rather "
        "than their own data. %s"
        % (
            components["households"],
            components["observations"],
            len(thin),
            LOW_RELIABILITY,
            components["note"],
        )
    )


def rank_churn(raw_order: Sequence[str], pooled_order: Sequence[str]) -> dict[str, Any]:
    """How far the pooled ranking moves from the unpooled one.

    A large churn is the finding: it is the size of the error the current
    unpooled leaderboard is making, expressed in positions.
    """
    positions = {identifier: index for index, identifier in enumerate(raw_order)}
    moves = []
    for index, identifier in enumerate(pooled_order):
        before = positions.get(identifier, index)
        moves.append(abs(before - index))
    changed = sum(1 for move in moves if move > 0)
    return {
        "changed": changed,
        "total": len(pooled_order),
        "share": changed / len(pooled_order) if pooled_order else 0.0,
        "largest_move": max(moves) if moves else 0,
        "mean_move": statistics.fmean(moves) if moves else 0.0,
    }


def rank_with_uncertainty(
    estimates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank only where posterior intervals separate; otherwise report a tie.

    Ordering by point estimate reports positions 3 and 4 as distinct when their
    intervals overlap almost entirely. That ordering is stable in the display
    and unstable in reality — it will flip next month and the user will read
    the flip as change.
    """
    ordered = sorted(estimates, key=lambda entry: entry["pooled_mean"])
    bands: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    anchor_upper = None

    for entry in ordered:
        if not current:
            current = [entry]
            anchor_upper = entry["upper"]
            continue
        # Compared against the band's anchor, not against a running maximum.
        # A running maximum chains: A overlaps B, B overlaps C, and C ends up
        # in a band with an A it is cleanly separated from, so on a wide panel
        # every household collapses into one band and the output says nothing.
        # The anchor keeps a band to "not separated from the household that
        # opened it", which is the claim a tie band is actually making.
        if anchor_upper is not None and entry["lower"] <= anchor_upper:
            current.append(entry)
        else:
            bands.append(_band(current, len(bands) + 1))
            current = [entry]
            anchor_upper = entry["upper"]
    if current:
        bands.append(_band(current, len(bands) + 1))
    return bands


def _band(members: Sequence[Mapping[str, Any]], position: int) -> dict[str, Any]:
    return {
        "band": position,
        "ids": [item["id"] for item in members],
        "labels": [item["label"] for item in members],
        "lowest": min(item["pooled_mean"] for item in members),
        "highest": max(item["pooled_mean"] for item in members),
        "separated": len(members) == 1,
    }


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def percentile_of(
    identifier: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Percentile from the pooled estimates, with the uncertainty attached.

    The unpooled version of this is what every "you are in the top 20%" message
    is currently computed from, and for a household with two assessments it is
    a statement about two months of weather.
    """
    estimates = result["estimates"]
    target = next((entry for entry in estimates if entry["id"] == identifier), None)
    if target is None:
        raise PoolingError("Household '%s' is not in this panel." % identifier)

    total = len(estimates)
    below = sum(1 for entry in estimates if entry["pooled_mean"] < target["pooled_mean"])
    raw_below = sum(1 for entry in estimates if entry["raw_mean"] < target["raw_mean"])

    lower_rank = sum(1 for entry in estimates if entry["pooled_mean"] < target["lower"])
    upper_rank = sum(1 for entry in estimates if entry["pooled_mean"] < target["upper"])

    return {
        "id": identifier,
        "pooled_percentile": below / total * 100.0,
        "raw_percentile": raw_below / total * 100.0,
        "percentile_low": lower_rank / total * 100.0,
        "percentile_high": upper_rank / total * 100.0,
        "reliability": target["reliability"],
        "n": target["n"],
        "headline": (
            "%.0fth percentile pooled (%.0fth unpooled), and anywhere between "
            "the %.0fth and the %.0fth once the uncertainty in %d observation(s) "
            "is carried through."
            % (
                below / total * 100.0,
                raw_below / total * 100.0,
                lower_rank / total * 100.0,
                upper_rank / total * 100.0,
                target["n"],
            )
        ),
    }


# ---------------------------------------------------------------------------
# Trend, scaled to the household's own noise
# ---------------------------------------------------------------------------


def trend_direction(
    household: Mapping[str, Any],
    within_variance: float,
    alpha: float = TREND_ALPHA,
) -> dict[str, Any]:
    """Slope test against the panel's within-household noise.

    Replaces a fixed threshold that has no idea how noisy the household is. A
    50 kg rule fires on nothing at all for a household whose month-to-month
    swing is 300 kg, and fails to fire on a real change for one whose swing is
    20 kg.

    The pooled within-household variance is used rather than the household's
    own, deliberately: at three observations a household's own variance
    estimate is worse than the noise it is trying to measure, and borrowing the
    panel's is the same move that motivates the rest of this module.

    "Insufficient data" is a first-class answer. `analyse_trend()` currently
    returns "stable" for both "measured as stable" and "we have no idea", and
    those are not the same claim.
    """
    observations = household["observations"]
    n = len(observations)
    if n < MIN_TREND_POINTS:
        return {
            "direction": "insufficient_data",
            "n": n,
            "slope": None,
            "p_value": None,
            "headline": (
                "%d observation(s) — not enough to distinguish a trend from "
                "noise. This is not the same as 'stable'." % n
            ),
        }

    xs = list(range(n))
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(observations)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        raise PoolingError("Degenerate time index.")

    slope = sum(
        (xs[index] - mean_x) * (observations[index] - mean_y) for index in range(n)
    ) / sxx

    if within_variance <= 0:
        return {
            "direction": "improving" if slope < 0 else ("worsening" if slope > 0 else "stable"),
            "n": n,
            "slope": slope,
            "p_value": 0.0 if slope != 0 else 1.0,
            "headline": "No within-household noise in this panel; the slope is exact.",
        }

    standard_error = math.sqrt(within_variance / sxx)
    statistic = slope / standard_error if standard_error > 0 else 0.0
    degrees = max(1, n - 2)
    p_value = two_sided_p(statistic, degrees)

    if p_value >= alpha:
        direction = "stable"
    else:
        direction = "improving" if slope < 0 else "worsening"

    detectable = 1.959964 * standard_error
    return {
        "direction": direction,
        "n": n,
        "slope": slope,
        "standard_error": standard_error,
        "t_statistic": statistic,
        "degrees_of_freedom": degrees,
        "p_value": p_value,
        "minimum_detectable_slope": detectable,
        "headline": (
            "%s at %.0f per period (p=%.3f)."
            % (direction.capitalize(), slope, p_value)
            if direction in ("improving", "worsening")
            else "No detectable trend (slope %.0f per period, p=%.3f). Nothing "
            "smaller than %.0f per period could have been detected from %d "
            "observations at this noise level."
            % (slope, p_value, detectable, n)
        ),
    }


# ---------------------------------------------------------------------------
# Badges with hysteresis
# ---------------------------------------------------------------------------


def badge_for(
    estimate: Mapping[str, Any],
    thresholds: Sequence[tuple[str, float]] = DEFAULT_BADGES,
    current_badge: str | None = None,
) -> dict[str, Any]:
    """Award on the interval, retain on the point estimate.

    A hard cut on a noisy statistic makes a household gain and lose a badge on
    measurement noise, which is wrong and — for a gamification feature —
    actively counterproductive. So a new badge requires the whole posterior
    interval to clear the threshold, and an existing one is kept until the
    point estimate crosses back.
    """
    ordered = sorted(thresholds, key=lambda item: item[1])

    earned = None
    for name, ceiling in ordered:
        if estimate["upper"] <= ceiling:
            earned = name
            break

    provisional = None
    for name, ceiling in ordered:
        if estimate["pooled_mean"] <= ceiling:
            provisional = name
            break

    levels = [name for name, _ in ordered]

    def _rank(name: str | None) -> int | None:
        return levels.index(name) if name in levels else None

    held = _rank(current_badge)
    still_qualifies = (
        held is not None
        and provisional is not None
        and _rank(provisional) <= held
    )

    if still_qualifies:
        # The held badge is checked before the outright award, not after. A
        # household holding Platinum whose interval only clears Gold still has
        # a point estimate inside Platinum, and demoting it there is exactly
        # the revocation-on-noise this function exists to prevent.
        if earned is not None and _rank(earned) <= held:
            awarded, reason = earned, "interval_clears"
        else:
            awarded, reason = current_badge, "retained"
    elif earned is not None:
        awarded = earned
        reason = "downgraded" if held is not None else "interval_clears"
    elif provisional is not None:
        awarded = provisional
        reason = "downgraded" if held is not None else "point_estimate_only"
    else:
        awarded, reason = None, "none"

    return {
        "badge": awarded,
        "provisional": provisional,
        "earned_outright": earned,
        "reason": reason,
        "held_on_hysteresis": reason == "retained",
        "headline": _badge_headline(awarded, reason, current_badge),
    }


def _badge_headline(
    awarded: str | None,
    reason: str,
    current_badge: str | None,
) -> str:
    if awarded is None:
        return "No badge."
    if reason == "interval_clears":
        return "%s — the whole interval clears the threshold." % awarded
    if reason == "retained":
        return (
            "%s, retained. The point estimate still qualifies and the interval "
            "no longer clears outright; revoking on noise is worse than "
            "holding." % awarded
        )
    if reason == "downgraded":
        return (
            "%s, down from %s. The estimate has moved past the threshold, not "
            "wobbled around it." % (awarded, current_badge)
        )
    return "%s on the point estimate alone — not yet earned outright." % awarded

# ---------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------


def get_pooling_notes(result: Mapping[str, Any]) -> list[str]:
    """Plain-language readings of a pooled panel."""
    notes: list[str] = [result["headline"]]
    components = result["components"]

    if components["negative_component"]:
        notes.append(
            "Negative between-household variance is not a bug. It says the "
            "households in this panel are indistinguishable from one another "
            "once month-to-month noise is accounted for, so ranking them is "
            "ranking noise."
        )

    notes.append(
        "Intraclass correlation %.2f — that share of the observed spread is "
        "genuine difference between households; the rest is within-household "
        "noise that a single-month comparison would report as difference."
        % components["intraclass_correlation"]
    )

    churn = result["rank_churn"]
    notes.append(
        "Pooling moves %d of %d households in the ranking, the largest by %d "
        "position(s). That is the size of the error the unpooled leaderboard "
        "is making."
        % (churn["changed"], churn["total"], churn["largest_move"])
    )

    unseparated = [band for band in result["ranking"] if not band["separated"]]
    if unseparated:
        largest = max(unseparated, key=lambda band: len(band["ids"]))
        notes.append(
            "The largest tie band contains %d households whose intervals "
            "overlap. Ordering them would invent a distinction the data does "
            "not support." % len(largest["ids"])
        )

    thin = [entry for entry in result["estimates"] if entry["mostly_group"]]
    if thin:
        notes.append(
            "%d household(s) have reliability below %.2f. Their reported figure "
            "is mostly a statement about the group, and should be labelled that "
            "way rather than presented as a measurement of them."
            % (len(thin), LOW_RELIABILITY)
        )
    return notes


def summarise(result: Mapping[str, Any]) -> str:
    """One line for a log or a saved-panel list."""
    components = result["components"]
    churn = result["rank_churn"]
    return (
        "%d households | ICC %.2f | between SD %.0f, within SD %.0f | "
        "%d/%d ranks moved"
        % (
            components["households"],
            components["intraclass_correlation"],
            components["between_sd"],
            components["within_sd"],
            churn["changed"],
            churn["total"],
        )
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _connect():
    return sqlite3.connect(DB_NAME)


def _ensure_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_pooled_panels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            label TEXT NOT NULL,
            households INTEGER NOT NULL,
            intraclass_correlation REAL NOT NULL,
            ranks_moved INTEGER NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_benchmark_pooled_panels_user
        ON benchmark_pooled_panels (user_id)
        """
    )


def save_panel(user_id: Any, result: Mapping[str, Any], label: str = "") -> int | None:
    """Persist a pooled panel. None if storage is unavailable."""
    if not user_id or not result.get("estimates"):
        return None
    try:
        with _connect() as conn:
            _ensure_tables(conn)
            cursor = conn.execute(
                """
                INSERT INTO benchmark_pooled_panels
                    (user_id, label, households, intraclass_correlation,
                     ranks_moved, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_id),
                    str(label or "panel"),
                    int(result["components"]["households"]),
                    float(result["components"]["intraclass_correlation"]),
                    int(result["rank_churn"]["changed"]),
                    json.dumps(result, default=str),
                ),
            )
            return cursor.lastrowid
    except sqlite3.Error:
        return None


def get_panels(user_id: Any, limit: int = 25) -> list[dict[str, Any]]:
    """Most recent saved panels for one user."""
    if not user_id:
        return []
    try:
        with _connect() as conn:
            _ensure_tables(conn)
            rows = conn.execute(
                """
                SELECT id, label, households, intraclass_correlation, ranks_moved,
                       payload, created_at
                FROM benchmark_pooled_panels
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(user_id), int(limit)),
            ).fetchall()
    except sqlite3.Error:
        return []

    panels = []
    for row in rows:
        try:
            payload = json.loads(row[5])
        except (TypeError, ValueError):
            payload = {}
        panels.append(
            {
                "id": row[0],
                "label": row[1],
                "households": row[2],
                "intraclass_correlation": row[3],
                "ranks_moved": row[4],
                "payload": payload,
                "created_at": row[6],
            }
        )
    return panels


def delete_panel(user_id: Any, panel_id: int) -> bool:
    """Remove one saved panel belonging to this user."""
    if not user_id:
        return False
    try:
        with _connect() as conn:
            _ensure_tables(conn)
            cursor = conn.execute(
                "DELETE FROM benchmark_pooled_panels WHERE user_id = ? AND id = ?",
                (str(user_id), int(panel_id)),
            )
            return cursor.rowcount > 0
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------------------
# Worked example
# ---------------------------------------------------------------------------


def demo_panel(
    households: int = 40,
    between_sd: float = 900.0,
    within_sd: float = 700.0,
    grand_mean: float = 3800.0,
    min_history: int = 1,
    max_history: int = 36,
    seed: int = 20241118,
) -> list[dict[str, Any]]:
    """A panel with wildly uneven history lengths, which is the realistic case.

    Households get between one and three years of assessments, drawn so that
    short histories are common — which is what an app's user base looks like,
    and it is why the unpooled leaderboard is topped by people who have barely
    used it.

    The true household levels are drawn from the same distribution regardless
    of history length, so any relationship between "few observations" and
    "extreme rank" in the unpooled ranking is entirely an artefact.
    """
    import random

    rng = random.Random(seed)
    panel = []
    for index in range(int(households)):
        true_level = rng.gauss(grand_mean, between_sd)
        # Short histories deliberately over-represented.
        span = min(
            max_history,
            max(min_history, int(min_history + rng.expovariate(1.0 / 8.0))),
        )
        observations = [
            max(200.0, rng.gauss(true_level, within_sd)) for _ in range(span)
        ]
        panel.append(
            build_household("h%02d" % index, observations, label="Household %d" % index)
        )
    return panel
