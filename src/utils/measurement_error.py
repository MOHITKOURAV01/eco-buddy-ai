"""Measurement error in self-reported activity data, and what it does to
every relationship this app estimates.

Almost every number here is recalled rather than measured. `src/data/data_quality.py`
checks those numbers for outliers, duplicates, implausible jumps and staleness,
all of which catch *gross* errors, and hands the survivors downstream as though
they were measurements.

They are measurements plus a recall error, and that error has a consequence
nothing in this repo corrects for: it biases every estimated relationship
toward zero, by a known factor, in a known direction.

This is not the missing-data problem. `src/data/imputation_bias.py` handles
values that are absent. This is about values that are present and wrong by a
random amount, which is a different failure with a different fix.

The arithmetic
--------------
Write a reported value as truth plus noise, ``X = T + e``, with ``e``
independent of ``T``. Then a regression of ``Y`` on ``X`` does not recover the
slope of ``Y`` on ``T``. It recovers::

    beta_observed = beta_true * lambda

    lambda = var(T) / (var(T) + var(e))

``lambda`` is the reliability: the share of the reported variance that is real.
It is between zero and one, so the observed slope is always closer to zero than
the truth. Not sometimes, not on average — always, and by exactly that factor.

Two facts make this correctable rather than merely regrettable:

*Repeat measurements identify the error variance.* Two reports of the same
period differ only by noise, so half the variance of their difference is
``var(e)``, and the rest follows.

*A validation subsample identifies it directly.* Regress a trusted value on the
reported one and the slope of that regression **is** ``lambda`` — because
``cov(X, W) = var(T)`` when ``W`` is the truth, and dividing by ``var(X)``
gives the reliability. The calibration equation and the attenuation factor are
the same object seen twice.

Why the direction matters here
------------------------------
The bias runs toward "your action did nothing". An app whose purpose is to show
people their choices matter is systematically understating the effect of those
choices, and understating it most in the categories people recall worst. Of all
the directions this error could take, it has taken the worst one.

Refusals
--------
No disattenuation from a guessed reliability. A guessed ``lambda`` produces a
confidently wrong correction, which is worse than the attenuated estimate it
replaces, because the attenuated estimate at least errs in a known direction.

No correction when the error is differential. The formula above assumes ``e``
is independent of ``T``. People under-report meat and over-report cycling;
where the validation data says the error tracks the truth, the correction does
not apply and returning one anyway is the failure this module exists to
prevent.

No correction factor above a stated ceiling. A reliability near zero divides by
nearly nothing, and the result is arithmetic rather than an estimate.

Self-contained by design: the distribution functions below are duplicated from
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

# Sample requirements -------------------------------------------------------
MIN_REPEAT_PAIRS = 8
MIN_VALIDATION_PAIRS = 10
MIN_REGRESSION_POINTS = 4
MAX_RECORDS = 200000

# Reliability bands ---------------------------------------------------------
# Below the floor the correction is division by nearly nothing.
RELIABILITY_FLOOR = 0.20
POOR_RELIABILITY = 0.50
GOOD_RELIABILITY = 0.80

# Inference -----------------------------------------------------------------
DEFAULT_ALPHA = 0.05
# The differential-error test is one the analysis wants to pass, so it runs at
# a lenient alpha on purpose: a lenient threshold rejects more often, and
# rejecting blocks the correction. The error is pushed toward refusing.
DIFFERENTIAL_ALPHA = 0.10
DEFAULT_CONFIDENCE = 0.95
CONFIDENCE_Z = {0.80: 1.281552, 0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}

# Heaping -------------------------------------------------------------------
ROUND_BASES = (5, 10, 25, 50, 100)
# Whipple's index: 100 means no digit preference, 175 is the conventional
# "rough data" line in demographic practice.
WHIPPLE_ROUGH = 175.0

# SIMEX ---------------------------------------------------------------------
SIMEX_LAMBDAS = (0.5, 1.0, 1.5, 2.0)
SIMEX_REPLICATES = 40


class CalibrationError(ValueError):
    """Raised when a correction cannot be supported by the data supplied."""


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
        raise CalibrationError(
            "Confidence must be one of %s."
            % ", ".join(str(value) for value in sorted(CONFIDENCE_Z))
        )
    return CONFIDENCE_Z[key]


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
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
        if abs(delta - 1.0) < 3.0e-12:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(value: float, degrees_of_freedom: float) -> float:
    """Student-t cumulative distribution."""
    if degrees_of_freedom <= 0:
        raise CalibrationError("Degrees of freedom must be positive.")
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * betai(degrees_of_freedom / 2.0, 0.5, x)
    return 1.0 - tail if value > 0 else tail


def two_sided_p(statistic: float, degrees_of_freedom: float) -> float:
    """Two-sided p-value for a t statistic."""
    if degrees_of_freedom <= 0:
        return 1.0
    return 2.0 * (1.0 - t_cdf(abs(statistic), degrees_of_freedom))


def chi_square_p(statistic: float, degrees_of_freedom: int) -> float:
    """Upper tail of the chi-square distribution, by the regularised gamma.

    Series and continued-fraction expansions, which is the standard pair; the
    series converges for small arguments and the fraction for large ones.
    """
    if degrees_of_freedom <= 0:
        raise CalibrationError("Degrees of freedom must be positive.")
    if statistic <= 0:
        return 1.0

    shape = degrees_of_freedom / 2.0
    x = statistic / 2.0

    if x < shape + 1.0:
        term = 1.0 / shape
        total = term
        power = shape
        for _ in range(500):
            power += 1.0
            term *= x / power
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        lower = total * math.exp(-x + shape * math.log(x) - math.lgamma(shape))
        return max(0.0, min(1.0, 1.0 - lower))

    tiny = 1e-300
    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, 500):
        an = -index * (index - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    upper = h * math.exp(-x + shape * math.log(x) - math.lgamma(shape))
    return max(0.0, min(1.0, upper))


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)


# ---------------------------------------------------------------------------
# Ordinary least squares, because everything below is a slope
# ---------------------------------------------------------------------------


def ols(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    """Simple linear regression with the standard errors.

    Returned rather than imported because the correction below needs the
    slope's variance as well as the slope, and every wrapper in this repo
    returns only the slope.
    """
    if len(xs) != len(ys):
        raise CalibrationError("x and y must be the same length.")
    n = len(xs)
    if n < MIN_REGRESSION_POINTS:
        raise CalibrationError(
            "Need at least %d points to fit a slope and estimate its error."
            % MIN_REGRESSION_POINTS
        )

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        raise CalibrationError(
            "The predictor does not vary, so no slope is identified."
        )

    sxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    residuals = [ys[i] - (intercept + slope * xs[i]) for i in range(n)]
    degrees = n - 2
    residual_variance = (
        sum(value * value for value in residuals) / degrees if degrees > 0 else 0.0
    )
    slope_variance = residual_variance / sxx
    slope_se = math.sqrt(max(slope_variance, 0.0))

    syy = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1.0 - (sum(v * v for v in residuals) / syy) if syy > 0 else 0.0

    statistic = slope / slope_se if slope_se > 0 else 0.0
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "slope_se": slope_se,
        "slope_variance": slope_variance,
        "degrees_of_freedom": degrees,
        "t_statistic": statistic,
        "p_value": two_sided_p(statistic, degrees) if degrees > 0 else 1.0,
        "r_squared": r_squared,
        "residual_variance": residual_variance,
    }


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation."""
    if len(xs) != len(ys):
        raise CalibrationError("x and y must be the same length.")
    if len(xs) < 2:
        raise CalibrationError("Correlation needs at least two points.")
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        raise CalibrationError("A variable does not vary; correlation is undefined.")
    sxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(xs)))
    return sxy / math.sqrt(sxx * syy)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def build_record(
    identifier: Any,
    reported: float,
    repeat: float | None = None,
    validated: float | None = None,
    category: str = "general",
) -> dict[str, Any]:
    """One self-reported value, optionally with a repeat or a trusted value.

    `repeat` is a second recall of the same period — two reports differing only
    by noise. `validated` is a value from a source that does not have recall
    error in it: a utility bill, a receipt, an odometer.

    Both are optional because most records have neither. That is the situation
    the module is built for; the few records that do have one are what make the
    correction possible for all of them.
    """
    value = _finite(reported)
    if value is None:
        raise CalibrationError("Record '%s' has a non-numeric reported value." % identifier)

    repeat_value = None if repeat is None else _finite(repeat)
    if repeat is not None and repeat_value is None:
        raise CalibrationError("Record '%s' has a non-numeric repeat value." % identifier)

    validated_value = None if validated is None else _finite(validated)
    if validated is not None and validated_value is None:
        raise CalibrationError(
            "Record '%s' has a non-numeric validated value." % identifier
        )

    return {
        "id": str(identifier),
        "category": str(category or "general"),
        "reported": value,
        "repeat": repeat_value,
        "validated": validated_value,
        "has_repeat": repeat_value is not None,
        "has_validation": validated_value is not None,
    }


def _validate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise CalibrationError("No records supplied.")
    if len(records) > MAX_RECORDS:
        raise CalibrationError("At most %d records are supported." % MAX_RECORDS)
    cleaned = []
    for record in records:
        if not isinstance(record, Mapping) or "reported" not in record:
            raise CalibrationError("Records must be built with build_record().")
        cleaned.append(dict(record))
    return cleaned


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def reliability_from_repeats(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Error variance from repeat reports of the same period.

    Two reports of one truth differ only by noise, so::

        var(x1 - x2) = 2 * var(e)

    and the true variance is what is left of the total once ``var(e)`` is
    removed. A negative remainder is reported rather than floored: it says the
    within-pair disagreement is larger than the between-unit spread, meaning
    the variable carries no usable signal at all, and that is the finding.
    """
    cleaned = _validate_records(records)
    pairs = [record for record in cleaned if record["has_repeat"]]
    if len(pairs) < MIN_REPEAT_PAIRS:
        raise CalibrationError(
            "Need at least %d records with a repeat measurement to estimate the "
            "error variance; %d supplied. Without repeats there is nothing that "
            "separates noise from real variation."
            % (MIN_REPEAT_PAIRS, len(pairs))
        )

    differences = [record["reported"] - record["repeat"] for record in pairs]
    error_variance = _variance(differences) / 2.0

    everything = [record["reported"] for record in pairs] + [
        record["repeat"] for record in pairs
    ]
    total_variance = _variance(everything)
    true_variance = total_variance - error_variance
    degenerate = true_variance <= 0

    lam = 0.0 if degenerate or total_variance <= 0 else true_variance / total_variance

    # Bias in the mean difference is a separate finding from the variance: it
    # says the two recalls disagree systematically, not just noisily.
    mean_difference = statistics.fmean(differences)
    difference_se = (
        math.sqrt(_variance(differences) / len(differences))
        if len(differences) > 1
        else 0.0
    )
    statistic = mean_difference / difference_se if difference_se > 0 else 0.0

    return {
        "method": "repeats",
        "pairs": len(pairs),
        "reliability": lam,
        "error_variance": error_variance,
        "error_sd": math.sqrt(max(error_variance, 0.0)),
        "true_variance": max(true_variance, 0.0),
        "total_variance": total_variance,
        "raw_true_variance": true_variance,
        "degenerate": degenerate,
        "mean_difference": mean_difference,
        "difference_p": two_sided_p(statistic, len(differences) - 1)
        if len(differences) > 1
        else 1.0,
        # The sampling error of a variance-ratio estimate from n pairs. Used by
        # the delta method below, because a correction that treats lambda as
        # exact understates its own uncertainty.
        "reliability_se": _reliability_se(lam, len(pairs)),
        "note": (
            "Repeat reports disagree by more than the households differ from "
            "each other. This variable has no usable between-unit signal, and "
            "no correction can create one."
            if degenerate
            else "Reliability %.3f from %d repeat pairs: %.0f%% of the reported "
            "spread is real, the rest is recall noise."
            % (lam, len(pairs), lam * 100.0)
        ),
    }


def _reliability_se(lam: float, pairs: int) -> float:
    """Approximate standard error of an intraclass reliability from n pairs.

    The usual large-sample form for a two-rater ICC. Approximate on purpose —
    the delta-method correction below needs an order of magnitude for the
    uncertainty in lambda, not a fourth decimal place, and a more elaborate
    expression would imply a precision the estimate does not have.
    """
    if pairs < 3:
        return 0.0
    return (1.0 - lam * lam) * math.sqrt(2.0 / max(pairs - 2, 1))


def reliability_from_validation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reliability and the calibration equation, from a trusted subsample.

    Regressing the trusted value on the reported one gives a slope that *is*
    the reliability::

        cov(X, W) = cov(T + e, T) = var(T)
        slope     = cov(X, W) / var(X) = var(T) / (var(T) + var(e)) = lambda

    So the calibration equation and the attenuation factor are the same object.
    That is convenient and it is also the reason this route is preferred: one
    fit produces both, and neither depends on the repeats being independent.
    """
    cleaned = _validate_records(records)
    pairs = [record for record in cleaned if record["has_validation"]]
    if len(pairs) < MIN_VALIDATION_PAIRS:
        raise CalibrationError(
            "Need at least %d records with a validated value; %d supplied."
            % (MIN_VALIDATION_PAIRS, len(pairs))
        )

    reported = [record["reported"] for record in pairs]
    validated = [record["validated"] for record in pairs]
    fit = ols(reported, validated)

    lam = fit["slope"]
    return {
        "method": "validation",
        "pairs": len(pairs),
        "reliability": lam,
        "reliability_se": fit["slope_se"],
        "intercept": fit["intercept"],
        "slope": lam,
        "r_squared": fit["r_squared"],
        "residual_variance": fit["residual_variance"],
        "degrees_of_freedom": fit["degrees_of_freedom"],
        "mean_bias": statistics.fmean(
            [record["reported"] - record["validated"] for record in pairs]
        ),
        "out_of_range": not (0.0 < lam <= 1.0),
        "note": (
            "Calibration slope %.3f is outside (0, 1]. A reliability cannot be, "
            "so either the trusted values are not trusted or the error is not "
            "classical. No correction is derived from this fit." % lam
            if not (0.0 < lam <= 1.0)
            else "Reliability %.3f from %d validated records. Calibration: "
            "true = %.2f + %.3f * reported."
            % (lam, len(pairs), fit["intercept"], lam)
        ),
    }


def reliability_band(lam: float) -> str:
    """Plain label for a reliability, on the module's stated cuts."""
    if lam >= GOOD_RELIABILITY:
        return "good"
    if lam >= POOR_RELIABILITY:
        return "usable"
    if lam >= RELIABILITY_FLOOR:
        return "poor"
    return "unusable"


# ---------------------------------------------------------------------------
# The differential-error check, which gates everything else
# ---------------------------------------------------------------------------


def differential_error_test(
    records: Sequence[Mapping[str, Any]],
    alpha: float = DIFFERENTIAL_ALPHA,
) -> dict[str, Any]:
    """Is the recall error independent of the truth, or does it track it?

    Under classical error ``X - W = e`` is independent of ``W``, so regressing
    the error on the truth gives a zero slope. A non-zero slope is differential
    error: people under-report meat and over-report cycling, and the size of
    the misreport depends on how much there was.

    This matters because the attenuation formula is derived from independence.
    Under differential error the observed slope is not ``beta * lambda`` and
    dividing by ``lambda`` does not recover anything. So this runs first, and a
    failure blocks the correction rather than appearing as a caveat under it.
    """
    cleaned = _validate_records(records)
    pairs = [record for record in cleaned if record["has_validation"]]
    if len(pairs) < MIN_VALIDATION_PAIRS:
        raise CalibrationError(
            "The differential-error test needs at least %d validated records."
            % MIN_VALIDATION_PAIRS
        )

    truth = [record["validated"] for record in pairs]
    errors = [record["reported"] - record["validated"] for record in pairs]
    fit = ols(truth, errors)

    differential = fit["p_value"] < alpha
    direction = (
        "over-reported more at high values"
        if fit["slope"] > 0
        else "under-reported more at high values"
    )

    return {
        "pairs": len(pairs),
        "slope": fit["slope"],
        "slope_se": fit["slope_se"],
        "p_value": fit["p_value"],
        "alpha": alpha,
        "classical": not differential,
        "differential": differential,
        "mean_error": statistics.fmean(errors),
        "headline": (
            "Error depends on the true value (slope %.3f, p=%.4f): %s. The "
            "classical attenuation formula does not apply and no disattenuation "
            "is available from this data."
            % (fit["slope"], fit["p_value"], direction)
            if differential
            else "Error looks independent of the true value (slope %.3f, "
            "p=%.3f). Classical attenuation applies."
            % (fit["slope"], fit["p_value"])
        ),
    }


# ---------------------------------------------------------------------------
# Disattenuation
# ---------------------------------------------------------------------------


def disattenuate_slope(
    slope: float,
    slope_se: float,
    lam: float,
    reliability_se: float = 0.0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Undo the attenuation, and widen the interval for having done so.

    The point estimate is ``slope / lambda``. The interval is the part people
    get wrong: dividing the standard error by ``lambda`` treats the reliability
    as exact, and it is itself an estimate. The delta method gives::

        var(beta_c) ~= var(beta)/lambda^2 + beta^2 * var(lambda)/lambda^4

    The second term is often the larger one, which is the honest reason a
    corrected estimate is not simply a better estimate — it is a less biased
    estimate that is less precise, and both halves belong on screen.
    """
    lam = float(lam)
    if lam <= 0:
        raise CalibrationError(
            "A reliability of zero or below means the reported values carry no "
            "signal. There is nothing to correct."
        )
    if lam > 1.0:
        raise CalibrationError(
            "A reliability above one is not possible under classical error "
            "(%.3f supplied). Something upstream is wrong and correcting on it "
            "would propagate the error rather than remove it." % lam
        )
    if lam < RELIABILITY_FLOOR:
        raise CalibrationError(
            "Reliability %.3f is below the floor of %.2f. Dividing by it "
            "multiplies the estimate by more than %.0fx, which is arithmetic "
            "rather than measurement. Report the attenuated estimate and the "
            "reliability instead."
            % (lam, RELIABILITY_FLOOR, 1.0 / RELIABILITY_FLOOR)
        )

    z = _z_for(confidence)
    corrected = slope / lam

    variance = (slope_se**2) / (lam**2)
    if reliability_se > 0:
        variance += (slope**2) * (reliability_se**2) / (lam**4)
    corrected_se = math.sqrt(max(variance, 0.0))

    naive_se = slope_se / lam
    return {
        "observed_slope": slope,
        "corrected_slope": corrected,
        "reliability": lam,
        "correction_factor": 1.0 / lam,
        "attenuation_bias": corrected - slope,
        "corrected_se": corrected_se,
        "naive_corrected_se": naive_se,
        "se_inflation_from_lambda": (
            corrected_se / naive_se if naive_se > 0 else 1.0
        ),
        "lower": corrected - z * corrected_se,
        "upper": corrected + z * corrected_se,
        "confidence": confidence,
        "reliability_band": reliability_band(lam),
        "headline": (
            "Observed slope %.4f is attenuated by a factor of %.2f. The "
            "corrected slope is %.4f — %.0f%% larger — and its interval is "
            "%.1fx wider than dividing the standard error alone would suggest."
            % (
                slope,
                1.0 / lam,
                corrected,
                (abs(corrected) / abs(slope) - 1.0) * 100.0 if slope else 0.0,
                corrected_se / naive_se if naive_se > 0 else 1.0,
            )
        ),
    }


def disattenuate_correlation(
    observed: float,
    reliability_x: float,
    reliability_y: float,
) -> dict[str, Any]:
    """Correct a correlation for error in both variables.

    ``r_true = r_obs / sqrt(lambda_x * lambda_y)``. When both variables are
    self-reported the attenuation compounds, and a relationship reported as
    weak can be strong.

    A corrected value above one is not clamped. It means the reliabilities are
    too low to be consistent with the observed correlation — one of the three
    inputs is wrong — and returning 1.0 would hide that.
    """
    if not -1.0 <= observed <= 1.0:
        raise CalibrationError("An observed correlation must be in [-1, 1].")
    for name, value in (("x", reliability_x), ("y", reliability_y)):
        if not 0.0 < value <= 1.0:
            raise CalibrationError(
                "Reliability for %s must be in (0, 1]; %.3f supplied." % (name, value)
            )

    denominator = math.sqrt(reliability_x * reliability_y)
    corrected = observed / denominator
    impossible = abs(corrected) > 1.0

    return {
        "observed": observed,
        "corrected": corrected,
        "reliability_x": reliability_x,
        "reliability_y": reliability_y,
        "correction_factor": 1.0 / denominator,
        "impossible": impossible,
        "headline": (
            "Corrected correlation %.3f is outside [-1, 1], which is not a "
            "stronger relationship — it is a contradiction. The reliabilities "
            "and the observed correlation cannot all be right." % corrected
            if impossible
            else "Observed %.3f corrects to %.3f once error in both variables "
            "is removed (%.2fx)." % (observed, corrected, 1.0 / denominator)
        ),
    }


# ---------------------------------------------------------------------------
# Regression calibration
# ---------------------------------------------------------------------------


def regression_calibration(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit the trusted value on the reported one, then apply it everywhere.

    The standard approach when you can validate some of the data but not all of
    it, which is exactly this app's situation: a few records have a utility
    bill or a receipt behind them and most have a memory.
    """
    fit = reliability_from_validation(records)
    if fit["out_of_range"]:
        raise CalibrationError(fit["note"])

    return {
        "intercept": fit["intercept"],
        "slope": fit["slope"],
        "reliability": fit["reliability"],
        "reliability_se": fit["reliability_se"],
        "r_squared": fit["r_squared"],
        "pairs": fit["pairs"],
        "residual_sd": math.sqrt(max(fit["residual_variance"], 0.0)),
        "headline": fit["note"],
    }


def apply_calibration(
    calibration: Mapping[str, Any],
    values: Sequence[float],
) -> list[float]:
    """Map reported values onto the calibrated scale."""
    intercept = float(calibration["intercept"])
    slope = float(calibration["slope"])
    calibrated = []
    for value in values:
        number = _finite(value)
        if number is None:
            raise CalibrationError("Cannot calibrate a non-numeric value.")
        calibrated.append(intercept + slope * number)
    return calibrated


# ---------------------------------------------------------------------------
# SIMEX — the fallback where no validation subsample exists
# ---------------------------------------------------------------------------


def simex(
    xs: Sequence[float],
    ys: Sequence[float],
    error_variance: float,
    lambdas: Sequence[float] = SIMEX_LAMBDAS,
    replicates: int = SIMEX_REPLICATES,
    seed: int = 20240311,
) -> dict[str, Any]:
    """Simulation-extrapolation: add more error, watch the slope fall, extrapolate back.

    Add noise of variance ``lambda * var(e)`` to the predictor, refit, and the
    slope decays in a way that can be modelled. Extrapolating that curve to
    ``lambda = -1`` — the hypothetical case of no error at all — recovers the
    unattenuated slope.

    This is the weaker route and is labelled as such. It needs ``var(e)`` from
    somewhere, and if that number is a guess then so is the answer. Where a
    validation subsample exists, `regression_calibration` is strictly better
    because it estimates the error rather than assuming it.
    """
    import random

    if error_variance < 0:
        raise CalibrationError("Error variance cannot be negative.")
    if error_variance == 0:
        raise CalibrationError(
            "With no error variance there is nothing to extrapolate away from. "
            "The observed slope is already the answer."
        )
    if replicates < 2:
        raise CalibrationError("SIMEX needs at least two replicates per level.")

    rng = random.Random(seed)
    naive = ols(xs, ys)

    points = [(0.0, naive["slope"])]
    for level in lambdas:
        if level <= 0:
            raise CalibrationError("SIMEX levels must be positive.")
        scale = math.sqrt(level * error_variance)
        slopes = []
        for _ in range(int(replicates)):
            noisy = [x + rng.gauss(0.0, scale) for x in xs]
            slopes.append(ols(noisy, ys)["slope"])
        points.append((level, statistics.fmean(slopes)))

    # Quadratic in lambda, evaluated at -1. A quadratic is the conventional
    # extrapolant: linear understates the correction and anything higher-order
    # fits the simulation noise.
    coefficients = _quadratic_fit(
        [point[0] for point in points], [point[1] for point in points]
    )
    extrapolated = (
        coefficients[0] + coefficients[1] * (-1.0) + coefficients[2] * 1.0
    )

    return {
        "naive_slope": naive["slope"],
        "corrected_slope": extrapolated,
        "correction_factor": (
            extrapolated / naive["slope"] if naive["slope"] else float("nan")
        ),
        "curve": [{"lambda": level, "slope": slope} for level, slope in points],
        "coefficients": list(coefficients),
        "replicates": int(replicates),
        "error_variance": error_variance,
        "headline": (
            "SIMEX moves the slope from %.4f to %.4f. This assumes the error "
            "variance supplied (%.2f) is right; where it is a guess, so is the "
            "correction." % (naive["slope"], extrapolated, error_variance)
        ),
    }


def _quadratic_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """Least squares fit of ``a + b x + c x^2`` by normal equations."""
    n = len(xs)
    if n < 3:
        raise CalibrationError("A quadratic needs at least three points.")

    sums = [sum(x**power for x in xs) for power in range(5)]
    rhs = [
        sum(ys),
        sum(xs[i] * ys[i] for i in range(n)),
        sum(xs[i] ** 2 * ys[i] for i in range(n)),
    ]
    matrix = [
        [sums[0], sums[1], sums[2]],
        [sums[1], sums[2], sums[3]],
        [sums[2], sums[3], sums[4]],
    ]
    return tuple(_solve3(matrix, rhs))


def _solve3(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting on a 3x3 system."""
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise CalibrationError("Extrapolation system is singular.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, 3):
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, 4):
                augmented[row][k] -= factor * augmented[column][k]

    solution = [0.0, 0.0, 0.0]
    for row in (2, 1, 0):
        total = augmented[row][3] - sum(
            augmented[row][k] * solution[k] for k in range(row + 1, 3)
        )
        solution[row] = total / augmented[row][row]
    return solution


# ---------------------------------------------------------------------------
# Heaping and digit preference
# ---------------------------------------------------------------------------


def heaping_diagnostics(values: Sequence[float]) -> dict[str, Any]:
    """Is the reported precision real?

    Self-reported mileage clusters on multiples of 50; hours cluster on
    multiples of 5. `detect_outliers()` in `data_quality.py` sees nothing wrong
    with a heaped distribution because nothing is wrong with any individual
    value — the problem is in the joint distribution of the last digit.

    Whipple's index is the demographic convention: the share of values ending
    in 0 or 5 against the 20% expected under no preference, times 500. 100 is
    clean, 175 is the conventional line for rough data, 500 is total heaping.
    """
    numbers = []
    for value in values:
        number = _finite(value)
        if number is None:
            raise CalibrationError("Cannot diagnose heaping on non-numeric values.")
        numbers.append(number)

    if len(numbers) < MIN_REPEAT_PAIRS:
        raise CalibrationError(
            "Need at least %d values to say anything about digit preference."
            % MIN_REPEAT_PAIRS
        )

    integers = [int(round(number)) for number in numbers]
    digits = [abs(number) % 10 for number in integers]
    counts = [digits.count(digit) for digit in range(10)]
    expected = len(digits) / 10.0

    statistic = sum((count - expected) ** 2 / expected for count in counts)
    p_value = chi_square_p(statistic, 9)

    zero_or_five = counts[0] + counts[5]
    whipple = zero_or_five / len(digits) * 500.0

    multiples = {}
    for base in ROUND_BASES:
        share = sum(1 for number in integers if number % base == 0) / len(integers)
        multiples[base] = share

    heaped = p_value < DEFAULT_ALPHA and whipple > WHIPPLE_ROUGH
    effective_base = 1
    for base in ROUND_BASES:
        if multiples[base] > 0.80:
            effective_base = base

    return {
        "n": len(integers),
        "digit_counts": counts,
        "chi_square": statistic,
        "p_value": p_value,
        "whipple_index": whipple,
        "round_multiple_share": multiples,
        "heaped": heaped,
        "effective_precision": effective_base,
        "headline": (
            "Whipple index %.0f (p=%.4f): the values are heaped on round "
            "numbers. Reported precision is fictional; the effective precision "
            "is the nearest %d." % (whipple, p_value, effective_base)
            if heaped
            else "Whipple index %.0f (p=%.3f) — no material digit preference."
            % (whipple, p_value)
        ),
    }


# ---------------------------------------------------------------------------
# What error does to a total, as opposed to a slope
# ---------------------------------------------------------------------------


def propagate_to_total(
    components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Random error and systematic bias behave differently in a sum.

    Attenuation is a statement about slopes. For the footprint *total* the
    arithmetic is different and the difference is routinely confused: random
    errors partly cancel across categories, so the relative error of the total
    falls roughly as one over the square root of the number of components,
    while a systematic bias in each category adds up undiminished.

    So a footprint built from ten noisily-recalled categories can have a
    perfectly usable total and be useless for any comparison between
    categories — and if every category is under-reported by 15%, the total is
    under-reported by 15% no matter how many categories there are.
    """
    if not components:
        raise CalibrationError("No components supplied.")

    total = 0.0
    variance = 0.0
    bias = 0.0
    rows = []
    for component in components:
        value = _finite(component.get("value"))
        if value is None:
            raise CalibrationError("Component '%s' has no numeric value." % component)
        error_sd = _finite(component.get("error_sd", 0.0)) or 0.0
        if error_sd < 0:
            raise CalibrationError("An error standard deviation cannot be negative.")
        component_bias = _finite(component.get("bias", 0.0)) or 0.0

        total += value
        variance += error_sd**2
        bias += component_bias
        rows.append(
            {
                "name": str(component.get("name", "component")),
                "value": value,
                "error_sd": error_sd,
                "bias": component_bias,
            }
        )

    random_sd = math.sqrt(variance)
    summed_sd = sum(row["error_sd"] for row in rows)

    return {
        "components": rows,
        "total": total,
        "random_sd": random_sd,
        "sum_of_component_sds": summed_sd,
        "cancellation_factor": (random_sd / summed_sd) if summed_sd > 0 else 1.0,
        "systematic_bias": bias,
        "corrected_total": total - bias,
        "relative_random": (random_sd / total) if total else 0.0,
        "relative_bias": (bias / total) if total else 0.0,
        "bias_dominates": abs(bias) > random_sd,
        "headline": (
            "Total %.0f. Random error contributes +/- %.0f — only %.0f%% of the "
            "%.0f you would get by adding the component errors, because they "
            "partly cancel. Systematic bias contributes %.0f and does not "
            "cancel at all, %s."
            % (
                total,
                random_sd,
                (random_sd / summed_sd * 100.0) if summed_sd > 0 else 100.0,
                summed_sd,
                bias,
                "which makes it the larger problem here"
                if abs(bias) > random_sd
                else "though it is the smaller of the two here",
            )
        ),
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def analyse(
    records: Sequence[Mapping[str, Any]],
    slope: float | None = None,
    slope_se: float = 0.0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Everything the supplied records can support, and nothing more.

    The order is deliberate. The differential-error test runs before any
    correction, because a correction derived from a formula whose assumption
    has failed is worse than no correction — it is wrong in an unknown
    direction rather than a known one.
    """
    cleaned = _validate_records(records)
    reported = [record["reported"] for record in cleaned]

    result: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "records": len(cleaned),
        "with_repeat": sum(1 for record in cleaned if record["has_repeat"]),
        "with_validation": sum(1 for record in cleaned if record["has_validation"]),
        "confidence": confidence,
        "repeats": None,
        "validation": None,
        "differential": None,
        "calibration": None,
        "correction": None,
        "heaping": None,
        "blocked": None,
    }

    try:
        result["heaping"] = heaping_diagnostics(reported)
    except CalibrationError:
        result["heaping"] = None

    try:
        result["repeats"] = reliability_from_repeats(cleaned)
    except CalibrationError:
        result["repeats"] = None

    try:
        result["validation"] = reliability_from_validation(cleaned)
    except CalibrationError:
        result["validation"] = None

    if result["validation"] is not None:
        try:
            result["differential"] = differential_error_test(cleaned)
        except CalibrationError:
            result["differential"] = None

    # Validation is preferred over repeats: it estimates the error against a
    # known truth rather than against a second guess, and repeats share any
    # bias the respondent carries between the two occasions.
    chosen = None
    if result["validation"] is not None and not result["validation"]["out_of_range"]:
        chosen = result["validation"]
    elif result["repeats"] is not None and not result["repeats"]["degenerate"]:
        chosen = result["repeats"]

    if chosen is None:
        result["blocked"] = (
            "No usable reliability estimate. Without repeat or validated "
            "measurements the attenuation cannot be quantified, and a guessed "
            "reliability would produce a confidently wrong correction."
        )
        result["headline"] = result["blocked"]
        return result

    result["source"] = chosen["method"]
    result["reliability"] = chosen["reliability"]

    if result["differential"] is not None and result["differential"]["differential"]:
        result["blocked"] = result["differential"]["headline"]
        result["headline"] = result["blocked"]
        return result

    if chosen["method"] == "validation":
        try:
            result["calibration"] = regression_calibration(cleaned)
        except CalibrationError as error:
            result["calibration"] = None
            result["blocked"] = str(error)

    if slope is not None:
        try:
            result["correction"] = disattenuate_slope(
                slope,
                slope_se,
                chosen["reliability"],
                reliability_se=chosen.get("reliability_se", 0.0),
                confidence=confidence,
            )
        except CalibrationError as error:
            result["correction"] = None
            result["blocked"] = str(error)

    result["headline"] = (
        result["blocked"]
        if result["blocked"]
        else "Reliability %.3f from %d %s records (%s). Every slope estimated on "
        "this variable is attenuated by a factor of %.2f."
        % (
            chosen["reliability"],
            chosen["pairs"],
            chosen["method"],
            reliability_band(chosen["reliability"]),
            1.0 / chosen["reliability"] if chosen["reliability"] > 0 else float("inf"),
        )
    )
    return result


# ---------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------


def get_measurement_notes(result: Mapping[str, Any]) -> list[str]:
    """Plain-language readings of an analysis."""
    notes: list[str] = [result.get("headline", "")]

    if result.get("blocked"):
        notes.append(
            "No corrected estimate is reported. That is the result, not a "
            "missing result — the alternative is a number with no basis."
        )

    heaping = result.get("heaping")
    if heaping and heaping["heaped"]:
        notes.append(
            "Values are heaped on multiples of %d. Reported precision beyond "
            "that is invented, and any threshold sitting between two heaps will "
            "classify almost nobody." % heaping["effective_precision"]
        )

    repeats = result.get("repeats")
    if repeats and repeats.get("difference_p", 1.0) < DEFAULT_ALPHA:
        notes.append(
            "The two recalls disagree systematically, not just noisily (mean "
            "difference %.2f, p=%.4f). That is a bias between occasions, and "
            "the reliability estimate assumes there is none."
            % (repeats["mean_difference"], repeats["difference_p"])
        )

    validation = result.get("validation")
    if validation and abs(validation.get("mean_bias", 0.0)) > 0:
        notes.append(
            "Reported values sit %.2f from the trusted ones on average. "
            "Attenuation is about the spread; this is about the level, and the "
            "two need separate corrections." % validation["mean_bias"]
        )

    correction = result.get("correction")
    if correction:
        notes.append(
            "Correcting the slope multiplies it by %.2f and widens its interval "
            "by %.1fx beyond the naive division. A corrected estimate is less "
            "biased and less precise; both are true."
            % (
                correction["correction_factor"],
                correction["se_inflation_from_lambda"],
            )
        )
        if correction["reliability_band"] in ("poor", "unusable"):
            notes.append(
                "Reliability is in the '%s' band. The correction is large, and "
                "a large correction rests entirely on the reliability estimate "
                "being right." % correction["reliability_band"]
            )

    if result.get("with_validation", 0) == 0 and result.get("with_repeat", 0) > 0:
        notes.append(
            "This used repeat reports rather than validated values. Repeats "
            "share whatever bias the respondent carries between occasions, so "
            "they understate the error; a utility bill or a receipt would give "
            "a lower and more honest reliability."
        )

    return [note for note in notes if note]


def summarise(result: Mapping[str, Any]) -> str:
    """One line for a log or a saved-analysis list."""
    if result.get("blocked"):
        return "blocked: %s" % result["blocked"][:80]
    lam = result.get("reliability")
    return "%d records | %s | lambda %.3f | correction %.2fx" % (
        result.get("records", 0),
        result.get("source", "none"),
        lam if lam is not None else 0.0,
        (1.0 / lam) if lam else 0.0,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _connect():
    return sqlite3.connect(DB_NAME)


def _ensure_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS measurement_error_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            label TEXT NOT NULL,
            category TEXT NOT NULL,
            reliability REAL,
            method TEXT NOT NULL,
            blocked INTEGER NOT NULL DEFAULT 0,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_measurement_error_user
        ON measurement_error_analyses (user_id)
        """
    )


def save_analysis(
    user_id: Any,
    result: Mapping[str, Any],
    label: str = "",
    category: str = "general",
) -> int | None:
    """Persist an analysis. None if storage is unavailable."""
    if not user_id or not result.get("engine_version"):
        return None
    try:
        with _connect() as conn:
            _ensure_tables(conn)
            cursor = conn.execute(
                """
                INSERT INTO measurement_error_analyses
                    (user_id, label, category, reliability, method, blocked, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_id),
                    str(label or "analysis"),
                    str(category or "general"),
                    result.get("reliability"),
                    str(result.get("source", "none")),
                    1 if result.get("blocked") else 0,
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
                SELECT id, label, category, reliability, method, blocked, payload,
                       created_at
                FROM measurement_error_analyses
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
            payload = json.loads(row[6])
        except (TypeError, ValueError):
            payload = {}
        analyses.append(
            {
                "id": row[0],
                "label": row[1],
                "category": row[2],
                "reliability": row[3],
                "method": row[4],
                "blocked": bool(row[5]),
                "payload": payload,
                "created_at": row[7],
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
                "DELETE FROM measurement_error_analyses WHERE user_id = ? AND id = ?",
                (str(user_id), int(analysis_id)),
            )
            return cursor.rowcount > 0
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------------------
# Worked examples
# ---------------------------------------------------------------------------


def demo_records(
    count: int = 200,
    true_sd: float = 400.0,
    true_mean: float = 1200.0,
    error_sd: float = 300.0,
    validation_share: float = 0.20,
    repeat_share: float = 0.25,
    differential_slope: float = 0.0,
    heap_to: int = 0,
    seed: int = 20240311,
) -> list[dict[str, Any]]:
    """A realistic collection: mostly recall, a little of it checkable.

    `differential_slope` makes the error depend on the truth, which is what
    under-reporting meat looks like. `heap_to` rounds the reported values onto
    a grid, which is what recalled mileage looks like.
    """
    import random

    rng = random.Random(seed)
    records = []
    for index in range(int(count)):
        truth = rng.gauss(true_mean, true_sd)
        error = rng.gauss(0.0, error_sd) + differential_slope * (truth - true_mean)
        reported = truth + error
        if heap_to > 1:
            reported = round(reported / heap_to) * heap_to

        repeat = None
        if rng.random() < repeat_share:
            repeat = truth + rng.gauss(0.0, error_sd)
            if heap_to > 1:
                repeat = round(repeat / heap_to) * heap_to

        validated = truth if rng.random() < validation_share else None

        records.append(
            build_record(
                "r%04d" % index,
                reported,
                repeat=repeat,
                validated=validated,
                category="energy_kwh",
            )
        )
    return records


def demo_regression(
    records: Sequence[Mapping[str, Any]],
    true_slope: float = 0.5,
    outcome_noise: float = 50.0,
    seed: int = 8812,
) -> dict[str, Any]:
    """An outcome generated from the *truth*, so the attenuation is checkable.

    The reported predictor is what the app has. The true predictor is what
    generated the outcome. The gap between the two fitted slopes is exactly the
    bias this module exists to remove, and because the data generating process
    is known here, the correction can be checked against the answer.
    """
    import random

    rng = random.Random(seed)
    usable = [record for record in records if record["validated"] is not None]
    if len(usable) < MIN_REGRESSION_POINTS:
        raise CalibrationError("Not enough validated records to build the example.")

    truth = [record["validated"] for record in usable]
    reported = [record["reported"] for record in usable]
    outcome = [value * true_slope + rng.gauss(0.0, outcome_noise) for value in truth]

    on_truth = ols(truth, outcome)
    on_reported = ols(reported, outcome)
    return {
        "true_slope": true_slope,
        "fit_on_truth": on_truth,
        "fit_on_reported": on_reported,
        "attenuation_observed": (
            on_reported["slope"] / on_truth["slope"] if on_truth["slope"] else 0.0
        ),
        "reported": reported,
        "truth": truth,
        "outcome": outcome,
    }


def attenuation_table(
    slope: float = 1.0,
    reliabilities: Sequence[float] = (
        0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10,
    ),
) -> list[dict[str, Any]]:
    """What each reliability costs, as a table, for the page and the docs."""
    rows = []
    for lam in reliabilities:
        rows.append(
            {
                "reliability": lam,
                "observed_slope": slope * lam,
                "understatement": (1.0 - lam) * 100.0,
                "correction_factor": 1.0 / lam,
                "band": reliability_band(lam),
                "correctable": lam >= RELIABILITY_FLOOR,
            }
        )
    return rows


def make_component(
    name: str,
    value: float,
    error_sd: float = 0.0,
    bias: float = 0.0,
) -> dict[str, Any]:
    """One line of a footprint total, with its error and its bias."""
    return {
        "name": str(name),
        "value": float(value),
        "error_sd": float(error_sd),
        "bias": float(bias),
    }
