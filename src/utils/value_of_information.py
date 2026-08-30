"""Which uncertainty is worth resolving, as opposed to which one is largest.

`src/utils/global_sensitivity.py` answers "which parameter drives the spread in
the output" and answers it correctly. `src/utils/footprint_uncertainty.py`
reports how wide the answer is. Between them the app can tell a user that grid
intensity accounts for 60% of the variance in their footprint.

That is true, and it is not the question anyone is actually asking, which is:
**should I go and find out?**

Those are different questions with different answers, and the gap is not small.
A parameter can account for 70% of the output variance and be worth nothing to
measure, because the decision it feeds is the same across its whole plausible
range. A parameter can account for 3% of the variance and be worth a great
deal, because the decision flips somewhere inside that 3%.

Variance-based sensitivity ranks parameters by how much they move the *number*.
Value of information ranks them by how much they move the *decision*, and only
the second one tells anyone what to do next.

The structural difference
-------------------------
This module takes a **decision** — a set of options and a payoff — rather than
a model output. That is not a detail. Value of information is undefined without
a decision, and the temptation to compute it against a bare model output is
exactly the confusion the module exists to remove. So `analyse()` requires
options and refuses without them.

The quantities
--------------
``EVPI``   the value of resolving *all* uncertainty. An upper bound on every
           possible study, survey and sensor. Where it is below the cost of the
           cheapest measurement the discussion ends, and "act on what you have"
           is the correct output.

``EVPPI``  the value of resolving *one* parameter while the rest stay
           uncertain. This is the ranking that answers "what should I measure
           next", and it is routinely a different order from the variance
           ranking.

``EVSI``   the value of a *feasible* study rather than a perfect one — three
           months of meter readings, not omniscience. Bounded above by EVPPI,
           and the number a real decision needs.

``ENBS``   EVSI minus what the study costs, across sample sizes. Where the
           optimum is negative, the honest recommendation is not to collect the
           data.

All three are estimated by the same move: partition the draws on what the study
would reveal, take the best option *within* each partition, and average. What
you gain is the difference between choosing after seeing and choosing now.

Refusals
--------
No VOI without an explicit decision. No EVSI above its EVPPI and no EVPPI above
the EVPI — both are guaranteed by theory, so a violation is an estimation
failure and is reported as one rather than returned as a result. No
recommendation to collect data whose expected net benefit is negative.

Self-contained by design: nothing here imports the sensitivity module it
contrasts itself with, so both keep working independently.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
from typing import Any, Callable, Mapping, Sequence

DB_NAME = os.getenv("ECO_BUDDY_DB", "eco_buddy.db")

ENGINE_VERSION = "1.0.0"

# Requirements --------------------------------------------------------------
MIN_OPTIONS = 2
MIN_DRAWS = 200
MAX_DRAWS = 200000
MAX_OPTIONS = 50
MAX_PARAMETERS = 40

# Estimation ----------------------------------------------------------------
DEFAULT_DRAWS = 4000
# Partitions used to estimate a conditional expectation. Too few and the
# estimate is biased downward; too many and each bin is noise. Thirty is the
# usual working range for the single-parameter method.
DEFAULT_BINS = 30
# EVPPI and EVSI are non-negative by construction and estimated with noise, so
# small negative values are rounded to zero. Anything larger than this is an
# estimation failure and is reported rather than absorbed.
ESTIMATION_TOLERANCE = 0.02
# The max-of-conditional-means estimator is biased *upward* with finite bins:
# each bin's maximum picks up some of the sampling noise in that bin's means.
# So a parameter with a true EVPPI of zero estimates as a small positive
# number, and "resolving this cannot change the choice" has to be a claim about
# a share of EVPI rather than about an exact zero.
NEGLIGIBLE_SHARE_OF_EVPI = 0.10
# A parameter has to carry at least this much of the payoff variance before the
# gap between the two rankings is worth pointing at.
LOUD_VARIANCE_SHARE = 0.25

DISTRIBUTIONS = ("normal", "uniform", "triangular", "lognormal")

# EVSI needs a conjugate prior to update. Normal is the only one implemented,
# and asking for another is refused rather than approximated.
EVSI_DISTRIBUTIONS = ("normal",)


class VOIError(ValueError):
    """Raised when a value-of-information question cannot be answered as asked."""


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


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def build_parameter(
    name: str,
    distribution: str = "normal",
    mean: float = 0.0,
    sd: float = 1.0,
    low: float | None = None,
    high: float | None = None,
    mode: float | None = None,
) -> dict[str, Any]:
    """One uncertain input, with the distribution the caller believes about it.

    The distribution matters less than it looks. EVPPI depends on where the
    decision boundary falls relative to the mass, not on the exact shape, and a
    parameter whose whole range sits on one side of the boundary is worth
    nothing to measure however it is distributed.
    """
    if distribution not in DISTRIBUTIONS:
        raise VOIError(
            "Distribution must be one of %s." % ", ".join(DISTRIBUTIONS)
        )

    if distribution == "normal":
        value = _finite(sd)
        if value is None or value < 0:
            raise VOIError("Parameter '%s' needs a non-negative sd." % name)
        spec = {"mean": float(mean), "sd": value}
    elif distribution == "lognormal":
        value = _finite(sd)
        if value is None or value < 0:
            raise VOIError("Parameter '%s' needs a non-negative log sd." % name)
        spec = {"mean": float(mean), "sd": value}
    elif distribution == "uniform":
        if low is None or high is None or low >= high:
            raise VOIError("Parameter '%s' needs low < high." % name)
        spec = {"low": float(low), "high": float(high)}
    else:  # triangular
        if low is None or high is None or low >= high:
            raise VOIError("Parameter '%s' needs low < high." % name)
        peak = float(high + low) / 2.0 if mode is None else float(mode)
        if not low <= peak <= high:
            raise VOIError("Parameter '%s' needs low <= mode <= high." % name)
        spec = {"low": float(low), "high": float(high), "mode": peak}

    return {
        "name": str(name),
        "distribution": distribution,
        **spec,
        "updatable": distribution in EVSI_DISTRIBUTIONS,
    }


def sample_parameter(parameter: Mapping[str, Any], rng) -> float:
    """One draw."""
    kind = parameter["distribution"]
    if kind == "normal":
        return rng.gauss(parameter["mean"], parameter["sd"])
    if kind == "lognormal":
        return math.exp(rng.gauss(parameter["mean"], parameter["sd"]))
    if kind == "uniform":
        return rng.uniform(parameter["low"], parameter["high"])
    return rng.triangular(parameter["low"], parameter["high"], parameter["mode"])


def _validate_parameters(
    parameters: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not parameters:
        raise VOIError("At least one uncertain parameter is needed.")
    if len(parameters) > MAX_PARAMETERS:
        raise VOIError("At most %d parameters are supported." % MAX_PARAMETERS)
    names: set[str] = set()
    cleaned = []
    for parameter in parameters:
        if not isinstance(parameter, Mapping) or "distribution" not in parameter:
            raise VOIError("Parameters must be built with build_parameter().")
        if parameter["name"] in names:
            raise VOIError("Parameter '%s' appears twice." % parameter["name"])
        names.add(parameter["name"])
        cleaned.append(dict(parameter))
    return cleaned


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def build_option(
    name: str,
    payoff: Callable[[Mapping[str, float]], float],
    cost: float = 0.0,
) -> dict[str, Any]:
    """One choice, and what it is worth given the parameters.

    `payoff` is a callable so the module works on the app's real decisions —
    an abatement measure whose cost per tonne is a nonlinear function of the
    grid, a replacement date chosen by backward induction — rather than only on
    a linear scoring sheet.

    Higher payoff is better. Where the natural quantity is a cost, negate it,
    and say so in the name.
    """
    if not callable(payoff):
        raise VOIError("Option '%s' needs a callable payoff." % name)
    cost_value = _finite(cost)
    if cost_value is None:
        raise VOIError("Option '%s' has a non-numeric cost." % name)
    return {"name": str(name), "payoff": payoff, "cost": cost_value}


def _validate_options(options: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(options) < MIN_OPTIONS:
        raise VOIError(
            "Value of information is undefined without a decision. At least %d "
            "options are needed; a single option is not a choice, and there is "
            "nothing information could change." % MIN_OPTIONS
        )
    if len(options) > MAX_OPTIONS:
        raise VOIError("At most %d options are supported." % MAX_OPTIONS)
    names: set[str] = set()
    cleaned = []
    for option in options:
        if not isinstance(option, Mapping) or "payoff" not in option:
            raise VOIError("Options must be built with build_option().")
        if option["name"] in names:
            raise VOIError("Option '%s' appears twice." % option["name"])
        names.add(option["name"])
        cleaned.append(dict(option))
    return cleaned


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate(
    options: Sequence[Mapping[str, Any]],
    parameters: Sequence[Mapping[str, Any]],
    draws: int = DEFAULT_DRAWS,
    seed: int = 20240915,
) -> dict[str, Any]:
    """Draw the parameters, evaluate every option on every draw.

    Everything below reads this one matrix. Sharing the draws across options is
    not an optimisation — it is required, because the comparison between two
    options must hold the state of the world fixed.
    """
    import random

    cleaned_options = _validate_options(options)
    cleaned_parameters = _validate_parameters(parameters)
    if draws < MIN_DRAWS:
        raise VOIError(
            "At least %d draws are needed; the estimator partitions them and a "
            "small sample makes every partition noise." % MIN_DRAWS
        )
    if draws > MAX_DRAWS:
        raise VOIError("At most %d draws are supported." % MAX_DRAWS)

    rng = random.Random(seed)
    names = [parameter["name"] for parameter in cleaned_parameters]
    samples: dict[str, list[float]] = {name: [] for name in names}
    payoffs: list[list[float]] = []

    for _ in range(int(draws)):
        state = {}
        for parameter in cleaned_parameters:
            value = sample_parameter(parameter, rng)
            state[parameter["name"]] = value
            samples[parameter["name"]].append(value)

        row = []
        for option in cleaned_options:
            value = _finite(option["payoff"](state))
            if value is None:
                raise VOIError(
                    "Option '%s' returned a non-numeric payoff." % option["name"]
                )
            row.append(value - option["cost"])
        payoffs.append(row)

    return {
        "options": [option["name"] for option in cleaned_options],
        "parameters": names,
        "samples": samples,
        "payoffs": payoffs,
        "draws": int(draws),
    }


# ---------------------------------------------------------------------------
# The decision as it stands
# ---------------------------------------------------------------------------


def baseline_decision(simulation: Mapping[str, Any]) -> dict[str, Any]:
    """What to choose now, and how likely that is to be wrong.

    Two numbers that routinely disagree and are both needed. The probability of
    being wrong says how often; the expected opportunity loss says how much it
    costs when it happens. An option that is second-best by a rounding error
    ninety percent of the time is not a mistake worth avoiding, and an app that
    reports only the probability cannot say so.
    """
    payoffs = simulation["payoffs"]
    names = simulation["options"]
    draws = len(payoffs)

    means = [
        statistics.fmean([row[index] for row in payoffs]) for index in range(len(names))
    ]
    best_index = max(range(len(names)), key=lambda index: means[index])

    wins = [0] * len(names)
    regret_total = 0.0
    for row in payoffs:
        best = max(range(len(names)), key=lambda index: row[index])
        wins[best] += 1
        regret_total += row[best] - row[best_index]

    return {
        "options": names,
        "expected_payoffs": means,
        "recommended": names[best_index],
        "recommended_index": best_index,
        "expected_payoff": means[best_index],
        "probability_best": [count / draws for count in wins],
        "probability_recommended_is_best": wins[best_index] / draws,
        "probability_wrong": 1.0 - wins[best_index] / draws,
        "expected_opportunity_loss": regret_total / draws,
        "headline": (
            "Choose %s. It is the best option on %.0f%% of draws, and the "
            "expected cost of being wrong is %.4g."
            % (
                names[best_index],
                wins[best_index] / draws * 100.0,
                regret_total / draws,
            )
        ),
    }


# ---------------------------------------------------------------------------
# EVPI
# ---------------------------------------------------------------------------


def evpi(simulation: Mapping[str, Any]) -> dict[str, Any]:
    """The value of resolving everything. An upper bound on every possible study.

    ``E[max_d NB] - max_d E[NB]``: choosing after seeing the world, minus
    choosing before. It is exactly the expected opportunity loss of the current
    recommendation, which is why the two are reported together — a reader who
    finds that surprising has understood something useful about both.

    Where this is below the cost of the cheapest measurement, no study is worth
    running and the correct output is "act on what you have", stated plainly
    rather than as an absence.
    """
    payoffs = simulation["payoffs"]
    names = simulation["options"]
    draws = len(payoffs)

    means = [
        statistics.fmean([row[index] for row in payoffs]) for index in range(len(names))
    ]
    best_now = max(means)
    best_after = statistics.fmean([max(row) for row in payoffs])
    value = max(0.0, best_after - best_now)

    return {
        "evpi": value,
        "expected_payoff_now": best_now,
        "expected_payoff_with_perfect_information": best_after,
        "draws": draws,
        "headline": (
            "Resolving every uncertainty is worth %.4g. That is the ceiling on "
            "every study, survey and sensor that could ever be run on this "
            "decision." % value
        ),
    }


# ---------------------------------------------------------------------------
# EVPPI
# ---------------------------------------------------------------------------


def _conditional_value(
    conditioning: Sequence[float],
    payoffs: Sequence[Sequence[float]],
    bins: int,
) -> float:
    """``E_x[max_d E[NB | x]]``, by partitioning on the conditioning variable.

    Sort on the variable, split into bins, take the mean payoff of each option
    within a bin, and keep the best. Bin boundaries are extended to keep tied
    values together: a tie split across two bins makes the conditional means
    differ for a reason that has nothing to do with the variable, which inflates
    the estimate.

    The estimator is biased upward and it is worth being explicit about why: the
    maximum within each bin picks up that bin's sampling noise as well as its
    signal. A parameter whose true EVPPI is zero therefore estimates as a small
    positive number, shrinking as the draws grow. That is why "resolving this
    cannot change the choice" is expressed here as a share of EVPI rather than
    as an exact zero — an exact-zero test would never fire on real output.
    """
    count = len(conditioning)
    option_count = len(payoffs[0])
    order = sorted(range(count), key=lambda index: conditioning[index])
    bucket_count = max(1, min(int(bins), count // 10))
    size = count / bucket_count

    total = 0.0
    covered = 0
    start = 0
    for bucket in range(bucket_count):
        if start >= count:
            break
        end = int(round((bucket + 1) * size))
        end = max(end, start + 1)
        while end < count and conditioning[order[end]] == conditioning[order[end - 1]]:
            end += 1
        members = order[start:end]
        means = [
            statistics.fmean([payoffs[index][option] for index in members])
            for option in range(option_count)
        ]
        total += max(means) * len(members)
        covered += len(members)
        start = end

    return total / covered if covered else 0.0


def evppi(
    simulation: Mapping[str, Any],
    parameter: str,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """The value of resolving one parameter while the rest stay uncertain.

    This is the ranking that answers "what should I measure next", and it is
    routinely a different order from the variance ranking. A parameter can
    dominate the output variance and have an EVPPI of zero, because moving the
    number is not the same as moving the choice.
    """
    if parameter not in simulation["samples"]:
        raise VOIError("Parameter '%s' is not in this simulation." % parameter)

    payoffs = simulation["payoffs"]
    names = simulation["options"]
    means = [
        statistics.fmean([row[index] for row in payoffs]) for index in range(len(names))
    ]
    best_now = max(means)

    conditional = _conditional_value(simulation["samples"][parameter], payoffs, bins)
    raw = conditional - best_now
    ceiling = evpi(simulation)["evpi"]

    return {
        "parameter": parameter,
        "evppi": max(0.0, raw),
        "raw": raw,
        "share_of_evpi": (max(0.0, raw) / ceiling) if ceiling > 0 else 0.0,
        "evpi": ceiling,
        "bins": bins,
        # Both bounds hold by theory, so a violation beyond the tolerance is an
        # estimation failure and is flagged rather than absorbed.
        "below_zero": raw < -ESTIMATION_TOLERANCE * max(ceiling, 1e-9),
        "above_evpi": raw > ceiling * (1.0 + ESTIMATION_TOLERANCE) + 1e-9,
        "headline": (
            "Resolving '%s' alone is worth %.4g, which is %.0f%% of the value "
            "of resolving everything."
            % (
                parameter,
                max(0.0, raw),
                ((max(0.0, raw) / ceiling) * 100.0) if ceiling > 0 else 0.0,
            )
        ),
    }


def evppi_ranking(
    simulation: Mapping[str, Any],
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Every parameter, ordered by what resolving it is worth."""
    entries = [
        evppi(simulation, name, bins=bins) for name in simulation["parameters"]
    ]
    entries.sort(key=lambda entry: entry["evppi"], reverse=True)
    failures = [
        entry["parameter"]
        for entry in entries
        if entry["below_zero"] or entry["above_evpi"]
    ]
    return {
        "entries": entries,
        "order": [entry["parameter"] for entry in entries],
        "estimation_failures": failures,
        "headline": (
            "EVPPI estimates for %s fall outside the [0, EVPI] bounds that hold "
            "by theory. That is an estimation failure — more draws or fewer "
            "bins — not a finding." % ", ".join(failures)
            if failures
            else "Measuring '%s' is worth the most; measuring '%s' is worth the "
            "least."
            % (entries[0]["parameter"], entries[-1]["parameter"])
            if entries
            else "No parameters."
        ),
    }


# ---------------------------------------------------------------------------
# Variance sensitivity, for the contrast
# ---------------------------------------------------------------------------


def variance_ranking(
    simulation: Mapping[str, Any],
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """First-order variance share in the payoff of the currently chosen option.

    Included so the two rankings can be put side by side. This is the ordering
    a reader would naturally take from `global_sensitivity.py` and use as a
    measurement priority list, and it is wrong in both directions: it
    recommends parameters that cannot change the decision, and it hides
    parameters that can.
    """
    payoffs = simulation["payoffs"]
    decision = baseline_decision(simulation)
    chosen = decision["recommended_index"]
    outcome = [row[chosen] for row in payoffs]
    total = _variance(outcome)

    shares: dict[str, float] = {}
    for name in simulation["parameters"]:
        if total <= 0:
            shares[name] = 0.0
            continue
        values = simulation["samples"][name]
        order = sorted(range(len(values)), key=lambda index: values[index])
        bucket_count = max(2, min(int(bins), len(values) // 10))
        size = len(order) / bucket_count

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
            means.append(statistics.fmean([outcome[index] for index in members]))
            counts.append(len(members))
            start = end

        if len(means) < 2:
            shares[name] = 0.0
            continue
        grand = sum(means[i] * counts[i] for i in range(len(means))) / sum(counts)
        between = sum(
            counts[i] * (means[i] - grand) ** 2 for i in range(len(means))
        ) / sum(counts)
        shares[name] = max(0.0, min(1.0, between / total))

    order = sorted(shares, key=lambda name: shares[name], reverse=True)
    return {
        "shares": shares,
        "order": order,
        "outcome_variance": total,
        "headline": (
            "'%s' drives the most variance in the payoff of the chosen option "
            "(%.0f%%)." % (order[0], shares[order[0]] * 100.0)
            if order
            else "No parameters."
        ),
    }


def compare_rankings(
    simulation: Mapping[str, Any],
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Do the variance ranking and the decision ranking agree?

    Where they do not — and they usually do not — the difference is the whole
    argument for this module. The interesting case is a parameter at the top of
    the variance ranking with an EVPPI of zero: it moves the number a great
    deal and the choice not at all, so measuring it buys nothing.
    """
    decision_side = evppi_ranking(simulation, bins=bins)
    variance_side = variance_ranking(simulation, bins=bins)

    positions_variance = {
        name: index for index, name in enumerate(variance_side["order"])
    }
    positions_decision = {
        name: index for index, name in enumerate(decision_side["order"])
    }

    rows = []
    for name in simulation["parameters"]:
        entry = next(
            item for item in decision_side["entries"] if item["parameter"] == name
        )
        rows.append(
            {
                "parameter": name,
                "variance_share": variance_side["shares"][name],
                "variance_rank": positions_variance[name] + 1,
                "evppi": entry["evppi"],
                "decision_rank": positions_decision[name] + 1,
                "moved": abs(positions_variance[name] - positions_decision[name]),
                "share_of_evpi": entry["share_of_evpi"],
                "high_variance_no_value": (
                    variance_side["shares"][name] > LOUD_VARIANCE_SHARE
                    and entry["share_of_evpi"] < NEGLIGIBLE_SHARE_OF_EVPI
                ),
            }
        )
    rows.sort(key=lambda row: row["decision_rank"])

    wasted = [row for row in rows if row["high_variance_no_value"]]
    agree = variance_side["order"] == decision_side["order"]

    return {
        "rows": rows,
        "agree": agree,
        "wasted_measurements": [row["parameter"] for row in wasted],
        "largest_move": max((row["moved"] for row in rows), default=0),
        "headline": (
            "The two rankings agree. Here the variance ordering happens to be a "
            "usable measurement priority; that is not something to rely on."
            if agree
            else "The rankings disagree, the largest by %d positions.%s"
            % (
                max((row["moved"] for row in rows), default=0),
                (
                    " %s drives %.0f%% of the payoff variance and buys %.0f%% "
                    "of the decision value: measuring it cannot change the "
                    "choice."
                    % (
                        ", ".join(row["parameter"] for row in wasted),
                        sum(row["variance_share"] for row in wasted) * 100.0,
                        sum(row["share_of_evpi"] for row in wasted) * 100.0,
                    )
                    if wasted
                    else ""
                ),
            )
        ),
    }


# ---------------------------------------------------------------------------
# EVSI
# ---------------------------------------------------------------------------


def evsi(
    simulation: Mapping[str, Any],
    parameters: Sequence[Mapping[str, Any]],
    parameter: str,
    sample_size: int,
    measurement_sd: float,
    bins: int = DEFAULT_BINS,
    seed: int = 20240915,
) -> dict[str, Any]:
    """The value of a study you could actually run, rather than omniscience.

    A normal-normal update: with prior ``N(mu0, s0^2)`` and a study mean of
    ``n`` observations at per-observation sd ``sigma``, the posterior mean is
    the precision-weighted average of the two. Simulate the study result, form
    the posterior mean, and partition the draws on *that* instead of on the
    parameter itself.

    The two limits are the reason this is the right construction. At ``n = 0``
    the posterior mean is the prior mean for every draw, one partition, and the
    value is zero. As ``n`` grows the posterior mean converges on the parameter
    and the value converges on EVPPI. A study is worth somewhere between
    nothing and perfect information, and this says where.
    """
    import random

    cleaned = _validate_parameters(parameters)
    target = next((item for item in cleaned if item["name"] == parameter), None)
    if target is None:
        raise VOIError("Parameter '%s' is not in this decision." % parameter)
    if not target["updatable"]:
        raise VOIError(
            "EVSI needs a conjugate prior to update and only a normal prior is "
            "implemented. '%s' is %s. Approximating it would produce a number "
            "whose error nobody could bound."
            % (parameter, target["distribution"])
        )
    if sample_size < 0:
        raise VOIError("Sample size cannot be negative.")
    if measurement_sd <= 0:
        raise VOIError(
            "A measurement with no error is perfect information; use EVPPI."
        )

    prior_sd = target["sd"]
    if prior_sd <= 0:
        raise VOIError(
            "'%s' has no prior uncertainty, so there is nothing a study could "
            "resolve." % parameter
        )

    rng = random.Random(seed)
    prior_mean = target["mean"]
    values = simulation["samples"][parameter]

    posterior_means = []
    for value in values:
        if sample_size == 0:
            posterior_means.append(prior_mean)
            continue
        observed = rng.gauss(value, measurement_sd / math.sqrt(sample_size))
        prior_precision = 1.0 / (prior_sd**2)
        data_precision = sample_size / (measurement_sd**2)
        posterior_means.append(
            (prior_mean * prior_precision + observed * data_precision)
            / (prior_precision + data_precision)
        )

    payoffs = simulation["payoffs"]
    names = simulation["options"]
    best_now = max(
        statistics.fmean([row[index] for row in payoffs]) for index in range(len(names))
    )
    conditional = _conditional_value(posterior_means, payoffs, bins)
    raw = conditional - best_now

    ceiling = evppi(simulation, parameter, bins=bins)["evppi"]
    return {
        "parameter": parameter,
        "sample_size": int(sample_size),
        "measurement_sd": measurement_sd,
        "evsi": max(0.0, raw),
        "raw": raw,
        "evppi": ceiling,
        "share_of_evppi": (max(0.0, raw) / ceiling) if ceiling > 0 else 0.0,
        "above_evppi": raw > ceiling * (1.0 + ESTIMATION_TOLERANCE) + 1e-9,
        "headline": (
            "A study of %d observations on '%s' is worth %.4g — %.0f%% of what "
            "resolving it perfectly would be worth."
            % (
                sample_size,
                parameter,
                max(0.0, raw),
                ((max(0.0, raw) / ceiling) * 100.0) if ceiling > 0 else 0.0,
            )
        ),
    }


def expected_net_benefit_of_sampling(
    simulation: Mapping[str, Any],
    parameters: Sequence[Mapping[str, Any]],
    parameter: str,
    measurement_sd: float,
    sample_sizes: Sequence[int],
    fixed_cost: float = 0.0,
    cost_per_observation: float = 0.0,
    population: float = 1.0,
    bins: int = DEFAULT_BINS,
    seed: int = 20240915,
) -> dict[str, Any]:
    """Is the study worth what it costs?

    Measurement is not free and this app currently prices it at zero. Fitting a
    smart meter, keeping a food diary, digging out twelve months of bills — all
    have a real cost, and a recommendation to measure something should have to
    clear a bar like any other recommendation.

    Where the optimum is negative the answer is not to collect the data, and
    that is a useful result rather than a failure to produce one.
    """
    if fixed_cost < 0 or cost_per_observation < 0:
        raise VOIError("Costs cannot be negative.")
    if population <= 0:
        raise VOIError("Population must be positive.")
    if not sample_sizes:
        raise VOIError("At least one sample size is needed.")

    rows = []
    for size in sample_sizes:
        if size < 0:
            raise VOIError("Sample sizes cannot be negative.")
        study = evsi(
            simulation,
            parameters,
            parameter,
            size,
            measurement_sd,
            bins=bins,
            seed=seed,
        )
        cost = fixed_cost + cost_per_observation * size if size > 0 else 0.0
        benefit = study["evsi"] * population
        rows.append(
            {
                "sample_size": int(size),
                "evsi": study["evsi"],
                "population_evsi": benefit,
                "cost": cost,
                "net_benefit": benefit - cost,
            }
        )

    best = max(rows, key=lambda row: row["net_benefit"])
    worthwhile = best["net_benefit"] > 0 and best["sample_size"] > 0

    return {
        "parameter": parameter,
        "rows": rows,
        "optimum": best,
        "worthwhile": worthwhile,
        "population": population,
        "headline": (
            "The best study collects %d observations, at a cost of %.4g for an "
            "expected benefit of %.4g — a net gain of %.4g."
            % (
                best["sample_size"],
                best["cost"],
                best["population_evsi"],
                best["net_benefit"],
            )
            if worthwhile
            else "No study of '%s' pays for itself at these costs. The best "
            "available net benefit is %.4g. Act on the data you have."
            % (parameter, best["net_benefit"])
        ),
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def analyse(
    options: Sequence[Mapping[str, Any]],
    parameters: Sequence[Mapping[str, Any]],
    draws: int = DEFAULT_DRAWS,
    bins: int = DEFAULT_BINS,
    cheapest_measurement: float = 0.0,
    seed: int = 20240915,
) -> dict[str, Any]:
    """The decision, its uncertainty, and what resolving each input would buy."""
    simulation = simulate(options, parameters, draws=draws, seed=seed)
    decision = baseline_decision(simulation)
    ceiling = evpi(simulation)
    ranking = evppi_ranking(simulation, bins=bins)
    comparison = compare_rankings(simulation, bins=bins)

    act_now = ceiling["evpi"] <= cheapest_measurement
    return {
        "engine_version": ENGINE_VERSION,
        "options": simulation["options"],
        "parameters": simulation["parameters"],
        "draws": simulation["draws"],
        "bins": bins,
        "decision": decision,
        "evpi": ceiling,
        "evppi": ranking,
        "comparison": comparison,
        "cheapest_measurement": cheapest_measurement,
        "act_on_what_you_have": act_now,
        "headline": (
            "Resolving every uncertainty is worth %.4g, which is less than the "
            "%.4g the cheapest measurement costs. Act on what you have — no "
            "further data can pay for itself on this decision."
            % (ceiling["evpi"], cheapest_measurement)
            if act_now
            else "%s %s" % (decision["headline"], ranking["headline"])
        ),
    }


def get_voi_notes(result: Mapping[str, Any]) -> list[str]:
    """Plain-language readings of an analysis."""
    notes: list[str] = [result.get("headline", "")]

    decision = result.get("decision")
    if decision:
        notes.append(
            "The recommendation is wrong on %.0f%% of draws, and costs %.4g on "
            "average when it is. A recommendation made at 51%% confidence and "
            "one made at 99%% are displayed identically today."
            % (
                decision["probability_wrong"] * 100.0,
                decision["expected_opportunity_loss"],
            )
        )

    if result.get("act_on_what_you_have"):
        notes.append(
            "This is the result the app currently cannot produce: you have "
            "enough information to choose, and collecting more cannot change "
            "the answer."
        )

    comparison = result.get("comparison")
    if comparison:
        notes.append(comparison["headline"])
        if comparison["wasted_measurements"]:
            notes.append(
                "Taking the variance ranking as a measurement priority list — "
                "which is what a reader would naturally do with "
                "global_sensitivity.py — recommends measuring %s, and measuring "
                "it cannot change the choice."
                % ", ".join(comparison["wasted_measurements"])
            )

    ranking = result.get("evppi")
    if ranking and ranking.get("estimation_failures"):
        notes.append(
            "EVPPI for %s falls outside [0, EVPI], which cannot happen and "
            "therefore indicates too few draws or too many bins."
            % ", ".join(ranking["estimation_failures"])
        )

    return [note for note in notes if note]


def summarise(result: Mapping[str, Any]) -> str:
    """One line for a log or a saved-analysis list."""
    decision = result.get("decision") or {}
    ranking = result.get("evppi") or {}
    return "%d options, %d parameters | choose %s (%.0f%% right) | EVPI %.4g | measure %s" % (
        len(result.get("options", [])),
        len(result.get("parameters", [])),
        decision.get("recommended", "?"),
        decision.get("probability_recommended_is_best", 0.0) * 100.0,
        (result.get("evpi") or {}).get("evpi", 0.0),
        (ranking.get("order") or ["nothing"])[0],
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _connect():
    return sqlite3.connect(DB_NAME)


def _ensure_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS value_of_information_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            label TEXT NOT NULL,
            recommended TEXT NOT NULL,
            evpi REAL NOT NULL,
            top_parameter TEXT NOT NULL,
            act_now INTEGER NOT NULL DEFAULT 0,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_value_of_information_user
        ON value_of_information_analyses (user_id)
        """
    )


def save_analysis(user_id: Any, result: Mapping[str, Any], label: str = "") -> int | None:
    """Persist an analysis. None if storage is unavailable."""
    if not user_id or not result.get("engine_version"):
        return None
    try:
        with _connect() as conn:
            _ensure_tables(conn)
            order = (result.get("evppi") or {}).get("order") or ["none"]
            cursor = conn.execute(
                """
                INSERT INTO value_of_information_analyses
                    (user_id, label, recommended, evpi, top_parameter, act_now,
                     payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_id),
                    str(label or "decision"),
                    str((result.get("decision") or {}).get("recommended", "")),
                    float((result.get("evpi") or {}).get("evpi", 0.0)),
                    str(order[0]),
                    1 if result.get("act_on_what_you_have") else 0,
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
                SELECT id, label, recommended, evpi, top_parameter, act_now,
                       payload, created_at
                FROM value_of_information_analyses
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
                "recommended": row[2],
                "evpi": row[3],
                "top_parameter": row[4],
                "act_now": bool(row[5]),
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
                "DELETE FROM value_of_information_analyses "
                "WHERE user_id = ? AND id = ?",
                (str(user_id), int(analysis_id)),
            )
            return cursor.rowcount > 0
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------------------
# Worked examples
# ---------------------------------------------------------------------------


def demo_decision(
    decisive_spread: float = 30.0,
    loud_spread: float = 400.0,
    gap: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A decision built so the two rankings must disagree.

    Two parameters, and the contrast is the entire point:

    ``grid_intensity`` has a very large spread and enters **both** options
    identically. It moves the payoff of whichever option is chosen by a great
    deal and moves the difference between them not at all, so it dominates the
    variance ranking and has an EVPPI of exactly zero. Measuring it is a
    perfectly good use of the variance ranking and a complete waste of money.

    ``heat_pump_performance`` has a small spread and enters the two options with
    opposite signs, so its sign decides the choice. It is near the bottom of the
    variance ranking and at the top of the decision ranking.

    `gap` shifts one option's payoff away from the other. At a large enough gap
    the decision stops being close, every EVPPI collapses to zero, and the
    correct output becomes "act on what you have".
    """
    parameters = [
        build_parameter(
            "grid_intensity", "normal", mean=0.0, sd=loud_spread
        ),
        build_parameter(
            "heat_pump_performance", "normal", mean=0.0, sd=decisive_spread
        ),
    ]

    def heat_pump(state: Mapping[str, float]) -> float:
        return 1000.0 + gap + state["grid_intensity"] + state["heat_pump_performance"]

    def insulation(state: Mapping[str, float]) -> float:
        return 1000.0 + state["grid_intensity"] - state["heat_pump_performance"]

    options = [
        build_option("Heat pump", heat_pump),
        build_option("Insulation", insulation),
    ]
    return options, parameters


def demo_abatement_decision(
    seed: int = 20240915,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Three abatement measures ranked on point estimates, as this app does.

    `src/carbon/abatement_curve.py` orders measures by cost per tonne computed
    from uncertain inputs, and presents two measures whose intervals overlap as
    ranked. The decision-relevant question — is the *choice* uncertain, and
    would resolving one input settle it — is not asked there.
    """
    parameters = [
        build_parameter("grid_intensity", "normal", mean=250.0, sd=60.0),
        build_parameter("fuel_price", "normal", mean=1.0, sd=0.25),
        build_parameter("install_quality", "normal", mean=1.0, sd=0.15),
        build_parameter("occupancy", "uniform", low=0.7, high=1.3),
    ]

    def heat_pump(state: Mapping[str, float]) -> float:
        saved = 4500.0 * state["occupancy"] * state["install_quality"]
        return saved * (state["grid_intensity"] / 250.0) * 0.9 - 2600.0

    def insulation(state: Mapping[str, float]) -> float:
        saved = 3000.0 * state["occupancy"]
        return saved * state["fuel_price"] * 1.15 - 1400.0

    def solar(state: Mapping[str, float]) -> float:
        saved = 3400.0 * state["install_quality"]
        return saved * (state["grid_intensity"] / 250.0) - 2100.0

    options = [
        build_option("Heat pump", heat_pump),
        build_option("Insulation", insulation),
        build_option("Solar", solar),
    ]
    return options, parameters
