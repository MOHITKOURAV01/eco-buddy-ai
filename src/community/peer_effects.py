"""Peer effects, and the three different worlds that produce the same
correlation.

The community half of this app rests on a causal claim it never tests: that
being around greener people makes you greener. `block_leaderboard.py`,
`pledge_leaderboard.py`, `community_challenges.py` and `eco_social.py` all show
a user their group's behaviour on the premise that seeing it will change theirs.

There is a correlation to point at — members of a green block do have lower
footprints — and it is consistent with three worlds that nothing here can tell
apart:

**Endogenous effect.** The group's behaviour changes yours. This is the one the
feature assumes.

**Exogenous (contextual) effect.** The group's *characteristics* affect you.
Everyone on the block has a bus route and a solar co-op.

**Correlated effect.** Nothing affects anything. Green people move to green
blocks, and joined the green challenge for the same reason they were already
green.

The reflection problem
----------------------
Manski's result is not that separating these is hard. It is that in a group
where everyone is connected to everyone, the endogenous effect is **not
identified** — the coefficient does not exist to be estimated, and any number
produced by that regression is arithmetic.

This module makes that concrete rather than asserting it. Two mechanical
results, both derived and both checked in the tests:

*Including yourself in the peer mean gives a slope of one.* For a group of
size ``m`` with no peer effect at all, ``cov(y_i, groupmean) = var(y)/m`` and
``var(groupmean) = var(y)/m``, so the regression returns 1.0. Not approximately
— it is the same quantity over itself. That is the default construction
everywhere in this repo, and it manufactures an effect out of nothing.

*Group fixed effects on a leave-one-out outcome mean give exactly -(m-1).*
Within a group, ``loo_i = (m * ybar - y_i) / (m - 1)``, so demeaning both sides
leaves ``loo_i - loobar = -(y_i - ybar) / (m - 1)``. The regressor is a perfect
negative multiple of the outcome. The estimate carries no information about
peers whatsoever, and it will not look like an error — it will look like a
strong negative peer effect.

So the honest output for most community data in this app is "consistent with
influence and with sorting, and here is what would distinguish them". That is a
useful answer, and it is currently unavailable at any price.

What makes it identifiable
--------------------------
Intransitive network structure. If everyone in a group is connected to everyone
else, a friend's friends are your friends and there is no exclusion
restriction. Where the network has open triads — friends-of-friends who are not
your friends — their characteristics affect you only through your friend, and
that supplies the instrument. Bramoullé, Djebbari and Fortin.

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

# Design requirements -------------------------------------------------------
MIN_GROUPS = 4
MIN_MEMBERS = 20
MAX_MEMBERS = 100000

# Identification ------------------------------------------------------------
# Below this share of open triads there is no exclusion restriction to work
# with and the endogenous effect is not identified.
MIN_INTRANSITIVITY = 0.10
# Perfect collinearity never comes out exactly perfect in floating point.
COLLINEARITY_TOLERANCE = 1e-6

# Inference -----------------------------------------------------------------
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
# Two-sided 5% and 80% power, the conventional pair.
Z_ALPHA_TWO_SIDED = {0.01: 2.575829, 0.05: 1.959964, 0.10: 1.644854}
Z_POWER = {0.80: 0.841621, 0.90: 1.281552, 0.95: 1.644854}


class PeerEffectError(ValueError):
    """Raised when a design cannot support the claim being asked of it."""


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
        raise PeerEffectError("Degrees of freedom must be positive.")
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * betai(degrees_of_freedom / 2.0, 0.5, x)
    return 1.0 - tail if value > 0 else tail


def two_sided_p(statistic: float, degrees_of_freedom: float) -> float:
    """Two-sided p-value for a t statistic."""
    if degrees_of_freedom <= 0:
        return 1.0
    return 2.0 * (1.0 - t_cdf(abs(statistic), degrees_of_freedom))


def regress(
    xs: Sequence[float],
    ys: Sequence[float],
    clusters: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Simple regression, with cluster-robust errors when clusters are given.

    Clustering matters more here than almost anywhere else in this repo: the
    whole premise is that members of a group are not independent, so treating
    each member as an observation overstates the sample by roughly the group
    size.
    """
    if len(xs) != len(ys):
        raise PeerEffectError("x and y must be the same length.")
    n = len(xs)
    if n < 4:
        raise PeerEffectError("Need at least four observations to fit a slope.")

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        raise PeerEffectError("The regressor does not vary; no slope is identified.")

    slope = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / sxx
    intercept = mean_y - slope * mean_x
    residuals = [ys[i] - (intercept + slope * xs[i]) for i in range(n)]

    degrees = n - 2
    if clusters is None:
        residual_variance = (
            sum(value * value for value in residuals) / degrees if degrees > 0 else 0.0
        )
        variance = residual_variance / sxx
        cluster_count = None
    else:
        if len(clusters) != n:
            raise PeerEffectError("clusters must be the same length as the data.")
        by_cluster: dict[Any, float] = {}
        for index in range(n):
            key = clusters[index]
            by_cluster[key] = by_cluster.get(key, 0.0) + (
                xs[index] - mean_x
            ) * residuals[index]
        cluster_count = len(by_cluster)
        if cluster_count < 2:
            raise PeerEffectError(
                "Cluster-robust errors need at least two clusters. With one "
                "group there is no independent variation to estimate from."
            )
        meat = sum(value * value for value in by_cluster.values())
        # Small-cluster correction, the standard finite-sample adjustment.
        correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / max(n - 2, 1))
        variance = correction * meat / (sxx * sxx)
        degrees = cluster_count - 1

    standard_error = math.sqrt(max(variance, 0.0))
    statistic = slope / standard_error if standard_error > 0 else 0.0

    syy = sum((y - mean_y) ** 2 for y in ys)
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "standard_error": standard_error,
        "degrees_of_freedom": degrees,
        "t_statistic": statistic,
        "p_value": two_sided_p(statistic, degrees) if degrees > 0 else 1.0,
        "clusters": cluster_count,
        "r_squared": (
            1.0 - sum(value * value for value in residuals) / syy if syy > 0 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Members and networks
# ---------------------------------------------------------------------------


def build_member(
    identifier: Any,
    group: Any,
    outcome: float,
    baseline: float | None = None,
    peers: Sequence[Any] | None = None,
    treated: bool = False,
) -> dict[str, Any]:
    """One person, their group, their footprint, and who they are linked to.

    `baseline` is their outcome from before the group formed. It is optional
    and it is the single most valuable field here: without it, homophily and
    influence cannot be told apart at all.

    `peers` is an explicit link list. Where it is absent the group is treated
    as a clique — everyone linked to everyone — which is how the community
    features in this repo actually model a block or a challenge, and which is
    also precisely the structure in which nothing is identified.
    """
    value = _finite(outcome)
    if value is None:
        raise PeerEffectError("Member '%s' has a non-numeric outcome." % identifier)

    base = None if baseline is None else _finite(baseline)
    if baseline is not None and base is None:
        raise PeerEffectError("Member '%s' has a non-numeric baseline." % identifier)

    return {
        "id": str(identifier),
        "group": str(group),
        "outcome": value,
        "baseline": base,
        "peers": [str(peer) for peer in (peers or [])],
        "explicit_peers": peers is not None,
        "treated": bool(treated),
    }


def _validate_members(members: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(members) < MIN_MEMBERS:
        raise PeerEffectError(
            "Need at least %d members; %d supplied." % (MIN_MEMBERS, len(members))
        )
    if len(members) > MAX_MEMBERS:
        raise PeerEffectError("At most %d members are supported." % MAX_MEMBERS)

    seen: set[str] = set()
    cleaned = []
    for member in members:
        if not isinstance(member, Mapping) or "outcome" not in member:
            raise PeerEffectError("Members must be built with build_member().")
        if member["id"] in seen:
            raise PeerEffectError("Member '%s' appears twice." % member["id"])
        seen.add(member["id"])
        cleaned.append(dict(member))

    groups = {member["group"] for member in cleaned}
    if len(groups) < MIN_GROUPS:
        raise PeerEffectError(
            "Need at least %d groups to separate within-group from "
            "between-group variation; %d supplied." % (MIN_GROUPS, len(groups))
        )
    return cleaned


def adjacency(members: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Link list per member, with groups expanded to cliques where implicit.

    Links are symmetrised. A one-directional "I follow them" is treated as a
    link in both directions, because influence in an app like this runs both
    ways and asymmetry here would be a claim the data does not support.
    """
    known = {member["id"] for member in members}
    links: dict[str, set[str]] = {member["id"]: set() for member in members}

    by_group: dict[str, list[str]] = {}
    for member in members:
        by_group.setdefault(member["group"], []).append(member["id"])

    for member in members:
        if member["explicit_peers"]:
            for peer in member["peers"]:
                if peer in known and peer != member["id"]:
                    links[member["id"]].add(peer)
                    links[peer].add(member["id"])
        else:
            for other in by_group[member["group"]]:
                if other != member["id"]:
                    links[member["id"]].add(other)
                    links[other].add(member["id"])
    return links


# ---------------------------------------------------------------------------
# Peer means — where the mechanical artefact lives
# ---------------------------------------------------------------------------


def peer_means(
    members: Sequence[Mapping[str, Any]],
    field: str = "outcome",
    include_self: bool = False,
) -> dict[str, float | None]:
    """Mean of a member's peers, with or without the member in it.

    `include_self=True` reproduces the construction used throughout this repo's
    community features. It is offered so the artefact can be measured, not
    because it is ever the right choice.
    """
    cleaned = _validate_members(members)
    links = adjacency(cleaned)
    values = {member["id"]: member.get(field) for member in cleaned}

    result: dict[str, float | None] = {}
    for member in cleaned:
        identifier = member["id"]
        neighbours = set(links[identifier])
        if include_self:
            neighbours.add(identifier)
        usable = [
            values[peer]
            for peer in neighbours
            if values.get(peer) is not None
        ]
        result[identifier] = statistics.fmean(usable) if usable else None
    return result


def naive_peer_regression(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The construction the app currently uses: member regressed on group mean.

    With no peer effect at all this returns a slope of one, exactly, because::

        cov(y_i, groupmean) = var(y) / m
        var(groupmean)      = var(y) / m

    It is the same quantity divided by itself. The number is arithmetic, it is
    large, it is positive, and on a dashboard it is indistinguishable from a
    finding.
    """
    cleaned = _validate_members(members)
    means = peer_means(cleaned, include_self=True)

    pairs = [
        (means[member["id"]], member["outcome"])
        for member in cleaned
        if means[member["id"]] is not None
    ]
    if len(pairs) < 4:
        raise PeerEffectError("Not enough members with a computable group mean.")

    fit = regress(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        clusters=[member["group"] for member in cleaned if means[member["id"]] is not None],
    )
    return {
        "method": "self_included",
        "slope": fit["slope"],
        "standard_error": fit["standard_error"],
        "p_value": fit["p_value"],
        "n": fit["n"],
        "mechanical": True,
        "headline": (
            "Slope %.3f. This regression includes each member in their own peer "
            "mean, which returns a value near 1.0 whether or not any peer "
            "effect exists. It is not an estimate of anything."
            % fit["slope"]
        ),
    }


def leave_one_out_regression(
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Member regressed on the mean of their peers, self excluded.

    The single most important fix in this module and close to free. It removes
    the mechanical artefact and leaves the real problem: a positive slope here
    is still consistent with sorting, because members of a group share whatever
    made them a group.
    """
    cleaned = _validate_members(members)
    means = peer_means(cleaned, include_self=False)

    usable = [member for member in cleaned if means[member["id"]] is not None]
    if len(usable) < 4:
        raise PeerEffectError("Not enough members with at least one peer.")

    fit = regress(
        [means[member["id"]] for member in usable],
        [member["outcome"] for member in usable],
        clusters=[member["group"] for member in usable],
    )
    return {
        "method": "leave_one_out",
        "slope": fit["slope"],
        "standard_error": fit["standard_error"],
        "p_value": fit["p_value"],
        "clusters": fit["clusters"],
        "n": fit["n"],
        "confounded": True,
        "headline": (
            "Slope %.3f (clustered SE %.3f, p=%.4f) across %d groups. The "
            "mechanical artefact is gone. Sorting is not: members of a group "
            "share whatever made them a group, and this cannot tell that from "
            "influence."
            % (fit["slope"], fit["standard_error"], fit["p_value"], fit["clusters"] or 0)
        ),
    }


def within_group_regression(
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Group fixed effects on a leave-one-out outcome mean — and why it fails.

    Absorbing the group removes everything the group shares, which is where
    sorting hides. It also destroys the regressor, and this is worth being
    precise about because the failure is silent.

    Within a group, ``loo_i = (m * ybar - y_i) / (m - 1)``. Demeaning both
    sides gives ``loo_i - loobar = -(y_i - ybar) / (m - 1)``: the regressor is
    a perfect negative multiple of the outcome. The regression returns exactly
    ``-(m - 1)``, it has nothing to do with peers, and it looks like a strong
    negative peer effect.

    So this function computes it, detects the collinearity, and refuses. That
    refusal is the reflection problem in a form that can be put on a screen.
    """
    cleaned = _validate_members(members)
    means = peer_means(cleaned, include_self=False)
    usable = [member for member in cleaned if means[member["id"]] is not None]

    by_group: dict[str, list[dict[str, Any]]] = {}
    for member in usable:
        by_group.setdefault(member["group"], []).append(member)

    demeaned_x: list[float] = []
    demeaned_y: list[float] = []
    clusters: list[str] = []
    sizes: list[int] = []
    for group, entries in by_group.items():
        if len(entries) < 2:
            continue
        sizes.append(len(entries))
        mean_y = statistics.fmean([entry["outcome"] for entry in entries])
        mean_x = statistics.fmean([means[entry["id"]] for entry in entries])
        for entry in entries:
            demeaned_y.append(entry["outcome"] - mean_y)
            demeaned_x.append(means[entry["id"]] - mean_x)
            clusters.append(group)

    if len(demeaned_x) < 4:
        raise PeerEffectError("Not enough within-group variation to fit anything.")

    # Perfect collinearity check: is the demeaned regressor a fixed multiple of
    # the demeaned outcome? Under a clique structure it is, exactly.
    ratios = [
        demeaned_x[i] / demeaned_y[i]
        for i in range(len(demeaned_y))
        if abs(demeaned_y[i]) > 1e-9
    ]
    collinear = False
    ratio_spread = None
    if len(ratios) >= 2:
        ratio_spread = max(ratios) - min(ratios)
        collinear = ratio_spread < COLLINEARITY_TOLERANCE

    fit = regress(demeaned_x, demeaned_y, clusters=clusters)
    mean_size = statistics.fmean(sizes) if sizes else 0.0
    expected = -(mean_size - 1.0)

    return {
        "method": "within_group",
        "slope": fit["slope"],
        "standard_error": fit["standard_error"],
        "p_value": fit["p_value"],
        "mean_group_size": mean_size,
        "mechanical_value": expected,
        "collinear": collinear,
        "ratio_spread": ratio_spread,
        "identified": not collinear,
        "headline": (
            "Slope %.3f, which is -(m - 1) for a mean group size of %.1f. The "
            "demeaned peer mean is a perfect negative multiple of the demeaned "
            "outcome, so this coefficient contains no information about peers "
            "at all. This is the reflection problem, not a result."
            % (fit["slope"], mean_size)
            if collinear
            else "Slope %.3f (p=%.4f). The network is not a set of cliques, so "
            "the within-group regressor is not collinear with the outcome and "
            "this coefficient means something."
            % (fit["slope"], fit["p_value"])
        ),
    }


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def network_identification(
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Does the network have the structure that identifies an endogenous effect?

    In a clique, a friend's friends are your friends, so their characteristics
    reach you directly and there is no exclusion restriction. Where the network
    has *open* triads — i linked to j, j linked to k, i not linked to k — then
    k affects i only through j, and k's characteristics instrument for j's
    behaviour.

    The share of open triads is therefore the identifying variation, and where
    it is zero no estimator recovers an endogenous effect. Not "recovers it
    imprecisely" — there is nothing there.
    """
    cleaned = _validate_members(members)
    links = adjacency(cleaned)

    open_triads = 0
    closed_triads = 0
    for centre, neighbours in links.items():
        ordered = sorted(neighbours)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                if second in links.get(first, set()):
                    closed_triads += 1
                else:
                    open_triads += 1

    total = open_triads + closed_triads
    intransitivity = open_triads / total if total else 0.0

    degrees = [len(neighbours) for neighbours in links.values()]
    isolated = sum(1 for degree in degrees if degree == 0)
    identified = intransitivity >= MIN_INTRANSITIVITY

    return {
        "members": len(cleaned),
        "open_triads": open_triads,
        "closed_triads": closed_triads,
        "intransitivity": intransitivity,
        "mean_degree": statistics.fmean(degrees) if degrees else 0.0,
        "isolated": isolated,
        "identified": identified,
        "threshold": MIN_INTRANSITIVITY,
        "headline": (
            "%.1f%% of triads are open — enough intransitive structure to "
            "identify an endogenous effect. Friends-of-friends who are not "
            "friends supply the exclusion restriction."
            % (intransitivity * 100.0)
            if identified
            else "Only %.1f%% of triads are open. This network is close to a "
            "set of cliques, which is how a block or a challenge in this app is "
            "actually modelled, and in a clique the endogenous peer effect is "
            "not identified. No estimator recovers it, because there is nothing "
            "to recover." % (intransitivity * 100.0)
        ),
    }


# ---------------------------------------------------------------------------
# Homophily
# ---------------------------------------------------------------------------


def homophily_test(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Were they already similar before the group existed?

    Two friends whose footprints converge over a year look identical whether
    one persuaded the other or they were converging anyway. Baselines from
    before group formation are what separates the two, and the separation has
    to be made on the right pair of variables — which is the part that is easy
    to get wrong.

    *Sorting* is peers' pre-period level predicting **your pre-period level**.
    Nothing had happened yet, so any association there is selection by
    construction.

    *Influence* is peers' pre-period level predicting **your change**. That is
    the outcome with your own baseline partialled out of both sides, and it is
    the only part any peer effect could be.

    Testing sorting against the *outcome* instead — the obvious construction —
    conflates the two, because under a real peer effect peers' baselines
    predict your outcome exactly as they would under sorting. Both would be
    reported as sorting and every genuine effect would be blocked.
    """
    cleaned = _validate_members(members)
    with_baseline = [member for member in cleaned if member["baseline"] is not None]
    if len(with_baseline) < MIN_MEMBERS:
        raise PeerEffectError(
            "Homophily cannot be tested without baselines from before the group "
            "formed. This is the field that separates influence from sorting, "
            "and without it the two are indistinguishable at any sample size."
        )

    baseline_means = peer_means(cleaned, field="baseline", include_self=False)
    outcome_means = peer_means(cleaned, field="outcome", include_self=False)

    usable = [
        member
        for member in with_baseline
        if baseline_means[member["id"]] is not None
        and outcome_means[member["id"]] is not None
    ]
    if len(usable) < 4:
        raise PeerEffectError("Not enough members with both a baseline and peers.")

    clusters = [member["group"] for member in usable]
    own = [member["baseline"] for member in usable]
    peer_baseline = [baseline_means[member["id"]] for member in usable]

    # Sorting: peers' pre-period level against the member's own pre-period
    # level. Nothing has happened yet, so an association here is selection.
    sorting = regress(peer_baseline, own, clusters=clusters)

    # Influence: peers' pre-period level against the member's change, with
    # their own baseline partialled out of both sides.
    outcome_residual = _residualise([member["outcome"] for member in usable], own)
    peer_residual = _residualise(peer_baseline, own)
    influence = regress(peer_residual, outcome_residual, clusters=clusters)

    sorted_in = sorting["p_value"] < DEFAULT_ALPHA
    influenced = influence["p_value"] < DEFAULT_ALPHA

    return {
        "n": len(usable),
        "sorting_slope": sorting["slope"],
        "sorting_p": sorting["p_value"],
        "influence_slope": influence["slope"],
        "influence_p": influence["p_value"],
        "sorting_detected": sorted_in,
        "influence_detected": influenced,
        "headline": (
            "Peers were already alike before the group formed (slope %.3f, "
            "p=%.4f), so any cross-sectional association is at least partly "
            "selection. The part not explained by who they already were is "
            "%.3f (p=%.3f)."
            % (
                sorting["slope"],
                sorting["p_value"],
                influence["slope"],
                influence["p_value"],
            )
            if sorted_in
            else "No pre-existing similarity (p=%.3f): peers' baselines do not "
            "predict this member's baseline. The influence estimate with own "
            "baseline partialled out is %.3f (p=%.4f)."
            % (sorting["p_value"], influence["slope"], influence["p_value"])
        ),
    }


def _residualise(ys: Sequence[float], xs: Sequence[float]) -> list[float]:
    """Residuals of y on x, for partialling one variable out of another."""
    if len(ys) != len(xs):
        raise PeerEffectError("Cannot residualise series of different lengths.")
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return [y - mean_y for y in ys]
    slope = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(ys))) / sxx
    intercept = mean_y - slope * mean_x
    return [ys[i] - (intercept + slope * xs[i]) for i in range(len(ys))]


# ---------------------------------------------------------------------------
# Spillover
# ---------------------------------------------------------------------------


def spillover_exposure(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How contaminated is the control group?

    Where treated and untreated share a network, the untreated are partly
    treated. That violates SUTVA, and the direction of the resulting bias is
    worth stating plainly: contamination pulls the control group toward the
    treated group, which shrinks the measured difference. A null result on a
    contaminated design is therefore not evidence of no effect.
    """
    cleaned = _validate_members(members)
    links = adjacency(cleaned)
    treated = {member["id"] for member in cleaned if member["treated"]}
    if not treated:
        raise PeerEffectError("No treated members; there is no spillover to measure.")
    if len(treated) == len(cleaned):
        raise PeerEffectError("Everyone is treated; there is no control group.")

    exposures = []
    for member in cleaned:
        if member["treated"]:
            continue
        neighbours = links[member["id"]]
        share = (
            sum(1 for peer in neighbours if peer in treated) / len(neighbours)
            if neighbours
            else 0.0
        )
        exposures.append({"id": member["id"], "exposure": share})

    if not exposures:
        raise PeerEffectError("No control members to measure exposure for.")

    values = [entry["exposure"] for entry in exposures]
    mean_exposure = statistics.fmean(values)
    clean_controls = sum(1 for value in values if value == 0.0)

    return {
        "controls": len(exposures),
        "clean_controls": clean_controls,
        "clean_share": clean_controls / len(exposures),
        "mean_exposure": mean_exposure,
        "max_exposure": max(values),
        "exposures": exposures,
        "attenuation_expected": mean_exposure > 0,
        "headline": (
            "%d of %d controls have no treated peers. The average control has "
            "%.0f%% of their network treated, so the control group is partly "
            "treated and the measured difference is smaller than the real "
            "effect. A null here is not evidence of no effect."
            % (clean_controls, len(exposures), mean_exposure * 100.0)
            if mean_exposure > 0
            else "No control has a treated peer. The arms are separated and "
            "SUTVA holds for this design."
        ),
    }


# ---------------------------------------------------------------------------
# The design that would settle it
# ---------------------------------------------------------------------------


def encouragement_power(
    groups: int,
    group_size: int,
    outcome_sd: float,
    intraclass_correlation: float,
    compliance: float = 1.0,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> dict[str, Any]:
    """Minimum detectable effect for a randomised encouragement design.

    The clean answer to all of the above is to randomise who is shown a peer
    comparison. This does not need new data to *plan*, and planning it needs
    two things the app does not currently account for.

    *The design effect.* Randomising over groups gives you groups, not people.
    ``1 + (m - 1) * ICC`` is the factor by which the required sample grows, and
    at an ICC of 0.15 with groups of 20 that is a multiplier of nearly four.

    *Compliance.* Encouragement is not treatment. Dividing the
    intention-to-treat effect by the compliance rate gives the local average
    treatment effect, and the standard error divides too — so a design with 50%
    compliance needs four times the sample, not twice.
    """
    if groups < 2:
        raise PeerEffectError("A randomised design needs at least two groups.")
    if group_size < 1:
        raise PeerEffectError("Group size must be at least one.")
    if outcome_sd <= 0:
        raise PeerEffectError("Outcome standard deviation must be positive.")
    if not 0.0 <= intraclass_correlation < 1.0:
        raise PeerEffectError("The intraclass correlation must be in [0, 1).")
    if not 0.0 < compliance <= 1.0:
        raise PeerEffectError("Compliance must be in (0, 1].")
    if alpha not in Z_ALPHA_TWO_SIDED:
        raise PeerEffectError(
            "Alpha must be one of %s."
            % ", ".join(str(value) for value in sorted(Z_ALPHA_TWO_SIDED))
        )
    if power not in Z_POWER:
        raise PeerEffectError(
            "Power must be one of %s."
            % ", ".join(str(value) for value in sorted(Z_POWER))
        )

    total = groups * group_size
    design_effect = 1.0 + (group_size - 1) * intraclass_correlation
    effective = total / design_effect

    z_alpha = Z_ALPHA_TWO_SIDED[alpha]
    z_power = Z_POWER[power]
    # Two equal arms, so the variance of the difference carries a factor of 4.
    itt = (z_alpha + z_power) * outcome_sd * math.sqrt(4.0 / effective)
    late = itt / compliance

    return {
        "groups": groups,
        "group_size": group_size,
        "total_members": total,
        "design_effect": design_effect,
        "effective_sample": effective,
        "intraclass_correlation": intraclass_correlation,
        "compliance": compliance,
        "minimum_detectable_itt": itt,
        "minimum_detectable_late": late,
        "alpha": alpha,
        "power": power,
        "headline": (
            "%d groups of %d is %d people but only %.0f independent "
            "observations — the design effect is %.2f. At %.0f%% compliance the "
            "smallest peer effect this design could detect is %.1f."
            % (
                groups,
                group_size,
                total,
                effective,
                design_effect,
                compliance * 100.0,
                late,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def analyse(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything the design supports, and an explicit statement of what it does not.

    The order matters. Identification is checked before any endogenous effect is
    reported, because an estimate from a design that cannot identify one is not
    an imprecise estimate — it is a number with no referent.
    """
    cleaned = _validate_members(members)

    result: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "members": len(cleaned),
        "groups": len({member["group"] for member in cleaned}),
        "with_baseline": sum(1 for m in cleaned if m["baseline"] is not None),
        "treated": sum(1 for m in cleaned if m["treated"]),
        "identification": network_identification(cleaned),
        "naive": naive_peer_regression(cleaned),
        "leave_one_out": leave_one_out_regression(cleaned),
        "within_group": None,
        "homophily": None,
        "spillover": None,
        "endogenous_effect": None,
        "blocked": None,
    }

    try:
        result["within_group"] = within_group_regression(cleaned)
    except PeerEffectError:
        result["within_group"] = None

    try:
        result["homophily"] = homophily_test(cleaned)
    except PeerEffectError as error:
        result["homophily"] = None
        result["homophily_unavailable"] = str(error)

    try:
        result["spillover"] = spillover_exposure(cleaned)
    except PeerEffectError:
        result["spillover"] = None

    reasons = []
    if not result["identification"]["identified"]:
        reasons.append(
            "the network is close to a set of cliques, in which the endogenous "
            "effect is not identified"
        )
    if result["within_group"] is not None and result["within_group"]["collinear"]:
        reasons.append(
            "the within-group regressor is perfectly collinear with the outcome"
        )
    if result["homophily"] is None:
        reasons.append(
            "there are no pre-group baselines, so sorting and influence cannot "
            "be separated"
        )
    elif result["homophily"]["sorting_detected"]:
        reasons.append(
            "peers were already alike before the group formed, so the "
            "association is at least partly sorting"
        )

    if reasons:
        result["blocked"] = (
            "No endogenous peer effect is reported, because %s." % "; and ".join(reasons)
        )
        result["headline"] = result["blocked"]
    else:
        result["endogenous_effect"] = result["leave_one_out"]
        result["headline"] = (
            "An endogenous peer effect of %.3f (p=%.4f) survives every check "
            "this design can run: the network is intransitive enough to "
            "identify one, and peers were not already alike."
            % (
                result["leave_one_out"]["slope"],
                result["leave_one_out"]["p_value"],
            )
        )
    return result


# ---------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------


def get_peer_notes(result: Mapping[str, Any]) -> list[str]:
    """Plain-language readings of an analysis."""
    notes: list[str] = [result.get("headline", "")]

    naive = result.get("naive")
    loo = result.get("leave_one_out")
    if naive and loo:
        notes.append(
            "Including each member in their own peer mean gives %.3f. Excluding "
            "them gives %.3f. The first number is what the current construction "
            "produces and it is close to 1.0 whether or not any effect exists."
            % (naive["slope"], loo["slope"])
        )

    within = result.get("within_group")
    if within and within["collinear"]:
        notes.append(
            "Group fixed effects return %.2f, which is -(m - 1) for a mean group "
            "size of %.1f. That is not a negative peer effect; the regressor is "
            "a multiple of the outcome and the coefficient is arithmetic."
            % (within["slope"], within["mean_group_size"])
        )

    identification = result.get("identification")
    if identification and not identification["identified"]:
        notes.append(
            "Only %.1f%% of triads are open. Every group in this app's community "
            "features is modelled as a clique, and a clique supplies no "
            "exclusion restriction — so this is the normal case, not an "
            "unlucky one." % (identification["intransitivity"] * 100.0)
        )

    if result.get("homophily") is None and result.get("homophily_unavailable"):
        notes.append(
            "No pre-group baselines. Influence and sorting produce the same "
            "trajectory, and without a before-period nothing separates them at "
            "any sample size."
        )

    spillover = result.get("spillover")
    if spillover and spillover["attenuation_expected"]:
        notes.append(
            "The control group is %.0f%% exposed to treated peers. That pulls "
            "the arms together, so any measured effect understates the real one "
            "and a null result is not evidence of absence."
            % (spillover["mean_exposure"] * 100.0)
        )

    if result.get("blocked"):
        notes.append(
            "This is a result, not a missing result. 'Consistent with influence "
            "and with sorting' is what the data supports, and reporting a "
            "number instead would be reporting the design rather than the "
            "world."
        )
    return [note for note in notes if note]


def summarise(result: Mapping[str, Any]) -> str:
    """One line for a log or a saved-analysis list."""
    identification = result.get("identification") or {}
    return "%d members in %d groups | intransitivity %.2f | naive %.2f, LOO %.2f | %s" % (
        result.get("members", 0),
        result.get("groups", 0),
        identification.get("intransitivity", 0.0),
        (result.get("naive") or {}).get("slope", 0.0),
        (result.get("leave_one_out") or {}).get("slope", 0.0),
        "blocked" if result.get("blocked") else "estimated",
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _connect():
    return sqlite3.connect(DB_NAME)


def _ensure_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS peer_effect_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            label TEXT NOT NULL,
            members INTEGER NOT NULL,
            groups_count INTEGER NOT NULL,
            intransitivity REAL NOT NULL,
            blocked INTEGER NOT NULL DEFAULT 0,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_peer_effect_user
        ON peer_effect_analyses (user_id)
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
                INSERT INTO peer_effect_analyses
                    (user_id, label, members, groups_count, intransitivity,
                     blocked, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_id),
                    str(label or "analysis"),
                    int(result.get("members", 0)),
                    int(result.get("groups", 0)),
                    float((result.get("identification") or {}).get("intransitivity", 0.0)),
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
                SELECT id, label, members, groups_count, intransitivity, blocked,
                       payload, created_at
                FROM peer_effect_analyses
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
                "members": row[2],
                "groups": row[3],
                "intransitivity": row[4],
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
                "DELETE FROM peer_effect_analyses WHERE user_id = ? AND id = ?",
                (str(user_id), int(analysis_id)),
            )
            return cursor.rowcount > 0
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------------------
# Worked examples
# ---------------------------------------------------------------------------


def demo_cliques(
    groups: int = 12,
    group_size: int = 8,
    peer_effect: float = 0.0,
    group_sd: float = 0.0,
    individual_sd: float = 300.0,
    grand_mean: float = 3500.0,
    treated_groups: int = 0,
    seed: int = 20240519,
) -> list[dict[str, Any]]:
    """Cliques — how a block or a challenge is actually modelled in this app.

    `group_sd` is the correlated effect: how much groups genuinely differ for
    reasons that have nothing to do with influence. `peer_effect` is a
    one-round diffusion of peers' baselines into the outcome rather than an
    equilibrium of simultaneous influence, which is a simplification and is
    stated here rather than implied — the equilibrium version needs a solver
    and changes none of the identification conclusions.
    """
    import random

    rng = random.Random(seed)
    members = []
    baselines: dict[str, float] = {}
    rosters: dict[str, list[str]] = {}

    for index in range(int(groups)):
        group = "g%02d" % index
        level = grand_mean + rng.gauss(0.0, group_sd)
        rosters[group] = []
        for position in range(int(group_size)):
            identifier = "%s_m%02d" % (group, position)
            baselines[identifier] = level + rng.gauss(0.0, individual_sd)
            rosters[group].append(identifier)

    grand = statistics.fmean(list(baselines.values()))
    for index in range(int(groups)):
        group = "g%02d" % index
        treated = index < int(treated_groups)
        for identifier in rosters[group]:
            others = [
                baselines[peer] for peer in rosters[group] if peer != identifier
            ]
            peer_level = statistics.fmean(others) if others else grand
            outcome = baselines[identifier] + peer_effect * (peer_level - grand)
            members.append(
                build_member(
                    identifier,
                    group,
                    outcome,
                    baseline=baselines[identifier],
                    treated=treated,
                )
            )
    return members


def demo_open_network(
    members_count: int = 120,
    mean_degree: int = 6,
    peer_effect: float = 0.0,
    individual_sd: float = 300.0,
    grand_mean: float = 3500.0,
    groups: int = 8,
    seed: int = 20240519,
) -> list[dict[str, Any]]:
    """A network with open triads — the structure that identifies an effect.

    Links are drawn at random rather than by group, so a friend's friends are
    mostly not your friends. That is the exclusion restriction, and it is also
    not what any community feature in this repo builds.
    """
    import random

    rng = random.Random(seed)
    identifiers = ["n%03d" % index for index in range(int(members_count))]
    baselines = {
        identifier: grand_mean + rng.gauss(0.0, individual_sd)
        for identifier in identifiers
    }

    links: dict[str, set[str]] = {identifier: set() for identifier in identifiers}
    target = max(1, int(members_count * mean_degree / 2))
    for _ in range(target):
        first, second = rng.sample(identifiers, 2)
        links[first].add(second)
        links[second].add(first)

    grand = statistics.fmean(list(baselines.values()))
    members = []
    for index, identifier in enumerate(identifiers):
        neighbours = sorted(links[identifier])
        peer_level = (
            statistics.fmean([baselines[peer] for peer in neighbours])
            if neighbours
            else grand
        )
        outcome = baselines[identifier] + peer_effect * (peer_level - grand)
        members.append(
            build_member(
                identifier,
                "g%02d" % (index % int(groups)),
                outcome,
                baseline=baselines[identifier],
                peers=neighbours,
            )
        )
    return members


def demo_self_selected(
    groups: int = 12,
    group_size: int = 8,
    sorting_strength: float = 400.0,
    individual_sd: float = 250.0,
    seed: int = 20240519,
) -> list[dict[str, Any]]:
    """Opt-in groups: green people joined the green challenge.

    No influence at all — `peer_effect` is zero throughout. Every association
    this produces is sorting, which makes it the right test case for whether
    the module reports one.
    """
    return demo_cliques(
        groups=groups,
        group_size=group_size,
        peer_effect=0.0,
        group_sd=sorting_strength,
        individual_sd=individual_sd,
        seed=seed,
    )
