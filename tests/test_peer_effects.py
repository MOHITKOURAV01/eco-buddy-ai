"""Tests for the peer effect engine.

Two of the claims under test are *exact*, not statistical, and they are the
reason this module exists. Both are asserted to machine precision rather than
with a tolerance, because an approximate assertion would leave room for the
reader to think they are approximately true:

    Including a member in their own peer mean returns a slope of exactly 1.0,
    with no peer effect present anywhere in the data.

    Group fixed effects on a leave-one-out outcome mean return exactly
    -(m - 1), for every group size, because the demeaned regressor is a perfect
    negative multiple of the demeaned outcome.

The third claim is statistical: removing the self-inclusion artefact leaves a
positive coefficient on self-selected data where the true effect is zero, and
controlling for pre-group baselines removes it. That is sorting, and the test
asserts the whole sequence rather than any one number.

The refusals are tested as hard as the arithmetic, because the failure mode
here is not a wrong number — it is a number reported from a design in which the
quantity is not identified.
"""

import math
import os
import sqlite3
import tempfile

import pytest

from src.community import peer_effects
from src.community.peer_effects import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    ENGINE_VERSION,
    MIN_GROUPS,
    MIN_INTRANSITIVITY,
    MIN_MEMBERS,
    PeerEffectError,
    adjacency,
    analyse,
    betai,
    build_member,
    delete_analysis,
    demo_cliques,
    demo_open_network,
    demo_self_selected,
    encouragement_power,
    get_analyses,
    get_peer_notes,
    homophily_test,
    leave_one_out_regression,
    naive_peer_regression,
    network_identification,
    peer_means,
    regress,
    save_analysis,
    spillover_exposure,
    summarise,
    t_cdf,
    two_sided_p,
    within_group_regression,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(monkeypatch):
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    monkeypatch.setattr(peer_effects, "DB_NAME", path)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def _tiny_panel(groups=4, size=6):
    """Small but valid: enough groups and members to pass validation."""
    return demo_cliques(groups=groups, group_size=size, peer_effect=0.0, seed=1)


# ---------------------------------------------------------------------------
# build_member
# ---------------------------------------------------------------------------


def test_build_member_keeps_the_outcome():
    member = build_member("a", "g1", 3400.0)
    assert member["outcome"] == pytest.approx(3400.0)
    assert member["group"] == "g1"


def test_build_member_without_peers_is_implicit():
    assert build_member("a", "g1", 1.0)["explicit_peers"] is False


def test_an_empty_peer_list_is_still_explicit():
    """An empty list is a claim that they have no links; None is no claim."""
    assert build_member("a", "g1", 1.0, peers=[])["explicit_peers"] is True


def test_build_member_records_a_baseline():
    assert build_member("a", "g1", 1.0, baseline=2.0)["baseline"] == pytest.approx(2.0)


def test_build_member_rejects_a_non_numeric_outcome():
    with pytest.raises(PeerEffectError, match="non-numeric outcome"):
        build_member("a", "g1", "banana")


def test_build_member_rejects_a_non_numeric_baseline():
    with pytest.raises(PeerEffectError, match="non-numeric baseline"):
        build_member("a", "g1", 1.0, baseline=float("nan"))


def test_build_member_defaults_treated_to_false():
    assert build_member("a", "g1", 1.0)["treated"] is False


# ---------------------------------------------------------------------------
# Validation — the refusals
# ---------------------------------------------------------------------------


def test_too_few_members_is_refused():
    members = [build_member("m%d" % i, "g%d" % (i % 4), 1.0) for i in range(8)]
    with pytest.raises(PeerEffectError, match="at least %d members" % MIN_MEMBERS):
        peer_means(members)


def test_too_few_groups_is_refused():
    members = [build_member("m%d" % i, "g%d" % (i % 2), float(i)) for i in range(30)]
    with pytest.raises(PeerEffectError, match="at least %d groups" % MIN_GROUPS):
        peer_means(members)


def test_duplicate_member_ids_are_refused():
    members = _tiny_panel()
    members[3]["id"] = members[0]["id"]
    with pytest.raises(PeerEffectError, match="appears twice"):
        peer_means(members)


def test_raw_mappings_are_refused():
    members = _tiny_panel()
    members[2] = {"id": "x", "group": "g1"}
    with pytest.raises(PeerEffectError, match="build_member"):
        peer_means(members)


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------


def test_a_group_without_explicit_peers_becomes_a_clique():
    members = demo_cliques(groups=4, group_size=5, seed=2)
    links = adjacency(members)
    first = members[0]
    assert len(links[first["id"]]) == 4


def test_explicit_links_are_symmetrised():
    members = [
        build_member("a", "g1", 1.0, peers=["b"]),
        build_member("b", "g1", 2.0, peers=[]),
    ] + [build_member("m%d" % i, "g%d" % (2 + i % 3), float(i)) for i in range(20)]
    links = adjacency(members)
    assert "a" in links["b"]
    assert "b" in links["a"]


def test_links_to_unknown_members_are_dropped():
    members = [
        build_member("a", "g1", 1.0, peers=["ghost"]),
    ] + [
        build_member("m%d" % i, "g%d" % (1 + i % 4), float(i), peers=[])
        for i in range(24)
    ]
    assert adjacency(members)["a"] == set()


def test_an_implicit_group_mate_still_links_to_an_explicit_member():
    """Mixed declarations resolve toward the link existing, not away from it."""
    members = [
        build_member("a", "g1", 1.0, peers=[]),
        build_member("b", "g1", 2.0),
    ] + [build_member("m%d" % i, "g%d" % (2 + i % 3), float(i)) for i in range(20)]
    assert "b" in adjacency(members)["a"]


def test_a_member_is_never_their_own_peer():
    members = [
        build_member("a", "g1", 1.0, peers=["a", "b"]),
        build_member("b", "g1", 2.0, peers=[]),
    ] + [build_member("m%d" % i, "g%d" % (2 + i % 3), float(i)) for i in range(20)]
    assert "a" not in adjacency(members)["a"]


# ---------------------------------------------------------------------------
# Peer means
# ---------------------------------------------------------------------------


def test_leave_one_out_excludes_the_member():
    members = [
        build_member("a", "g1", 10.0),
        build_member("b", "g1", 20.0),
        build_member("c", "g1", 30.0),
    ] + [build_member("m%d" % i, "g%d" % (2 + i % 3), float(i)) for i in range(20)]
    means = peer_means(members, include_self=False)
    assert means["a"] == pytest.approx(25.0)


def test_self_inclusion_gives_the_group_mean():
    members = [
        build_member("a", "g1", 10.0),
        build_member("b", "g1", 20.0),
        build_member("c", "g1", 30.0),
    ] + [build_member("m%d" % i, "g%d" % (2 + i % 3), float(i)) for i in range(20)]
    means = peer_means(members, include_self=True)
    assert means["a"] == pytest.approx(20.0)
    assert means["b"] == pytest.approx(20.0)


def test_peer_means_can_read_the_baseline_field():
    members = demo_cliques(groups=4, group_size=5, seed=3)
    means = peer_means(members, field="baseline", include_self=False)
    assert all(value is not None for value in means.values())


def test_a_member_with_no_peers_has_no_mean():
    members = [build_member("lonely", "g1", 1.0, peers=[])] + [
        build_member("m%d" % i, "g%d" % (1 + i % 4), float(i), peers=[])
        for i in range(24)
    ]
    assert peer_means(members, include_self=False)["lonely"] is None


# ---------------------------------------------------------------------------
# The self-inclusion artefact — exact
# ---------------------------------------------------------------------------


def test_self_inclusion_returns_exactly_one_with_no_peer_effect():
    """cov(y, groupmean) = var(y)/m and var(groupmean) = var(y)/m."""
    members = demo_cliques(groups=15, group_size=8, peer_effect=0.0, group_sd=0.0, seed=4)
    assert naive_peer_regression(members)["slope"] == pytest.approx(1.0, abs=1e-9)


def test_self_inclusion_returns_one_regardless_of_group_size():
    for size in (3, 5, 12, 20):
        members = demo_cliques(groups=10, group_size=size, peer_effect=0.0, seed=5)
        assert naive_peer_regression(members)["slope"] == pytest.approx(1.0, abs=1e-9)


def test_self_inclusion_returns_one_regardless_of_sorting():
    for sorting in (0.0, 300.0, 1500.0):
        members = demo_cliques(groups=12, group_size=8, group_sd=sorting, seed=6)
        assert naive_peer_regression(members)["slope"] == pytest.approx(1.0, abs=1e-9)


def test_self_inclusion_returns_one_even_with_a_real_peer_effect():
    """It is not measuring the effect. It is not measuring anything."""
    members = demo_cliques(groups=12, group_size=8, peer_effect=0.7, seed=7)
    assert naive_peer_regression(members)["slope"] == pytest.approx(1.0, abs=1e-9)


def test_the_naive_regression_labels_itself_mechanical():
    result = naive_peer_regression(_tiny_panel(groups=8))
    assert result["mechanical"] is True
    assert "not an estimate of anything" in result["headline"]


# ---------------------------------------------------------------------------
# The group fixed effects artefact — exact
# ---------------------------------------------------------------------------


def test_group_fixed_effects_return_exactly_minus_m_minus_one():
    for size in (4, 6, 8, 15):
        members = demo_cliques(groups=20, group_size=size, group_sd=400.0, seed=8)
        result = within_group_regression(members)
        assert result["slope"] == pytest.approx(-(size - 1), abs=1e-6)
        assert result["mechanical_value"] == pytest.approx(-(size - 1))


def test_the_within_group_collinearity_is_detected():
    result = within_group_regression(demo_cliques(groups=15, group_size=8, seed=9))
    assert result["collinear"] is True
    assert result["identified"] is False
    assert "reflection problem" in result["headline"]


def test_the_within_group_artefact_is_unchanged_by_a_real_peer_effect():
    plain = within_group_regression(demo_cliques(groups=15, group_size=8, seed=10))
    effect = within_group_regression(
        demo_cliques(groups=15, group_size=8, peer_effect=0.8, seed=10)
    )
    assert plain["slope"] == pytest.approx(effect["slope"], abs=1e-6)


def test_within_group_is_not_collinear_on_an_open_network():
    result = within_group_regression(demo_open_network(members_count=200, seed=11))
    assert result["collinear"] is False
    assert result["identified"] is True


# ---------------------------------------------------------------------------
# Leave-one-out
# ---------------------------------------------------------------------------


def test_leave_one_out_goes_to_zero_with_no_effect_and_no_sorting():
    members = demo_cliques(
        groups=400, group_size=8, peer_effect=0.0, group_sd=0.0, seed=12
    )
    assert leave_one_out_regression(members)["slope"] == pytest.approx(0.0, abs=0.10)


def test_leave_one_out_is_far_from_the_self_included_estimate():
    members = demo_cliques(groups=300, group_size=8, peer_effect=0.0, group_sd=0.0, seed=13)
    naive = naive_peer_regression(members)["slope"]
    loo = leave_one_out_regression(members)["slope"]
    assert abs(naive - loo) > 0.5


def test_leave_one_out_rises_with_the_true_peer_effect():
    slopes = [
        leave_one_out_regression(
            demo_open_network(members_count=400, peer_effect=effect, seed=14)
        )["slope"]
        for effect in (0.0, 0.3, 0.6, 0.9)
    ]
    assert slopes == sorted(slopes)


def test_leave_one_out_is_positive_under_pure_sorting():
    """The true effect is zero and every association is selection."""
    members = demo_self_selected(groups=40, group_size=8, sorting_strength=600.0, seed=15)
    assert leave_one_out_regression(members)["slope"] > 0.3


def test_leave_one_out_labels_itself_confounded():
    result = leave_one_out_regression(_tiny_panel(groups=8))
    assert result["confounded"] is True
    assert "Sorting is not" in result["headline"]


def test_leave_one_out_clusters_by_group():
    result = leave_one_out_regression(demo_cliques(groups=20, group_size=8, seed=16))
    assert result["clusters"] == 20


# ---------------------------------------------------------------------------
# Clustered standard errors
# ---------------------------------------------------------------------------


def test_clustering_changes_the_standard_error():
    xs = [float(i % 10) for i in range(80)]
    ys = [x * 2.0 + (5.0 if i % 3 else -5.0) for i, x in enumerate(xs)]
    clusters = ["c%d" % (i // 8) for i in range(80)]
    plain = regress(xs, ys)
    clustered = regress(xs, ys, clusters=clusters)
    assert plain["slope"] == pytest.approx(clustered["slope"])
    assert plain["standard_error"] != pytest.approx(clustered["standard_error"])


def test_clustered_degrees_of_freedom_come_from_the_clusters():
    xs = [float(i % 10) for i in range(80)]
    ys = [x * 2.0 for x in xs]
    clusters = ["c%d" % (i // 8) for i in range(80)]
    assert regress(xs, ys, clusters=clusters)["degrees_of_freedom"] == 9


def test_a_single_cluster_is_refused():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    with pytest.raises(PeerEffectError, match="at least two clusters"):
        regress(xs, ys, clusters=["one"] * 5)


def test_mismatched_cluster_length_is_refused():
    with pytest.raises(PeerEffectError, match="same length as the data"):
        regress([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], clusters=["a"])


def test_regress_refuses_a_constant_regressor():
    with pytest.raises(PeerEffectError, match="does not vary"):
        regress([1.0] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_regress_refuses_mismatched_lengths():
    with pytest.raises(PeerEffectError, match="same length"):
        regress([1.0, 2.0], [1.0])


def test_regress_recovers_an_exact_line():
    fit = regress([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 5.0, 7.0, 9.0, 11.0])
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def test_cliques_have_no_open_triads():
    result = network_identification(demo_cliques(groups=10, group_size=6, seed=17))
    assert result["open_triads"] == 0
    assert result["intransitivity"] == pytest.approx(0.0)
    assert result["identified"] is False


def test_a_random_network_is_mostly_open_triads():
    result = network_identification(demo_open_network(members_count=200, seed=18))
    assert result["intransitivity"] > MIN_INTRANSITIVITY
    assert result["identified"] is True


def test_identification_explains_the_clique_case():
    result = network_identification(demo_cliques(groups=10, group_size=6, seed=19))
    assert "not identified" in result["headline"]
    assert "nothing to recover" in result["headline"]


def test_identification_reports_the_mean_degree():
    result = network_identification(demo_cliques(groups=10, group_size=7, seed=20))
    assert result["mean_degree"] == pytest.approx(6.0)


def test_identification_counts_isolated_members():
    members = [build_member("lonely", "g1", 1.0, peers=[])] + [
        build_member(
            "m%d" % i,
            "g%d" % (1 + i % 4),
            float(i),
            peers=["m%d" % ((i + 1) % 24)],
        )
        for i in range(24)
    ]
    assert network_identification(members)["isolated"] >= 1


# ---------------------------------------------------------------------------
# Homophily
# ---------------------------------------------------------------------------


def test_sorting_is_detected_when_groups_are_self_selected():
    members = demo_self_selected(groups=40, group_size=8, sorting_strength=600.0, seed=21)
    result = homophily_test(members)
    assert result["sorting_detected"] is True
    assert result["sorting_slope"] > 0


def test_pure_sorting_leaves_no_influence():
    """Zero true effect: everything the sorting slope found was who they already were."""
    members = demo_self_selected(groups=40, group_size=8, sorting_strength=600.0, seed=22)
    result = homophily_test(members)
    assert result["sorting_slope"] > 0.5
    assert result["influence_slope"] == pytest.approx(0.0, abs=1e-6)
    assert result["influence_detected"] is False


def test_influence_recovers_the_true_effect_on_an_open_network():
    """Peers' baselines against the member's change, own baseline partialled out."""
    for seed in range(5):
        result = homophily_test(
            demo_open_network(members_count=300, peer_effect=0.6, seed=seed)
        )
        assert result["influence_slope"] == pytest.approx(0.6, abs=0.02)


def test_testing_sorting_against_the_outcome_would_conflate_the_two():
    """The construction this module deliberately avoids.

    Under a real peer effect, peers' baselines predict the member's outcome
    exactly as they would under sorting. Testing on the outcome would report
    sorting here and block a genuine effect, which is why the test runs on the
    member's own baseline instead.
    """
    members = demo_open_network(members_count=300, peer_effect=0.6, seed=0)
    baseline_means = peer_means(members, field="baseline", include_self=False)
    usable = [m for m in members if baseline_means[m["id"]] is not None]
    conflated = regress(
        [baseline_means[m["id"]] for m in usable],
        [m["outcome"] for m in usable],
        clusters=[m["group"] for m in usable],
    )
    assert conflated["p_value"] < DEFAULT_ALPHA
    assert homophily_test(members)["sorting_detected"] is False


def test_no_sorting_is_found_when_groups_are_random():
    members = demo_cliques(groups=40, group_size=8, group_sd=0.0, seed=23)
    assert homophily_test(members)["sorting_detected"] is False


def test_homophily_is_refused_without_baselines():
    members = [
        build_member("m%d" % i, "g%d" % (i % 5), float(i % 17)) for i in range(40)
    ]
    with pytest.raises(PeerEffectError, match="without baselines"):
        homophily_test(members)


def test_the_homophily_refusal_says_why_it_matters():
    members = [
        build_member("m%d" % i, "g%d" % (i % 5), float(i % 17)) for i in range(40)
    ]
    with pytest.raises(PeerEffectError, match="indistinguishable at any sample size"):
        homophily_test(members)


# ---------------------------------------------------------------------------
# Spillover
# ---------------------------------------------------------------------------


def test_separated_arms_have_no_exposure():
    """Whole groups are treated, so within a clique nobody has a mixed network."""
    members = demo_cliques(groups=12, group_size=8, treated_groups=6, seed=24)
    result = spillover_exposure(members)
    assert result["mean_exposure"] == pytest.approx(0.0)
    assert result["attenuation_expected"] is False


def test_mixed_networks_produce_exposure():
    """Randomising within a group leaves every control surrounded by treated peers."""
    members = []
    for index in range(48):
        members.append(
            build_member(
                "m%02d" % index,
                "g%d" % (index // 8),
                float(1000 + index),
                treated=index % 2 == 0,
            )
        )
    result = spillover_exposure(members)
    assert result["mean_exposure"] > 0
    assert "not evidence of no effect" in result["headline"]


def test_spillover_needs_a_treated_arm():
    with pytest.raises(PeerEffectError, match="No treated members"):
        spillover_exposure(demo_cliques(groups=8, group_size=6, seed=25))


def test_spillover_needs_a_control_arm():
    members = demo_cliques(groups=8, group_size=6, treated_groups=8, seed=26)
    with pytest.raises(PeerEffectError, match="no control group"):
        spillover_exposure(members)


def test_exposure_is_reported_per_control():
    members = []
    for index in range(48):
        members.append(
            build_member(
                "m%02d" % index,
                "g%d" % (index // 8),
                float(1000 + index),
                treated=index % 3 == 0,
            )
        )
    result = spillover_exposure(members)
    assert len(result["exposures"]) == result["controls"]
    assert all(0.0 <= entry["exposure"] <= 1.0 for entry in result["exposures"])


# ---------------------------------------------------------------------------
# Encouragement design
# ---------------------------------------------------------------------------


def test_the_design_effect_is_one_plus_m_minus_one_times_icc():
    design = encouragement_power(
        groups=50, group_size=20, outcome_sd=900.0, intraclass_correlation=0.15
    )
    assert design["design_effect"] == pytest.approx(1.0 + 19 * 0.15)


def test_clustering_shrinks_the_effective_sample():
    design = encouragement_power(
        groups=50, group_size=20, outcome_sd=900.0, intraclass_correlation=0.15
    )
    assert design["effective_sample"] < design["total_members"]


def test_a_zero_icc_leaves_the_sample_intact():
    design = encouragement_power(
        groups=50, group_size=20, outcome_sd=900.0, intraclass_correlation=0.0
    )
    assert design["design_effect"] == pytest.approx(1.0)
    assert design["effective_sample"] == pytest.approx(design["total_members"])


def test_more_groups_detect_smaller_effects():
    small = encouragement_power(
        groups=20, group_size=20, outcome_sd=900.0, intraclass_correlation=0.1
    )
    large = encouragement_power(
        groups=200, group_size=20, outcome_sd=900.0, intraclass_correlation=0.1
    )
    assert large["minimum_detectable_late"] < small["minimum_detectable_itt"]


def test_the_detectable_effect_scales_as_one_over_root_n():
    base = encouragement_power(
        groups=50, group_size=1, outcome_sd=900.0, intraclass_correlation=0.0
    )
    quadrupled = encouragement_power(
        groups=200, group_size=1, outcome_sd=900.0, intraclass_correlation=0.0
    )
    assert quadrupled["minimum_detectable_itt"] == pytest.approx(
        base["minimum_detectable_itt"] / 2.0
    )


def test_partial_compliance_inflates_the_detectable_effect():
    full = encouragement_power(
        groups=50, group_size=20, outcome_sd=900.0, intraclass_correlation=0.1,
        compliance=1.0,
    )
    half = encouragement_power(
        groups=50, group_size=20, outcome_sd=900.0, intraclass_correlation=0.1,
        compliance=0.5,
    )
    assert half["minimum_detectable_late"] == pytest.approx(
        full["minimum_detectable_late"] * 2.0
    )


def test_higher_power_needs_a_larger_effect_to_detect():
    low = encouragement_power(
        groups=50, group_size=20, outcome_sd=900.0, intraclass_correlation=0.1,
        power=0.80,
    )
    high = encouragement_power(
        groups=50, group_size=20, outcome_sd=900.0, intraclass_correlation=0.1,
        power=0.95,
    )
    assert high["minimum_detectable_late"] > low["minimum_detectable_late"]


def test_the_design_refuses_a_single_group():
    with pytest.raises(PeerEffectError, match="at least two groups"):
        encouragement_power(1, 20, 900.0, 0.1)


def test_the_design_refuses_a_non_positive_outcome_sd():
    with pytest.raises(PeerEffectError, match="must be positive"):
        encouragement_power(50, 20, 0.0, 0.1)


def test_the_design_refuses_an_icc_of_one():
    with pytest.raises(PeerEffectError, match=r"in \[0, 1\)"):
        encouragement_power(50, 20, 900.0, 1.0)


def test_the_design_refuses_zero_compliance():
    with pytest.raises(PeerEffectError, match=r"in \(0, 1\]"):
        encouragement_power(50, 20, 900.0, 0.1, compliance=0.0)


def test_the_design_refuses_an_unsupported_alpha():
    with pytest.raises(PeerEffectError, match="Alpha must be"):
        encouragement_power(50, 20, 900.0, 0.1, alpha=0.037)


def test_the_design_refuses_an_unsupported_power():
    with pytest.raises(PeerEffectError, match="Power must be"):
        encouragement_power(50, 20, 900.0, 0.1, power=0.5)


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------


def test_cliques_block_the_endogenous_effect():
    result = analyse(demo_cliques(groups=20, group_size=8, peer_effect=0.6, seed=27))
    assert result["blocked"]
    assert result["endogenous_effect"] is None
    assert "cliques" in result["blocked"]


def test_self_selected_groups_block_on_sorting():
    result = analyse(demo_self_selected(groups=30, group_size=8, seed=28))
    assert result["blocked"]
    assert "already alike" in result["blocked"]


def test_missing_baselines_block_the_estimate():
    members = [
        build_member("m%d" % i, "g%d" % (i % 6), float(1000 + (i * 37) % 500))
        for i in range(60)
    ]
    result = analyse(members)
    assert result["blocked"]
    assert "no pre-group baselines" in result["blocked"]


def test_an_open_network_without_sorting_can_report_an_effect():
    result = analyse(demo_open_network(members_count=300, peer_effect=0.6, seed=0))
    assert result["blocked"] is None
    assert result["endogenous_effect"] is not None


def test_analyse_always_reports_both_constructions():
    result = analyse(demo_cliques(groups=15, group_size=8, seed=30))
    assert result["naive"]["slope"] == pytest.approx(1.0, abs=1e-9)
    assert result["leave_one_out"]["slope"] != pytest.approx(1.0, abs=1e-6)


def test_analyse_carries_the_engine_version():
    assert analyse(_tiny_panel(groups=8))["engine_version"] == ENGINE_VERSION


def test_analyse_counts_members_and_groups():
    result = analyse(demo_cliques(groups=9, group_size=7, seed=31))
    assert result["members"] == 63
    assert result["groups"] == 9


def test_analyse_lists_every_reason_it_blocked():
    result = analyse(demo_self_selected(groups=30, group_size=8, seed=32))
    assert "; and " in result["blocked"]


# ---------------------------------------------------------------------------
# Notes and summaries
# ---------------------------------------------------------------------------


def test_notes_lead_with_the_headline():
    result = analyse(_tiny_panel(groups=8))
    assert get_peer_notes(result)[0] == result["headline"]


def test_notes_compare_the_two_constructions():
    notes = get_peer_notes(analyse(demo_cliques(groups=15, group_size=8, seed=33)))
    assert any("close to 1.0 whether or not" in note for note in notes)


def test_notes_explain_the_minus_m_minus_one_result():
    notes = get_peer_notes(analyse(demo_cliques(groups=15, group_size=8, seed=34)))
    assert any("is arithmetic" in note for note in notes)


def test_notes_say_a_block_is_a_result():
    notes = get_peer_notes(analyse(demo_cliques(groups=15, group_size=8, seed=35)))
    assert any("not a missing result" in note for note in notes)


def test_notes_flag_missing_baselines():
    members = [
        build_member("m%d" % i, "g%d" % (i % 6), float(1000 + (i * 37) % 500))
        for i in range(60)
    ]
    notes = get_peer_notes(analyse(members))
    assert any("No pre-group baselines" in note for note in notes)


def test_summary_is_one_line():
    summary = summarise(analyse(_tiny_panel(groups=8)))
    assert "\n" not in summary
    assert "intransitivity" in summary


def test_summary_says_blocked_when_blocked():
    assert summarise(analyse(demo_cliques(groups=15, group_size=8, seed=36))).endswith(
        "blocked"
    )


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def test_t_cdf_is_a_half_at_zero():
    assert t_cdf(0.0, 10) == pytest.approx(0.5)


def test_two_sided_p_of_zero_is_one():
    assert two_sided_p(0.0, 10) == pytest.approx(1.0)


def test_two_sided_p_falls_as_the_statistic_grows():
    assert two_sided_p(4.0, 10) < two_sided_p(1.0, 10)


def test_betai_endpoints():
    assert betai(2.0, 3.0, 0.0) == pytest.approx(0.0)
    assert betai(2.0, 3.0, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def test_demo_cliques_is_deterministic_for_a_seed():
    first = demo_cliques(groups=6, group_size=5, seed=777)
    second = demo_cliques(groups=6, group_size=5, seed=777)
    assert [m["outcome"] for m in first] == [m["outcome"] for m in second]


def test_demo_cliques_honours_its_shape():
    members = demo_cliques(groups=7, group_size=9, seed=38)
    assert len(members) == 63
    assert len({member["group"] for member in members}) == 7


def test_demo_cliques_can_treat_whole_groups():
    members = demo_cliques(groups=10, group_size=5, treated_groups=4, seed=39)
    assert sum(1 for member in members if member["treated"]) == 20


def test_demo_open_network_gives_explicit_peers():
    members = demo_open_network(members_count=60, seed=40)
    assert all(member["explicit_peers"] for member in members)


def test_demo_self_selected_has_no_true_peer_effect():
    """Every association it produces must therefore be selection."""
    plain = demo_self_selected(
        groups=8, group_size=6, sorting_strength=0.0, individual_sd=250.0, seed=41
    )
    reference = demo_cliques(
        groups=8, group_size=6, peer_effect=0.0, group_sd=0.0,
        individual_sd=250.0, seed=41,
    )
    assert [m["outcome"] for m in plain] == pytest.approx(
        [m["outcome"] for m in reference]
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_read_back_an_analysis(temp_db):
    result = analyse(_tiny_panel(groups=8))
    analysis_id = save_analysis("user-1", result, label="Block A")
    assert analysis_id is not None

    analyses = get_analyses("user-1")
    assert len(analyses) == 1
    assert analyses[0]["label"] == "Block A"
    assert analyses[0]["payload"]["engine_version"] == ENGINE_VERSION


def test_a_blocked_analysis_is_stored_as_blocked(temp_db):
    save_analysis("user-1", analyse(demo_cliques(groups=15, group_size=8, seed=42)))
    assert get_analyses("user-1")[0]["blocked"] is True


def test_analyses_are_scoped_to_their_user(temp_db):
    save_analysis("user-1", analyse(_tiny_panel(groups=8)))
    assert get_analyses("user-2") == []


def test_saving_without_a_user_is_a_no_op(temp_db):
    assert save_analysis("", analyse(_tiny_panel(groups=8))) is None


def test_saving_a_non_result_is_a_no_op(temp_db):
    assert save_analysis("user-1", {}) is None


def test_delete_removes_only_the_named_analysis(temp_db):
    result = analyse(_tiny_panel(groups=8))
    first = save_analysis("user-1", result, label="one")
    save_analysis("user-1", result, label="two")
    assert delete_analysis("user-1", first) is True
    remaining = get_analyses("user-1")
    assert len(remaining) == 1
    assert remaining[0]["label"] == "two"


def test_delete_refuses_another_users_analysis(temp_db):
    analysis_id = save_analysis("user-1", analyse(_tiny_panel(groups=8)))
    assert delete_analysis("user-2", analysis_id) is False


def test_delete_without_a_user_is_false(temp_db):
    assert delete_analysis("", 1) is False


def test_reads_without_a_user_are_empty(temp_db):
    assert get_analyses(None) == []


def test_analyses_come_back_newest_first(temp_db):
    result = analyse(_tiny_panel(groups=8))
    save_analysis("user-1", result, label="older")
    save_analysis("user-1", result, label="newer")
    assert [entry["label"] for entry in get_analyses("user-1")] == ["newer", "older"]


def test_storage_failure_is_swallowed_not_raised(monkeypatch):
    """A dashboard must render when the database is unavailable."""

    def explode(*_args, **_kwargs):
        raise sqlite3.Error("disk is on fire")

    monkeypatch.setattr(peer_effects, "_connect", explode)
    assert save_analysis("user-1", analyse(_tiny_panel(groups=8))) is None
    assert get_analyses("user-1") == []
    assert delete_analysis("user-1", 1) is False


def test_a_corrupt_payload_reads_back_as_empty(temp_db):
    save_analysis("user-1", analyse(_tiny_panel(groups=8)))
    with sqlite3.connect(temp_db) as conn:
        conn.execute("UPDATE peer_effect_analyses SET payload = 'not json'")
    assert get_analyses("user-1")[0]["payload"] == {}
