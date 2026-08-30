"""Did the group change them, or did it collect them?

Members of a green block do have lower footprints. That correlation is
consistent with influence, with shared circumstances, and with green people
moving to green blocks — and no community feature in this app can tell the
three apart.

This page shows two mechanical artefacts in the current construction, both
exact rather than approximate, and then says what the data can and cannot
support.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles.theme import apply_theme
from src.community.peer_effects import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    MIN_INTRANSITIVITY,
    PeerEffectError,
    analyse,
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
    save_analysis,
    spillover_exposure,
    summarise,
    within_group_regression,
)

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>🕸️ Peer Effects</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "The community features assume that seeing your group's behaviour changes "
    "yours. There is a correlation. There are also two other explanations for "
    "it, and one artefact that manufactures it out of nothing."
)

with st.expander("Three worlds, one correlation"):
    st.markdown(
        """
**Endogenous effect.** The group's behaviour changes yours. This is the one the
feature assumes.

**Exogenous (contextual) effect.** The group's *characteristics* affect you.
Everyone on the block has a bus route and a solar co-op.

**Correlated effect.** Nothing affects anything. Green people move to green
blocks, and joined the green challenge for the same reason they were already
green.

**The reflection problem.** Manski's result is not that separating these is
hard. In a group where everyone is connected to everyone, the endogenous effect
is *not identified* — the coefficient does not exist to be estimated. Two
mechanical consequences follow, and both are exact:

*Including yourself in the peer mean returns a slope of one.* For a group of
size `m` with no peer effect at all, `cov(y, groupmean) = var(y)/m` and
`var(groupmean) = var(y)/m`. The same quantity over itself. That is the default
construction in this repo's community features.

*Group fixed effects on a leave-one-out mean return exactly `-(m - 1)`.*
Within a group, `loo = (m*ybar - y)/(m - 1)`, so demeaning leaves the regressor
a perfect negative multiple of the outcome. It will not look like an error — it
will look like a strong negative peer effect.

**What makes it identifiable.** Intransitive structure. If a friend's friends
are your friends, their characteristics reach you directly and there is no
exclusion restriction. Where the network has open triads, a
friend-of-a-friend affects you only through your friend, and that is the
instrument.

**Homophily and influence produce the same trajectory.** Two friends whose
footprints converge look identical whether one persuaded the other or they were
converging anyway. Baselines from before the group formed are what separates
them, and without a before-period nothing does — at any sample size.

**The honest output is usually a refusal.** "Consistent with influence and with
sorting, and here is what would distinguish them" is what the data supports.
        """
    )

(
    tab_artefact,
    tab_identification,
    tab_sorting,
    tab_design,
    tab_saved,
) = st.tabs(
    ["The artefact", "Identification", "Sorting", "The experiment", "Saved"]
)


with tab_artefact:
    st.subheader("Two numbers that are arithmetic, not findings")
    columns = st.columns(3)
    groups = columns[0].slider("Groups", 6, 60, 12, key="art_groups")
    size = columns[1].slider("Members per group", 3, 20, 8, key="art_size")
    sorting = columns[2].slider(
        "Sorting into groups",
        0.0,
        800.0,
        0.0,
        step=50.0,
        key="art_sorting",
        help="How much groups genuinely differ for reasons unrelated to influence.",
    )

    st.caption(
        "The true peer effect below is **zero**. Everything the estimators "
        "report is therefore either sorting or arithmetic, which is what makes "
        "this a usable test of the estimators."
    )

    members = demo_cliques(
        groups=groups, group_size=size, peer_effect=0.0, group_sd=sorting
    )
    st.session_state["pe_members"] = members

    try:
        naive = naive_peer_regression(members)
        loo = leave_one_out_regression(members)
        within = within_group_regression(members)
    except PeerEffectError as error:
        st.error(str(error))
        st.stop()

    head = st.columns(4)
    head[0].metric("True peer effect", "0.000")
    head[1].metric("Self included", "%.3f" % naive["slope"])
    head[2].metric("Leave-one-out", "%.3f" % loo["slope"])
    head[3].metric("Group fixed effects", "%.2f" % within["slope"])

    st.error(naive["headline"])
    if within["collinear"]:
        st.error(within["headline"])
    else:
        st.info(within["headline"])
    st.info(loo["headline"])

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=["Self included", "Leave-one-out", "Group fixed effects"],
            y=[naive["slope"], loo["slope"], within["slope"]],
            marker_color=["#C1443C", "#2E86AB", "#C1443C"],
        )
    )
    figure.add_hline(y=0.0, line_dash="dash", annotation_text="the truth")
    figure.update_layout(
        title="Same data, no peer effect in it, three answers",
        yaxis_title="Estimated peer coefficient",
        height=430,
    )
    st.plotly_chart(figure, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Construction": "Member on their group mean, self included",
                    "Estimate": round(naive["slope"], 4),
                    "What it actually is": "cov/var of the same quantity: 1.0 by "
                    "construction.",
                },
                {
                    "Construction": "Member on their peers' mean, self excluded",
                    "Estimate": round(loo["slope"], 4),
                    "What it actually is": "Real, and confounded by whatever made "
                    "the group a group.",
                },
                {
                    "Construction": "Group fixed effects on the peers' mean",
                    "Estimate": round(within["slope"], 4),
                    "What it actually is": "-(m - 1) = %.0f. The regressor is a "
                    "multiple of the outcome." % within["mechanical_value"],
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Where each member sits against their peers**")
    excluded = peer_means(members, include_self=False)
    included = peer_means(members, include_self=True)
    scatter = go.Figure()
    scatter.add_trace(
        go.Scatter(
            x=[included[member["id"]] for member in members],
            y=[member["outcome"] for member in members],
            mode="markers",
            name="Self included",
            marker={"size": 7, "color": "#C1443C", "opacity": 0.7},
        )
    )
    scatter.add_trace(
        go.Scatter(
            x=[excluded[member["id"]] for member in members],
            y=[member["outcome"] for member in members],
            mode="markers",
            name="Leave-one-out",
            marker={"size": 7, "color": "#2E86AB", "opacity": 0.7},
        )
    )
    scatter.update_layout(
        title="The red cloud is a line because each point helped compute its own x",
        xaxis_title="Peer mean",
        yaxis_title="Member's outcome",
        height=450,
    )
    st.plotly_chart(scatter, use_container_width=True)

with tab_identification:
    st.subheader("Does this network identify anything?")
    structure = st.radio(
        "Network structure",
        ["Cliques (how this app models a block)", "Open network (random links)"],
        horizontal=True,
    )

    if structure.startswith("Cliques"):
        network = demo_cliques(groups=14, group_size=8, peer_effect=0.3, group_sd=200.0)
    else:
        network = demo_open_network(members_count=200, peer_effect=0.3)

    identification = network_identification(network)

    head = st.columns(4)
    head[0].metric("Open triads", identification["open_triads"])
    head[1].metric("Closed triads", identification["closed_triads"])
    head[2].metric("Intransitivity", "%.1f%%" % (identification["intransitivity"] * 100))
    head[3].metric("Mean degree", "%.1f" % identification["mean_degree"])

    if identification["identified"]:
        st.success(identification["headline"])
    else:
        st.error(identification["headline"])

    st.caption(
        "The threshold is %.0f%% open triads. Below it there is no exclusion "
        "restriction and no estimator recovers an endogenous effect — not "
        "imprecisely, but at all."
        % (MIN_INTRANSITIVITY * 100)
    )

    triads = go.Figure()
    triads.add_trace(
        go.Bar(
            x=["Open (identifying)", "Closed (not identifying)"],
            y=[identification["open_triads"], identification["closed_triads"]],
            marker_color=["#2E7D32", "#C1443C"],
        )
    )
    triads.update_layout(
        title="Only open triads carry identifying variation",
        yaxis_title="Triads",
        height=380,
    )
    st.plotly_chart(triads, use_container_width=True)

    st.markdown(
        """
In a clique, a friend's friends are your friends. Their characteristics reach
you directly as well as through your friend, so there is nothing that affects
your friend and not you — no instrument.

In an open triad, a friend-of-a-friend affects you only through your friend.
Their characteristics are the exclusion restriction, and that is the whole of
what makes a linear-in-means model estimable.

Every group in this app's community features is a clique.
        """
    )

with tab_sorting:
    st.subheader("Were they already alike?")
    strength = st.slider(
        "How strongly people sort into groups",
        0.0,
        900.0,
        400.0,
        step=50.0,
        help="Zero means groups are random. High means green people join green groups.",
    )
    selected = demo_self_selected(sorting_strength=strength)

    try:
        loo = leave_one_out_regression(selected)
        homophily = homophily_test(selected)
    except PeerEffectError as error:
        st.error(str(error))
        st.stop()

    st.caption(
        "The true peer effect here is **zero** and every group is opt-in, which "
        "is how every challenge and pledge group in this app is formed."
    )

    head = st.columns(4)
    head[0].metric("True peer effect", "0.000")
    head[1].metric("Leave-one-out estimate", "%.3f" % loo["slope"])
    head[2].metric("Sorting slope", "%.3f" % homophily["sorting_slope"])
    head[3].metric(
        "Influence", "%.3f" % homophily["influence_slope"]
    )

    if homophily["sorting_detected"]:
        st.error(homophily["headline"])
    else:
        st.success(homophily["headline"])

    st.warning(
        "**The leave-one-out estimate is %.3f and the true effect is zero.** "
        "Removing the self-inclusion artefact does not remove sorting; nothing "
        "in a cross-section does." % loo["slope"]
    )

    bars = go.Figure()
    bars.add_trace(
        go.Bar(
            x=[
                "Leave-one-out",
                "Sorting (peers' baseline on own baseline)",
                "Influence (own baseline partialled out)",
                "Truth",
            ],
            y=[
                loo["slope"],
                homophily["sorting_slope"],
                homophily["influence_slope"],
                0.0,
            ],
            marker_color=["#C1443C", "#C1443C", "#2E86AB", "#2E7D32"],
        )
    )
    bars.update_layout(
        title="They were already alike. Nothing is left once that is taken out.",
        yaxis_title="Coefficient",
        height=420,
    )
    st.plotly_chart(bars, use_container_width=True)

    st.markdown("**Full analysis**")
    report = analyse(selected)
    st.session_state["pe_report"] = report

    if report.get("blocked"):
        st.error(report["headline"])
    else:
        st.success(report["headline"])

    for note in get_peer_notes(report):
        st.markdown("- %s" % note)

    st.caption(summarise(report))

    label = st.text_input("Label", value="Opt-in challenge groups")
    if st.button("Save analysis", type="primary"):
        saved = save_analysis(user_id, report, label=label)
        if saved:
            st.success("Saved as #%d." % saved)
        else:
            st.warning("Could not save — storage is unavailable.")

    st.markdown("**Spillover into the control group**")
    treated_groups = st.slider("Groups shown the comparison", 1, 11, 5)
    contaminated = demo_cliques(
        groups=12, group_size=8, peer_effect=0.0, treated_groups=treated_groups
    )
    try:
        spill = spillover_exposure(contaminated)
    except PeerEffectError as error:
        st.info(str(error))
    else:
        columns = st.columns(3)
        columns[0].metric("Controls", spill["controls"])
        columns[1].metric("Uncontaminated", spill["clean_controls"])
        columns[2].metric("Mean exposure", "%.0f%%" % (spill["mean_exposure"] * 100))
        if spill["attenuation_expected"]:
            st.warning(spill["headline"])
        else:
            st.success(spill["headline"])

with tab_design:
    st.subheader("The experiment that would settle it")
    st.markdown(
        "Randomise who is shown a peer comparison, then estimate the effect on "
        "the randomisation rather than on the correlation. Two things this app "
        "does not currently account for decide whether it is worth running."
    )

    columns = st.columns(2)
    groups = columns[0].slider("Groups randomised", 4, 400, 60)
    size = columns[0].slider("Members per group", 2, 100, 20)
    icc = columns[1].slider(
        "Intraclass correlation",
        0.0,
        0.5,
        0.15,
        step=0.01,
        help="How much members of a group resemble each other.",
    )
    compliance = columns[1].slider(
        "Compliance",
        0.1,
        1.0,
        0.5,
        step=0.05,
        help="Share of those encouraged who actually engage with the comparison.",
    )

    outcome_sd = st.slider("Outcome standard deviation", 100.0, 2000.0, 900.0, 50.0)
    alpha = st.select_slider("Alpha", options=[0.01, 0.05, 0.10], value=DEFAULT_ALPHA)
    power = st.select_slider("Power", options=[0.80, 0.90, 0.95], value=DEFAULT_POWER)

    try:
        design = encouragement_power(
            groups=groups,
            group_size=size,
            outcome_sd=outcome_sd,
            intraclass_correlation=icc,
            compliance=compliance,
            alpha=alpha,
            power=power,
        )
    except PeerEffectError as error:
        st.error(str(error))
        st.stop()

    head = st.columns(4)
    head[0].metric("People", design["total_members"])
    head[1].metric("Effective sample", "%.0f" % design["effective_sample"])
    head[2].metric("Design effect", "%.2f" % design["design_effect"])
    head[3].metric("Detectable effect", "%.1f" % design["minimum_detectable_late"])

    st.info(design["headline"])

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Ignoring": "Nothing",
                    "Effective sample": round(design["effective_sample"], 0),
                    "Detectable effect": round(design["minimum_detectable_late"], 1),
                },
                {
                    "Ignoring": "Clustering",
                    "Effective sample": design["total_members"],
                    "Detectable effect": round(
                        design["minimum_detectable_late"]
                        / (design["design_effect"] ** 0.5),
                        1,
                    ),
                },
                {
                    "Ignoring": "Clustering and compliance",
                    "Effective sample": design["total_members"],
                    "Detectable effect": round(
                        design["minimum_detectable_itt"]
                        / (design["design_effect"] ** 0.5),
                        1,
                    ),
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Randomising over groups gives you groups, not people. And encouragement "
        "is not treatment: dividing by a %.0f%% compliance rate means the design "
        "needs %.1fx the sample, not %.1fx."
        % (
            compliance * 100,
            1.0 / (compliance**2),
            1.0 / compliance,
        )
    )

    curve = go.Figure()
    sizes = list(range(10, 401, 10))
    curve.add_trace(
        go.Scatter(
            x=sizes,
            y=[
                encouragement_power(
                    groups=count,
                    group_size=size,
                    outcome_sd=outcome_sd,
                    intraclass_correlation=icc,
                    compliance=compliance,
                    alpha=alpha,
                    power=power,
                )["minimum_detectable_late"]
                for count in sizes
            ],
            mode="lines",
            line={"color": "#2E86AB", "width": 3},
            name="With clustering",
        )
    )
    curve.add_vline(x=groups, line_dash="dash", annotation_text="chosen")
    curve.update_layout(
        title="Smallest peer effect the design could detect",
        xaxis_title="Groups randomised",
        yaxis_title="Detectable effect",
        height=420,
    )
    st.plotly_chart(curve, use_container_width=True)

with tab_saved:
    st.subheader("Saved analyses")
    analyses = get_analyses(user_id)
    if not analyses:
        st.info("Nothing saved yet.")
    else:
        for entry in analyses:
            title = "#%d — %s (%s)" % (
                entry["id"],
                entry["label"],
                entry["created_at"],
            )
            with st.expander(title):
                columns = st.columns(3)
                columns[0].metric("Members", entry["members"])
                columns[1].metric("Groups", entry["groups"])
                columns[2].metric(
                    "Intransitivity", "%.2f" % (entry["intransitivity"] or 0.0)
                )
                if entry["blocked"]:
                    st.error("No endogenous effect was reported for this design.")
                payload = entry.get("payload") or {}
                if payload:
                    st.caption(summarise(payload))
                if st.button("Delete", key="pe_delete_%d" % entry["id"]):
                    if delete_analysis(user_id, entry["id"]):
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.warning("Could not delete.")

st.divider()
st.caption(
    "Manski's reflection problem and the linear-in-means model; "
    "Bramoullé–Djebbari–Fortin identification through intransitive network "
    "structure; homophily versus contagion; randomised encouragement designs "
    "and LATE; SUTVA violation under partial interference. Standard library "
    "only — no dependency added."
)
