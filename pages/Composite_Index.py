"""Seven numbers that sum to 100, and nothing that checks them.

`confidence_scoring.py` weights seven factors, adds them, and cuts the result
into three bands at 85 and 60. `eco_score.py` adds three category scores on
three different scales. Both are composite indicators, and neither runs any of
the standard checks a composite indicator gets.

This page runs them: what the weights actually did, whether a defensible
alternative weighting changes the answer, and what the choice to add rather
than multiply is quietly asserting.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles.theme import apply_theme
from src.utils.composite_index import (
    APP_CONFIDENCE_BANDS,
    APP_CONFIDENCE_WEIGHTS,
    DEFAULT_DRAWS,
    MIN_BAND_PROBABILITY,
    NORMALISATIONS,
    REDUNDANCY_THRESHOLD,
    CompositeError,
    analyse,
    band_probabilities,
    build_index,
    delete_analysis,
    demo_compensating_index,
    demo_confidence_index,
    demo_eco_score,
    get_analyses,
    get_index_notes,
    raw_sum_effective_weights,
    save_analysis,
    summarise,
)

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>⚖️ Composite Index</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "The scores in this app are weighted sums. The weights are not wrong — the "
    "point is that nothing tests what happens if they are different."
)

with st.expander("Three choices this app makes without making them"):
    st.markdown(
        """
**Normalisation.** Adding numbers on different scales lets the units do the
weighting. In `eco_score.py` a diet factor runs 1.5–7.0, a transport factor
0.1–8.0, and `energy_kwh * 0.5` runs into the hundreds. Choosing "none" is
still choosing.

**Aggregation.** A linear sum is fully compensatory: a perfect score on one
component buys back a zero on another. That may be the intended policy. It is
not a policy anyone chose — it is what addition does — and for a *confidence*
score it is close to indefensible, because an assessment with unknown emission
factors is not rescued by having every field filled in.

**Weighting.** In a linear sum, influence depends on variance as much as on the
nominal weight. A component with weight 20 that barely varies contributes almost
nothing to the ranking; one with weight 5 and a wide spread can dominate it.
The standard diagnostic is the Pearson correlation ratio:

    S_i = Var(E[Y | X_i]) / Var(Y)

The dictionary says what was intended. This says what happened.

**The question that matters.** Would a defensible alternative weighting change
the answer? If the ranking is stable across reasonable weightings, the exact
numbers do not matter and the score is robust. If it is not, the score is a
statement about the weights rather than about the user — and that fact belongs
next to it.
        """
    )

(
    tab_confidence,
    tab_scales,
    tab_compensation,
    tab_bands,
    tab_saved,
) = st.tabs(
    ["Effective weights", "Scales", "Compensation", "Bands", "Saved"]
)


with tab_confidence:
    st.subheader("What the confidence weights actually did")
    st.caption(
        "Built from the real `CONFIDENCE_WEIGHTS` in "
        "`src/carbon/confidence_scoring.py`. Two components — "
        "`input_completeness` and `category_coverage` — are close to the same "
        "measurement there, and between them they carry 35 of the 100 points."
    )

    columns = st.columns(3)
    units = columns[0].slider("Assessments", 10, 200, 40, key="ci_units")
    normalisation = columns[1].selectbox(
        "Normalisation", list(NORMALISATIONS), index=0, key="ci_norm"
    )
    draws = columns[2].slider("Reweighting draws", 100, 2000, DEFAULT_DRAWS, 100)

    names, components = demo_confidence_index(units=units)
    try:
        report = analyse(names, components, normalisation=normalisation, draws=draws)
    except CompositeError as error:
        st.error(str(error))
        st.stop()

    st.session_state["ci_report"] = report
    effective = report["effective_weights"]

    head = st.columns(4)
    head[0].metric("Components", len(components))
    head[1].metric("Assessments", units)
    head[2].metric("Unstable ranks", report["sensitivity"]["unstable"])
    head[3].metric(
        "Most overworked",
        "%.2fx" % effective["most_overworked_ratio"],
        help="'%s' — effective weight over nominal." % effective["most_overworked"],
    )

    st.info(effective["headline"])

    figure = go.Figure()
    order = sorted(
        effective["nominal"], key=lambda name: effective["nominal"][name], reverse=True
    )
    figure.add_trace(
        go.Bar(
            x=order,
            y=[effective["nominal"][name] * 100 for name in order],
            name="Nominal",
            marker_color="#999999",
        )
    )
    figure.add_trace(
        go.Bar(
            x=order,
            y=[effective["effective"][name] * 100 for name in order],
            name="Effective",
            marker_color="#2E86AB",
        )
    )
    figure.update_layout(
        title="What the weights say against what the ranking did",
        yaxis_title="Share of influence (%)",
        barmode="group",
        height=440,
    )
    st.plotly_chart(figure, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Component": name,
                    "Nominal": APP_CONFIDENCE_WEIGHTS.get(name, 0.0),
                    "Nominal %": round(effective["nominal"][name] * 100, 1),
                    "Effective %": round(effective["effective"][name] * 100, 1),
                    "Ratio": round(effective["ratios"][name], 2),
                }
                for name in order
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Redundancy**")
    correlations = report["correlations"]
    if correlations["redundant_pairs"]:
        st.warning(correlations["headline"])
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Component A": pair["components"][0],
                        "Component B": pair["components"][1],
                        "Correlation": round(pair["correlation"], 3),
                        "Combined weight": pair["combined_weight"],
                    }
                    for pair in correlations["redundant_pairs"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success(correlations["headline"])
    st.caption(
        "Two components correlated above %.2f are one construct wearing two "
        "hats, and their combined weight buys it that much influence — not "
        "twice that much." % REDUNDANCY_THRESHOLD
    )

    st.markdown("**Would a different weighting change the answer?**")
    sensitivity = report["sensitivity"]
    if sensitivity["robust"]:
        st.success(sensitivity["headline"])
    else:
        st.error(sensitivity["headline"])

    top = sensitivity["intervals"][:20]
    interval = go.Figure()
    interval.add_trace(
        go.Scatter(
            x=[entry["baseline_rank"] for entry in top],
            y=[entry["unit"] for entry in top],
            mode="markers",
            marker={"size": 9, "color": "#2E86AB"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [entry["upper"] - entry["baseline_rank"] for entry in top],
                "arrayminus": [
                    entry["baseline_rank"] - entry["lower"] for entry in top
                ],
            },
            name="Rank interval",
        )
    )
    interval.update_layout(
        title="Rank under %d defensible reweightings" % sensitivity["draws"],
        xaxis_title="Rank",
        height=max(420, 22 * len(top)),
    )
    st.plotly_chart(interval, use_container_width=True)

    for note in get_index_notes(report):
        st.markdown("- %s" % note)

    st.caption(summarise(report))

    label = st.text_input("Label", value="Confidence score")
    if st.button("Save analysis", type="primary"):
        saved = save_analysis(user_id, report, label=label)
        if saved:
            st.success("Saved as #%d." % saved)
        else:
            st.warning("Could not save — storage is unavailable.")

with tab_scales:
    st.subheader("When the units do the weighting")
    st.caption(
        "Three components on the scales `src/calculators/eco_score.py` actually "
        "adds: a diet factor around 1.5–7.0, a transport factor around 0.1–8.0, "
        "and `energy_kwh * 0.5` running into the hundreds. All three carry "
        "equal nominal weight."
    )

    names, components = demo_eco_score(units=60)
    try:
        mismatch = raw_sum_effective_weights(components)
        normalised = build_index(names, components, normalisation="minmax")
    except CompositeError as error:
        st.error(str(error))
        st.stop()

    st.error(mismatch["headline"])

    unnormalised_weights = mismatch["effective_weights"]
    normalised_weights = normalised["effective_weights"]

    comparison = go.Figure()
    order = ["diet", "transport", "energy"]
    comparison.add_trace(
        go.Bar(
            x=order,
            y=[unnormalised_weights["nominal"][name] * 100 for name in order],
            name="Nominal",
            marker_color="#999999",
        )
    )
    comparison.add_trace(
        go.Bar(
            x=order,
            y=[unnormalised_weights["effective"][name] * 100 for name in order],
            name="Effective, added as-is",
            marker_color="#C1443C",
        )
    )
    comparison.add_trace(
        go.Bar(
            x=order,
            y=[normalised_weights["effective"][name] * 100 for name in order],
            name="Effective, min-max normalised",
            marker_color="#2E7D32",
        )
    )
    comparison.update_layout(
        title="Equal nominal weights, one component doing most of the work",
        yaxis_title="Share of influence (%)",
        barmode="group",
        height=440,
    )
    st.plotly_chart(comparison, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Component": name,
                    "Nominal %": round(unnormalised_weights["nominal"][name] * 100, 1),
                    "Added as-is %": round(
                        unnormalised_weights["effective"][name] * 100, 1
                    ),
                    "Normalised %": round(
                        normalised_weights["effective"][name] * 100, 1
                    ),
                }
                for name in order
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The `categories_processed` field in that module's output implies a "
        "balance the arithmetic does not have. Normalisation is what gives the "
        "nominal weights their meaning back — which is why the fix has to run "
        "before the sum, not after it."
    )

    st.markdown("**The four normalisations are not interchangeable**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Method": "minmax",
                    "Keeps": "Shape and relative distance",
                    "Costs": "One outlier compresses everyone else.",
                },
                {
                    "Method": "zscore",
                    "Keeps": "Distance in standard deviations",
                    "Costs": "Assumes the spread is meaningful; unbounded.",
                },
                {
                    "Method": "rank",
                    "Keeps": "Order only",
                    "Costs": "A unit twice as good is one position better.",
                },
                {
                    "Method": "distance_to_reference",
                    "Keeps": "Distance to a target",
                    "Costs": "Needs a defensible target; the only one whose "
                    "values do not move when a unit joins.",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_compensation:
    st.subheader("Is a zero allowed to be bought back?")
    st.caption(
        "A set built so that compensation matters: some units are uniformly "
        "mediocre, others are excellent on two components and near-zero on a "
        "third. A linear sum ranks them together. A geometric mean does not."
    )

    names, components = demo_compensating_index(units=30)
    try:
        report = analyse(names, components)
    except CompositeError as error:
        st.error(str(error))
        st.stop()

    head = st.columns(3)
    head[0].metric(
        "Linear vs geometric",
        "%d reversals" % report["linear_vs_geometric"]["count"],
    )
    head[1].metric(
        "Linear vs outranking",
        "%d reversals" % report["linear_vs_outranking"]["count"],
    )
    head[2].metric(
        "Dominance violations", report["dominance"]["count"]
    )

    st.warning(report["linear_vs_geometric"]["headline"])
    st.info(report["linear_vs_outranking"]["headline"])
    if report["dominance"]["clean"]:
        st.success(report["dominance"]["headline"])
    else:
        st.error(report["dominance"]["headline"])

    rows = []
    for index, unit in enumerate(report["units"]):
        rows.append(
            {
                "Unit": unit,
                "Linear rank": report["linear"]["ranks"][index],
                "Geometric rank": report["geometric"]["ranks"][index],
                "Outranking rank": report["non_compensatory"]["ranks"][index],
                "Moved": abs(
                    report["linear"]["ranks"][index]
                    - report["geometric"]["ranks"][index]
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("Linear rank")
    st.dataframe(frame, use_container_width=True, hide_index=True)

    movement = go.Figure()
    movement.add_trace(
        go.Scatter(
            x=[row["Linear rank"] for row in rows],
            y=[row["Geometric rank"] for row in rows],
            mode="markers",
            marker={"size": 10, "color": "#2E86AB"},
            text=[row["Unit"] for row in rows],
            name="Units",
        )
    )
    movement.add_trace(
        go.Scatter(
            x=[1, len(rows)],
            y=[1, len(rows)],
            mode="lines",
            line={"dash": "dash", "color": "#888"},
            name="No change",
        )
    )
    movement.update_layout(
        title="Points off the line are units whose weakness was compensated",
        xaxis_title="Rank under a linear sum",
        yaxis_title="Rank under a geometric mean",
        height=450,
    )
    st.plotly_chart(movement, use_container_width=True)

    st.markdown(
        """
Every reversal is a case where one unit's shortfall was allowed to be bought
back under one rule and not the other. Which rule is right is a value judgement
and cannot be settled by arithmetic — but it has to be *made*, and adding is
not the same as deciding that compensation is acceptable.

For a confidence score it is probably the geometric mean: an assessment with
unknown emission factors is not rescued by having every field filled in.
        """
    )

with tab_bands:
    st.subheader("84.9 is Medium and 85.1 is High")
    st.caption(
        "Nothing about the underlying data changes at 85. Reporting the "
        "probability of each band under a defensible reweighting is what makes "
        "the arbitrariness of the cut visible."
    )

    units = st.slider("Assessments", 10, 120, 30, key="band_units")
    names, components = demo_confidence_index(units=units)

    try:
        bands = band_probabilities(names, components, bands=APP_CONFIDENCE_BANDS)
    except CompositeError as error:
        st.error(str(error))
        st.stop()

    head = st.columns(3)
    head[0].metric("Assessments", len(bands["rows"]))
    head[1].metric("Borderline", bands["borderline"])
    head[2].metric("Threshold", "%.0f%%" % (MIN_BAND_PROBABILITY * 100))

    if bands["borderline"]:
        st.warning(bands["headline"])
    else:
        st.success(bands["headline"])

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Assessment": row["unit"],
                    "Modal band": row["modal_band"],
                    "Confidence in the label": "%.0f%%" % (row["modal_probability"] * 100),
                    **{
                        label: round(row["probabilities"][label], 3)
                        for label in bands["bands"]
                    },
                    "Reportable": "yes" if row["confident"] else "borderline",
                }
                for row in bands["rows"]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    stacked = go.Figure()
    colours = {"High": "#2E7D32", "Medium": "#E8A33D", "Low": "#C1443C"}
    for label in bands["bands"]:
        stacked.add_trace(
            go.Bar(
                x=[row["unit"] for row in bands["rows"]],
                y=[row["probabilities"][label] for row in bands["rows"]],
                name=label,
                marker_color=colours.get(label, "#999999"),
            )
        )
    stacked.update_layout(
        title="Band probability per assessment, not the band it happened to land in",
        yaxis_title="Probability",
        barmode="stack",
        height=460,
        xaxis={"tickangle": -60},
    )
    st.plotly_chart(stacked, use_container_width=True)

    st.markdown("**The bands in use**")
    st.dataframe(
        pd.DataFrame(
            [{"Threshold": threshold, "Label": label} for threshold, label in APP_CONFIDENCE_BANDS]
        ),
        use_container_width=True,
        hide_index=True,
    )

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
                columns[0].metric("Units", entry["units"])
                columns[1].metric("Normalisation", entry["normalisation"])
                columns[2].metric("Unstable ranks", entry["unstable_ranks"])
                payload = entry.get("payload") or {}
                if payload:
                    st.caption(summarise(payload))
                if st.button("Delete", key="ci_delete_%d" % entry["id"]):
                    if delete_analysis(user_id, entry["id"]):
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.warning("Could not delete.")

st.divider()
st.caption(
    "OECD/JRC Handbook on Constructing Composite Indicators; compensatory "
    "versus non-compensatory aggregation; Pearson correlation ratio for "
    "effective weights; Dirichlet weight uncertainty and rank intervals. "
    "Standard library only — no dependency added."
)
