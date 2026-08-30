"""Which uncertainty is worth resolving, not which one is largest.

`global_sensitivity.py` can tell a user that grid intensity accounts for 60% of
the variance in their footprint. That is true, and it does not answer the
question they are asking, which is whether to go and find out.

A parameter can drive most of the variance and be worth nothing to measure,
because the decision it feeds is the same across its whole range. This page
computes both rankings and shows where they part company.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles.theme import apply_theme
from src.utils.value_of_information import (
    DEFAULT_BINS,
    DEFAULT_DRAWS,
    LOUD_VARIANCE_SHARE,
    NEGLIGIBLE_SHARE_OF_EVPI,
    VOIError,
    analyse,
    delete_analysis,
    demo_abatement_decision,
    demo_decision,
    evppi,
    expected_net_benefit_of_sampling,
    get_analyses,
    get_voi_notes,
    save_analysis,
    simulate,
    summarise,
)

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>🔎 Value of Information</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "Variance-based sensitivity ranks parameters by how much they move the "
    "number. This ranks them by how much they move the decision, and only the "
    "second one tells anyone what to do next."
)

with st.expander("Why the two rankings are different questions"):
    st.markdown(
        """
**Moving the number is not moving the choice.** A parameter can account for 70%
of the output variance and be worth nothing to measure, because the decision it
feeds is the same across its entire plausible range. A parameter can account for
3% of the variance and be worth a great deal, because the choice flips somewhere
inside that 3%.

**Value of information needs a decision.** It is undefined without one, and the
temptation to compute it against a bare model output is exactly the confusion
this page exists to remove. So it takes a set of options and a payoff rather
than a model.

**EVPI** — the value of resolving *everything*. An upper bound on every study,
survey and sensor that could ever be run. Where it is below the cost of the
cheapest measurement, no data collection can pay for itself and the correct
output is "act on what you have" — a result this app currently cannot produce.

**EVPPI** — the value of resolving *one* parameter while the rest stay
uncertain. This is the ranking that answers "what should I measure next".

**EVSI** — the value of a study you could actually run: three months of meter
readings, not omniscience. Bounded above by EVPPI. At a sample size of zero it
is zero; as the sample grows it converges on EVPPI.

**ENBS** — EVSI minus what the study costs. Measurement is not free and this app
currently prices it at zero. A recommendation to measure something is a
recommendation like any other and should have to clear a bar.

**Two ways of being wrong.** The probability the recommendation is not best says
how *often*; the expected opportunity loss says how *much* when it happens. They
routinely disagree, and an option that is second-best by a rounding error is not
a mistake worth avoiding.

**The estimator is biased upward.** Partitioning the draws and taking the best
option within each bin picks up some of that bin's sampling noise, so a
parameter whose true EVPPI is zero estimates as a small positive number that
shrinks as the draws grow. "This cannot change the choice" is therefore stated
as a share of EVPI, not as an exact zero.
        """
    )

(
    tab_contrast,
    tab_abatement,
    tab_study,
    tab_saved,
) = st.tabs(
    ["The two rankings", "Abatement measures", "Is a study worth it?", "Saved"]
)


with tab_contrast:
    st.subheader("One parameter drives the variance. The other drives the choice.")
    st.markdown(
        "Two options and two parameters, built so the rankings must disagree. "
        "`grid_intensity` enters **both** options identically, so it moves the "
        "payoff a great deal and the difference between the options not at all. "
        "`heat_pump_performance` enters them with opposite signs, so its sign "
        "decides the choice."
    )

    columns = st.columns(3)
    loud = columns[0].slider("grid_intensity spread", 50.0, 1200.0, 400.0, 50.0)
    decisive = columns[1].slider(
        "heat_pump_performance spread", 5.0, 200.0, 30.0, 5.0
    )
    gap = columns[2].slider(
        "Gap between the options",
        0.0,
        400.0,
        0.0,
        10.0,
        help="Push the options apart and the decision stops being close.",
    )

    draws = st.select_slider(
        "Draws", options=[1000, 2000, 4000, 8000, 20000], value=DEFAULT_DRAWS
    )
    cheapest = st.number_input("Cost of the cheapest measurement", value=0.0, step=1.0)

    options, parameters = demo_decision(
        decisive_spread=decisive, loud_spread=loud, gap=gap
    )
    try:
        report = analyse(
            options,
            parameters,
            draws=draws,
            cheapest_measurement=cheapest,
        )
    except VOIError as error:
        st.error(str(error))
        st.stop()

    st.session_state["voi_report"] = report
    decision = report["decision"]

    head = st.columns(4)
    head[0].metric("Choose", decision["recommended"])
    head[1].metric(
        "Chance that is wrong", "%.0f%%" % (decision["probability_wrong"] * 100)
    )
    head[2].metric(
        "Cost when it is", "%.3g" % decision["expected_opportunity_loss"]
    )
    head[3].metric("EVPI", "%.3g" % report["evpi"]["evpi"])

    if report["act_on_what_you_have"]:
        st.success(report["headline"])
    else:
        st.info(decision["headline"])
        st.warning(report["comparison"]["headline"])

    rows = report["comparison"]["rows"]
    comparison = go.Figure()
    comparison.add_trace(
        go.Bar(
            x=[row["parameter"] for row in rows],
            y=[row["variance_share"] * 100 for row in rows],
            name="Share of payoff variance (%)",
            marker_color="#C1443C",
        )
    )
    comparison.add_trace(
        go.Bar(
            x=[row["parameter"] for row in rows],
            y=[row["share_of_evpi"] * 100 for row in rows],
            name="Share of decision value (%)",
            marker_color="#2E86AB",
        )
    )
    comparison.update_layout(
        title="Moving the number against moving the choice",
        yaxis_title="Share (%)",
        barmode="group",
        height=430,
    )
    st.plotly_chart(comparison, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Parameter": row["parameter"],
                    "Variance share": "%.1f%%" % (row["variance_share"] * 100),
                    "Variance rank": row["variance_rank"],
                    "EVPPI": round(row["evppi"], 4),
                    "Share of EVPI": "%.0f%%" % (row["share_of_evpi"] * 100),
                    "Decision rank": row["decision_rank"],
                    "Worth measuring": "no" if row["high_variance_no_value"] else "yes",
                }
                for row in rows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "A parameter is marked not worth measuring when it carries more than "
        "%.0f%% of the payoff variance and buys less than %.0f%% of the "
        "decision value. The threshold is a share rather than an exact zero "
        "because the estimator is biased upward — increase the draws and watch "
        "the small EVPPIs fall."
        % (LOUD_VARIANCE_SHARE * 100, NEGLIGIBLE_SHARE_OF_EVPI * 100)
    )

    st.markdown("**Probability each option is best**")
    probabilities = go.Figure()
    probabilities.add_trace(
        go.Bar(
            x=report["options"],
            y=[value * 100 for value in decision["probability_best"]],
            marker_color="#2E86AB",
        )
    )
    probabilities.add_hline(y=50.0, line_dash="dot")
    probabilities.update_layout(
        title="A recommendation at 51%% and one at 99%% look identical today",
        yaxis_title="Probability best (%)",
        height=380,
    )
    st.plotly_chart(probabilities, use_container_width=True)

    for note in get_voi_notes(report):
        st.markdown("- %s" % note)

    st.caption(summarise(report))

    label = st.text_input("Label", value="Heat pump vs insulation")
    if st.button("Save analysis", type="primary"):
        saved = save_analysis(user_id, report, label=label)
        if saved:
            st.success("Saved as #%d." % saved)
        else:
            st.warning("Could not save — storage is unavailable.")

with tab_abatement:
    st.subheader("Three measures, ranked on point estimates")
    st.markdown(
        "`src/carbon/abatement_curve.py` orders measures by cost per tonne "
        "computed from uncertain inputs, and presents two measures whose "
        "intervals overlap as ranked. The decision-relevant question — is the "
        "*choice* uncertain, and would resolving one input settle it — is not "
        "asked there."
    )

    draws = st.select_slider(
        "Draws", options=[2000, 4000, 8000, 20000], value=8000, key="ab_draws"
    )
    options, parameters = demo_abatement_decision()
    try:
        report = analyse(options, parameters, draws=draws)
    except VOIError as error:
        st.error(str(error))
        st.stop()

    st.session_state["voi_abatement"] = report
    st.session_state["voi_abatement_params"] = parameters
    decision = report["decision"]

    head = st.columns(4)
    head[0].metric("Recommended", decision["recommended"])
    head[1].metric(
        "Probability it is best",
        "%.0f%%" % (decision["probability_recommended_is_best"] * 100),
    )
    head[2].metric("EVPI", "%.4g" % report["evpi"]["evpi"])
    head[3].metric("Measure next", report["evppi"]["order"][0])

    st.info(decision["headline"])
    st.warning(report["comparison"]["headline"])

    payoffs = go.Figure()
    payoffs.add_trace(
        go.Bar(
            x=report["options"],
            y=decision["expected_payoffs"],
            marker_color="#2E86AB",
            name="Expected payoff",
        )
    )
    payoffs.update_layout(
        title="Expected payoff by measure",
        yaxis_title="Net benefit",
        height=390,
    )
    st.plotly_chart(payoffs, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Measure": name,
                    "Expected payoff": round(decision["expected_payoffs"][index], 1),
                    "Probability best": "%.0f%%"
                    % (decision["probability_best"][index] * 100),
                }
                for index, name in enumerate(report["options"])
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Where the two rankings disagree**")
    rows = report["comparison"]["rows"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Parameter": row["parameter"],
                    "Variance rank": row["variance_rank"],
                    "Variance share": "%.1f%%" % (row["variance_share"] * 100),
                    "Decision rank": row["decision_rank"],
                    "EVPPI": round(row["evppi"], 1),
                    "Positions moved": row["moved"],
                }
                for row in rows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    slope = go.Figure()
    for row in rows:
        slope.add_trace(
            go.Scatter(
                x=["Variance ranking", "Decision ranking"],
                y=[row["variance_rank"], row["decision_rank"]],
                mode="lines+markers+text",
                name=row["parameter"],
                text=[row["parameter"], row["parameter"]],
                textposition="middle right",
            )
        )
    slope.update_layout(
        title="Reading the variance ranking as a measurement priority list",
        yaxis={"title": "Rank", "autorange": "reversed"},
        height=430,
        showlegend=False,
    )
    st.plotly_chart(slope, use_container_width=True)

    for note in get_voi_notes(report):
        st.markdown("- %s" % note)

with tab_study:
    st.subheader("Would the study pay for itself?")
    report = st.session_state.get("voi_abatement")
    parameters = st.session_state.get("voi_abatement_params")
    if not report or not parameters:
        st.info("Open the abatement tab first.")
    else:
        updatable = [
            entry["parameter"]
            for entry in report["evppi"]["entries"]
            if next(
                item["updatable"]
                for item in parameters
                if item["name"] == entry["parameter"]
            )
        ]
        if not updatable:
            st.warning(
                "No parameter in this decision has a normal prior, and EVSI "
                "needs a conjugate prior to update. Approximating it would "
                "produce a number whose error nobody could bound."
            )
        else:
            parameter = st.selectbox("Parameter to study", updatable)
            columns = st.columns(3)
            measurement = columns[0].number_input(
                "Per-observation measurement sd", value=60.0, min_value=0.1, step=5.0
            )
            fixed = columns[1].number_input("Fixed study cost", value=200.0, step=25.0)
            per_unit = columns[2].number_input(
                "Cost per observation", value=8.0, step=1.0
            )
            population = st.number_input(
                "Households the decision applies to", value=1.0, min_value=1.0, step=1.0
            )

            options, params = demo_abatement_decision()
            simulation = simulate(options, params, draws=report["draws"])
            sizes = [0, 1, 2, 5, 10, 20, 40, 80, 160, 320]

            try:
                study = expected_net_benefit_of_sampling(
                    simulation,
                    params,
                    parameter,
                    measurement_sd=measurement,
                    sample_sizes=sizes,
                    fixed_cost=fixed,
                    cost_per_observation=per_unit,
                    population=population,
                    bins=DEFAULT_BINS,
                )
                ceiling = evppi(simulation, parameter)
            except VOIError as error:
                st.error(str(error))
                st.stop()

            head = st.columns(4)
            head[0].metric("EVPPI (perfect)", "%.4g" % ceiling["evppi"])
            head[1].metric("Best sample size", study["optimum"]["sample_size"])
            head[2].metric("Its EVSI", "%.4g" % study["optimum"]["evsi"])
            head[3].metric("Net benefit", "%.4g" % study["optimum"]["net_benefit"])

            if study["worthwhile"]:
                st.success(study["headline"])
            else:
                st.error(study["headline"])

            curve = go.Figure()
            curve.add_trace(
                go.Scatter(
                    x=[row["sample_size"] for row in study["rows"]],
                    y=[row["population_evsi"] for row in study["rows"]],
                    mode="lines+markers",
                    name="Value of the study",
                    line={"color": "#2E86AB", "width": 3},
                )
            )
            curve.add_trace(
                go.Scatter(
                    x=[row["sample_size"] for row in study["rows"]],
                    y=[row["cost"] for row in study["rows"]],
                    mode="lines",
                    name="What it costs",
                    line={"color": "#C1443C", "dash": "dash"},
                )
            )
            curve.add_hline(
                y=ceiling["evppi"] * population,
                line_dash="dot",
                annotation_text="EVPPI ceiling",
            )
            curve.update_layout(
                title="EVSI rises toward the EVPPI ceiling; the cost does not stop",
                xaxis_title="Observations collected",
                yaxis_title="Value",
                height=440,
            )
            st.plotly_chart(curve, use_container_width=True)

            net = go.Figure()
            net.add_trace(
                go.Bar(
                    x=[row["sample_size"] for row in study["rows"]],
                    y=[row["net_benefit"] for row in study["rows"]],
                    marker_color=[
                        "#2E7D32" if row["net_benefit"] > 0 else "#C1443C"
                        for row in study["rows"]
                    ],
                )
            )
            net.add_hline(y=0.0, line_dash="dash")
            net.update_layout(
                title="Expected net benefit of sampling",
                xaxis_title="Observations collected",
                yaxis_title="Net benefit",
                height=400,
            )
            st.plotly_chart(net, use_container_width=True)

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Observations": row["sample_size"],
                            "EVSI": round(row["evsi"], 2),
                            "Value to population": round(row["population_evsi"], 1),
                            "Cost": round(row["cost"], 1),
                            "Net benefit": round(row["net_benefit"], 1),
                        }
                        for row in study["rows"]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "At zero observations EVSI is zero. As the sample grows it "
                "converges on EVPPI and never exceeds it, because no study can "
                "be worth more than knowing the answer. Where the best net "
                "benefit is negative the recommendation is not to collect the "
                "data — which is a result, not a failure to produce one."
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
                columns[0].metric("Recommended", entry["recommended"] or "—")
                columns[1].metric("EVPI", "%.4g" % (entry["evpi"] or 0.0))
                columns[2].metric("Measure next", entry["top_parameter"])
                if entry["act_now"]:
                    st.success(
                        "No measurement can pay for itself on this decision."
                    )
                payload = entry.get("payload") or {}
                if payload:
                    st.caption(summarise(payload))
                if st.button("Delete", key="voi_delete_%d" % entry["id"]):
                    if delete_analysis(user_id, entry["id"]):
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.warning("Could not delete.")

st.divider()
st.caption(
    "Expected value of perfect information; EVPPI by the single-parameter "
    "partition method; EVSI through a normal-normal preposterior update; "
    "expected net benefit of sampling; opportunity loss under uncertainty. "
    "Standard library only — no dependency added."
)
