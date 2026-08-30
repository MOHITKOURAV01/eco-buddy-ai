"""Every number here was remembered, and remembering adds noise.

Noise in a predictor does not widen an estimate — it shrinks it, toward zero,
by a factor that can be calculated. This page calculates it, checks the
assumption that makes the correction valid, and refuses when the assumption
fails.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles.theme import apply_theme
from src.utils.measurement_error import (
    DEFAULT_CONFIDENCE,
    GOOD_RELIABILITY,
    POOR_RELIABILITY,
    RELIABILITY_FLOOR,
    CalibrationError,
    analyse,
    attenuation_table,
    delete_analysis,
    demo_records,
    demo_regression,
    disattenuate_correlation,
    disattenuate_slope,
    get_analyses,
    get_measurement_notes,
    heaping_diagnostics,
    make_component,
    propagate_to_total,
    reliability_band,
    simex,
    summarise,
    save_analysis,
)

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>📐 Measurement Error</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "Recall error in a predictor biases every slope estimated from it toward "
    "zero — always, and by a factor that can be worked out."
)

with st.expander("Why this is a bias and not just uncertainty"):
    st.markdown(
        """
**The arithmetic.** Write a reported value as truth plus noise, `X = T + e`.
A regression of `Y` on `X` does not recover the slope of `Y` on `T`. It
recovers:

    beta_observed = beta_true * lambda
    lambda        = var(T) / (var(T) + var(e))

`lambda` is between zero and one, so the observed slope is always closer to
zero than the truth. Not on average — always.

**Error in the outcome and error in the predictor do opposite things.** Noise
in `Y` widens the interval and leaves the estimate unbiased. Noise in `X`
biases the estimate. Only one of those is fixable after the fact, and the app
currently treats them alike.

**The direction is the worst available one.** The bias runs toward "your action
did nothing", and it runs hardest in the categories people recall worst. An app
built to show people their choices matter is understating those choices,
systematically.

**Two things identify the error.** Repeat reports of the same period differ
only by noise, so half the variance of their difference is `var(e)`. And a
trusted value — a bill, a receipt, an odometer — identifies it directly:
regressing the trusted value on the reported one gives a slope that *is*
`lambda`, because `cov(X, W) = var(T)`. The calibration equation and the
attenuation factor are the same object.

**Differential error breaks all of it.** The formula assumes the error is
independent of the truth. People under-report meat and over-report cycling, and
where the error tracks the truth the correction recovers nothing. So that test
runs first and a failure blocks the estimate rather than appearing as a caveat
under it.

**A corrected estimate is less biased and less precise.** Dividing the standard
error by `lambda` treats the reliability as exact. It is an estimate too, and
the delta method carries its uncertainty through. Both halves are reported,
because a correction presented without the widened interval is a different kind
of overconfidence.

**This is not the missing-data problem.** `src/data/imputation_bias.py` handles
values that are absent. These are values that are present and wrong by a random
amount.
        """
    )

(
    tab_reliability,
    tab_correction,
    tab_heaping,
    tab_total,
    tab_saved,
) = st.tabs(
    ["Reliability", "Correction", "Heaping", "Totals", "Saved"]
)


def _record_controls(prefix):
    columns = st.columns(4)
    count = columns[0].slider("Records", 40, 600, 200, key="%s_count" % prefix)
    error = columns[1].slider(
        "Recall error SD",
        0.0,
        800.0,
        300.0,
        step=25.0,
        key="%s_error" % prefix,
        help="How far a remembered value sits from the real one.",
    )
    validation = columns[2].slider(
        "Share with a trusted value",
        0.0,
        0.6,
        0.20,
        step=0.05,
        key="%s_validation" % prefix,
        help="Bills, receipts, odometer readings.",
    )
    differential = columns[3].slider(
        "Differential error",
        0.0,
        0.8,
        0.0,
        step=0.05,
        key="%s_differential" % prefix,
        help="How much the misreport grows with the true value.",
    )
    return demo_records(
        count=count,
        error_sd=error,
        validation_share=validation,
        differential_slope=differential,
    )


with tab_reliability:
    st.subheader("How much of the reported spread is real?")
    records = _record_controls("rel")
    st.session_state["me_records"] = records

    confidence = st.select_slider(
        "Interval confidence",
        options=[0.80, 0.90, 0.95, 0.99],
        value=DEFAULT_CONFIDENCE,
    )

    result = analyse(records, confidence=confidence)
    st.session_state["me_result"] = result

    head = st.columns(4)
    head[0].metric("Records", result["records"])
    head[1].metric("With a repeat", result["with_repeat"])
    head[2].metric("With a trusted value", result["with_validation"])
    head[3].metric(
        "lambda",
        "%.3f" % result["reliability"] if result.get("reliability") else "—",
    )

    if result.get("blocked"):
        st.error(result["headline"])
    else:
        st.success(result["headline"])

    if result.get("differential"):
        differential = result["differential"]
        if differential["differential"]:
            st.error(differential["headline"])
        else:
            st.info(differential["headline"])
        st.caption(
            "This test runs at p = 0.10 rather than 0.05, on purpose. It is a "
            "test the analysis wants to pass, so the lenient threshold rejects "
            "more often and rejecting is what blocks the correction. The error "
            "is pushed toward refusing."
        )

    rows = []
    if result.get("repeats"):
        repeats = result["repeats"]
        rows.append(
            {
                "Source": "Repeat reports",
                "Pairs": repeats["pairs"],
                "lambda": round(repeats["reliability"], 3),
                "Error SD": round(repeats["error_sd"], 1),
                "Assumes": "The two recalls share no bias.",
            }
        )
    if result.get("validation"):
        validation = result["validation"]
        rows.append(
            {
                "Source": "Trusted values",
                "Pairs": validation["pairs"],
                "lambda": round(validation["reliability"], 3),
                "Error SD": round(
                    (validation["residual_variance"]) ** 0.5, 1
                ),
                "Assumes": "The trusted value has no error of its own.",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Where both are available the trusted values win: repeats share "
            "whatever bias the respondent carries between occasions, so they "
            "flatter the reliability."
        )

    st.markdown("**What each reliability costs**")
    table = attenuation_table()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "lambda": row["reliability"],
                    "A true slope of 1.0 reads as": round(row["observed_slope"], 3),
                    "Understated by": "%.0f%%" % row["understatement"],
                    "Correction": "%.2fx" % row["correction_factor"],
                    "Band": row["band"],
                    "Correctable": "yes" if row["correctable"] else "no",
                }
                for row in table
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Below lambda = %.2f the correction divides by nearly nothing and the "
        "module refuses. The right output there is the attenuated estimate with "
        "its reliability attached, not a large number with no basis."
        % RELIABILITY_FLOOR
    )

    for note in get_measurement_notes(result):
        st.markdown("- %s" % note)

    label = st.text_input("Label", value="Energy kWh recall")
    if st.button("Save analysis", type="primary"):
        saved = save_analysis(user_id, result, label=label, category="energy_kwh")
        if saved:
            st.success("Saved as #%d." % saved)
        else:
            st.warning("Could not save — storage is unavailable.")

with tab_correction:
    st.subheader("The slope, before and after")
    records = st.session_state.get("me_records")
    result = st.session_state.get("me_result")
    if not records or not result:
        st.info("Build a dataset on the first tab.")
    elif result.get("blocked"):
        st.error(result["headline"])
        st.info(
            "No corrected slope is reported. That is the result, not a missing "
            "result — the alternative is a number derived from a formula whose "
            "assumption has failed."
        )
    else:
        try:
            example = demo_regression(records)
        except CalibrationError as error:
            st.error(str(error))
            st.stop()

        observed = example["fit_on_reported"]
        actual = example["fit_on_truth"]

        try:
            correction = disattenuate_slope(
                observed["slope"],
                observed["slope_se"],
                result["reliability"],
                reliability_se=(result.get("validation") or {}).get(
                    "reliability_se", 0.0
                ),
                confidence=result["confidence"],
            )
        except CalibrationError as error:
            st.error(str(error))
            st.stop()

        st.caption(
            "The outcome here is generated from the *true* predictor, so the "
            "answer is known and the correction can be checked against it. That "
            "is the only reason a demonstration is worth anything."
        )

        head = st.columns(4)
        head[0].metric("True slope", "%.4f" % example["true_slope"])
        head[1].metric("Fitted on reported", "%.4f" % observed["slope"])
        head[2].metric("Corrected", "%.4f" % correction["corrected_slope"])
        head[3].metric("Fitted on truth", "%.4f" % actual["slope"])

        st.warning(
            "**The uncorrected slope understates the relationship by %.0f%%.** "
            "Not by chance, and not in a direction that varies."
            % ((1.0 - example["attenuation_observed"]) * 100.0)
        )

        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=["Fitted on reported", "Corrected", "Fitted on truth"],
                y=[
                    observed["slope"],
                    correction["corrected_slope"],
                    actual["slope"],
                ],
                marker_color=["#C1443C", "#2E86AB", "#2E7D32"],
                error_y={
                    "type": "data",
                    "array": [
                        1.96 * observed["slope_se"],
                        correction["upper"] - correction["corrected_slope"],
                        1.96 * actual["slope_se"],
                    ],
                },
            )
        )
        figure.add_hline(
            y=example["true_slope"],
            line_dash="dash",
            annotation_text="truth",
        )
        figure.update_layout(
            title="Same outcome, three predictors",
            yaxis_title="Slope",
            height=420,
        )
        st.plotly_chart(figure, use_container_width=True)

        st.info(correction["headline"])

        st.markdown("**Standard errors**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Method": "Divide the SE by lambda",
                        "Std. error": round(correction["naive_corrected_se"], 5),
                        "Assumes": "The reliability is known exactly.",
                    },
                    {
                        "Method": "Delta method (used)",
                        "Std. error": round(correction["corrected_se"], 5),
                        "Assumes": "The reliability is itself estimated.",
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "The delta-method error is %.2fx the naive one. That gap is the "
            "price of the correction, and it is often the larger term."
            % correction["se_inflation_from_lambda"]
        )

        st.markdown("**Correlations attenuate twice**")
        columns = st.columns(3)
        observed_r = columns[0].slider("Observed correlation", 0.0, 0.95, 0.35, 0.05)
        rel_x = columns[1].slider("lambda for x", 0.25, 1.0, 0.70, 0.05)
        rel_y = columns[2].slider("lambda for y", 0.25, 1.0, 0.70, 0.05)
        pair = disattenuate_correlation(observed_r, rel_x, rel_y)
        if pair["impossible"]:
            st.error(pair["headline"])
        else:
            st.success(pair["headline"])
        st.caption(
            "When both variables are self-reported the attenuation compounds, "
            "so a relationship reported as weak can be genuinely strong."
        )

        st.markdown("**SIMEX — where no trusted values exist**")
        st.caption(
            "Add more error, watch the slope fall, extrapolate the curve back to "
            "no error at all. It needs the error variance from somewhere, and "
            "where that is a guess so is the answer — which is why it is the "
            "fallback and not the default."
        )
        assumed = st.slider("Assumed error SD", 50.0, 800.0, 300.0, 25.0)
        try:
            curve = simex(example["reported"], example["outcome"], assumed**2)
        except CalibrationError as error:
            st.error(str(error))
        else:
            simex_figure = go.Figure()
            simex_figure.add_trace(
                go.Scatter(
                    x=[point["lambda"] for point in curve["curve"]],
                    y=[point["slope"] for point in curve["curve"]],
                    mode="lines+markers",
                    name="Simulated",
                    line={"color": "#2E86AB"},
                )
            )
            simex_figure.add_trace(
                go.Scatter(
                    x=[-1.0],
                    y=[curve["corrected_slope"]],
                    mode="markers",
                    marker={"size": 14, "color": "#C1443C", "symbol": "star"},
                    name="Extrapolated",
                )
            )
            simex_figure.add_hline(
                y=example["true_slope"], line_dash="dash", annotation_text="truth"
            )
            simex_figure.update_layout(
                title="Extrapolating past zero error",
                xaxis_title="Added error (multiples of var(e))",
                yaxis_title="Slope",
                height=400,
            )
            st.plotly_chart(simex_figure, use_container_width=True)
            st.caption(curve["headline"])

with tab_heaping:
    st.subheader("Is the reported precision real?")
    st.markdown(
        "Recalled mileage clusters on multiples of 50 and hours cluster on "
        "multiples of 5. `detect_outliers()` sees nothing wrong, because "
        "nothing is wrong with any individual value — the problem is in the "
        "distribution of the last digit."
    )

    grid = st.select_slider(
        "Round reports to the nearest",
        options=[0, 5, 10, 25, 50, 100],
        value=50,
        format_func=lambda value: "not rounded" if value == 0 else str(value),
    )
    sample = demo_records(count=400, heap_to=grid, seed=4242)
    values = [record["reported"] for record in sample]

    try:
        diagnosis = heaping_diagnostics(values)
    except CalibrationError as error:
        st.error(str(error))
        st.stop()

    head = st.columns(3)
    head[0].metric("Whipple index", "%.0f" % diagnosis["whipple_index"])
    head[1].metric("p", "%.4f" % diagnosis["p_value"])
    head[2].metric("Effective precision", diagnosis["effective_precision"])

    if diagnosis["heaped"]:
        st.error(diagnosis["headline"])
    else:
        st.success(diagnosis["headline"])

    digits = go.Figure()
    digits.add_trace(
        go.Bar(
            x=list(range(10)),
            y=diagnosis["digit_counts"],
            marker_color="#2E86AB",
            name="Observed",
        )
    )
    digits.add_hline(
        y=diagnosis["n"] / 10.0,
        line_dash="dash",
        line_color="#C1443C",
        annotation_text="expected",
    )
    digits.update_layout(
        title="Final digit of every reported value",
        xaxis_title="Final digit",
        yaxis_title="Count",
        height=400,
    )
    st.plotly_chart(digits, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {"Multiple of": base, "Share of values": round(share, 3)}
                for base, share in diagnosis["round_multiple_share"].items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "100 is a clean Whipple index and 175 is the conventional line for "
        "rough data. A heaped variable cannot support a threshold that falls "
        "between two heaps, because almost nobody will land there."
    )

with tab_total:
    st.subheader("A total is not a slope")
    st.markdown(
        "Attenuation is about relationships. For the footprint total the "
        "arithmetic is different, and confusing the two is common: random "
        "errors partly cancel across categories, and a systematic bias in every "
        "category does not cancel at all."
    )

    columns = st.columns(2)
    noise = columns[0].slider("Random error, per category (%)", 0, 60, 12)
    bias = columns[1].slider("Under-reporting, per category (%)", 0, 40, 8)

    base = [
        ("Home energy", 3000.0),
        ("Transport", 2000.0),
        ("Food", 1500.0),
        ("Goods", 900.0),
        ("Services", 600.0),
    ]
    components = [
        make_component(
            name,
            value,
            error_sd=value * noise / 100.0,
            bias=-value * bias / 100.0,
        )
        for name, value in base
    ]
    totals = propagate_to_total(components)

    head = st.columns(4)
    head[0].metric("Reported total", "%.0f" % totals["total"])
    head[1].metric("Random error", "± %.0f" % totals["random_sd"])
    head[2].metric("Systematic bias", "%.0f" % totals["systematic_bias"])
    head[3].metric("Corrected total", "%.0f" % totals["corrected_total"])

    if totals["bias_dominates"]:
        st.error(totals["headline"])
    else:
        st.info(totals["headline"])

    cancel = go.Figure()
    cancel.add_trace(
        go.Bar(
            x=["Component errors added", "Random error on the total", "Systematic bias"],
            y=[
                totals["sum_of_component_sds"],
                totals["random_sd"],
                abs(totals["systematic_bias"]),
            ],
            marker_color=["#999999", "#2E86AB", "#C1443C"],
        )
    )
    cancel.update_layout(
        title="Random error cancels across categories. Bias does not.",
        yaxis_title="kg CO2e",
        height=400,
    )
    st.plotly_chart(cancel, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Category": row["name"],
                    "Reported": round(row["value"], 0),
                    "Random error SD": round(row["error_sd"], 0),
                    "Bias": round(row["bias"], 0),
                }
                for row in totals["components"]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "A footprint built from five noisily-recalled categories can have a "
        "perfectly usable total and still be useless for any comparison "
        "*between* categories. And if every category is under-reported by the "
        "same fraction, so is the total — no amount of averaging helps."
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
                columns[0].metric("Method", entry["method"])
                columns[1].metric(
                    "lambda",
                    "%.3f" % entry["reliability"] if entry["reliability"] else "—",
                )
                columns[2].metric(
                    "Band",
                    reliability_band(entry["reliability"])
                    if entry["reliability"]
                    else "—",
                )
                if entry["blocked"]:
                    st.error("Correction was blocked for this analysis.")
                payload = entry.get("payload") or {}
                if payload:
                    st.caption(summarise(payload))
                if st.button("Delete", key="me_delete_%d" % entry["id"]):
                    if delete_analysis(user_id, entry["id"]):
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.warning("Could not delete.")

st.divider()
st.caption(
    "Classical and differential measurement error; regression dilution and "
    "disattenuation; regression calibration on a validation subsample; SIMEX; "
    "Whipple's index for digit preference. Bands: good ≥ %.2f, usable ≥ %.2f, "
    "correction refused below %.2f. Standard library only — no dependency added."
    % (GOOD_RELIABILITY, POOR_RELIABILITY, RELIABILITY_FLOOR)
)
