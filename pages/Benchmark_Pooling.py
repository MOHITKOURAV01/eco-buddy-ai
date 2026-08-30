"""The top of the leaderboard is short histories, not low footprints.

`carbon_benchmarking.py` gives a household with two assessments the same
percentile, the same trend label, the same rank and the same badge it gives a
household with forty. The two numbers are not the same kind of object and the
app renders them identically.

This page shrinks each household toward the group by an amount set by how noisy
its own history is, then re-ranks, and shows how far the ranking moved. The
size of that move is the size of the error the unpooled leaderboard is making.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles.theme import apply_theme
from src.carbon.benchmark_pooling import (
    DEFAULT_BADGES,
    DEFAULT_CONFIDENCE,
    HIGH_RELIABILITY,
    LOW_RELIABILITY,
    PoolingError,
    badge_for,
    delete_panel,
    demo_panel,
    get_panels,
    get_pooling_notes,
    percentile_of,
    pool_panel,
    reliability,
    save_panel,
    summarise,
    trend_direction,
)

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>📊 Benchmark Pooling</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "A household with two assessments and a household with forty get the same "
    "percentile, the same badge and the same place in the ranking. Only one of "
    "them has been measured."
)

with st.expander("Why an unpooled leaderboard ranks history length"):
    st.markdown(
        """
**Extremes in a ranking are mostly small samples.** The top of any leaderboard
built from raw means is populated by whoever has the fewest, luckiest
observations. This is not a subtlety — it is the most reliable artefact in
ranked data, and the leaderboard reproduces it faithfully every time it renders.

**The fix is to shrink, by an amount the data chooses.** Split the observed
spread into between-household variance (real differences) and within-household
variance (month-to-month noise). Then for a household with `n` observations:

    lambda = tau^2 / (tau^2 + sigma^2 / n)
    pooled = lambda * own_mean + (1 - lambda) * grand_mean

`lambda` is the weight on the household's own data. It goes to zero for one
noisy observation and to one for a long clean history, and it is the honest
answer to "how much of this number is you and how much is the group".

**A rank implies a separation that usually is not there.** Ordering by point
estimate reports positions 3 and 4 as distinct when their intervals overlap
almost entirely. That ordering is stable in the display and unstable in
reality — it flips next month and the user reads the flip as change. Where
intervals overlap this page reports a tie band instead.

**Badges are hard cuts on a noisy statistic.** A household oscillating around a
boundary gains and loses a badge on measurement noise. A new badge here
requires the whole interval to clear the threshold; an existing one is kept
until the point estimate crosses back.

**"Stable" and "we have no idea" are not the same claim.** A fixed 50 kg trend
threshold fires on nothing at all for a household whose month-to-month swing is
300 kg. The slope test below is scaled to the panel's own noise, and
"insufficient data" is a first-class answer.

**The refusals.** No pooling against a group too small to estimate
between-household variance. A negative variance component is reported rather
than floored at zero — it means the households are not measurably different
from each other, which is a finding about the panel. And no pooled estimate is
shown without its reliability, because a pooled estimate with `lambda` near
zero is a statement about the group.
        """
    )

tab_panel, tab_ranking, tab_household, tab_saved = st.tabs(
    ["Panel & shrinkage", "Ranking", "One household", "Saved panels"]
)


def _panel_from_controls(prefix):
    columns = st.columns(4)
    households = columns[0].slider(
        "Households", 6, 200, 40, key="%s_households" % prefix
    )
    between = columns[1].slider(
        "Between-household SD",
        100.0,
        2000.0,
        900.0,
        step=50.0,
        key="%s_between" % prefix,
        help="How different households genuinely are from each other.",
    )
    within = columns[2].slider(
        "Within-household SD",
        50.0,
        2000.0,
        700.0,
        step=50.0,
        key="%s_within" % prefix,
        help="Month-to-month noise inside a single household.",
    )
    max_history = columns[3].slider(
        "Longest history (months)", 6, 60, 36, key="%s_history" % prefix
    )
    return demo_panel(
        households=households,
        between_sd=between,
        within_sd=within,
        max_history=max_history,
    )


with tab_panel:
    st.subheader("How much of the spread is real difference?")
    panel = _panel_from_controls("panel")
    st.session_state["bp_panel"] = panel

    confidence = st.select_slider(
        "Interval confidence",
        options=[0.80, 0.90, 0.95, 0.99],
        value=DEFAULT_CONFIDENCE,
    )

    try:
        result = pool_panel(panel, confidence=confidence)
    except PoolingError as error:
        st.error(str(error))
        st.stop()

    st.session_state["bp_result"] = result
    components = result["components"]

    head = st.columns(4)
    head[0].metric("Households", components["households"])
    head[1].metric("Observations", components["observations"])
    head[2].metric("Between-household SD", "%.0f" % components["between_sd"])
    head[3].metric("Within-household SD", "%.0f" % components["within_sd"])

    if components["negative_component"]:
        st.error(components["note"])
    else:
        st.info(components["note"])

    st.metric(
        "Intraclass correlation",
        "%.2f" % components["intraclass_correlation"],
        help=(
            "The share of observed spread that is genuine difference between "
            "households. The rest is noise a single-month comparison would "
            "report as difference."
        ),
    )

    st.markdown("**Shrinkage against history length**")
    curve = go.Figure()
    counts = list(range(1, 37))
    curve.add_trace(
        go.Scatter(
            x=counts,
            y=[
                reliability(
                    n,
                    components["between_variance"],
                    components["within_variance"],
                )
                for n in counts
            ],
            mode="lines",
            name="lambda",
            line={"color": "#2E86AB", "width": 3},
        )
    )
    curve.add_hline(
        y=LOW_RELIABILITY,
        line_dash="dot",
        annotation_text="mostly the group",
    )
    curve.add_hline(
        y=HIGH_RELIABILITY,
        line_dash="dot",
        annotation_text="mostly their own data",
    )
    curve.update_layout(
        title="Weight on a household's own data, by number of assessments",
        xaxis_title="Assessments",
        yaxis_title="lambda",
        yaxis_range=[0, 1],
        height=380,
    )
    st.plotly_chart(curve, use_container_width=True)

    st.markdown("**Raw against pooled**")
    scatter = go.Figure()
    scatter.add_trace(
        go.Scatter(
            x=[entry["raw_mean"] for entry in result["estimates"]],
            y=[entry["pooled_mean"] for entry in result["estimates"]],
            mode="markers",
            marker={
                "size": 10,
                "color": [entry["reliability"] for entry in result["estimates"]],
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {"title": "lambda"},
            },
            text=[
                "%s — %d assessment(s)" % (entry["label"], entry["n"])
                for entry in result["estimates"]
            ],
            name="Households",
        )
    )
    extremes = [entry["raw_mean"] for entry in result["estimates"]]
    scatter.add_trace(
        go.Scatter(
            x=[min(extremes), max(extremes)],
            y=[min(extremes), max(extremes)],
            mode="lines",
            line={"dash": "dash", "color": "#888"},
            name="No shrinkage",
        )
    )
    scatter.update_layout(
        title="Points far below the dashed line were being ranked on luck",
        xaxis_title="Raw mean",
        yaxis_title="Pooled mean",
        height=430,
    )
    st.plotly_chart(scatter, use_container_width=True)

    for note in get_pooling_notes(result):
        st.markdown("- %s" % note)

    st.caption(summarise(result))

    label = st.text_input("Label for this panel", value="Neighbourhood panel")
    if st.button("Save panel", type="primary"):
        saved = save_panel(user_id, result, label=label)
        if saved:
            st.success("Saved as #%d." % saved)
        else:
            st.warning("Could not save — storage is unavailable.")

with tab_ranking:
    st.subheader("Rank only where the intervals separate")
    result = st.session_state.get("bp_result")
    if not result:
        st.info("Build a panel on the first tab.")
    else:
        churn = result["rank_churn"]
        head = st.columns(3)
        head[0].metric("Ranks moved by pooling", "%d / %d" % (churn["changed"], churn["total"]))
        head[1].metric("Largest move", "%d positions" % churn["largest_move"])
        head[2].metric("Tie bands", len(result["ranking"]))

        st.warning(
            "**Pooling moves %.0f%% of the ranking.** That is the size of the "
            "error the unpooled leaderboard is making, expressed in positions."
            % (churn["share"] * 100.0)
        )

        rows = []
        for band in result["ranking"]:
            rows.append(
                {
                    "Band": band["band"],
                    "Households": len(band["ids"]),
                    "Range": "%.0f – %.0f" % (band["lowest"], band["highest"]),
                    "Separated": "yes" if band["separated"] else "tied",
                    "Members": ", ".join(band["labels"][:6])
                    + ("…" if len(band["labels"]) > 6 else ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "A band groups households whose posterior intervals overlap the "
            "one that opened the band. Ordering within a band would invent a "
            "distinction the data does not support."
        )

        st.markdown("**Pooled estimates with intervals**")
        top = result["estimates"][:25]
        interval = go.Figure()
        interval.add_trace(
            go.Scatter(
                x=[entry["pooled_mean"] for entry in top],
                y=[entry["label"] for entry in top],
                mode="markers",
                marker={"size": 9, "color": "#2E86AB"},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [entry["upper"] - entry["pooled_mean"] for entry in top],
                    "arrayminus": [entry["pooled_mean"] - entry["lower"] for entry in top],
                },
                name="Pooled",
            )
        )
        interval.add_trace(
            go.Scatter(
                x=[entry["raw_mean"] for entry in top],
                y=[entry["label"] for entry in top],
                mode="markers",
                marker={"size": 7, "color": "#C1443C", "symbol": "x"},
                name="Raw",
            )
        )
        interval.update_layout(
            title="The leaders on the raw estimate are not the leaders once shrunk",
            xaxis_title="Annual footprint",
            height=max(420, 22 * len(top)),
        )
        st.plotly_chart(interval, use_container_width=True)

        st.markdown("**Every household**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Household": entry["label"],
                        "n": entry["n"],
                        "Raw": round(entry["raw_mean"], 1),
                        "Pooled": round(entry["pooled_mean"], 1),
                        "Shrunk by": round(entry["shrinkage"], 1),
                        "lambda": round(entry["reliability"], 3),
                        "Reads as": entry["reliability_band"].replace("_", " "),
                    }
                    for entry in result["estimates"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_household:
    st.subheader("What this household's number actually says")
    result = st.session_state.get("bp_result")
    panel = st.session_state.get("bp_panel")
    if not result or not panel:
        st.info("Build a panel on the first tab.")
    else:
        options = {entry["label"]: entry["id"] for entry in result["estimates"]}
        choice = st.selectbox("Household", list(options))
        identifier = options[choice]
        estimate = next(
            entry for entry in result["estimates"] if entry["id"] == identifier
        )
        household = next(entry for entry in panel if entry["id"] == identifier)

        head = st.columns(4)
        head[0].metric("Assessments", estimate["n"])
        head[1].metric("Raw mean", "%.0f" % estimate["raw_mean"])
        head[2].metric(
            "Pooled",
            "%.0f" % estimate["pooled_mean"],
            delta="%.0f" % -estimate["shrinkage"],
        )
        head[3].metric("lambda", "%.2f" % estimate["reliability"])

        if estimate["mostly_group"]:
            st.error(
                "Reliability %.2f — this figure is mostly the group mean. It is "
                "a statement about the neighbourhood, not a measurement of this "
                "household, and it should be labelled that way wherever it "
                "appears." % estimate["reliability"]
            )
        elif estimate["reliability"] < HIGH_RELIABILITY:
            st.warning(
                "Reliability %.2f — part own data, part group. The interval is "
                "the honest width." % estimate["reliability"]
            )
        else:
            st.success(
                "Reliability %.2f — enough history that the pooled figure is "
                "essentially this household's own." % estimate["reliability"]
            )

        st.caption(
            "%.0f – %.0f at %d%% confidence."
            % (estimate["lower"], estimate["upper"], round(result["confidence"] * 100))
        )

        st.markdown("**Percentile**")
        place = percentile_of(identifier, result)
        st.info(place["headline"])

        st.markdown("**Trend**")
        trend = trend_direction(household, result["components"]["within_variance"])
        if trend["direction"] == "insufficient_data":
            st.warning(trend["headline"])
        elif trend["direction"] == "stable":
            st.info(trend["headline"])
        else:
            st.success(trend["headline"])

        history = go.Figure()
        history.add_trace(
            go.Scatter(
                x=list(range(1, estimate["n"] + 1)),
                y=household["observations"],
                mode="lines+markers",
                name="Assessments",
                line={"color": "#2E86AB"},
            )
        )
        history.add_hline(
            y=estimate["raw_mean"],
            line_dash="dot",
            line_color="#C1443C",
            annotation_text="raw mean",
        )
        history.add_hline(
            y=estimate["pooled_mean"],
            line_dash="dash",
            line_color="#2E7D32",
            annotation_text="pooled",
        )
        history.add_hline(
            y=result["components"]["grand_mean"],
            line_dash="dot",
            line_color="#888",
            annotation_text="group mean",
        )
        history.update_layout(
            title="History, and where the pooled estimate sits in it",
            xaxis_title="Assessment",
            yaxis_title="Annual footprint",
            height=420,
        )
        st.plotly_chart(history, use_container_width=True)

        st.markdown("**Badge**")
        current = st.selectbox(
            "Badge currently held",
            ["(none)"] + [name for name, _ in DEFAULT_BADGES],
            help="Hysteresis only applies to a badge already awarded.",
        )
        verdict = badge_for(
            estimate,
            current_badge=None if current == "(none)" else current,
        )
        if verdict["held_on_hysteresis"]:
            st.warning(verdict["headline"])
        elif verdict["badge"]:
            st.success(verdict["headline"])
        else:
            st.info(verdict["headline"])

        st.dataframe(
            pd.DataFrame(
                [
                    {"Badge": name, "Ceiling": ceiling}
                    for name, ceiling in DEFAULT_BADGES
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_saved:
    st.subheader("Saved panels")
    panels = get_panels(user_id)
    if not panels:
        st.info("Nothing saved yet.")
    else:
        for entry in panels:
            with st.expander(
                "#%d — %s (%s)" % (entry["id"], entry["label"], entry["created_at"])
            ):
                columns = st.columns(3)
                columns[0].metric("Households", entry["households"])
                columns[1].metric(
                    "ICC", "%.2f" % (entry["intraclass_correlation"] or 0.0)
                )
                columns[2].metric("Ranks moved", entry["ranks_moved"])
                payload = entry.get("payload") or {}
                if payload.get("headline"):
                    st.caption(payload["headline"])
                if st.button("Delete", key="bp_delete_%d" % entry["id"]):
                    if delete_panel(user_id, entry["id"]):
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.warning("Could not delete.")

st.divider()
st.caption(
    "Empirical-Bayes shrinkage by method of moments on a one-way random-effects "
    "model. James–Stein estimation; reliability weighting; ranking under "
    "posterior uncertainty. Standard library only — no dependency added."
)
