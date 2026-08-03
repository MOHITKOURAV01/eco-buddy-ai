import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from offset_quality import (
    AVOIDANCE,
    DEFAULT_REGISTRY,
    PROJECT_TYPES,
    REGISTRIES,
    REMOVAL,
    TYPICAL_BUFFER_SHARE,
    assess_credit,
    default_buffer_share,
    delete_holding,
    get_holdings,
    get_offset_advice,
    get_project_type,
    list_project_types,
    list_registries,
    mitigation_hierarchy,
    portfolio_summary,
    recommend_portfolio,
    save_holding,
)
from styles.theme import apply_theme

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

CURRENT_YEAR = datetime.date.today().year

st.markdown(
    "<div class='section-header'>🧾 Carbon Offset Quality Auditor</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "One tonne of carbon credit is not one thing. This page scores what you are "
    "about to buy, converts the tonnes you pay for into the tonnes that are "
    "plausibly real, and keeps offsetting in its proper place — last."
)

st.markdown("---")
st.markdown("### 🔍 Assess an Offer")

project_col, registry_col = st.columns(2)
project_type = project_col.selectbox(
    "Project type",
    [project["name"] for project in list_project_types()],
    help="What the project actually does.",
)
registry = registry_col.selectbox(
    "Standard / registry",
    [item["name"] for item in list_registries()],
    index=len(REGISTRIES) - 1,
    help="Who has independently checked this credit.",
)

project = get_project_type(project_type)
st.caption(
    f"**{'Removal' if project['kind'] == REMOVAL else 'Avoidance'}** — {project['note']} "
    f"Typical market price around {project['typical_price']:.0f} per tonne; "
    f"below {project['floor_price']:.0f} the claimed tonne cannot plausibly be delivered."
)
st.caption(f"**{registry}** — {REGISTRIES[registry]['note']}")

tonnes_col, price_col, vintage_col, buffer_col = st.columns(4)
tonnes = tonnes_col.number_input(
    "Tonnes offered", min_value=0.0, max_value=10000.0, value=5.0, step=0.5
)
price = price_col.number_input(
    "Price per tonne", min_value=0.0, max_value=5000.0,
    value=float(project["typical_price"]), step=1.0,
)
vintage = vintage_col.number_input(
    "Vintage year", min_value=1995, max_value=CURRENT_YEAR, value=CURRENT_YEAR, step=1,
    help="The year the reduction is claimed to have happened.",
)
buffer_share = buffer_col.slider(
    "Registry buffer pool", min_value=0.0, max_value=0.5,
    value=float(default_buffer_share(project_type)), step=0.01,
    help=(
        "Credits held back to cover reversals. Good for reversible projects — "
        "and not counted towards your tonnes, because it is insurance."
    ),
)

assessment = assess_credit(
    project_type, tonnes, price, registry, vintage, CURRENT_YEAR, buffer_share
)

if not assessment or assessment["nominal_tonnes"] <= 0:
    st.info("Enter a tonnage above to assess the offer.")
    st.stop()

grade_colours = {"A": "#2f5e32", "B": "#5f8f36", "C": "#c9a227", "D": "#c1662f", "F": "#a02c2c"}

st.markdown("### 📋 Verdict")

verdict_cols = st.columns(4)
verdict_cols[0].metric("Quality score", f"{assessment['score']}/100")
verdict_cols[1].metric("Grade", assessment["grade"])
shortfall = assessment["nominal_tonnes"] - assessment["effective_tonnes"]
verdict_cols[2].metric(
    "Tonnes you actually get",
    f"{assessment['effective_tonnes']:,.2f}",
    f"-{shortfall:,.2f}",
)
verdict_cols[3].metric(
    "Real cost per tonne",
    f"{assessment['cost_per_effective_tonne']:,.0f}" if assessment["cost_per_effective_tonne"] else "—",
    f"listed at {assessment['price_per_tonne']:,.0f}",
)

st.markdown(
    f"<div style='padding:0.75rem 1rem;border-radius:0.5rem;background:{grade_colours.get(assessment['grade'], '#666')};"
    f"color:white;font-weight:600'>{assessment['grade']} — {assessment['verdict']}</div>",
    unsafe_allow_html=True,
)

if assessment["warnings"]:
    st.markdown("#### ⚠️ What to know before buying")
    for warning in assessment["warnings"]:
        st.warning(warning)

factor_col, component_col = st.columns(2)

with factor_col:
    factors = assessment["factors"]
    waterfall = go.Figure(
        go.Bar(
            x=list(factors.values()),
            y=[name.replace("_", " ").title() for name in factors],
            orientation="h",
            marker_color="#5f8f36",
        )
    )
    waterfall.update_layout(
        title="What survives each discount (1.0 = nothing lost)",
        xaxis_range=[0, 1],
        yaxis_title="",
    )
    st.plotly_chart(waterfall, use_container_width=True)

with component_col:
    components = assessment["components"]
    radar = go.Figure(
        go.Scatterpolar(
            r=[round(value * 100, 1) for value in components.values()],
            theta=[name.replace("_", " ").title() for name in components],
            fill="toself",
            line_color="#2f5e32",
        )
    )
    radar.update_layout(
        title="Quality by dimension",
        polar=dict(radialaxis=dict(range=[0, 100])),
        showlegend=False,
    )
    st.plotly_chart(radar, use_container_width=True)

st.caption(
    f"You would pay **{assessment['total_spend']:,.0f}** for "
    f"{assessment['nominal_tonnes']:,.2f} nominal tonnes and receive roughly "
    f"**{assessment['effective_tonnes']:,.2f}** real ones "
    f"({assessment['delivery_ratio'] * 100:.0f}% delivery). Storage is credited for "
    f"about {assessment['permanence_years']:,} years."
)

st.markdown("---")
st.markdown("### 🪜 Reduce First")

hierarchy_cols = st.columns(3)
footprint = hierarchy_cols[0].number_input(
    "Your annual footprint (kg CO₂e)", min_value=0.0, max_value=200000.0,
    value=float(st.session_state.get("last_footprint_kg", 8000.0)), step=100.0,
)
reduced = hierarchy_cols[1].number_input(
    "Reduced so far this year (kg)", min_value=0.0, max_value=200000.0, value=500.0, step=100.0,
    help="Emissions you have actually cut — not bought back.",
)
offset_tonnes = hierarchy_cols[2].number_input(
    "Tonnes offset this year", min_value=0.0, max_value=500.0,
    value=float(assessment["nominal_tonnes"]), step=0.5,
)

hierarchy = mitigation_hierarchy(footprint, reduced, offset_tonnes)

if hierarchy["status"] == "REDUCTION_LED":
    st.success(hierarchy["message"])
elif hierarchy["status"] == "NOTHING_YET":
    st.info(hierarchy["message"])
else:
    st.warning(hierarchy["message"])

hierarchy_frame = pd.DataFrame(
    [
        {"Action": "Reduced", "kg CO₂e": hierarchy["reduced_kg"]},
        {"Action": "Offset", "kg CO₂e": hierarchy["offset_kg"]},
        {"Action": "Still emitted", "kg CO₂e": max(0.0, hierarchy["residual_kg"] - hierarchy["offset_kg"])},
    ]
)
hierarchy_fig = px.bar(
    hierarchy_frame, x="kg CO₂e", y="Action", orientation="h",
    color="Action",
    color_discrete_map={
        "Reduced": "#2f5e32", "Offset": "#c9a227", "Still emitted": "#a02c2c",
    },
    title="Where your footprint went",
)
hierarchy_fig.update_layout(showlegend=False, yaxis_title="")
st.plotly_chart(hierarchy_fig, use_container_width=True)

st.markdown("---")
st.markdown("### 🧮 What Would a Budget Buy?")

budget_col, target_col, preference_col = st.columns(3)
budget = budget_col.number_input(
    "Budget", min_value=0.0, max_value=100000.0, value=500.0, step=50.0
)
target = target_col.number_input(
    "Tonnes you want covered", min_value=0.0, max_value=1000.0,
    value=round(max(0.0, hierarchy["residual_kg"]) / 1000, 1), step=0.5,
)
preference = preference_col.slider(
    "Share spent on removals", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Removals are durable and expensive; avoidance is cheap and contestable.",
)

recommendation = recommend_portfolio(budget, target, preference, CURRENT_YEAR)

if recommendation["allocations"]:
    allocation_frame = pd.DataFrame(
        [
            {
                "Project": row["project_type"],
                "Type": row["kind"].title(),
                "Share": f"{row['share_pct']}%",
                "Spend": round(row["spend"]),
                "Price/t": row["price_per_tonne"],
                "Nominal t": row["nominal_tonnes"],
                "Effective t": row["effective_tonnes"],
                "Grade": row["grade"],
            }
            for row in recommendation["allocations"]
        ]
    )
    st.dataframe(allocation_frame, use_container_width=True, hide_index=True)

    if recommendation["affordable"]:
        st.success(recommendation["note"])
    else:
        st.warning(
            f"{recommendation['note']} Shortfall: "
            f"{recommendation['shortfall_tonnes']:,.2f} effective tonnes."
        )
else:
    st.info(recommendation["note"])

st.markdown("---")
st.markdown("### 💾 Your Offset Portfolio")

with st.form("save_offset_holding"):
    label = st.text_input("Name this purchase", value=f"{project_type} {vintage}")
    if st.form_submit_button("Add to my portfolio", use_container_width=True):
        if save_holding(user_id, label, assessment):
            st.success(f"Saved **{label}**.")
            st.rerun()
        else:
            st.error("Could not save that holding. Please try again.")

holdings = get_holdings(user_id)
if holdings:
    portfolio = [
        {
            "project_type": holding["project_type"],
            "kind": holding["kind"],
            "score": holding["score"],
            "grade": holding["grade"],
            "nominal_tonnes": holding["nominal_tonnes"],
            "effective_tonnes": holding["effective_tonnes"],
            "total_spend": holding["total_spend"],
            "permanence_years": holding["permanence_years"],
        }
        for holding in holdings
    ]
    summary = portfolio_summary(portfolio)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Tonnes paid for", f"{summary['nominal_tonnes']:,.2f}")
    summary_cols[1].metric(
        "Tonnes likely real", f"{summary['effective_tonnes']:,.2f}",
        f"-{summary['shortfall_tonnes']:,.2f}",
    )
    summary_cols[2].metric("Portfolio grade", f"{summary['grade']} ({summary['weighted_score']}/100)")
    summary_cols[3].metric("Removal share", f"{summary['removal_share_pct']}%")

    holdings_frame = pd.DataFrame(
        [
            {
                "Purchase": holding["label"],
                "Project": holding["project_type"],
                "Registry": holding["registry"],
                "Type": holding["kind"].title(),
                "Nominal t": holding["nominal_tonnes"],
                "Effective t": holding["effective_tonnes"],
                "Spend": round(holding["total_spend"]),
                "Grade": holding["grade"],
            }
            for holding in holdings
        ]
    )
    st.dataframe(holdings_frame, use_container_width=True, hide_index=True)

    delivery_fig = go.Figure()
    delivery_fig.add_bar(
        name="Paid for", x=holdings_frame["Purchase"], y=holdings_frame["Nominal t"],
        marker_color="#c9a227",
    )
    delivery_fig.add_bar(
        name="Likely real", x=holdings_frame["Purchase"], y=holdings_frame["Effective t"],
        marker_color="#2f5e32",
    )
    delivery_fig.update_layout(
        barmode="group", title="What you paid for vs what you probably got",
        yaxis_title="tonnes CO₂e", xaxis_title="",
    )
    st.plotly_chart(delivery_fig, use_container_width=True)

    remove_col, _ = st.columns([2, 3])
    to_remove = remove_col.selectbox(
        "Remove a holding", holdings,
        format_func=lambda holding: f"{holding['label']} ({holding['grade']})",
    )
    if remove_col.button("Delete holding", use_container_width=True):
        if delete_holding(to_remove["id"]):
            st.success("Holding removed.")
            st.rerun()
        else:
            st.error("Could not remove that holding.")

    st.markdown("### 💡 What Would Improve This")
    for tip in get_offset_advice(summary, hierarchy):
        st.markdown(f"- {tip}")
else:
    st.info("No holdings saved yet. Assess an offer above and add it to see the portfolio view.")

with st.expander("📐 How these numbers are worked out"):
    st.markdown(
        f"""
**Effective tonnes** — the honest version of what you bought:

```
effective = nominal
          × additionality confidence
          × durability discount
          × measurement confidence
          × (1 − leakage)
          × (1 − registry buffer share)
          × (1 − vintage penalty)
```

Every factor is between 0 and 1, so the effective figure can never exceed the
nominal one.

**Additionality** — the confidence the reduction would not have happened
anyway. Grid-connected wind and solar score lowest here: on most grids they
are now the cheapest generation available and were getting built regardless.

**Durability** — carbon released again in thirty years was rented, not offset.
The discount is logarithmic against a 1,000-year geological reference, because
the difference between 10 and 100 years matters far more than between 5,000
and 10,000.

**Measurement** — whether the tonnage is physically measured or modelled from
a counterfactual. Avoided-deforestation baselines are the hardest case: they
rest on what *would* have happened, which nobody can observe.

**Registry weight** — how much independent scrutiny the credit has had. An
unverified credit is multiplied by {REGISTRIES[DEFAULT_REGISTRY]['weight']:.2f},
because nobody has checked it.

**Buffer pool** — credits a registry holds back to cover reversals. It is a
good sign for reversible projects, but it is insurance, so it is not delivered
to you as tonnes. Geological storage is charged no buffer, because it does not
carry the risk.

**Price floor** — below the floor price, the claimed tonne cannot plausibly be
paid for. In this market, cheap is the most visible warning sign there is.
        """
    )
