import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sequestration import (
    DEFAULT_HORIZON_YEARS,
    DEFAULT_SURVIVAL_RATE,
    MAX_HORIZON_YEARS,
    PLANTING_TYPES,
    build_plan_summary,
    capacity_for_area,
    delete_planting_plan,
    design_plan,
    get_planting_plans,
    get_planting_tips,
    get_planting_type,
    list_planting_types,
    offset_share,
    save_planting_plan,
)
from styles.theme import apply_theme

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>🌳 Tree & Garden Sequestration Planner</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "\"Plant a tree\" is the most repeated piece of advice in this app and the "
    "only one with no arithmetic behind it. This page supplies the arithmetic — "
    "including the parts that are less comfortable."
)
st.info(
    "**Two things the usual advice leaves out.** Sequestration is an S-curve, "
    "not a flat rate: a sapling absorbs almost nothing for years. And a garden "
    "is a fixed area — for most gardens the honest answer to \"how long until "
    "this offsets my footprint?\" is that it does not. This page will tell you so."
)

st.markdown("---")
st.markdown("### 📐 Your Space")

area_col, horizon_col, survival_col = st.columns(3)
area_m2 = area_col.number_input(
    "Plantable area (m²)", min_value=0.0, max_value=100000.0, value=120.0, step=10.0
)
horizon_years = horizon_col.number_input(
    "Look ahead (years)",
    min_value=1,
    max_value=MAX_HORIZON_YEARS,
    value=DEFAULT_HORIZON_YEARS,
    step=5,
)
survival_rate = survival_col.slider(
    "Survival rate",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_SURVIVAL_RATE,
    step=0.05,
    help="Not everything planted survives. Establishment failure is the main cause.",
)

annual_footprint = st.number_input(
    "Your annual carbon footprint (kg CO₂e)",
    min_value=0.0,
    max_value=1000000.0,
    value=5000.0,
    step=100.0,
    help="Used to work out what share of your emissions this planting covers.",
)

st.markdown("---")
st.markdown("### 🌱 What Would You Plant?")

suggest_col, goal_col = st.columns([1, 2])
goal = goal_col.radio(
    "If you want a suggestion, optimise for",
    ["balanced", "fast", "capacity"],
    horizontal=True,
    format_func=lambda value: {
        "balanced": "Balanced",
        "fast": "Early uptake",
        "capacity": "Mature capacity",
    }[value],
)
if suggest_col.button("Suggest a plan", use_container_width=True):
    st.session_state["suggested_plan"] = design_plan(area_m2, goal)

suggested = {
    entry["planting_type"]: entry["count"]
    for entry in st.session_state.get("suggested_plan", [])
}

plan = []
for info in list_planting_types():
    name = info["name"]
    fits = capacity_for_area(area_m2, name)
    entry_col, info_col = st.columns([1, 3])
    count = entry_col.number_input(
        name,
        min_value=0,
        max_value=max(1, fits) if fits else 1000,
        value=int(suggested.get(name, 0)),
        step=1,
        key=f"plant_{name}",
    )
    info_col.caption(
        f"{info['note']} Mature uptake **{info['mature_rate_kg']} kg CO₂/yr** after "
        f"**{info['years_to_maturity']} years**, needs **{info['spacing_m2']} m²** each — "
        f"your space fits **{fits}**."
    )
    if count > 0:
        plan.append({"planting_type": name, "count": count})

if not plan:
    st.warning("Set a count above one to build a plan, or use the suggestion button.")
    st.stop()

summary = build_plan_summary(plan, area_m2, annual_footprint, horizon_years, survival_rate)

if not summary["fits"]:
    st.error(
        f"This plan needs **{summary['area_used_m2']:,.0f} m²** but you have "
        f"**{area_m2:,.0f} m²**. Crowded plantings compete and none of them reach "
        "the mature rates below."
    )

st.markdown("---")
st.markdown("### 📈 The Curve")

years = list(range(1, summary["years"] + 1))

figure = go.Figure()
figure.add_trace(
    go.Scatter(
        x=years,
        y=summary["cumulative_curve"],
        mode="lines",
        name="Cumulative CO₂ absorbed",
        line=dict(color="#2f5e32", width=3),
        fill="tozeroy",
        fillcolor="rgba(120, 169, 69, 0.20)",
    )
)
figure.add_trace(
    go.Scatter(
        x=years,
        y=summary["curve"],
        mode="lines",
        name="Absorbed that year",
        line=dict(color="#78a945", width=2, dash="dot"),
        yaxis="y2",
    )
)

if annual_footprint > 0:
    figure.add_hline(
        y=annual_footprint,
        line_dash="dash",
        line_color="#c0392b",
        annotation_text="One year of your emissions",
    )

figure.update_layout(
    height=440,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Years after planting",
    yaxis_title="Cumulative kg CO₂",
    yaxis2=dict(title="kg CO₂ that year", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(figure, use_container_width=True)

st.caption(
    f"Year one absorbs **{summary['curve'][0]:,.1f} kg**; year {summary['years']} "
    f"absorbs **{summary['curve'][-1]:,.1f} kg**. That gap is the S-curve, and it is "
    "the reason a flat annual multiplier gives people a number that never arrives."
)

st.markdown("---")
st.markdown("### 🎯 What It Actually Delivers")

metric_columns = st.columns(4)
metric_columns[0].metric(
    f"Net over {summary['years']} years", f"{summary['net_co2_kg']:,.0f} kg CO₂"
)
metric_columns[1].metric("At full maturity", f"{summary['mature_annual_kg']:,.0f} kg/yr")
metric_columns[2].metric(
    "Covers this share of a year",
    f"{summary['offset_share_at_horizon_pct']:.1f}%",
)
metric_columns[3].metric("Biodiversity", f"{summary['biodiversity_score']:.0f}/100")

if summary["years_to_offset"] is None:
    st.error(
        f"**This planting will not offset your {annual_footprint:,.0f} kg annual "
        f"footprint within {summary['years']} years.** At full maturity it absorbs "
        f"about {summary['mature_annual_kg']:,.0f} kg a year. That is not an argument "
        "against planting — it is an argument against treating planting as a "
        "substitute for reducing emissions in the first place."
    )
else:
    st.success(
        f"**This planting cancels one year of your emissions in year "
        f"{summary['years_to_offset']}.** Worth noting the wait — that is the part "
        "the usual advice never mentions."
    )

st.caption(
    f"Gross uptake {summary['gross_co2_kg']:,.0f} kg, less survival losses at "
    f"{survival_rate:.0%} and {summary['maintenance_co2_kg']:,.0f} kg of maintenance "
    "emissions from watering, pruning and mowing."
)

st.markdown("---")
st.markdown("### 🗓️ Milestones")

milestone_years = [year for year in (1, 5, 10, 20, 30, 40) if year <= summary["years"]]
milestone_frame = pd.DataFrame(
    [
        {
            "Year": year,
            "Absorbed that year": f"{summary['curve'][year - 1]:,.1f} kg",
            "Cumulative": f"{summary['cumulative_curve'][year - 1]:,.0f} kg",
            "Share of one year's footprint": (
                f"{offset_share(plan, annual_footprint, year, survival_rate):.1f}%"
            ),
        }
        for year in milestone_years
    ]
)
st.dataframe(milestone_frame, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 🐝 Your Plan")

plan_frame = pd.DataFrame(
    [
        {
            "Planting": entry["planting_type"],
            "Count": entry["count"],
            "Space used": f"{get_planting_type(entry['planting_type'])['spacing_m2'] * entry['count']:,.0f} m²",
            "Mature uptake": (
                f"{get_planting_type(entry['planting_type'])['mature_rate_kg'] * entry['count']:,.0f} kg/yr"
            ),
            "Native habitat": (
                "Yes" if get_planting_type(entry["planting_type"])["native_habitat"] else "No"
            ),
        }
        for entry in summary["plan"]
    ]
)
st.dataframe(plan_frame, use_container_width=True, hide_index=True)
st.caption(
    f"Using {summary['area_used_m2']:,.0f} m² of your {area_m2:,.0f} m². The "
    "biodiversity score rewards mixing species and choosing natives — optimising "
    "carbon alone would hand you a monoculture, which is a worse garden."
)

st.markdown("---")
st.markdown("### 💡 What To Do About It")
for tip in get_planting_tips(summary, annual_footprint):
    st.markdown(f"- {tip}")

st.markdown("---")
st.markdown("### 💾 Save This Plan")

name_col, save_col = st.columns([3, 1])
plan_name = name_col.text_input("Plan name", value="My garden")
if save_col.button("Save plan", use_container_width=True):
    if save_planting_plan(user_id, plan_name, summary, area_m2):
        st.success("Plan saved.")
    else:
        st.error("Could not save that plan.")

saved_plans = get_planting_plans(user_id)
if saved_plans:
    st.markdown("#### Saved plans")
    for record in saved_plans:
        detail_col, delete_col = st.columns([5, 1])
        offset_text = (
            f"offsets a year in year {record['years_to_offset']}"
            if record["years_to_offset"]
            else "does not offset a full year"
        )
        detail_col.markdown(
            f"**{record['plan_name']}** — {record['area_m2']:,.0f} m² · "
            f"{record['net_co2_kg']:,.0f} kg net over {record['horizon_years']} years · "
            f"{offset_text} · biodiversity {record['biodiversity_score']:.0f}/100"
        )
        if delete_col.button("Delete", key=f"delete_plan_{record['id']}"):
            delete_planting_plan(record["id"])
            st.rerun()
else:
    st.caption("No saved plans yet.")

st.caption(
    "Sequestration rates, maturity timelines and spacing follow published "
    "forestry and urban-tree carbon guidance. Uptake uses a logistic growth "
    "curve rather than a flat annual rate; the model and every assumption are "
    "documented inline in `sequestration.py`."
)
