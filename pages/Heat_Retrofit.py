import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from heat_retrofit import (
    AIRTIGHTNESS_LEVELS,
    CLIMATE_ZONES,
    DEFAULT_CLIMATE_ZONE,
    DEFAULT_EMITTER,
    DOOR_TYPES,
    EMITTER_TYPES,
    FLOOR_TYPES,
    FUELS,
    GLAZING_TYPES,
    HEATING_SYSTEMS,
    MONTHS,
    RETROFIT_MEASURES,
    ROOF_TYPES,
    WALL_TYPES,
    annual_degree_days,
    build_retrofit_plan,
    compare_systems,
    delete_retrofit_plan,
    estimate_envelope,
    fabric_first_check,
    get_retrofit_advice,
    get_retrofit_plans,
    heat_loss_coefficient,
    measure_applies,
    measure_cost,
    monthly_heat_demand_kwh,
    rank_measures,
    save_retrofit_plan,
)
from styles.theme import apply_theme

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>🏚️ Home Heat Loss & Retrofit Planner</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "The Home Energy Audit measures your appliances. This measures the building — "
    "where the heat escapes, which fix is worth doing first, and whether a heat "
    "pump would actually work here yet."
)

st.markdown("---")
st.markdown("### 🏠 The Building")

shape_col, storey_col, attached_col, zone_col = st.columns(4)
floor_area = shape_col.number_input(
    "Total floor area (m²)", min_value=20.0, max_value=1000.0, value=120.0, step=5.0,
    help="Add up every heated floor.",
)
storeys = storey_col.number_input("Storeys", min_value=1, max_value=5, value=2, step=1)
attached = attached_col.selectbox(
    "Walls shared with neighbours",
    [0, 1, 2, 3],
    format_func=lambda count: {
        0: "0 — detached", 1: "1 — semi-detached",
        2: "2 — mid-terrace", 3: "3 — flat, three sides sheltered",
    }[count],
)
climate_zone = zone_col.selectbox(
    "Climate", list(CLIMATE_ZONES.keys()),
    index=list(CLIMATE_ZONES.keys()).index(DEFAULT_CLIMATE_ZONE),
)

envelope = estimate_envelope(floor_area, storeys, attached)

st.caption(
    f"Estimated envelope — walls **{envelope['wall_m2']:,.0f} m²**, roof "
    f"**{envelope['roof_m2']:,.0f} m²**, windows **{envelope['glazing_m2']:,.0f} m²**, "
    f"volume **{envelope['volume_m3']:,.0f} m³**. "
    f"{annual_degree_days(climate_zone):,.0f} heating degree days a year."
)

st.markdown("### 🧱 How It Is Built")

fabric_left, fabric_right = st.columns(2)
fabric = {
    "wall": fabric_left.selectbox("Walls", list(WALL_TYPES.keys())),
    "roof": fabric_left.selectbox("Roof / loft", list(ROOF_TYPES.keys())),
    "floor": fabric_left.selectbox("Ground floor", list(FLOOR_TYPES.keys())),
    "glazing": fabric_right.selectbox("Windows", list(GLAZING_TYPES.keys())),
    "door": fabric_right.selectbox("External doors", list(DOOR_TYPES.keys())),
    "airtightness": fabric_right.selectbox("Draughtiness", list(AIRTIGHTNESS_LEVELS.keys())),
}

system_col, emitter_col = st.columns(2)
system_name = system_col.selectbox("Current heating system", list(HEATING_SYSTEMS.keys()))
emitter = emitter_col.selectbox(
    "Heat emitters", list(EMITTER_TYPES.keys()),
    index=list(EMITTER_TYPES.keys()).index(DEFAULT_EMITTER),
)

with st.expander("💷 Use my own fuel prices"):
    st.caption("Defaults are indicative. Enter what you actually pay per kWh.")
    fuel_overrides = {}
    price_columns = st.columns(len(FUELS))
    for index, (fuel_name, fuel) in enumerate(FUELS.items()):
        price = price_columns[index].number_input(
            fuel_name, min_value=0.0, max_value=2.0,
            value=float(fuel["price_per_kwh"]), step=0.01, format="%.3f",
            key=f"price_{fuel_name}",
        )
        fuel_overrides[fuel_name] = {"price_per_kwh": price}

losses = heat_loss_coefficient(envelope, fabric)

st.markdown("---")
st.markdown("### 🌡️ Where the Heat Goes")

loss_frame = pd.DataFrame(
    [
        {"Element": element.title(), "W/K": value}
        for element, value in losses["breakdown"].items()
        if value > 0
    ]
).sort_values("W/K", ascending=False)

loss_col, metric_col = st.columns([3, 2])
with loss_col:
    fig = px.bar(
        loss_frame, x="W/K", y="Element", orientation="h",
        title="Heat loss by element (watts per degree of temperature difference)",
        color="W/K", color_continuous_scale="Reds",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with metric_col:
    baseline_plan = build_retrofit_plan(
        envelope, fabric, [], climate_zone, system_name, emitter, fuel_overrides
    )
    st.metric("Heat loss coefficient", f"{losses['total_w_per_k']:,.0f} W/K")
    st.metric("Annual heat demand", f"{baseline_plan['baseline']['demand_kwh']:,.0f} kWh")
    st.metric("Peak load (design day)", f"{baseline_plan['baseline']['peak_load_w'] / 1000:,.1f} kW")
    st.metric("Heating emissions", f"{baseline_plan['baseline']['co2_kg']:,.0f} kg CO₂e")
    st.caption(
        f"Biggest single loss: **{losses['worst_element']}**. "
        f"Ventilation accounts for {losses['ventilation_w_per_k']:,.0f} W/K of the total."
    )

monthly = monthly_heat_demand_kwh(losses["total_w_per_k"], climate_zone)
season_fig = px.area(
    pd.DataFrame({"Month": MONTHS, "kWh": monthly}),
    x="Month", y="kWh", title="Heating demand through the year",
)
st.plotly_chart(season_fig, use_container_width=True)

st.markdown("---")
st.markdown("### 🔧 What To Do First")

ranking = rank_measures(envelope, fabric, climate_zone, system_name, fuel_overrides)

if not ranking["measures"]:
    st.success(
        "There is nothing left in the measure list for this house — the fabric is "
        "already at or beyond every upgrade modelled here."
    )
    selected_measures = []
else:
    ranked_frame = pd.DataFrame(
        [
            {
                "Measure": row["measure"],
                "Cost": round(row["cost"]),
                "kWh saved / yr": round(row["saved_kwh"]),
                "kg CO₂e saved / yr": round(row["saved_co2_kg"]),
                "Cost per kWh saved": row["cost_per_kwh_saved"],
                "Payback (yrs)": row["payback_years"],
            }
            for row in ranking["measures"]
        ]
    )
    st.dataframe(ranked_frame, use_container_width=True, hide_index=True)
    st.caption(
        "Ranked by cost per kilowatt-hour saved, each measure evaluated against "
        "your house as it is today — not against a brochure house."
    )

    for row in ranking["measures"][:3]:
        st.markdown(f"- **{row['measure']}** — {row['note']}")

    selected_measures = st.multiselect(
        "Build a plan from these measures (order matters — savings never double-count)",
        [row["measure"] for row in ranking["measures"]],
        default=[row["measure"] for row in ranking["measures"][:2]],
    )

plan = build_retrofit_plan(
    envelope, fabric, selected_measures, climate_zone, system_name, emitter, fuel_overrides
)

if plan["steps"]:
    st.markdown("### 📉 Your Plan")

    plan_cols = st.columns(4)
    plan_cols[0].metric("Upfront cost", f"{plan['total_cost']:,.0f}")
    plan_cols[1].metric(
        "Heat demand", f"{plan['after']['demand_kwh']:,.0f} kWh",
        f"-{plan['demand_saved_pct']}%",
    )
    plan_cols[2].metric(
        "Running cost", f"{plan['after']['cost_per_year']:,.0f}/yr",
        f"-{plan['annual_saving']:,.0f}/yr",
    )
    plan_cols[3].metric(
        "Emissions", f"{plan['after']['co2_kg']:,.0f} kg",
        f"-{plan['co2_saved_kg']:,.0f} kg",
    )

    if plan["payback_years"]:
        st.info(f"Simple payback: **{plan['payback_years']} years** at the fuel prices above.")
    else:
        st.info(
            "These measures do not pay back on fuel savings alone at current prices. "
            "They still buy comfort, quieter rooms and a house that holds heat in a cold snap."
        )

    step_frame = pd.DataFrame(
        [
            {
                "Step": index + 1,
                "Measure": step["measure"],
                "Cost": round(step["cost"]),
                "Demand after (kWh)": round(step["demand_kwh"]),
                "Flow temp needed (°C)": step["flow_temperature_c"],
                "Saved (kWh)": round(step["step_saved_kwh"]),
            }
            for index, step in enumerate(plan["steps"])
        ]
    )
    st.dataframe(step_frame, use_container_width=True, hide_index=True)

    waterfall = go.Figure(
        go.Scatter(
            x=["Today"] + [step["measure"] for step in plan["steps"]],
            y=[plan["baseline"]["demand_kwh"]] + [step["demand_kwh"] for step in plan["steps"]],
            mode="lines+markers",
            line=dict(color="#2f5e32", width=3),
        )
    )
    waterfall.update_layout(
        title="Heat demand as each measure lands",
        yaxis_title="kWh per year",
        xaxis_title="",
    )
    st.plotly_chart(waterfall, use_container_width=True)

st.markdown("---")
st.markdown("### ♨️ Would a Heat Pump Work Here?")

check = fabric_first_check(envelope, fabric, climate_zone, emitter, selected_measures)

pump_cols = st.columns(4)
pump_cols[0].metric("Seasonal COP today", check["cop_now"])
pump_cols[1].metric(
    "COP after the fabric work", check["cop_after_fabric"],
    f"{check['cop_gain']:+.2f}",
)
pump_cols[2].metric(
    "Flow temperature needed", f"{check['flow_after_c']} °C",
    f"{check['flow_after_c'] - check['flow_now_c']:+.1f} °C",
)
pump_cols[3].metric("Smaller unit you could buy", f"{check['smaller_unit_kw']:,.1f} kW")

st.info(check["verdict"])
st.caption(
    "A heat pump's efficiency is set by how hot the water has to be. Insulation "
    "lowers the peak load, the same radiators then satisfy it with cooler water, "
    "and the COP rises. That chain is why the order of the work matters."
)

st.markdown("### ⚖️ Every System, Same House")

comparison = compare_systems(
    plan["after"]["demand_kwh"], climate_zone, check["flow_after_c"], fuel_overrides
)
comparison_frame = pd.DataFrame(
    [
        {
            "System": row["system"],
            "Fuel": row["fuel"],
            "Efficiency / COP": row["efficiency"],
            "Fuel used (kWh)": round(row["fuel_kwh"]),
            "Running cost": round(row["cost"]),
            "kg CO₂e": round(row["co2_kg"]),
        }
        for row in comparison
    ]
)
st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
st.caption(
    "Compared against the heat demand *after* your selected measures, at the flow "
    "temperature that house would then need."
)

st.markdown("---")
st.markdown("### 💾 Saved Plans")

with st.form("save_retrofit_plan"):
    plan_name = st.text_input("Name this plan", value="My home")
    if st.form_submit_button("Save plan", use_container_width=True):
        if save_retrofit_plan(user_id, plan_name, plan, floor_area):
            st.success(f"Saved **{plan_name}**.")
            st.rerun()
        else:
            st.error("Could not save that plan. Please try again.")

saved = get_retrofit_plans(user_id)
if saved:
    saved_frame = pd.DataFrame(
        [
            {
                "Plan": row["plan_name"],
                "Climate": row["climate_zone"],
                "System": row["system"],
                "Before (kWh)": round(row["baseline_demand_kwh"]),
                "After (kWh)": round(row["final_demand_kwh"]),
                "kg CO₂e saved": round(row["co2_saved_kg"]),
                "Cost": round(row["total_cost"]),
                "Payback (yrs)": row["payback_years"],
            }
            for row in saved
        ]
    )
    st.dataframe(saved_frame, use_container_width=True, hide_index=True)

    remove_col, _ = st.columns([2, 3])
    to_remove = remove_col.selectbox(
        "Remove a plan", saved, format_func=lambda row: row["plan_name"]
    )
    if remove_col.button("Delete plan", use_container_width=True):
        if delete_retrofit_plan(to_remove["id"]):
            st.success("Plan removed.")
            st.rerun()
        else:
            st.error("Could not remove that plan.")
else:
    st.info("No plans saved yet.")

st.markdown("### 💡 Advice For This House")
for tip in get_retrofit_advice(plan):
    st.markdown(f"- {tip}")

with st.expander("📐 How these numbers are worked out"):
    st.markdown(
        """
**Heat loss coefficient** — every element's area multiplied by its U-value
(watts lost per square metre per degree), plus ventilation loss of
0.33 × air changes per hour × volume.

**Annual demand** — the degree-day method: `HLC × heating degree days × 24 ÷ 1000`.
Degree days are computed from monthly mean outdoor temperatures against a
15.5 °C base, which already allows for the heat given off by people, cooking
and sunlight.

**Peak load** — the same coefficient against the design cold-snap temperature
for your climate. This is what a heating system has to be sized for.

**Flow temperature** — radiator output scales with the temperature difference
to the room raised to about 1.3. Existing radiators were sized to heat the
house as it was, so once the fabric improves, the same radiators satisfy the
smaller load with cooler water.

**Heat pump COP** — a Carnot-fraction model: the theoretical limit between
source and water temperature, multiplied by how close real machines get to it.
Ground loops use a stable 8 °C source instead of tracking air temperature.

**Costs** — indicative installed costs, editable above. Payback is simple
payback on fuel saving alone and ignores comfort, which is usually the reason
people are glad they did the work.
        """
    )
