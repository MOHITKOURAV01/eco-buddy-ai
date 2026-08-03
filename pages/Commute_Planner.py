import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from commute_planner import (
    DEFAULT_GRID_INTENSITY,
    DEFAULT_MODE,
    DEFAULT_SEASON,
    HOME_WORKING_KWH,
    OFFICE_FIXED_KWH_PER_DESK_DAY,
    OFFICE_MARGINAL_KWH_PER_PERSON_DAY,
    SEASONS,
    TRAVEL_MODES,
    WORKING_DAYS_PER_WEEK,
    WORKING_WEEKS_PER_YEAR,
    annual_summary,
    best_schedule,
    compare_modes,
    consolidation_benefit,
    delete_commute_plan,
    get_commute_advice,
    get_commute_plans,
    is_shareable,
    save_commute_plan,
    weekly_plan,
)
from styles.theme import apply_theme

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>🏢 Hybrid Commute Planner</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "Route Planning prices one journey. This prices the **pattern** — the commute, "
    "the office building and the home office together, because a day worked at "
    "home is not a day with no emissions."
)

st.markdown("---")
st.markdown("### 🚗 Your Commute")

mode_col, distance_col, days_col, season_col = st.columns(4)
mode = mode_col.selectbox(
    "Usual mode", list(TRAVEL_MODES.keys()),
    index=list(TRAVEL_MODES.keys()).index(DEFAULT_MODE),
)
distance = distance_col.number_input(
    "One-way distance (km)", min_value=0.0, max_value=300.0, value=18.0, step=0.5
)
days_in_office = days_col.slider(
    "Days in the office per week", min_value=0, max_value=WORKING_DAYS_PER_WEEK, value=3
)
season = season_col.selectbox(
    "Season", SEASONS, index=SEASONS.index(DEFAULT_SEASON)
)

occupants = 1
if is_shareable(mode):
    occupants = st.slider(
        "People in the car", min_value=1, max_value=5, value=1,
        help="Sharing divides the journey's emissions between everyone in the vehicle.",
    )

st.caption(f"**{mode}** — {TRAVEL_MODES[mode]['note']}")

with st.expander("⚙️ Building and grid assumptions"):
    assumption_cols = st.columns(3)
    grid_intensity = assumption_cols[0].number_input(
        "Grid intensity (kg CO₂e/kWh)", min_value=0.0, max_value=1.5,
        value=DEFAULT_GRID_INTENSITY, step=0.01, format="%.3f",
    )
    home_kwh = assumption_cols[1].number_input(
        f"Home energy per WFH day in {season} (kWh)", min_value=0.0, max_value=40.0,
        value=float(HOME_WORKING_KWH[season]), step=0.1,
    )
    office_open_days = assumption_cols[2].slider(
        "Days the office is open", min_value=0, max_value=WORKING_DAYS_PER_WEEK,
        value=WORKING_DAYS_PER_WEEK,
        help="If the whole team works the same days, the building can shut on the others.",
    )
    st.caption(
        f"Office energy is split into a fixed **{OFFICE_FIXED_KWH_PER_DESK_DAY} kWh** "
        f"per desk for every day the building is *open* — spent whether you attend "
        f"or not — plus **{OFFICE_MARGINAL_KWH_PER_PERSON_DAY} kWh** for each day "
        f"you actually go in."
    )

plan = weekly_plan(
    days_in_office, distance, mode, occupants, season,
    office_open_days, grid_intensity, home_kwh,
)

st.markdown("---")
st.markdown("### 📊 Your Week")

week_cols = st.columns(4)
week_cols[0].metric("Weekly total", f"{plan['total_kg']:,.1f} kg CO₂e")
week_cols[1].metric("Commuting", f"{plan['commute_kg']:,.1f} kg", f"{plan['commute_share_pct']}% of week")
week_cols[2].metric("Office building", f"{plan['office']['co2_kg']:,.1f} kg")
week_cols[3].metric("Working from home", f"{plan['home']['co2_kg']:,.1f} kg")

split_frame = pd.DataFrame(
    [
        {"Source": "Commuting", "kg CO₂e": plan["commute_kg"]},
        {"Source": "Office building", "kg CO₂e": plan["office"]["co2_kg"]},
        {"Source": "Home working", "kg CO₂e": plan["home"]["co2_kg"]},
    ]
)

split_col, leg_col = st.columns([1, 1])
with split_col:
    split_fig = px.pie(
        split_frame, names="Source", values="kg CO₂e", hole=0.45,
        title="Where the week's emissions actually come from",
        color="Source",
        color_discrete_map={
            "Commuting": "#c1662f", "Office building": "#5f8f36", "Home working": "#c9a227",
        },
    )
    split_fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(split_fig, use_container_width=True)

with leg_col:
    leg = plan["leg"]
    st.markdown("**One leg of your journey**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Component": "Running emissions", "kg CO₂e": leg["running_kg"]},
                {"Component": "Cold start", "kg CO₂e": leg["cold_start_kg"]},
                {"Component": "Total per leg", "kg CO₂e": leg["total_kg"]},
            ]
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"{leg['kg_per_km']:.3f} kg per km · {leg['minutes']:.0f} minutes each way · "
        f"{plan['minutes_travelled']:.0f} minutes travelled this week."
    )
    if leg["cold_start_kg"] > 0:
        st.caption(
            "The cold start is charged per journey, not per kilometre — which is why "
            "a short drive is so much worse per kilometre than a long one."
        )

st.markdown("---")
st.markdown("### 🗓️ How Many Days Should You Go In?")

comparison = best_schedule(
    distance, mode, occupants, season, office_open_days, grid_intensity, home_kwh
)

schedule_frame = pd.DataFrame(
    [
        {
            "Days in office": item["days_in_office"],
            "Commuting": item["commute_kg"],
            "Office": item["office"]["co2_kg"],
            "Home": item["home"]["co2_kg"],
            "Weekly total": item["total_kg"],
        }
        for item in comparison["schedules"]
    ]
)

stacked = go.Figure()
for column, colour in (
    ("Commuting", "#c1662f"), ("Office", "#5f8f36"), ("Home", "#c9a227")
):
    stacked.add_bar(
        name=column, x=schedule_frame["Days in office"], y=schedule_frame[column],
        marker_color=colour,
    )
stacked.update_layout(
    barmode="stack",
    title="Weekly emissions for every attendance pattern",
    xaxis_title="Days in the office",
    yaxis_title="kg CO₂e per week",
)
st.plotly_chart(stacked, use_container_width=True)

best = comparison["best"]
if best["days_in_office"] == days_in_office:
    st.success(
        f"Your current pattern ({days_in_office} day(s) in) is already the lowest "
        f"of the six options, given this commute and these buildings."
    )
else:
    st.info(
        f"**{best['days_in_office']} day(s) in the office** would be lowest for you — "
        f"{best['total_kg']:,.1f} kg a week against your current {plan['total_kg']:,.1f} kg. "
        f"Across the year, {comparison['annual_spread_kg']:,.0f} kg separates the best "
        f"pattern from the worst."
    )

if not comparison["home_is_better"]:
    st.caption(
        "Note that going in *more* wins here. That happens when the journey is short "
        "and the season is cold: heating a home that would otherwise be empty costs "
        "more than the commute does."
    )

st.markdown("### 🤝 Anchor Days")

consolidation = consolidation_benefit(
    days_in_office, distance, mode, occupants, season, grid_intensity, home_kwh
)

if consolidation["worth_doing"]:
    anchor_cols = st.columns(3)
    anchor_cols[0].metric("Scattered across the week", f"{consolidation['scattered_kg']:,.1f} kg")
    anchor_cols[1].metric(
        "Everyone on the same days", f"{consolidation['consolidated_kg']:,.1f} kg",
        f"-{consolidation['saving_kg']:,.1f} kg/week",
    )
    anchor_cols[2].metric("Annual saving", f"{consolidation['annual_saving_kg']:,.0f} kg")
    st.caption(
        f"With {consolidation['days_in_office']} attendance days consolidated, the "
        f"building can close for {consolidation['closed_days']} — and nobody has to "
        f"travel any differently. This is the largest lever in hybrid work and it is "
        f"almost never counted."
    )
else:
    st.caption(
        "No consolidation saving at this attendance level — the office is open every "
        "day you attend regardless."
    )

st.markdown("---")
st.markdown("### 🚌 Could You Travel Differently?")

mode_rows = compare_modes(distance, occupants, season, mode)
mode_frame = pd.DataFrame(
    [
        {
            "Mode": row["mode"] + ("" if row["feasible"] else "  (not realistic at this distance)"),
            "kg CO₂e per leg": row["total_kg"],
            "Minutes each way": row["minutes"],
            "Saving vs now": f"{row['saving_pct']}%",
        }
        for row in mode_rows
    ]
)
st.dataframe(mode_frame, use_container_width=True, hide_index=True)

if is_shareable(mode) and occupants == 1:
    shared = weekly_plan(
        days_in_office, distance, mode, 2, season, office_open_days, grid_intensity, home_kwh
    )
    st.caption(
        f"Sharing the car with one colleague would take this week from "
        f"{plan['total_kg']:,.1f} kg to {shared['total_kg']:,.1f} kg — the largest "
        f"change available without changing mode at all."
    )

st.markdown("---")
st.markdown("### 📅 Across the Whole Year")

annual = annual_summary(days_in_office, distance, mode, occupants, office_open_days, grid_intensity)
annual_frame = pd.DataFrame(
    [
        {
            "Season": row["season"],
            "Working weeks": row["weeks"],
            "kg per week": row["weekly_kg"],
            "kg in season": row["season_kg"],
        }
        for row in annual["seasons"]
    ]
)
st.dataframe(annual_frame, use_container_width=True, hide_index=True)
st.caption(
    f"**{annual['annual_kg']:,.0f} kg CO₂e** across {annual['weeks_counted']} working "
    f"weeks. Worst season: **{annual['worst_season']}** — the home-working penalty is "
    f"roughly three times larger in winter than in summer, so a single-season answer "
    f"would have been misleading."
)

st.markdown("---")
st.markdown("### 💾 Saved Patterns")

with st.form("save_commute_plan"):
    plan_name = st.text_input("Name this pattern", value=f"{days_in_office} days by {mode.lower()}")
    if st.form_submit_button("Save pattern", use_container_width=True):
        if save_commute_plan(user_id, plan_name, plan, distance):
            st.success(f"Saved **{plan_name}**.")
            st.rerun()
        else:
            st.error("Could not save that pattern. Please try again.")

saved = get_commute_plans(user_id)
if saved:
    saved_frame = pd.DataFrame(
        [
            {
                "Pattern": row["plan_name"],
                "Mode": row["mode"],
                "km each way": row["distance_km"],
                "Office days": row["days_in_office"],
                "Season": row["season"],
                "kg/week": round(row["weekly_kg"], 1),
                "kg/year": round(row["annual_kg"]),
            }
            for row in saved
        ]
    )
    st.dataframe(saved_frame, use_container_width=True, hide_index=True)

    remove_col, _ = st.columns([2, 3])
    to_remove = remove_col.selectbox(
        "Remove a pattern", saved, format_func=lambda row: row["plan_name"]
    )
    if remove_col.button("Delete pattern", use_container_width=True):
        if delete_commute_plan(to_remove["id"]):
            st.success("Pattern removed.")
            st.rerun()
        else:
            st.error("Could not remove that pattern.")
else:
    st.info("No patterns saved yet.")

st.markdown("### 💡 What Would Actually Help")
for tip in get_commute_advice(plan, comparison, consolidation):
    st.markdown(f"- {tip}")

with st.expander("📐 How these numbers are worked out"):
    st.markdown(
        f"""
**The week** — `commute travel + office building + home working`. All three
terms are real, and leaving out either building is what makes most hybrid-work
advice wrong.

**Commute** — distance × the mode's factor, doubled for the return leg and
multiplied by the days attended. Private vehicles are per *vehicle*, so sharing
divides them; buses and trains are already per passenger, so claiming a share
of one would be double counting.

**Cold start** — a fixed penalty per journey, charged while the engine runs
rich and the catalyst is still cold. Because it is per start rather than per
kilometre, it dominates a short drive. Under {2.0:.0f} km the engine never
reaches operating temperature at all, so the penalty is larger again.

**Office** — {OFFICE_FIXED_KWH_PER_DESK_DAY} kWh per desk for every day the
building is *open*, plus {OFFICE_MARGINAL_KWH_PER_PERSON_DAY} kWh for each day
you attend. The fixed share is the whole point: one person staying home does
not stop the building being heated and lit. Only closing it does.

**Home** — the extra household energy of a day worked at home rather than an
empty house: {HOME_WORKING_KWH['Winter']} kWh in winter down to
{HOME_WORKING_KWH['Summer']} kWh in summer.

**The year** — seasons weighted by working weeks, because the home-working
penalty is roughly three times larger in winter than in summer.
        """
    )
