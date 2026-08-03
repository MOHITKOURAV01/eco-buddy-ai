import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from flight_footprint import (
    AIRPORTS,
    CABIN_CLASSES,
    DEFAULT_CABIN,
    DEFAULT_LOAD_FACTOR,
    PERSONAL_ANNUAL_BUDGET_KG,
    RADIATIVE_FORCING_MULTIPLIER,
    RF_RANGE,
    annual_summary,
    budget_share,
    compare_cabins,
    compare_routings,
    compare_to_alternatives,
    delete_trip,
    estimate_route,
    estimate_trip,
    get_airport,
    get_reduction_tips,
    get_trips,
    save_trip,
    trips_within_budget,
)
from styles.theme import apply_theme

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>✈️ Flight Footprint</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "The main assessment counts flights. This page weighs them — by distance, "
    "by cabin, by connection, and including the non-CO₂ warming that happens "
    "at cruise altitude."
)

AIRPORT_LABELS = {
    code: f"{code} — {details['city']}, {details['country']}"
    for code, details in sorted(AIRPORTS.items())
}
CODES = list(AIRPORT_LABELS.keys())


def _label(code):
    return AIRPORT_LABELS.get(code, code)


st.markdown("---")
st.markdown("### 🗺️ The Journey")

mode = st.radio(
    "How would you like to describe the flight?",
    ["Pick airports", "Enter a distance"],
    horizontal=True,
)

estimate = None
distance_for_comparison = 0.0

if mode == "Pick airports":
    origin_col, destination_col, via_col = st.columns(3)
    origin = origin_col.selectbox(
        "From", CODES, index=CODES.index("LHR") if "LHR" in CODES else 0,
        format_func=_label,
    )
    destination = destination_col.selectbox(
        "To", CODES, index=CODES.index("JFK") if "JFK" in CODES else 1,
        format_func=_label,
    )
    via = via_col.selectbox(
        "Connecting via (optional)",
        ["Direct"] + [code for code in CODES if code not in (origin, destination)],
        format_func=lambda code: "Direct flight" if code == "Direct" else _label(code),
    )
else:
    origin = destination = via = None
    manual_distance = st.number_input(
        "One-way distance (km)", min_value=0.0, max_value=20000.0, value=1500.0, step=50.0,
        help="The flown distance for a single direction.",
    )

st.markdown("### 💺 The Booking")

cabin_col, trip_col, people_col, load_col = st.columns(4)
cabin = cabin_col.selectbox("Cabin", list(CABIN_CLASSES.keys()))
round_trip = trip_col.selectbox("Trip type", ["Return", "One way"]) == "Return"
passengers = people_col.number_input(
    "Passengers", min_value=1, max_value=12, value=1, step=1,
    help="Everyone travelling on the same booking.",
)
load_factor = load_col.slider(
    "Aircraft load factor", min_value=0.30, max_value=1.00, value=DEFAULT_LOAD_FACTOR,
    step=0.01, help="How full the aircraft is. An emptier cabin means more fuel per person.",
)

st.caption(f"**{cabin}** — {CABIN_CLASSES[cabin]['note']}")

include_rf = st.checkbox(
    f"Include non-CO₂ warming (×{RADIATIVE_FORCING_MULTIPLIER} on the cruise portion)",
    value=True,
    help=(
        "Contrails, soot and nitrogen oxides at altitude roughly double aviation's "
        f"warming effect. Published estimates range from ×{RF_RANGE[0]} to ×{RF_RANGE[1]}."
    ),
)

if mode == "Pick airports":
    estimate = estimate_route(
        origin,
        destination,
        via=None if via == "Direct" else [via],
        cabin=cabin,
        round_trip=round_trip,
        passengers=passengers,
        include_radiative_forcing=include_rf,
        load_factor=load_factor,
    )
    if estimate:
        distance_for_comparison = estimate["direct_distance_km"] or 0.0
else:
    estimate = estimate_trip(
        [manual_distance],
        cabin=cabin,
        round_trip=round_trip,
        passengers=passengers,
        include_radiative_forcing=include_rf,
        load_factor=load_factor,
    )
    distance_for_comparison = manual_distance

if not estimate or estimate["co2e_kg"] <= 0:
    st.info("Choose two different airports, or enter a distance, to see the footprint.")
    st.stop()

st.markdown("---")
st.markdown("### 📊 The Footprint")

metric_cols = st.columns(4)
metric_cols[0].metric("Total CO₂e", f"{estimate['co2e_kg']:,.0f} kg")
metric_cols[1].metric("CO₂ only", f"{estimate['co2_kg']:,.0f} kg")
metric_cols[2].metric("Non-CO₂ warming", f"{estimate['non_co2_kg']:,.0f} kg")
metric_cols[3].metric("Distance flown", f"{estimate['distance_km']:,.0f} km")

share = budget_share(estimate["co2e_kg"])
allowance_trips = trips_within_budget(estimate["co2e_kg"])
st.progress(min(1.0, share / 100))
if allowance_trips:
    st.caption(
        f"This one journey is **{share}%** of a 1.5 °C-consistent personal allowance "
        f"of {PERSONAL_ANNUAL_BUDGET_KG:,.0f} kg CO₂e a year — about "
        f"**{allowance_trips:g}** such trips would use the entire year's budget for "
        f"everything you do."
    )

breakdown = pd.DataFrame(
    [
        {"Component": "Take-off & landing", "kg CO₂e": sum(leg["lto_kg"] for leg in estimate["legs"]) * estimate["passengers"]},
        {"Component": "Cruise CO₂", "kg CO₂e": sum(leg["cruise_kg"] for leg in estimate["legs"]) * estimate["passengers"]},
        {"Component": "Non-CO₂ (contrails, NOx)", "kg CO₂e": estimate["non_co2_kg"]},
    ]
)
breakdown = breakdown[breakdown["kg CO₂e"] > 0]

chart_col, legs_col = st.columns([1, 1])
with chart_col:
    fig = px.pie(
        breakdown, names="Component", values="kg CO₂e", hole=0.45,
        title="Where the warming comes from",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with legs_col:
    legs_table = pd.DataFrame(
        [
            {
                "Leg": index + 1,
                "Distance (km)": leg["distance_km"],
                "Haul": leg["haul"].title(),
                "kg CO₂e": round(leg["co2_kg"] + leg["non_co2_kg"], 1),
                "kg per km": leg["kg_per_km"],
            }
            for index, leg in enumerate(estimate["legs"])
        ]
    )
    st.markdown("**Leg by leg**")
    st.dataframe(legs_table, use_container_width=True, hide_index=True)
    st.caption(
        "Every leg carries a fixed take-off charge, which is why short hops and "
        "connections cost so much per kilometre."
    )

st.markdown("---")
st.markdown("### 💺 What the Cabin Costs")

cabin_options = compare_cabins(
    distance_for_comparison or estimate["distance_km"] / (2 if round_trip else 1),
    round_trip=round_trip,
)
cabin_frame = pd.DataFrame(cabin_options)
cabin_fig = px.bar(
    cabin_frame, x="cabin", y="co2e_kg",
    labels={"cabin": "", "co2e_kg": "kg CO₂e"},
    title="Same aircraft, same route, different seat",
)
st.plotly_chart(cabin_fig, use_container_width=True)
st.caption(
    "Premium seats are charged for the floor space they occupy — the aircraft's "
    "emissions divided over fewer passengers."
)

if mode == "Pick airports" and via != "Direct":
    comparison = compare_routings(origin, destination, via, cabin=cabin, round_trip=round_trip)
    if comparison:
        st.markdown("### 🔀 The Cost of Connecting")
        connect_cols = st.columns(3)
        connect_cols[0].metric("Extra distance", f"{comparison['extra_km']:,.0f} km")
        connect_cols[1].metric(
            "Extra CO₂e", f"{comparison['extra_kg']:,.0f} kg", f"+{comparison['extra_pct']}%"
        )
        connect_cols[2].metric("Extra take-offs", comparison["extra_takeoffs"])
        st.caption(
            f"Flying {_label(origin)} → {_label(destination)} via {_label(via)} adds "
            f"both kilometres and a second take-off cycle in each direction."
        )

st.markdown("---")
st.markdown("### 🚄 Could You Get There Another Way?")

one_way_km = distance_for_comparison or (
    estimate["distance_km"] / (2 if round_trip else 1) / max(1, estimate["passengers"])
)
alternatives = compare_to_alternatives(one_way_km, estimate["co2e_kg"] / (2 if round_trip else 1))
alt_frame = pd.DataFrame(
    [
        {
            "Mode": row["mode"] + ("" if row["plausible_at_this_distance"] else "  (not realistic at this distance)"),
            "kg CO₂e": row["co2e_kg"],
            "Saving vs flying": f"{row['saving_pct']}%",
        }
        for row in alternatives
    ]
)
st.dataframe(alt_frame, use_container_width=True, hide_index=True)
st.caption(f"Compared over a single {one_way_km:,.0f} km leg.")

st.markdown("---")
st.markdown("### 💾 Your Flying Year")

with st.form("save_flight_trip"):
    default_name = (
        f"{origin} → {destination}" if mode == "Pick airports" else f"{manual_distance:,.0f} km trip"
    )
    trip_label = st.text_input("Name this trip", value=default_name)
    if st.form_submit_button("Add to my flying year", use_container_width=True):
        if save_trip(user_id, trip_label, estimate):
            st.success(f"Saved **{trip_label}** — {estimate['co2e_kg']:,.0f} kg CO₂e.")
            st.rerun()
        else:
            st.error("Could not save that trip. Please try again.")

saved = get_trips(user_id)
if saved:
    summary = annual_summary(saved)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Trips logged", summary["trip_count"])
    summary_cols[1].metric("Total CO₂e", f"{summary['co2e_kg']:,.0f} kg")
    summary_cols[2].metric("Distance flown", f"{summary['distance_km']:,.0f} km")
    summary_cols[3].metric("Of annual allowance", f"{summary['budget_share_pct']}%")

    if summary["over_budget"]:
        st.error(
            "Your flying alone exceeds a 1.5 °C-consistent annual allowance for "
            "everything — housing, food, and all other transport included."
        )

    trips_frame = pd.DataFrame(
        [
            {
                "Trip": trip["label"],
                "Route": trip["route"],
                "Cabin": trip["cabin"],
                "Return": "Yes" if trip["round_trip"] else "No",
                "km": round(trip["distance_km"]),
                "kg CO₂e": round(trip["co2e_kg"]),
            }
            for trip in saved
        ]
    )
    st.dataframe(trips_frame, use_container_width=True, hide_index=True)

    waterfall = go.Figure(
        go.Bar(
            x=[trip["label"] for trip in saved],
            y=[trip["co2e_kg"] for trip in saved],
            marker_color="#2f5e32",
        )
    )
    waterfall.add_hline(
        y=PERSONAL_ANNUAL_BUDGET_KG,
        line_dash="dash",
        annotation_text="Full personal annual allowance",
    )
    waterfall.update_layout(
        title="Every trip against the annual allowance",
        yaxis_title="kg CO₂e",
        xaxis_title="",
    )
    st.plotly_chart(waterfall, use_container_width=True)

    remove_col, _ = st.columns([2, 3])
    to_remove = remove_col.selectbox(
        "Remove a trip",
        saved,
        format_func=lambda trip: f"{trip['label']} ({trip['co2e_kg']:,.0f} kg)",
    )
    if remove_col.button("Delete trip", use_container_width=True):
        if delete_trip(to_remove["id"]):
            st.success("Trip removed.")
            st.rerun()
        else:
            st.error("Could not remove that trip.")

    st.markdown("### 💡 What Would Actually Help")
    for tip in get_reduction_tips(summary):
        st.markdown(f"- {tip}")
else:
    st.info("No trips saved yet. Add one above to build a picture of your flying year.")

with st.expander("📐 How these numbers are worked out"):
    st.markdown(
        f"""
**Distance** — great-circle distance between airport coordinates, plus a
95 km allowance for real-world routing and holding.

**Fuel burn** — a fixed take-off and landing charge per leg, plus a cruise rate
per kilometre that depends on the haul length. Short flights are far worse per
kilometre because the take-off is amortised over fewer kilometres.

**Cabin** — a seat-space multiplier: a lie-flat business seat occupies roughly
three economy seats, so it carries roughly three economy seats' emissions. On
short-haul aircraft the multiplier is capped, because premium cabins there are
barely different from economy.

**Non-CO₂ warming** — contrail cirrus, soot and nitrogen oxides at cruise
altitude, applied as ×{RADIATIVE_FORCING_MULTIPLIER} on the *cruise* portion
only. Published estimates span ×{RF_RANGE[0]} to ×{RF_RANGE[1]}, and the
uncertainty is genuine — which is why the CO₂-only figure is shown alongside it.

**Load factor** — the same fuel divided across however many passengers are
actually on board.
        """
    )
