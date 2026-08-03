import pandas as pd
import plotly.express as px
import streamlit as st

from delivery_footprint import (
    DEFAULT_LAST_MILE_KM,
    DEFAULT_PARCEL_SIZE,
    DEFAULT_SPEED,
    DEFAULT_VEHICLE,
    LAST_MILE_VEHICLES,
    PACKAGING_MATERIALS,
    PARCEL_SIZES,
    SHIPPING_SPEEDS,
    annual_footprint,
    click_and_collect,
    compare_scenarios,
    consolidation_saving,
    delete_delivery_profile,
    get_delivery_profiles,
    get_delivery_tips,
    optimise_orders,
    parcel_footprint,
    save_delivery_profile,
)
from styles.theme import apply_theme

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>📦 Delivery & Packaging Footprint</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "The transport section counts the trips **you** take. It does not count the "
    "trips taken *for* you. A household that stopped driving to the shops has "
    "not removed those journeys — it has outsourced them."
)
st.caption(
    "Delivery usually does beat driving. But that advantage collapses at "
    "same-day speed, with a high return rate, when nobody is home, and when you "
    "drive specially to collect. This page models all four."
)

st.markdown("---")
st.markdown("### 🛒 Your Ordering Habits")

orders_col, distance_col, size_col = st.columns(3)
orders_per_year = orders_col.number_input(
    "Online orders per year", min_value=0, max_value=2000, value=30, step=1
)
distance_km = distance_col.number_input(
    "Depot to your door (km)",
    min_value=0.0,
    max_value=500.0,
    value=DEFAULT_LAST_MILE_KM,
    step=1.0,
    help="The final leg only — roughly 10-15 km for most addresses.",
)
parcel_size = size_col.selectbox(
    "Typical parcel size",
    list(PARCEL_SIZES.keys()),
    index=list(PARCEL_SIZES.keys()).index(DEFAULT_PARCEL_SIZE),
)

speed_col, vehicle_col, attempts_col = st.columns(3)
speed = speed_col.selectbox(
    "Usual shipping speed",
    list(SHIPPING_SPEEDS.keys()),
    index=list(SHIPPING_SPEEDS.keys()).index(DEFAULT_SPEED),
)
vehicle = vehicle_col.selectbox(
    "Delivery vehicle",
    list(LAST_MILE_VEHICLES.keys()),
    index=list(LAST_MILE_VEHICLES.keys()).index(DEFAULT_VEHICLE),
)
attempts = attempts_col.number_input(
    "Delivery attempts per parcel",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="More than one means the last mile happened more than once.",
)

st.caption(
    f"**{speed}** — {SHIPPING_SPEEDS[speed]['note']}  \n"
    f"**{vehicle}** — {LAST_MILE_VEHICLES[vehicle]['note']}  \n"
    f"**{parcel_size}** — {PARCEL_SIZES[parcel_size]['note']}"
)

materials = st.multiselect(
    "Packaging materials you usually receive",
    list(PACKAGING_MATERIALS.keys()),
    default=["Corrugated cardboard"],
)

return_col, resale_col, embodied_col = st.columns(3)
return_rate = return_col.slider(
    "Share of orders you return", min_value=0.0, max_value=1.0, value=0.1, step=0.05
)
resale_probability = resale_col.slider(
    "Share of returns that get resold",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Not everything sent back is resold. The rest is written off entirely.",
)
item_embodied = embodied_col.number_input(
    "Typical item's own footprint (kg CO₂e)",
    min_value=0.0,
    max_value=1000.0,
    value=8.0,
    step=1.0,
)

profile = {
    "orders_per_year": orders_per_year,
    "distance_km": distance_km,
    "vehicle": vehicle,
    "speed": speed,
    "parcel_size": parcel_size,
    "materials": materials,
    "attempts": attempts,
    "return_rate": return_rate,
    "resale_probability": resale_probability,
    "item_embodied_co2": item_embodied,
}

result = annual_footprint(profile)

if result["total_co2_kg"] <= 0:
    st.info("Set at least one order a year to see the breakdown.")
    st.stop()

st.markdown("---")
st.markdown("### 📊 What It Adds Up To")

metric_columns = st.columns(4)
metric_columns[0].metric("Annual total", f"{result['total_co2_kg']:,.1f} kg CO₂e")
metric_columns[1].metric("Per parcel", f"{result['per_parcel_co2_kg']:.2f} kg")
metric_columns[2].metric("Returns' share", f"{result['returns_share_pct']:.0f}%")
metric_columns[3].metric("Flown freight", f"{result['air_co2_kg']:,.1f} kg")

if result["air_co2_kg"] > 0:
    st.warning(
        "Part of your freight is travelling by air. Expedited shipping is the "
        "only reason that happens — standard delivery removes it entirely."
    )

breakdown_frame = pd.DataFrame(
    [
        {"Source": name, "kg CO₂e": value}
        for name, value in result["breakdown"].items()
        if value > 0
    ]
)
breakdown_figure = px.bar(
    breakdown_frame.sort_values("kg CO₂e"),
    x="kg CO₂e",
    y="Source",
    orientation="h",
    color="kg CO₂e",
    color_continuous_scale="Greens",
)
breakdown_figure.update_layout(
    height=340, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False
)
st.plotly_chart(breakdown_figure, use_container_width=True)

st.markdown("---")
st.markdown("### 🧮 What Batching Would Save")

st.markdown(
    "Same goods, fewer vans. This is the biggest free lever on the page — "
    "nothing about what you buy has to change."
)

batch_col, _ = st.columns([2, 1])
batched_orders = batch_col.slider(
    "If those items arrived in this many shipments instead",
    min_value=1,
    max_value=max(2, orders_per_year),
    value=max(1, orders_per_year // 2),
    step=1,
)

per_parcel = parcel_footprint(
    distance_km, vehicle, speed, parcel_size, materials, attempts
)["co2_kg"]
consolidated = consolidation_saving(orders_per_year, orders_per_year, per_parcel)
batched_profile = dict(profile)
batched_profile["orders_per_year"] = batched_orders
comparison = compare_scenarios(profile, batched_profile)

batch_columns = st.columns(3)
batch_columns[0].metric("Now", f"{comparison['before_co2_kg']:,.1f} kg CO₂e")
batch_columns[1].metric(
    f"At {batched_orders} shipments", f"{comparison['after_co2_kg']:,.1f} kg CO₂e"
)
batch_columns[2].metric(
    "Saved", f"{comparison['difference_kg']:,.1f} kg", f"{comparison['change_pct']:.0f}%"
)

st.caption(
    f"Collapsing all {orders_per_year} orders into a single shipment would be the "
    f"theoretical maximum: {consolidated['saving_kg']:,.1f} kg CO₂e, or "
    f"{consolidated['saving_pct']:.0f}% of the transport term."
)

st.markdown("---")
st.markdown("### 🚗 Would Collecting It Yourself Be Better?")

collect_col, trip_col = st.columns(2)
collect_distance = collect_col.number_input(
    "Distance to the pickup point (km, one way)",
    min_value=0.0,
    max_value=200.0,
    value=4.0,
    step=0.5,
)
dedicated = trip_col.checkbox(
    "I would drive there specially", value=True,
    help="If you were passing anyway, the trip costs nothing extra.",
)
collect_vehicle = st.selectbox(
    "How you would get there",
    list(LAST_MILE_VEHICLES.keys()),
    index=list(LAST_MILE_VEHICLES.keys()).index("Private car (collection)"),
)

collection = click_and_collect(collect_distance, collect_vehicle, dedicated)

if collection["better_than_delivery"]:
    st.success(
        f"Collecting wins: **{collection['collection_co2_kg']:.3f} kg CO₂e** against "
        f"**{collection['home_delivery_co2_kg']:.3f} kg** for home delivery."
    )
else:
    st.error(
        f"Collecting is **worse** here: {collection['collection_co2_kg']:.3f} kg CO₂e "
        f"against {collection['home_delivery_co2_kg']:.3f} kg for home delivery, a "
        f"difference of {collection['difference_kg']:.3f} kg. A whole car for one "
        "parcel rarely beats a van that was already on your street."
    )

st.markdown("---")
st.markdown("### ✅ What Would Actually Help")

options = optimise_orders(profile)
if options:
    options_frame = pd.DataFrame(
        [
            {
                "Change": item["action"],
                "Saves": f"{item['saving_kg']:,.1f} kg CO₂e/yr",
                "Why": item["note"],
            }
            for item in options
        ]
    )
    st.dataframe(options_frame, use_container_width=True, hide_index=True)
else:
    st.success("No further reductions found — your ordering habits are already tight.")

st.markdown("---")
st.markdown("### 💡 Tips")
for tip in get_delivery_tips(result):
    st.markdown(f"- {tip}")

st.markdown("---")
st.markdown("### 💾 Save This Profile")

name_col, save_col = st.columns([3, 1])
profile_name = name_col.text_input("Profile name", value="My ordering")
if save_col.button("Save profile", use_container_width=True):
    if save_delivery_profile(user_id, profile_name, profile):
        st.success("Profile saved.")
    else:
        st.error("Could not save that profile.")

saved_profiles = get_delivery_profiles(user_id)
if saved_profiles:
    st.markdown("#### Saved profiles")
    for record in saved_profiles:
        detail_col, delete_col = st.columns([5, 1])
        detail_col.markdown(
            f"**{record['profile_name']}** — {record['orders_per_year']} orders/yr · "
            f"{record['speed']} · {record['total_co2_kg']:,.1f} kg CO₂e a year · "
            f"returns {record['returns_share_pct']:.0f}%"
        )
        if delete_col.button("Delete", key=f"delete_profile_{record['id']}"):
            delete_delivery_profile(record["id"])
            st.rerun()
else:
    st.caption("No saved profiles yet.")

st.caption(
    "Last-mile intensities, speed multipliers and packaging factors follow "
    "published freight and e-commerce logistics studies. Vehicle figures are "
    "per parcel, so the van's efficiency is already amortised over its drops. "
    "Assumptions are documented inline in `delivery_footprint.py`."
)
