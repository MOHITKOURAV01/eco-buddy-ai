import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from food_waste import (
    DEFAULT_DISPOSAL,
    DEFAULT_STORAGE,
    DISPOSAL_ROUTES,
    FOOD_ITEMS,
    MEAL_KG,
    STORAGE_LOCATIONS,
    at_risk_items,
    avoidable_split,
    best_storage,
    compare_disposal_routes,
    delete_waste_entry,
    get_food_item,
    get_waste_log,
    get_waste_tips,
    list_categories,
    list_food_items,
    log_waste,
    over_purchase_diagnosis,
    shelf_life_days,
    spoilage_risk,
    summarise_log,
    undercount_vs_disposal_only,
    waste_footprint,
)
from styles.theme import apply_theme

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>🥬 Food Waste Footprint</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "The Waste Footprint page counts what happens to food *after* you bin it. "
    "This one counts what it took to grow, water, ship and chill the food nobody "
    "ate — which is almost always the larger number."
)

st.markdown("---")
st.markdown("### 🗑️ Log Something You Threw Away")

category_col, item_col, kg_col, route_col = st.columns(4)
category = category_col.selectbox("Category", ["All"] + list_categories())
items = [
    item["name"]
    for item in list_food_items(None if category == "All" else category)
]
item_name = item_col.selectbox("What was it?", items)
kg = kg_col.number_input(
    "How much? (kg)", min_value=0.0, max_value=50.0, value=0.4, step=0.05,
    help=f"A typical cooked meal is around {MEAL_KG} kg.",
)
disposal = route_col.selectbox(
    "Where did it go?", list(DISPOSAL_ROUTES.keys()),
    index=list(DISPOSAL_ROUTES.keys()).index(DEFAULT_DISPOSAL),
)

reason = st.text_input(
    "Why did it get thrown away? (optional)",
    placeholder="Cooked too much / forgot about it / past the date",
)

footprint = waste_footprint(item_name, kg, disposal)
item = get_food_item(item_name)

if not footprint or footprint["total_kg"] <= 0:
    st.info("Enter a quantity above to see what it cost.")
    st.stop()

st.markdown("### 📊 What That Cost")

cost_cols = st.columns(4)
cost_cols[0].metric("Total CO₂e", f"{footprint['co2_kg']:,.2f} kg")
cost_cols[1].metric("Water", f"{footprint['water_litres']:,.0f} L")
cost_cols[2].metric("Money", f"{footprint['money']:,.2f}")
cost_cols[3].metric("Meals' worth", f"{footprint['meals_equivalent']:,.1f}")

split_col, disposal_col = st.columns(2)

with split_col:
    split_frame = pd.DataFrame(
        [
            {"Source": "Growing, shipping, chilling", "kg CO₂e": footprint["production_co2_kg"]},
            {"Source": f"Disposal ({disposal})", "kg CO₂e": footprint["disposal_co2_kg"]},
        ]
    )
    fig = px.bar(
        split_frame, x="kg CO₂e", y="Source", orientation="h",
        title="Where the emissions actually are",
        color="Source",
        color_discrete_map={
            "Growing, shipping, chilling": "#a02c2c",
            f"Disposal ({disposal})": "#5f8f36",
        },
    )
    fig.update_layout(showlegend=False, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**{footprint['production_share_pct']}%** of this footprint was spent before "
        f"the food ever reached your bin. No disposal route can recover it."
    )

with disposal_col:
    route_rows = compare_disposal_routes(item_name, kg)
    route_frame = pd.DataFrame(
        [
            {
                "Route": row["route"],
                "Disposal kg CO₂e": row["disposal_co2_kg"],
                "Total kg CO₂e": row["total_co2_kg"],
            }
            for row in route_rows
        ]
    )
    st.markdown("**If it had gone somewhere else**")
    st.dataframe(route_frame, use_container_width=True, hide_index=True)
    best_route, worst_route = route_rows[0], route_rows[-1]
    st.caption(
        f"Best to worst route changes this by "
        f"{worst_route['total_co2_kg'] - best_route['total_co2_kg']:,.2f} kg. "
        f"Not buying it would have changed it by {footprint['co2_kg']:,.2f} kg."
    )

split = avoidable_split(item_name, kg)
if split["unavoidable_kg"] > 0:
    st.caption(
        f"Of the {split['total_kg']:.2f} kg, about **{split['avoidable_kg']:.2f} kg was "
        f"edible** and {split['unavoidable_kg']:.2f} kg was peel, bone or shell that was "
        f"never going to be eaten. Only the edible part is charged to production here."
    )

with st.form("log_food_waste"):
    if st.form_submit_button("Add to my waste log", use_container_width=True):
        if log_waste(user_id, footprint, reason):
            st.success(f"Logged {kg:,.2f} kg of {item_name.lower()}.")
            st.rerun()
        else:
            st.error("Could not log that. Please try again.")

st.markdown("---")
st.markdown("### 🧊 What's About To Go")

st.caption(
    "List what is currently in the house and how long you have had it. The ranking "
    "is by carbon at stake, not by weight — half a kilo of beef outranks two kilos "
    "of potatoes."
)

inventory_frame = st.data_editor(
    pd.DataFrame(
        [
            {"Item": "Chicken", "kg": 0.5, "Days held": 3, "Stored in": "Fridge"},
            {"Item": "Leafy salad", "kg": 0.2, "Days held": 3, "Stored in": "Fridge"},
            {"Item": "Bread", "kg": 0.8, "Days held": 3, "Stored in": "Counter / pantry"},
        ]
    ),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Item": st.column_config.SelectboxColumn(options=list(FOOD_ITEMS.keys())),
        "Stored in": st.column_config.SelectboxColumn(options=list(STORAGE_LOCATIONS.keys())),
    },
    key="food_inventory",
)

inventory = [
    {
        "item": row["Item"],
        "kg": row["kg"],
        "days_held": row["Days held"],
        "storage": row["Stored in"],
    }
    for _, row in inventory_frame.iterrows()
    if row.get("Item")
]

risks = at_risk_items(inventory)
if risks:
    risk_frame = pd.DataFrame(
        [
            {
                "Item": entry["item"],
                "kg": entry["kg"],
                "Days held": entry["days_held"],
                "Lasts (days)": entry["shelf_life_days"],
                "Spoilage risk": f"{entry['risk'] * 100:.0f}%",
                "kg CO₂e at risk": entry["co2_at_risk_kg"],
                "Value at risk": entry["value_at_risk"],
            }
            for entry in risks
        ]
    )
    st.dataframe(risk_frame, use_container_width=True, hide_index=True)

    urgent = [entry for entry in risks if entry["risk"] >= 0.5]
    if urgent:
        st.warning(
            "Use or freeze today: "
            + ", ".join(f"**{entry['item']}**" for entry in urgent[:4])
        )

    storage_advice = [best_storage(entry["item"]) for entry in risks[:4]]
    for advice in storage_advice:
        if advice and advice["gain_days"] > 0 and advice["freezable"]:
            st.caption(
                f"**{advice['item']}** keeps longest in the {advice['best'].lower()} — "
                f"{advice['days']:,.0f} days, {advice['gain_days']:,.0f} more than on the counter."
            )
else:
    st.info("Add items above to see what is most at risk.")

st.markdown("---")
st.markdown("### 🛒 How Much of the Shopping Gets Eaten?")

bought_col, wasted_col = st.columns(2)
bought = bought_col.number_input(
    "Food bought this week (kg)", min_value=0.0, max_value=200.0, value=18.0, step=0.5
)
wasted = wasted_col.number_input(
    "Of that, thrown away (kg)", min_value=0.0, max_value=200.0, value=2.5, step=0.1
)

diagnosis = over_purchase_diagnosis(bought, wasted)
if diagnosis["waste_share_pct"] >= 25:
    st.error(diagnosis["verdict"])
elif diagnosis["waste_share_pct"] >= 12:
    st.warning(diagnosis["verdict"])
else:
    st.success(diagnosis["verdict"])

st.markdown("---")
st.markdown("### 📒 Your Waste Log")

weeks_logged = st.number_input(
    "Weeks this log covers", min_value=1, max_value=104, value=1, step=1,
    help="Used to scale the annual figures.",
)

entries = get_waste_log(user_id)
if entries:
    summary = summarise_log(entries, weeks_logged)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Food wasted", f"{summary['total_kg']:,.1f} kg")
    summary_cols[1].metric("CO₂e", f"{summary['co2_kg']:,.1f} kg")
    summary_cols[2].metric("Money", f"{summary['money']:,.0f}")
    summary_cols[3].metric("Per year", f"{summary['annual_co2_kg']:,.0f} kg CO₂e")

    undercount = undercount_vs_disposal_only(summary)
    if undercount and undercount["multiple"]:
        st.info(
            f"Counted the way the Waste Footprint page counts it — disposal only — this "
            f"would read **{undercount['disposal_only_kg']:,.1f} kg CO₂e**. Counting the "
            f"food's production as well makes it **{undercount['full_kg']:,.1f} kg**, "
            f"about **{undercount['multiple']}×** larger."
        )

    log_frame = pd.DataFrame(
        [
            {
                "Item": entry["item"],
                "kg": round(entry["total_kg"], 2),
                "Route": entry["disposal"],
                "kg CO₂e": round(entry["co2_kg"], 2),
                "Money": round(entry["money"], 2),
                "Reason": entry["reason"] or "—",
            }
            for entry in entries
        ]
    )
    st.dataframe(log_frame, use_container_width=True, hide_index=True)

    if summary["by_category"]:
        category_frame = pd.DataFrame(
            [
                {"Category": name, "kg CO₂e": values["co2_kg"], "kg": values["kg"]}
                for name, values in summary["by_category"].items()
            ]
        ).sort_values("kg CO₂e", ascending=False)
        category_fig = px.bar(
            category_frame, x="Category", y="kg CO₂e",
            title="Which food does the damage",
            color="kg CO₂e", color_continuous_scale="Reds",
        )
        category_fig.update_layout(coloraxis_showscale=False, xaxis_title="")
        st.plotly_chart(category_fig, use_container_width=True)
        st.caption(
            "Weight and carbon rank differently. A little meat usually outweighs a "
            "lot of vegetables."
        )

    remove_col, _ = st.columns([2, 3])
    to_remove = remove_col.selectbox(
        "Remove an entry", entries,
        format_func=lambda entry: f"{entry['item']} — {entry['total_kg']:.2f} kg",
    )
    if remove_col.button("Delete entry", use_container_width=True):
        if delete_waste_entry(to_remove["id"]):
            st.success("Entry removed.")
            st.rerun()
        else:
            st.error("Could not remove that entry.")

    st.markdown("### 💡 What Would Actually Help")
    for tip in get_waste_tips(summary, risks):
        st.markdown(f"- {tip}")
else:
    st.info("Nothing logged yet. Add something above to start building the picture.")

with st.expander("📐 How these numbers are worked out"):
    st.markdown(
        f"""
**The footprint of wasted food**

```
waste footprint = production footprint of what was thrown out
                + emissions of the disposal route it went to
```

The second term is what the app already counted. The first is usually ten to
sixty times larger, because the food was grown, watered, fertilised, processed,
refrigerated, packaged and driven before anyone decided not to eat it.

**Avoidable vs unavoidable** — banana skins, bones and eggshells were never
going to be eaten; they are the cost of the meal, not a failure. Only the
edible share is charged to production. Disposal emissions apply to everything
that goes in the bin, peel included.

**Disposal routes** — landfill is worst because buried food rots without
oxygen and produces methane. Kerbside collection usually means anaerobic
digestion, which captures that methane as fuel. The table above deliberately
shows how *little* the route changes the total: composting is worth doing,
and it is not a substitute for buying less.

**Shelf life and spoilage** — each item has a baseline counter life, multiplied
by where you keep it (fridge ×{STORAGE_LOCATIONS['Fridge']}, freezer
×{STORAGE_LOCATIONS['Freezer']}). Risk stays low for most of an item's life and
then climbs steeply once it passes it, which is how food actually behaves.
Freezing is not offered for items it would ruin.

**Sources** — production footprints from Poore & Nemecek (2018); water
footprints from Mekonnen & Hoekstra (2011); disposal factors from standard
waste-treatment inventories. Every figure is documented inline in the module.
        """
    )
