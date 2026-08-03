import pandas as pd
import plotly.express as px
import streamlit as st

from wardrobe import (
    CONDITIONS,
    DEAD_STOCK_WEAR_THRESHOLD,
    DEFAULT_WASH_TEMPERATURE,
    DEFAULT_WEARS_PER_WASH,
    FIBRES,
    GARMENT_TYPES,
    WASH_TEMPERATURES,
    carbon_per_wear,
    compare_purchase,
    cost_per_wear,
    delete_garment,
    extend_life_saving,
    find_dead_stock,
    get_expected_wears,
    get_garments,
    get_wardrobe_tips,
    lifetime_footprint,
    list_fibres,
    log_wear,
    save_garment,
    wardrobe_summary,
)
from styles.theme import apply_theme

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>👕 Wardrobe & Textile Footprint</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "The Waste Footprint page counts the clothes you throw away. This one counts "
    "the ones still hanging up — because almost all of a garment's impact was "
    "spent before you ever wore it, and the only way to earn it back is to wear it."
)

st.markdown("---")
st.markdown("### ➕ Add a Garment")

with st.form("add_garment"):
    name_col, category_col = st.columns(2)
    garment_name = name_col.text_input("Name it", value="", placeholder="Blue linen shirt")
    category = category_col.selectbox("Type", list(GARMENT_TYPES.keys()))

    fibre_col, condition_col, price_col = st.columns(3)
    fibre = fibre_col.selectbox("Main fibre", list(FIBRES.keys()))
    condition = condition_col.selectbox("How you got it", list(CONDITIONS.keys()))
    price = price_col.number_input("Price paid", min_value=0.0, max_value=100000.0, value=25.0, step=5.0)

    wears_col, temp_col, per_wash_col = st.columns(3)
    wears = wears_col.number_input(
        "Times worn so far", min_value=0, max_value=10000, value=0, step=1
    )
    wash_temp = temp_col.selectbox(
        "Wash temperature",
        list(WASH_TEMPERATURES.keys()),
        index=list(WASH_TEMPERATURES.keys()).index(DEFAULT_WASH_TEMPERATURE),
    )
    wears_per_wash = per_wash_col.number_input(
        "Wears between washes",
        min_value=1,
        max_value=50,
        value=DEFAULT_WEARS_PER_WASH,
        step=1,
    )

    dry_col, iron_col = st.columns(2)
    tumble_dried = dry_col.checkbox("Tumble dried")
    ironed = iron_col.checkbox("Ironed")

    st.caption(
        f"**{fibre}** — {FIBRES[fibre]['note']}  \n"
        f"**{condition}** — {CONDITIONS[condition]['note']}  \n"
        f"A {category.lower()} typically gives about "
        f"{get_expected_wears(category)} wears before it is worn out."
    )

    if st.form_submit_button("Add to wardrobe", use_container_width=True):
        new_garment = {
            "name": garment_name or category,
            "category": category,
            "fibre": fibre,
            "condition": condition,
            "price": price,
            "wears": wears,
            "wash_temp": wash_temp,
            "wears_per_wash": wears_per_wash,
            "tumble_dried": tumble_dried,
            "ironed": ironed,
        }
        if save_garment(user_id, new_garment):
            st.success(f"Added {new_garment['name']}.")
            st.rerun()
        else:
            st.error("Could not save that garment.")

garments = get_garments(user_id)

if not garments:
    st.info("Add a few garments above to see where your wardrobe's carbon sits.")
    st.stop()

summary = wardrobe_summary(garments)

st.markdown("---")
st.markdown("### 📊 Your Wardrobe")

metric_columns = st.columns(4)
metric_columns[0].metric("Items", summary["item_count"])
metric_columns[1].metric("Total footprint", f"{summary['total_co2_kg']:.1f} kg CO₂e")
metric_columns[2].metric("Utilisation", f"{summary['utilisation_score']:.0f}/100")
metric_columns[3].metric(
    "Carbon per wear",
    f"{summary['carbon_per_wear_kg']:.3f} kg" if summary["carbon_per_wear_kg"] else "—",
)

if summary["utilisation_score"] < 30:
    st.warning(
        "A low utilisation score means the carbon is already spent and you have "
        "not got much back for it yet. Wearing what is already hanging there is "
        "the cheapest reduction available to you."
    )

water_col, spend_col = st.columns(2)
water_col.metric("Water embodied", f"{summary['total_water_l']:,.0f} L")
spend_col.metric("Total spend", f"{summary['total_spend']:,.2f}")

chart_col, split_col = st.columns([3, 2])

fibre_frame = pd.DataFrame(
    [{"Fibre": name, "kg CO₂e": value} for name, value in summary["fibre_split"].items()]
)
if not fibre_frame.empty:
    fibre_figure = px.bar(
        fibre_frame.sort_values("kg CO₂e", ascending=True),
        x="kg CO₂e",
        y="Fibre",
        orientation="h",
        color="kg CO₂e",
        color_continuous_scale="Greens",
    )
    fibre_figure.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False
    )
    chart_col.plotly_chart(fibre_figure, use_container_width=True)

condition_frame = pd.DataFrame(
    [
        {"How acquired": name, "Items": count}
        for name, count in summary["condition_split"].items()
    ]
)
if not condition_frame.empty:
    condition_figure = px.pie(
        condition_frame,
        names="How acquired",
        values="Items",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Greens_r,
    )
    condition_figure.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    split_col.plotly_chart(condition_figure, use_container_width=True)

st.markdown("---")
st.markdown("### 🧾 Every Item, By Carbon Per Wear")

rows = []
for record in garments:
    result = lifetime_footprint(record)
    per_wear = carbon_per_wear(record)
    money_per_wear = cost_per_wear(record.get("price", 0), record.get("wears", 0))
    rows.append(
        {
            "Item": result["name"],
            "Type": result["category"],
            "Fibre": result["fibre"],
            "How acquired": result["condition"],
            "Worn": result["wears"],
            "Total CO₂e": f"{result['total_co2_kg']:.2f} kg",
            "Per wear": f"{per_wear:.3f} kg" if per_wear is not None else "never worn",
            "Cost per wear": f"{money_per_wear:.2f}" if money_per_wear is not None else "—",
        }
    )

table = pd.DataFrame(rows)
st.dataframe(table, use_container_width=True, hide_index=True)

st.caption(
    "Carbon per wear is the number that matters. A heavy wool coat worn for a "
    "decade beats a light polyester top worn eight times, and only this column "
    "shows that."
)

st.markdown("#### Log a wear or remove an item")
for record in garments:
    detail_col, wear_col, delete_col = st.columns([5, 1, 1])
    per_wear = carbon_per_wear(record)
    detail_col.markdown(
        f"**{record['name']}** — {record['fibre']}, worn {record['wears']}× · "
        + (f"{per_wear:.3f} kg CO₂e per wear" if per_wear is not None else "never worn")
    )
    if wear_col.button("Wore it", key=f"wear_{record['id']}"):
        log_wear(record["id"])
        st.rerun()
    if delete_col.button("Remove", key=f"delete_garment_{record['id']}"):
        delete_garment(record["id"])
        st.rerun()

dead_stock = find_dead_stock(garments)
if dead_stock:
    st.markdown("---")
    st.markdown("### 💤 Dead Stock")
    st.markdown(
        f"{len(dead_stock)} item(s) worn {DEAD_STOCK_WEAR_THRESHOLD} times or fewer. "
        "The carbon behind these is already spent — wearing them costs nothing more."
    )
    dead_frame = pd.DataFrame(
        [
            {
                "Item": item["name"],
                "Worn": item["wears"],
                "Carbon going to waste": f"{item['embodied_co2_kg']:.2f} kg CO₂e",
            }
            for item in dead_stock
        ]
    )
    st.dataframe(dead_frame, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### ⏳ What If You Just Kept Them Longer?")

extra_years = st.slider(
    "Extra years of use", min_value=0.0, max_value=5.0, value=1.0, step=0.5
)
extension = extend_life_saving(garments, extra_years)

extend_columns = st.columns(3)
extend_columns[0].metric("CO₂e avoided", f"{extension['co2_saved_kg']:.1f} kg")
extend_columns[1].metric("Water avoided", f"{extension['water_saved_l']:,.0f} L")
extend_columns[2].metric(
    "Replacements avoided", f"{extension['replacements_avoided']:.1f}"
)
st.caption(
    "Modelled as avoided replacement purchases at each garment's own new-item "
    "footprint. This is consistently worth more than any fibre choice made at the till."
)

st.markdown("---")
st.markdown("### 🤔 Should You Buy It?")

buy_category_col, buy_fibre_col, buy_wears_col = st.columns(3)
buy_category = buy_category_col.selectbox(
    "Thinking about a…", list(GARMENT_TYPES.keys()), key="compare_category"
)
buy_fibre = buy_fibre_col.selectbox(
    "Made of", [item["name"] for item in list_fibres()], key="compare_fibre"
)
buy_wears = buy_wears_col.number_input(
    "Honestly, how many wears?",
    min_value=1,
    max_value=2000,
    value=get_expected_wears(buy_category),
    step=5,
)

comparison = compare_purchase(buy_category, buy_fibre, expected_wears=buy_wears)
comparison_frame = pd.DataFrame(
    [
        {
            "Option": item["condition"],
            "Total CO₂e": f"{item['total_co2_kg']:.2f} kg",
            "Per wear": f"{item['carbon_per_wear_kg']:.4f} kg"
            if item["carbon_per_wear_kg"] is not None
            else "—",
            "Why": item["note"],
        }
        for item in comparison
    ]
)
st.dataframe(comparison_frame, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 💡 What To Do About It")
for tip in get_wardrobe_tips(summary):
    st.markdown(f"- {tip}")

st.caption(
    "Fibre carbon and water intensities follow published textile LCA ranges. "
    "Figures are per kilogram of finished fabric and are documented inline in "
    "`wardrobe.py` so they can be revised as better data lands."
)
