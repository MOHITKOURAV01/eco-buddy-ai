import pandas as pd
import plotly.express as px
import streamlit as st

from pet_footprint import (
    DEFAULT_BAG_TYPE,
    DEFAULT_HUMAN_BASELINE,
    FOOD_TYPES,
    HUMAN_DIET_BASELINES,
    LITTER_TYPES,
    SPECIES_PROFILES,
    WASTE_BAG_TYPES,
    compare_to_human_diet,
    delete_pet,
    get_pet_tips,
    get_pets,
    get_species_profile,
    household_pawprint,
    is_obligate_carnivore,
    list_food_types,
    portion_check,
    reduction_options,
    save_pet,
    total_pawprint,
)
from styles.theme import apply_theme

user_id = st.session_state.get("user_id")
if not user_id:
    st.warning("Please log in from the main application page.")
    st.stop()

apply_theme()

st.markdown(
    "<div class='section-header'>🐾 Pet Carbon Pawprint</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "The main assessment counts what **you** eat. It has never asked about the "
    "animals in your home, which means a household with a large dog is quietly "
    "under-reported by a figure that can rival a short-haul flight."
)
st.caption(
    "This page is about arithmetic, not guilt. It takes pet ownership as given "
    "and shows which levers actually move the number — which is almost always "
    "food type first, and portion size second."
)

st.markdown("---")
st.markdown("### ➕ Add a Pet")

with st.form("add_pet"):
    name_col, species_col = st.columns(2)
    pet_name = name_col.text_input("Name", value="", placeholder="Rex")
    species = species_col.selectbox("Species and size", list(SPECIES_PROFILES.keys()))

    profile = get_species_profile(species)
    st.caption(
        f"**{species}** — {profile['note']} "
        f"Typical intake is about **{profile['daily_grams']}g a day**."
    )

    if is_obligate_carnivore(species):
        st.info(
            "Cats are obligate carnivores. Plant-based feed is not nutritionally "
            "appropriate for them, so this page will never suggest it — "
            "by-product-based food is the realistic lever instead."
        )

    food_col, grams_col = st.columns(2)
    food_type = food_col.selectbox(
        "Food type", [item["name"] for item in list_food_types()]
    )
    daily_grams = grams_col.number_input(
        "Food per day (grams)",
        min_value=0.0,
        max_value=5000.0,
        value=float(profile["daily_grams"]),
        step=10.0,
        help="Weigh it rather than eyeballing a scoop — overfeeding is common.",
    )
    st.caption(f"**{food_type}** — {FOOD_TYPES[food_type]['note']}")

    if profile["litter"]:
        litter_col, litter_kg_col = st.columns(2)
        litter_type = litter_col.selectbox("Litter type", list(LITTER_TYPES.keys()))
        litter_kg = litter_kg_col.number_input(
            "Litter used per month (kg)", min_value=0.0, max_value=100.0, value=8.0, step=0.5
        )
        st.caption(f"**{litter_type}** — {LITTER_TYPES[litter_type]['note']}")
    else:
        litter_type = list(LITTER_TYPES.keys())[0]
        litter_kg = 0.0

    bags_col, bag_type_col, bedding_col = st.columns(3)
    bags_per_week = bags_col.number_input(
        "Waste bags per week", min_value=0, max_value=100, value=7, step=1
    )
    bag_type = bag_type_col.selectbox(
        "Bag material",
        list(WASTE_BAG_TYPES.keys()),
        index=list(WASTE_BAG_TYPES.keys()).index(DEFAULT_BAG_TYPE),
    )
    bedding_kg = bedding_col.number_input(
        "Bedding replaced per year (kg)", min_value=0.0, max_value=100.0, value=1.0, step=0.5
    )

    toys_col, accessories_col, vet_col, groom_col = st.columns(4)
    toys = toys_col.number_input("Toys a year", min_value=0, max_value=200, value=4, step=1)
    accessories = accessories_col.number_input(
        "Accessories a year", min_value=0, max_value=100, value=1, step=1
    )
    vet_visits = vet_col.number_input(
        "Vet visits a year", min_value=0, max_value=100, value=2, step=1
    )
    grooming = groom_col.number_input(
        "Grooming a year", min_value=0, max_value=100, value=0, step=1
    )

    if st.form_submit_button("Add pet", use_container_width=True):
        new_pet = {
            "name": pet_name or species,
            "species": species,
            "food_type": food_type,
            "daily_grams": daily_grams,
            "litter_type": litter_type,
            "litter_kg_per_month": litter_kg,
            "bags_per_week": bags_per_week,
            "bag_type": bag_type,
            "bedding_kg_per_year": bedding_kg,
            "toys_per_year": toys,
            "accessories_per_year": accessories,
            "vet_visits": vet_visits,
            "grooming_visits": grooming,
        }
        if save_pet(user_id, new_pet):
            st.success(f"Added {new_pet['name']}.")
            st.rerun()
        else:
            st.error("Could not save that pet.")

pets = get_pets(user_id)

if not pets:
    st.info("Add a pet above to see what the animals in your home actually cost.")
    st.stop()

household = household_pawprint(pets)

st.markdown("---")
st.markdown("### 📊 The Household Pawprint")

baseline = st.selectbox(
    "Compare against a human diet of",
    list(HUMAN_DIET_BASELINES.keys()),
    index=list(HUMAN_DIET_BASELINES.keys()).index(DEFAULT_HUMAN_BASELINE),
)
comparison = compare_to_human_diet(household["total_co2_kg"], baseline)

metric_columns = st.columns(4)
metric_columns[0].metric("Animals", household["pet_count"])
metric_columns[1].metric("Annual total", f"{household['total_co2_kg']:,.0f} kg CO₂e")
metric_columns[2].metric("Food's share", f"{household['food_share_pct']:.0f}%")
metric_columns[3].metric(
    "Vs one human diet", f"{comparison['share_of_human_diet_pct']:.0f}%"
)

st.caption(
    f"Your pets emit the equivalent of **{comparison['human_diet_equivalent']:.2f}×** "
    f"a {baseline.lower()} human diet ({comparison['baseline_kg']:,.0f} kg CO₂e a year)."
)

chart_col, pet_col = st.columns(2)

breakdown_frame = pd.DataFrame(
    [
        {"Category": name, "kg CO₂e": value}
        for name, value in household["breakdown"].items()
        if value > 0
    ]
)
if not breakdown_frame.empty:
    breakdown_figure = px.pie(
        breakdown_frame,
        names="Category",
        values="kg CO₂e",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Greens_r,
    )
    breakdown_figure.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    chart_col.plotly_chart(breakdown_figure, use_container_width=True)

pet_frame = pd.DataFrame(
    [{"Pet": pet["name"], "kg CO₂e": pet["total_co2_kg"]} for pet in household["pets"]]
)
pet_figure = px.bar(
    pet_frame.sort_values("kg CO₂e"),
    x="kg CO₂e",
    y="Pet",
    orientation="h",
    color="kg CO₂e",
    color_continuous_scale="Greens",
)
pet_figure.update_layout(
    height=360, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False
)
pet_col.plotly_chart(pet_figure, use_container_width=True)

st.markdown("---")
st.markdown("### 🐕 Each Animal")

for record in pets:
    result = total_pawprint(record)
    check = portion_check(result["species"], result["daily_grams"])

    with st.expander(
        f"{result['name']} — {result['total_co2_kg']:,.0f} kg CO₂e a year", expanded=True
    ):
        detail_columns = st.columns(4)
        detail_columns[0].metric("Food", f"{result['food_co2_kg']:,.0f} kg")
        detail_columns[1].metric("Litter", f"{result['litter_co2_kg']:,.0f} kg")
        detail_columns[2].metric("Consumables", f"{result['consumables_co2_kg']:,.0f} kg")
        detail_columns[3].metric("Vet / grooming", f"{result['vet_co2_kg']:,.0f} kg")

        if check["status"] == "over":
            st.warning(
                f"**Portion check:** {check['actual_grams']:.0f}g a day against a "
                f"profile of {check['expected_grams']}g — about "
                f"{check['ratio']:.1f}× what this animal needs. That is "
                f"{check['excess_food_kg_per_year']:.0f} kg of food a year that "
                "harms the animal as well as the footprint."
            )
        elif check["status"] == "under":
            st.info(
                f"**Portion check:** {check['actual_grams']:.0f}g a day is below the "
                f"{check['expected_grams']}g profile. Worth confirming with a vet — "
                "this page optimises carbon, not welfare."
            )
        else:
            st.success(
                f"**Portion check:** {check['actual_grams']:.0f}g a day is in line "
                f"with the {check['expected_grams']}g profile."
            )

        options = reduction_options(record)
        if options:
            st.markdown("**What would actually help**")
            options_frame = pd.DataFrame(
                [
                    {
                        "Change": item["action"],
                        "Category": item["category"],
                        "Saves": f"{item['saving_kg']:,.0f} kg CO₂e/yr",
                        "Why": item["note"],
                    }
                    for item in options
                ]
            )
            st.dataframe(options_frame, use_container_width=True, hide_index=True)
        else:
            st.caption("No further reductions found — this animal is already well set up.")

        if st.button("Remove this pet", key=f"delete_pet_{record['id']}"):
            delete_pet(record["id"])
            st.rerun()

st.markdown("---")
st.markdown("### 💡 What To Do About It")
for tip in get_pet_tips(household):
    st.markdown(f"- {tip}")

st.caption(
    "Pet food factors follow published pet-nutrition LCA literature, including "
    "the distinction between feed made from human-grade cuts and feed made from "
    "by-products — material that was produced regardless of whether any pet ate "
    "it. Assumptions are documented inline in `pet_footprint.py`."
)
