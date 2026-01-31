import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
GITHUB_LINK = "https://github.com/malihashafi20"
PAGE_TITLE = "Pakistan Rent Estimator"

st.set_page_config(page_title=PAGE_TITLE, page_icon="🇵🇰", layout="centered")

# --- 2. STYLING ---
st.markdown("""
    <style>
    /* Main Action Button */
    .stButton>button { width: 100%; background-color: #006400; color: white; font-weight: bold; border-radius: 8px; }
    .stButton>button:hover { background-color: #004d00; color: white; }
    
    /* Result Box */
    .result-box { background-color: #f0fff4; padding: 20px; border-radius: 10px; border-left: 6px solid #006400; text-align: center; }
    .result-text { color: grey; margin: 0; font-size: 1.1em; }
    .result-value { color: #006400; margin: 0; font-size: 2.5em; font-weight: bold; }
    
    /* Sidebar GitHub Button Style */
    .github-btn {
        display: inline-block;
        width: 100%;
        padding: 10px;
        background-color: #24292e;
        color: white !important;
        text-align: center;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        border: 1px solid #444;
        transition: 0.3s;
    }
    .github-btn:hover {
        background-color: #444;
        border-color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD DATA ---
@st.cache_resource
def load_data():
    try:
        with open('Model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

data = load_data()
if not data:
    st.error("⚠️ Model missing. Please run 'train_model.py' first.")
    st.stop()

model = data['model']
ui_map = data['ui_map']
sample_df = data['sample_data']

# --- 4. SIDEBAR (REDESIGNED) ---
with st.sidebar:
    # 1. Logo Section
    try:
        st.image("logo.png", width=100)
    except:
        st.warning("⚠️ Logo not found. Save image as 'logo.png'.")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # 2. About Section (Clean Container)
    with st.container():
        st.info("This AI model estimates fair market rent using **2025 Zameen.com data** from major Pakistani cities.")
        
        # Styled GitHub Link
        st.markdown(f"""
            <a href="{GITHUB_LINK}" target="_blank" class="github-btn">
                🔗 View Source on GitHub
            </a>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Market Trends Plot (Dark Mode Optimized)
    st.markdown("### 📊 Market Trends")
    st.caption("Rent distribution per city (PKR)")

    # Set dark theme for the plot to match the app
    plt.style.use("dark_background") 
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Create Boxplot
    sns.boxplot(
        x='City', 
        y='price_clean', 
        data=sample_df, 
        ax=ax, 
        showfliers=False, 
        palette="viridis", # "viridis" pops better on dark backgrounds
        linewidth=1
    )
    
    # Clean up the chart
    ax.set_xlabel("")
    ax.set_ylabel("Rent (PKR)", fontsize=9, color="white")
    ax.tick_params(axis='x', rotation=45, colors="white")
    ax.tick_params(axis='y', colors="white")
    
    # Remove top and right borders for a cleaner look
    sns.despine(top=True, right=True, left=True, bottom=False)
    
    # Make background transparent so it fits any theme perfectly
    fig.patch.set_alpha(0.0) 
    ax.patch.set_alpha(0.0)
    
    st.pyplot(fig, use_container_width=True)

# --- 5. MAIN INTERFACE ---
st.title("🇵🇰 Pakistan Rent Estimator")
st.caption("Real-World AI Valuation for 2025 Market")
st.markdown("---")

# ROW 1: LOCATION
c1, c2 = st.columns(2)
with c1:
    city = st.selectbox("📍 Select City", list(ui_map.keys()))
with c2:
    area = st.selectbox("🏙️ Select Area / Sector", ui_map[city])

st.markdown("<br>", unsafe_allow_html=True)

# ROW 2: PLOT SIZE
st.subheader("🏠 Property Configuration")

size_labels = {
    1: "1 Marla", 2: "2 Marla", 3: "3 Marla", 4: "4 Marla",
    5: "5 Marla", 6: "6 Marla", 7: "7 Marla", 8: "8 Marla",
    9: "9 Marla", 10: "10 Marla",
    20: "1 Kanal", 40: "2 Kanal"
}

selected_size_key = st.select_slider(
    "Plot Size",
    options=list(size_labels.keys()),
    value=5,
    format_func=lambda x: size_labels[x]
)

# --- STRICT BUSINESS RULES & NOTES ---
# Define limits
if selected_size_key == 1:
    max_beds = 1
    max_baths = 1
    note_msg = "📝 **Note:** In 1 Marla, you can have **1 Bedroom and 1 Bathroom only**."
    note_type = "warning"

elif selected_size_key == 2:
    max_beds = 2
    max_baths = 2
    note_msg = "📝 **Note:** In 2 Marla, maximum limit is **2 Bedrooms and 2 Bathrooms**."
    note_type = "warning"

elif selected_size_key <= 10:
    max_beds = selected_size_key
    max_baths = selected_size_key + 1
    note_msg = f"📝 **Note:** For {size_labels[selected_size_key]}, standard limit is **{max_beds} Beds** & **{max_baths} Baths**."
    note_type = "info"

elif selected_size_key == 20: # 1 Kanal
    max_beds = 10
    max_baths = 11
    note_msg = "📝 **Note:** 1 Kanal allows up to **10 Beds & 11 Baths**."
    note_type = "info"

else: # 2 Kanal
    max_beds = 15
    max_baths = 16
    note_msg = "📝 **Note:** 2 Kanal allows up to **15 Beds & 16 Baths**."
    note_type = "info"

# Display the Note
if note_type == "warning":
    st.warning(note_msg)
else:
    st.info(note_msg)


# --- ROW 3: DYNAMIC SLIDERS ---
c3, c4 = st.columns(2)

with c3:
    if max_beds == 1:
        # Fixed display for 1 Marla
        st.write("**Bedroom**")
        st.button("1", key="btn_bed_1", disabled=True) 
        beds = 1
    else:
        # Dynamic slider
        beds = st.slider("Bedrooms", 1, max_beds, 1, key=f"bed_slider_{selected_size_key}")

with c4:
    if max_baths == 1:
        # Fixed display for 1 Marla
        st.write("**Bathroom**")
        st.button("1", key="btn_bath_1", disabled=True)
        baths = 1
    else:
        baths = st.slider("Bathrooms", 1, max_baths, 1, key=f"bath_slider_{selected_size_key}")

furnished = st.toggle("Furnished Property", value=False)

# --- 6. PREDICTION ---
if st.button("Calculate Rent"):
    input_df = pd.DataFrame(0, index=[0], columns=data['features'])
    input_df['marla_clean'] = selected_size_key
    input_df['bed_clean'] = beds
    input_df['bath_clean'] = baths
    input_df['is_furnished'] = 1 if furnished else 0

    if f"City_{city}" in input_df.columns:
        input_df[f"City_{city}"] = 1
    if f"Detailed_Area_{area}" in input_df.columns:
        input_df[f"Detailed_Area_{area}"] = 1

    pred = model.predict(input_df)[0]

    # Post-processing logic
    if selected_size_key == 1 and pred < 10000:
        pred = 12000
    if pred < 0:
        pred = 5000 * selected_size_key

    st.markdown(f"""
        <div class="result-box">
            <p class="result-text">Estimated Monthly Rent</p>
            <p class="result-value">PKR {int(pred):,}</p>
        </div>
    """, unsafe_allow_html=True)

    st.caption(f"📈 **Market Insight:** A {size_labels[selected_size_key]} property in {city} typically yields this rent.")

    # CSV Download Logic
    csv = pd.DataFrame({
        'City': [city],
        'Area': [area],
        'Size': [size_labels[selected_size_key]],
        'Bedrooms': [beds],
        'Bathrooms': [baths],
        'Rent': [int(pred)]
    }).to_csv(index=False).encode('utf-8')

    st.download_button("📥 Download ", csv, "rent_quote.csv", "text/csv")