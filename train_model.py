import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle
import re

print("🚀 Starting Pipeline with DATA REPAIR...")

# 1. Load Data
df = pd.read_csv('zameen_rentals_data.csv')

# --- 🚨 CRITICAL DATA REPAIR FUNCTION ---
def fix_shifted_columns(df):
    """
    Detects rows where 'Washrooms' column contains 'Marla' or 'Kanal'.
    In these rows:
    1. Move the value from 'Washrooms' -> 'Marla' (Area)
    2. Assume 'Washrooms' = 'Bedrooms' (since the real washroom count is lost/overwritten)
    """
    # Find rows where Washrooms looks like text (e.g., "1.8 Kanal", "10 Marla")
    bad_rows = df['Washrooms'].astype(str).str.contains('Marla|Kanal', case=False, na=False)
    
    if bad_rows.sum() > 0:
        print(f"⚠️ FOUND {bad_rows.sum()} CORRUPTED ROWS! Fixing them now...")
        
        # 1. Move Area info to the correct column
        df.loc[bad_rows, 'Marla'] = df.loc[bad_rows, 'Washrooms']
        
        # 2. Fill missing Washrooms with Bedroom count (Best guess heuristic)
        df.loc[bad_rows, 'Washrooms'] = df.loc[bad_rows, 'Bedrooms']
        
        print("✅ Fixed. '1.8 Kanal' is now in the Marla column, not Washrooms.")
    return df

# APPLY THE FIX IMMEDIATELY
df = fix_shifted_columns(df)

# --- STANDARD CLEANING & GEOGRAPHY ---
AREA_TO_CITY = {
    'Clifton': 'Karachi', 'Gulshan-e-Iqbal': 'Karachi', 'DHA City Karachi': 'Karachi',
    'North Nazimabad': 'Karachi', 'Malir': 'Karachi', 'PECHS': 'Karachi', 'Scheme 33': 'Karachi',
    'Gulberg': 'Lahore', 'Johar Town': 'Lahore', 'Wapda Town': 'Lahore', 'Model Town': 'Lahore',
    'DHA Defence': 'Lahore', 'Askari': 'Lahore', 'Valencia': 'Lahore',
    'Bahria Town Rawalpindi': 'Rawalpindi', 'Chaklala': 'Rawalpindi', 'Adiala': 'Rawalpindi',
    'Bahria Enclave': 'Islamabad', 'Bani Gala': 'Islamabad', 'DHA Defence Phase 2': 'Islamabad'
}
KNOWN_CITIES = ['Islamabad', 'Lahore', 'Karachi', 'Rawalpindi', 'Faisalabad', 'Multan', 'Gujranwala', 'Sialkot', 'Peshawar', 'Hyderabad', 'Quetta']

def parse_location(row):
    loc_str = str(row['Location'])
    loc_lower = loc_str.lower()
    
    # Detect City
    city = "Other"
    for k in KNOWN_CITIES:
        if k.lower() in loc_lower:
            city = k
            break
    if city == "Other":
        for area, mapped_city in AREA_TO_CITY.items():
            if area.lower() in loc_lower:
                city = mapped_city
                break
    if city == "Other" and re.search(r'\b[e-i]-\d{1,2}', loc_lower):
        city = "Islamabad"

    # Extract Specific Area
    if city != "Other":
        pattern = re.compile(city, re.IGNORECASE)
        area_part = pattern.sub("", loc_str)
    else:
        area_part = loc_str
    
    parts = [p.strip() for p in area_part.split(',') if p.strip()]
    final_area = " - ".join(dict.fromkeys(parts)) 
    
    if not final_area: final_area = "Main Area"
    return pd.Series([city, final_area])

print("🔹 Mapping Geographies...")
df[['City', 'Detailed_Area']] = df.apply(parse_location, axis=1)
df = df[df['City'] != 'Other']

# --- VALUE PARSING ---
def clean_currency(val):
    if not isinstance(val, str): return val
    val = val.lower().replace(',', '')
    try:
        if 'crore' in val: return float(re.findall(r"[\d\.]+", val)[0]) * 10000000
        elif 'lakh' in val: return float(re.findall(r"[\d\.]+", val)[0]) * 100000
        elif 'thousand' in val: return float(re.findall(r"[\d\.]+", val)[0]) * 1000
        return float(val)
    except: return np.nan

def clean_marla(val):
    if not isinstance(val, str): return val
    val = val.lower()
    try:
        num = float(re.findall(r"[\d\.]+", val)[0])
        if 'kanal' in val: return num * 20
        return num
    except: return np.nan

df['price_clean'] = df['Price'].apply(clean_currency)
df['marla_clean'] = df['Marla'].apply(clean_marla)
df['bed_clean'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
df['bath_clean'] = pd.to_numeric(df['Washrooms'], errors='coerce') # NOW this works because text is gone
df['is_furnished'] = df['Details'].str.lower().str.contains('furnished', na=False).astype(int)

df = df.dropna(subset=['price_clean', 'marla_clean', 'bed_clean'])
df = df[(df['price_clean'] < 5000000) & (df['marla_clean'] < 80)]

# Filter Areas
valid_areas = df['Detailed_Area'].value_counts()
valid_areas = valid_areas[valid_areas >= 3].index
df = df[df['Detailed_Area'].isin(valid_areas)]

# --- TRAINING ---
print("🔹 Training Model (Clean Data)...")
features = ['marla_clean', 'bed_clean', 'bath_clean', 'is_furnished']
X = df[features + ['City', 'Detailed_Area']]
X = pd.get_dummies(X, columns=['City', 'Detailed_Area'])
y = df['price_clean']

model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1)
model.fit(X, y)

# Save
ui_map = {city: sorted(df[df['City'] == city]['Detailed_Area'].unique()) for city in sorted(df['City'].unique())}
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).head(8)

artifact = {
    'model': model,
    'features': X.columns.tolist(),
    'ui_map': ui_map,
    'sample_data': df[['City', 'price_clean']].sample(500),
    'feature_importance': feat_imp
}

with open('Model.pkl', 'wb') as f:
    pickle.dump(artifact, f)

print("✅ DONE! Model Saved (Data Corruption Fixed).")