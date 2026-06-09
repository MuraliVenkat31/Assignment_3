import streamlit as st

# ── MUST be first Streamlit call ───────────────────────────────────────────────
st.set_page_config(
    page_title="Car Dekho - Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import pickle
import os

# ══════════════════════════════════════════════════════════════════════════════
# THEME / CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}
.block-container { padding: 0 2rem 3rem; max-width: 1200px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d0d18 0%, #12121f 60%, #1a0a0a 100%);
    border-bottom: 1px solid #ff4c1520;
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -2rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, #ff4c1508, transparent);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #ff4c15;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    letter-spacing: 0.04em;
    line-height: 1;
    color: #fff;
    margin: 0;
}
.hero-title span { color: #ff4c15; }
.hero-sub {
    color: #666;
    font-size: 0.95rem;
    font-weight: 300;
    margin-top: 0.6rem;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff4c15;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #ff4c1520;
}

/* ── Input card wrapper ── */
.input-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Result card ── */
.result-wrap {
    background: linear-gradient(135deg, #120a05, #0f0f1a);
    border: 1px solid #ff4c1530;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, #ff4c1510, transparent 65%);
    pointer-events: none;
}
.result-eyebrow {
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    letter-spacing: 0.04em;
    line-height: 1;
    color: #ff4c15;
    text-shadow: 0 0 40px #ff4c1540;
}
.result-range {
    color: #555;
    font-size: 0.82rem;
    margin-top: 0.5rem;
    font-weight: 300;
}
.result-model-badge {
    display: inline-block;
    background: #1a1a2a;
    border: 1px solid #2a2a3e;
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.72rem;
    color: #888;
    letter-spacing: 0.08em;
    margin-top: 1rem;
}

/* ── Stat pills ── */
.stat-row { display: flex; gap: 0.75rem; margin-top: 1.5rem; }
.stat-pill {
    flex: 1;
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 0.9rem 0.75rem;
    text-align: center;
}
.stat-pill-label {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.3rem;
}
.stat-pill-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    color: #e8e8f0;
    letter-spacing: 0.06em;
}

/* ── Feature grid ── */
.feat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-top: 1.25rem;
}
.feat-item {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.feat-key  { font-size: 0.72rem; color: #666; text-transform: uppercase; letter-spacing: 0.07em; }
.feat-val  { font-size: 0.82rem; color: #e8e8f0; font-weight: 500; }

/* ── Streamlit widget overrides ── */
.stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #888 !important;
}
div[data-baseweb="select"] > div {
    background: #0f0f1a !important;
    border-color: #1e1e2e !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div > div { background: #ff4c15 !important; }

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: #ff4c15 !important;
    color: #fff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.2rem !important;
    letter-spacing: 0.12em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px #ff4c1530 !important;
}
.stButton > button:hover {
    background: #e03d08 !important;
    box-shadow: 0 6px 32px #ff4c1550 !important;
    transform: translateY(-1px) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080810 !important;
    border-right: 1px solid #1a1a28;
}
section[data-testid="stSidebar"] .block-container { padding: 1rem 1rem 2rem; }

/* ── Divider ── */
hr { border-color: #1e1e2e !important; margin: 1.5rem 0 !important; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING MAPS  (single source of truth — used for both display & prediction)
# ══════════════════════════════════════════════════════════════════════════════
ENCODE = {
    'model': {
        'Maruti':0,'Ford':1,'Tata':2,'Hyundai':3,'Jeep':4,'Datsun':5,
        'Honda':6,'Mahindra':7,'Renault':8,'Mercedes-Benz':9,'Audi':10,
        'BMW':11,'Toyota':12,'Mini':13,'Kia':14,'Skoda':15,'Volkswagen':16,
        'Nissan':17,'Fiat':18,'Mahindra Ssangyong':19,'Mitsubishi':20,
        'Jaguar':21,'Land Rover':22,'Chevrolet':23,'Mahindra Renault':24,
        'Volvo':25,'Isuzu':26,'Lexus':27,'Porsche':28,
    },
    'fuel_type':          {'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4},
    'transmission':       {'Manual':0,'Automatic':1},
    'city':               {'Bangalore':0,'Chennai':1,'Delhi':2,'Hyderabad':3,'Jaipur':4,'Kolkata':5},
    'insurance_validity': {'Third Party insurance':0,'Comprehensive':1,'Third Party':2,'Zero Dep':3},
    'body_type': {
        'Hatchback':0,'SUV':1,'Sedan':2,'MUV':3,'Convertibles':4,
        'Minivans':5,'Coupe':6,'Wagon':7,'Pickup Trucks':8,
    },
}

FEATURE_ORDER = ['city','fuel_type','kms_driven','transmission','ownerNo',
                 'model','model_year','mileage','Engine','insurance_validity','body_type']


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ASSETS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_model(path="xgb_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner="Loading dataset…")
def load_data(path="df.csv"):
    return pd.read_csv(path)

model_loaded, data_loaded = True, True
try:
    xgb_model = load_model()
except Exception as e:
    st.error(f"❌ Could not load **xgb_model.pkl**: {e}")
    model_loaded = False

try:
    df = load_data()
    # Normalise city casing so lookups are consistent
    if 'city' in df.columns:
        df['city'] = df['city'].str.title()
except Exception as e:
    st.error(f"❌ Could not load **df.csv**: {e}")
    data_loaded = False
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-title">Car<span>Value</span> </div>
    <div class="hero-sub">Instant used-car price estimates powered by XGBoost · 66% R²</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — inputs
# ══════════════════════════════════════════════════════════════════════════════
def safe_unique(col, fallback):
    if data_loaded and col in df.columns:
        vals = sorted(df[col].dropna().unique().tolist())
        return vals if vals else fallback
    return fallback

with st.sidebar:
    st.markdown("### 🔧 Car Details")
    st.markdown("---")

    city             = st.selectbox("📍 City",               safe_unique('city',               list(ENCODE['city'].keys())))
    model_brand      = st.selectbox("🚘 Brand / Model",      safe_unique('model',              list(ENCODE['model'].keys())))
    body_type        = st.selectbox("🚙 Body Type",          safe_unique('body_type',          list(ENCODE['body_type'].keys())))
    fuel_type        = st.selectbox("⛽ Fuel Type",          safe_unique('fuel_type',          list(ENCODE['fuel_type'].keys())))
    transmission     = st.selectbox("⚙️ Transmission",       safe_unique('transmission',       list(ENCODE['transmission'].keys())))
    insurance_validity = st.selectbox("🛡️ Insurance",        safe_unique('insurance_validity', list(ENCODE['insurance_validity'].keys())))
    ownerNo          = st.selectbox("👤 No. of Owners",      sorted(safe_unique('ownerNo', [1,2,3,4,5])))

    st.markdown("---")
    model_year  = st.slider("📅 Year of Manufacture", 1994, 2024, 2019)
    kms_driven  = st.slider("🛣️ Kms Driven",          500,  200000, 45000, step=500)
    mileage     = st.slider("💧 Mileage (km/l)",       5,    45,    18)
    engine_cc   = st.slider("🔩 Engine (CC)",          600,  5000,  1200, step=50)

    st.markdown("---")
    predict_btn = st.button("⚡ Predict Price")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — two columns
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.1, 1], gap="large")

# ── Left: summary of inputs ──────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">Your Car Summary</div>', unsafe_allow_html=True)

    car_age = 2024 - model_year
    kms_label = f"{kms_driven:,} km"
    age_label = f"{car_age} yr{'s' if car_age != 1 else ''} old"

    st.markdown(f"""
    <div class="feat-grid">
        <div class="feat-item"><span class="feat-key">City</span>      <span class="feat-val">{city}</span></div>
        <div class="feat-item"><span class="feat-key">Brand</span>     <span class="feat-val">{model_brand}</span></div>
        <div class="feat-item"><span class="feat-key">Body</span>      <span class="feat-val">{body_type}</span></div>
        <div class="feat-item"><span class="feat-key">Fuel</span>      <span class="feat-val">{fuel_type}</span></div>
        <div class="feat-item"><span class="feat-key">Gearbox</span>   <span class="feat-val">{transmission}</span></div>
        <div class="feat-item"><span class="feat-key">Owners</span>    <span class="feat-val">{ownerNo}</span></div>
        <div class="feat-item"><span class="feat-key">Year</span>      <span class="feat-val">{model_year} ({age_label})</span></div>
        <div class="feat-item"><span class="feat-key">Kms</span>       <span class="feat-val">{kms_label}</span></div>
        <div class="feat-item"><span class="feat-key">Mileage</span>   <span class="feat-val">{mileage} km/l</span></div>
        <div class="feat-item"><span class="feat-key">Engine</span>    <span class="feat-val">{engine_cc} CC</span></div>
        <div class="feat-item"><span class="feat-key">Insurance</span> <span class="feat-val">{insurance_validity}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Depreciation tip
    if car_age > 5:
        dep_pct = min(car_age * 8, 70)
        st.markdown(f"""
        <div style="margin-top:1rem; background:#0f0f1a; border:1px solid #1e1e2e;
                    border-left:3px solid #ff4c15; border-radius:0 8px 8px 0;
                    padding:0.75rem 1rem; font-size:0.82rem; color:#888;">
        ⚠️ {car_age}-year-old car — estimated depreciation ~{dep_pct}% from new.
        </div>""", unsafe_allow_html=True)


# ── Right: prediction result ─────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">Estimated Value</div>', unsafe_allow_html=True)

    if predict_btn:
        if not model_loaded:
            st.error("Model not loaded. Please check xgb_model.pkl.")
        else:
            # ── Build & encode input row ──────────────────────────────────────
            def encode_val(col, val):
                mapping = ENCODE.get(col, {})
                if mapping:
                    # case-insensitive fallback
                    if val in mapping:
                        return mapping[val]
                    for k, v in mapping.items():
                        if k.lower() == str(val).lower():
                            return v
                    st.warning(f"'{val}' not found in {col} encoding — using 0.")
                    return 0
                return val   # numeric column, return as-is

            raw = {
                'city':               city,
                'fuel_type':          fuel_type,
                'kms_driven':         kms_driven,
                'transmission':       transmission,
                'ownerNo':            ownerNo,
                'model':              model_brand,
                'model_year':         model_year,
                'mileage':            mileage,
                'Engine':             engine_cc,
                'insurance_validity': insurance_validity,
                'body_type':          body_type,
            }

            encoded = {col: encode_val(col, raw[col]) for col in FEATURE_ORDER}
            input_df = pd.DataFrame([encoded], columns=FEATURE_ORDER)

            try:
                price = xgb_model.predict(input_df)[0]
                lakh  = price / 100_000
                low   = lakh * 0.92
                high  = lakh * 1.08

                st.markdown(f"""
                <div class="result-wrap">
                    <div class="result-eyebrow">Predicted Market Price</div>
                    <div class="result-price">₹{lakh:.2f}L</div>
                    <div class="result-range">Likely range: ₹{low:.2f}L – ₹{high:.2f}L</div>
                    <div class="result-model-badge">XGBoost · Test R² 66.3%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-pill">
                        <div class="stat-pill-label">Per Month EMI ~</div>
                        <div class="stat-pill-val">₹{(price*0.02/1000):.1f}K</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-pill-label">Car Age</div>
                        <div class="stat-pill-val">{car_age} YRS</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-pill-label">Kms / Year</div>
                        <div class="stat-pill-val">{int(kms_driven/max(car_age,1)):,}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.write("Input sent to model:", input_df)

    else:
        st.markdown("""
        <div class="result-wrap" style="opacity:0.5">
            <div class="result-eyebrow">Predicted Market Price</div>
            <div class="result-price" style="color:#333">₹ — L</div>
            <div class="result-range">Fill in the details and click Predict</div>
        </div>
        """, unsafe_allow_html=True)

