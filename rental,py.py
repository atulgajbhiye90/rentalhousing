# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# --------------------------------------------
# App Configuration
# --------------------------------------------
st.set_page_config(page_title="🏠 Rental Price Prediction - India", layout="wide")
st.title("🏠 Rental Property Price Predictor (India)")
st.markdown("Upload your dataset or use the default one to analyze and predict house rent across Indian cities.")

# --------------------------------------------
# File Upload & Loader
# --------------------------------------------
@st.cache_data
def load_from_bytes(bytes_data, filename: str) -> pd.DataFrame:
    from io import BytesIO
    bio = BytesIO(bytes_data)
    try:
        if filename.lower().endswith(".csv"):
            return pd.read_csv(bio)
        elif filename.lower().endswith(".xls"):
            return pd.read_excel(bio, engine="xlrd")
        elif filename.lower().endswith(".xlsx"):
            return pd.read_excel(bio, engine="openpyxl")
        else:
            st.error("Unsupported file type. Please upload .csv or .xlsx/.xls")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (.csv, .xls, .xlsx)",
    type=["csv", "xls", "xlsx"],
    help="Upload your rental dataset, or leave empty to use the built-in file."
)

# Load dataset
if uploaded_file is not None:
    df = load_from_bytes(uploaded_file.read(), uploaded_file.name)
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
else:
    try:
        df = pd.read_excel("rental_property_dataset_india.xlsx", engine="openpyxl")
        st.sidebar.info("Using built-in dataset: rental_property_dataset_india.xlsx")
    except Exception:
        try:
            df = pd.read_csv("rental_property_dataset_india.csv")
            st.sidebar.info("Using built-in dataset: rental_property_dataset_india.csv")
        except Exception:
            st.error("No dataset found. Please upload a valid dataset.")
            st.stop()

st.write(f"**Dataset loaded with {df.shape[0]:,} rows and {df.shape[1]} columns.**")
st.dataframe(df.head(10))

# --------------------------------------------
# Auto-detect key columns
# --------------------------------------------
def find_column(df, possible_names):
    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None

city_col = find_column(df, ["city", "location", "place", "area_name"])
ptype_col = find_column(df, ["property_type", "type", "house_type"])
furn_col = find_column(df, ["furnishing", "furnish", "furnishing_status"])
bed_col = find_column(df, ["bedroom", "bedrooms", "bhk"])
bath_col = find_column(df, ["bathroom", "bathrooms"])
area_col = find_column(df, ["area", "size", "sqft"])
price_col = find_column(df, ["price", "rent", "monthly_rent"])

st.markdown("### 🧭 Detected Columns")
st.write({
    "City": city_col,
    "Property Type": ptype_col,
    "Furnishing": furn_col,
    "Bedrooms": bed_col,
    "Bathrooms": bath_col,
    "Area": area_col,
    "Price": price_col,
})

if not price_col:
    st.error("❌ Could not detect target column (price/rent). Please rename it.")
    st.stop()

# --------------------------------------------
# Data Preprocessing
# --------------------------------------------
df = df.dropna(subset=[price_col])
features = [c for c in [city_col, ptype_col, furn_col, bed_col, bath_col, area_col] if c]
target = price_col

# Keep original values for form dropdowns
original_values = {}
for col in [city_col, ptype_col, furn_col]:
    if col and df[col].dtype == "object":
        original_values[col] = sorted(df[col].unique())

df = df[features + [target]].copy()

# Encode categorical columns
encoders = {}
for col in [city_col, ptype_col, furn_col]:
    if col and df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
st.success(f"✅ Model trained successfully — MAE: {mae:,.2f}")

# --------------------------------------------
# Manual Prediction Form
# --------------------------------------------
st.markdown("## 🧮 Predict House Rent (Manual Entry)")
with st.form("manual_prediction"):
    col1, col2, col3 = st.columns(3)
    with col1:
        if city_col and city_col in original_values:
            city = st.selectbox("City", original_values[city_col])
        else:
            city = st.text_input("City (type manually)")
        if ptype_col and ptype_col in original_values:
            property_type = st.selectbox("Property Type", original_values[ptype_col])
        else:
            property_type = st.text_input("Property Type")
    with col2:
        if furn_col and furn_col in original_values:
            furnishing = st.selectbox("Furnishing", original_values[furn_col])
        else:
            furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
        bedrooms = st.number_input("Bedrooms", 1, 10, 2)
    with col3:
        bathrooms = st.number_input("Bathrooms", 1, 10, 2)
        area = st.number_input("Area (sqft)", 100, 10000, 1000)

    submitted = st.form_submit_button("🔍 Predict Rent")

    if submitted:
        try:
            # Build input row with proper values
            input_row = []
            
            for feat in features:
                if feat == city_col:
                    if feat in encoders:
                        val = str(city)
                        if val in encoders[feat].classes_:
                            input_row.append(encoders[feat].transform([val])[0])
                        else:
                            st.warning(f"City '{city}' not in training data. Using closest match.")
                            input_row.append(0)
                    else:
                        input_row.append(city)
                        
                elif feat == ptype_col:
                    if feat in encoders:
                        val = str(property_type)
                        if val in encoders[feat].classes_:
                            input_row.append(encoders[feat].transform([val])[0])
                        else:
                            st.warning(f"Property type '{property_type}' not in training data.")
                            input_row.append(0)
                    else:
                        input_row.append(property_type)
                        
                elif feat == furn_col:
                    if feat in encoders:
                        val = str(furnishing)
                        if val in encoders[feat].classes_:
                            input_row.append(encoders[feat].transform([val])[0])
                        else:
                            st.warning(f"Furnishing '{furnishing}' not in training data.")
                            input_row.append(0)
                    else:
                        input_row.append(furnishing)
                        
                elif feat == bed_col:
                    input_row.append(int(bedrooms))
                elif feat == bath_col:
                    input_row.append(int(bathrooms))
                elif feat == area_col:
                    input_row.append(float(area))

            # Create input DataFrame with exact same structure as training data
            input_df = pd.DataFrame([input_row], columns=features)
            
            # Ensure all numeric columns are proper numeric types
            for col in input_df.columns:
                if col in [bed_col, bath_col, area_col]:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            
            prediction = model.predict(input_df)[0]

            st.success(f"🏡 **Estimated Rent/Price: ₹{prediction:,.0f}**")
            st.caption("This is a model-based estimate — actual prices may vary by area and amenities.")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.write("Debug info:")
            st.write(f"Features expected: {features}")
            st.write(f"Input shape: {input_df.shape if 'input_df' in locals() else 'Not created'}")

st.markdown("---")
st.markdown("© 2025 Rental Price Prediction App — Powered by Streamlit & scikit-learn")
