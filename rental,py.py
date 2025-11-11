# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="🏠 Rental Property Analysis - India", layout="wide")
sns.set(style="whitegrid")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_from_bytes(bytes_data: bytes, filename: str) -> pd.DataFrame:
    """Read uploaded Excel/CSV file."""
    name = filename.lower()
    bio = BytesIO(bytes_data)
    try:
        if name.endswith(".csv"):
            return pd.read_csv(bio)
        elif name.endswith(".xls"):
            return pd.read_excel(bio, engine="xlrd")
        elif name.endswith(".xlsx"):
            return pd.read_excel(bio, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return pd.DataFrame()

@st.cache_data
def load_repo_dataset() -> pd.DataFrame:
    """Load default dataset from repo."""
    for fname, engine in [
        ("rental_property_dataset_india.xls", "xlrd"),
        ("rental_property_dataset_india.xlsx", "openpyxl"),
        ("rental_property_dataset_india.csv", None),
    ]:
        try:
            if fname.endswith(".csv"):
                return pd.read_csv(fname)
            elif engine:
                return pd.read_excel(fname, engine=engine)
            else:
                return pd.read_excel(fname)
        except Exception:
            continue
    return pd.DataFrame()

def encode_data(df, target_col):
    """Encode categorical columns for modeling."""
    df = df.copy()
    le_dict = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    return df, le_dict

# -------------------------
# Sidebar - dataset upload
# -------------------------
st.sidebar.header("📂 Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload dataset (.xls, .xlsx, .csv)", type=["xls", "xlsx", "csv"]
)

if uploaded_file is not None:
    df = load_from_bytes(uploaded_file.read(), uploaded_file.name)
else:
    df = load_repo_dataset()

if df.empty:
    st.error("❌ No dataset found. Upload one or include it in the repo.")
    st.stop()

st.title("🏠 Rental Property Market Analysis - India")
st.write(f"Loaded dataset with **{df.shape[0]:,} rows** and **{df.shape[1]} columns**")

# -------------------------
# Data Overview
# -------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10))
st.write("Columns:", list(df.columns))

# Detect potential target column
target_candidates = [c for c in df.columns if "price" in c.lower() or "rent" in c.lower()]
target = st.selectbox("Select Target Column (Price/Rent)", target_candidates)

# Clean numeric & categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# -------------------------
# Quick visualization
# -------------------------
if target:
    st.subheader(f"📈 Distribution of {target}")
    fig, ax = plt.subplots()
    sns.histplot(df[target], kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------
# Train simple model
# -------------------------
st.subheader("🤖 Model Training")
if target:
    st.write("Training model to predict:", target)
    df_encoded, le_dict = encode_data(df, target)
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    st.success(f"✅ Model trained successfully! MAE: {mae:.2f}")

# -------------------------
# Manual prediction input form
# -------------------------
st.subheader("🏡 Predict Rent/Price for New Property (Manual Input)")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.selectbox("City", sorted(df["city"].dropna().unique()))
        property_type = st.selectbox("Property Type", sorted(df["property_type"].dropna().unique()))
    with col2:
        furnishing = st.selectbox(
            "Furnishing", sorted(df["furnishing"].dropna().unique())
            if "furnishing" in df.columns else ["Furnished", "Semi-Furnished", "Unfurnished"]
        )
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
    with col3:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        area = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1000)
    
    submitted = st.form_submit_button("🔍 Predict Rent/Price")

if submitted:
    # Prepare single-row DataFrame
    input_dict = {
        "city": [city],
        "property_type": [property_type],
        "furnishing": [furnishing],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "area": [area],
    }

    input_df = pd.DataFrame(input_dict)

    # Ensure all model columns exist
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0  # default for missing

    # Encode using same encoders
    for col, le in le_dict.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    # Predict
    prediction = model.predict(input_df[X.columns])[0]
    st.success(f"💰 **Predicted {target}: ₹{prediction:,.0f}**")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.write("Made with ❤️ using Streamlit | Data Source: Rental Property Dataset (India)")
