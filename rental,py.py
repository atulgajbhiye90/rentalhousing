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
# Utilities
# -------------------------
@st.cache_data
def load_from_bytes(bytes_data: bytes, filename: str) -> pd.DataFrame:
    if bytes_data is None:
        return pd.DataFrame()
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
        st.error(f"Error reading uploaded file: {e}")
    return pd.DataFrame()

@st.cache_data
def load_repo_dataset() -> pd.DataFrame:
    candidates = [
        ("rental_property_dataset_india.xls", "xlrd"),
        ("rental_property_dataset_india.xlsx", "openpyxl"),
        ("rental_property_dataset_india.csv", None),
    ]
    for fname, engine in candidates:
        try:
            if fname.endswith(".csv"):
                df = pd.read_csv(fname)
            elif engine:
                df = pd.read_excel(fname, engine=engine)
            else:
                df = pd.read_excel(fname)
            st.sidebar.success(f"Loaded dataset from repo: {fname}")
            return df
        except Exception:
            continue
    return pd.DataFrame()

def find_column(df, keywords):
    """Return first column name in df whose name contains any of the keywords (case-insensitive)."""
    cols = df.columns.astype(str)
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    return None

def normalize_colnames(df):
    """Trim whitespace from column names."""
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

def safe_label_encode_series(series, le=None):
    """
    Fit or update a LabelEncoder for a pandas Series.
    Returns encoder and encoded series.
    """
    s = series.fillna("").astype(str)
    if le is None:
        le = LabelEncoder()
        le.fit(s)
        encoded = le.transform(s)
        return le, encoded
    else:
        # update encoder classes if new categories exist
        existing = set(le.classes_)
        new_cats = [x for x in s.unique() if x not in existing]
        if new_cats:
            le.classes_ = np.concatenate([le.classes_, np.array(new_cats, dtype=object)])
        encoded = le.transform(s)
        return le, encoded

def ensure_cols_present(input_df, model_cols):
    """Ensure input_df has all model_cols. Add missing with 0 or median fallback."""
    df = input_df.copy()
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
    # order columns
    return df[model_cols]

# -------------------------
# Sidebar - dataset upload
# -------------------------
st.sidebar.header("📂 Dataset (upload optional)")
uploaded_file = st.sidebar.file_uploader(
    "Upload dataset (.xls, .xlsx, .csv)", type=["xls", "xlsx", "csv"]
)

if uploaded_file is not None:
    df = load_from_bytes(uploaded_file.read(), uploaded_file.name)
else:
    df = load_repo_dataset()

if df is None or df.empty:
    st.error("No dataset found. Upload a dataset or add rental_property_dataset_india.xls/.xlsx/.csv to the repo root.")
    st.stop()

df = normalize_colnames(df)

st.title("🏠 Rental Property Analysis & Manual Rent Prediction (India)")
st.write(f"Dataset: **{df.shape[0]:,} rows × {df.shape[1]:,} columns**")

# -------------------------
# Auto-detect columns
# -------------------------
# Common possibilities
city_col = find_column(df, ["city", "location", "place", "area_name", "town"])
ptype_col = find_column(df, ["property_type", "type", "flat_type", "house_type"])
