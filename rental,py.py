# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Rental Property Analysis", layout="wide")
sns.set(style="whitegrid")

# -------------------------
# Helpers: read uploaded or repo file
# -------------------------
@st.cache_data
def load_from_bytes(bytes_data: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded Excel/CSV file from bytes and return a DataFrame."""
    if not filename or bytes_data is None:
        return pd.DataFrame()
    name = filename.lower()
    bio = BytesIO(bytes_data)
    try:
        if name.endswith(".csv"):
            return pd.read_csv(bio)
        elif name.endswith(".xls"):
            # requires xlrd==1.2.0 in requirements.txt for .xls
            return pd.read_excel(bio, engine="xlrd")
        elif name.endswith(".xlsx"):
            return pd.read_excel(bio, engine="openpyxl")
        else:
            # fallback: let pandas try to infer
            return pd.read_excel(bio)
    except Exception as e:
        st.error(f"Could not read uploaded file `{filename}`: {e}")
        return pd.DataFrame()

@st.cache_data
def load_repo_dataset() -> pd.DataFrame:
    """Try loading dataset from the repository root with several fallbacks."""
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
            st.sidebar.success(f"Loaded repo dataset: {fname}")
            return df
        except Exception:
            continue
    return pd.DataFrame()

# -------------------------
# Sidebar: file uploader + options
# -------------------------
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload a dataset (.xls, .xlsx, .csv) — optional",
    type=["xls", "xlsx", "csv"],
    help="Uploaded file will be used for this session only. If not uploaded, the bundled repo dataset will be used."
)

df = pd.DataFrame()
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    st.sidebar.success(f"Uploaded: {uploaded_file.name} ({len(file_bytes):,} bytes)")
    df = load_from_bytes(file_bytes, uploaded_file.name)
else:
    df = load_repo_dataset()

if df is None or df.empty:
    st.sidebar.warning("No dataset found. Upload a dataset or add one to the repo root as rental_property_dataset_india.xls/.xlsx/.csv")
    st.title("Rental Property — No dataset")
    st.write("Please upload a dataset using the sidebar uploader or add a dataset file to the repository root.")
    st.stop()

# -------------------------
# Basic info & preview
# -------------------------
st.title("🏠 Rental Property — Quick Analysis")
st.write(f"Using dataset with **{df.shape[0]:,} rows** and **{df.shape[1]:,} columns**.")
st.subheader("Columns")
st.write(df.columns.tolist())

st.subheader("Preview (first 20 rows)")
st.dataframe(df.head(20))

# -------------------------
# Quick cleaning: normalize column names
# -------------------------
def normalize_cols(df_):
    df = df_.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

df = normalize_cols(df)

# -------------------------
# Identify numeric columns and target candidates
# -------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_candidates = [c for c in df.columns if "price" in c.lower() or "rent" in c.lower() or "price_per" in c.lower()]
if not target_candidates:
    # fallback: if there's a numeric column named 'price' or 'rent'
    for opt in ["price", "rent", "rent_per_sqft", "price_per_sqft"]:
        if opt in df.columns:
            target_candidates.append(opt)

# -------------------------
# Filters (sidebar)
# -------------------------
st.sidebar.header("Filters")
filters = {}
if "city" in df.columns:
    cities = ["All"] + sorted(df["city"].dropna().astype(str).unique().tolist())
    sel_city = st.sidebar.selectbox("City", cities)
    if sel_city != "All":
        filters["city"] = sel_city

if "property_type" in df.columns:
    types = ["All"] + sorted(df["property_type"].dropna().astype(str).unique().tolist())
    sel_type = st.sidebar.selectbox("Property type", types)
    if sel_type != "All":
        filters["property_type"] = sel_type

# Additional numeric range filter example: area
if "area" in df.columns and pd.api.types.is_numeric_dtype(df["area"]):
    min_area = int(df["area"].min(skipna=True)) if not np.isnan(df["area"].min(skipna=True)) else 0
    max_area = int(df["area"].max(skipna=True)) if not np.isnan(df["area"].max(skipna=True)) else 0
    if min_area < max_area:
        r = st.sidebar.slider("Area range", min_area, max_area, (min_area, max_area))
        filters["area_range"] = r

# apply filters
filtered = df.copy()
for k, v in filters.items():
    if k == "area_range":
        lo, hi = v
        filtered = filtered[(filtered["area"] >= lo) & (filtered["area"] <= hi)]
    else:
        filtered = filtered[filtered[k] == v]

st.subheader("Filtered sample")
st.write(f"{filtered.shape[0]:,} rows after filters")
st.dataframe(filtered.head(50))

# -------------------------
# Numeric summary & plots
# -------------------------
st.subheader("Numeric summary")
if num_cols:
    st.write(filtered[num_cols].describe())
else:
    st.write("No numeric columns detected.")

# Distribution plot for chosen target (if exists)
if target_candidates:
    target = st.selectbox("Choose a target column for distribution / modeling", target_candidates)
    if target in filtered.columns:
        st.subheader(f"Distribution of `{target}`")
        fig, ax = plt.subplots()
        sns.histplot(filtered[target].dropna(), kde=True, ax=ax)
        ax.set_xlabel(target)
        st.pyplot(fig)
else:
    st.info("No target-like columns (price/rent) detected. The automatic model demo will be disabled.")

# Scatter: price vs area if both exist
if ("area" in filtered.columns) and target_candidates and (target in filtered.columns) and pd.api.types.is_numeric_dtype(filtered["area"]):
    st.subheader(f"{target} vs area")
    fig2, ax2 = plt.subplots()
    ax2.scatter(filtered["area"].dropna(), filtered[target].loc[filtered["area"].dropna().index], alpha=0.6)
    ax2.set_xlabel("area")
    ax2.set_ylabel(target)
    st.pyplot(fig2)

# Correlation heatmap (small subset for speed)
st.subheader("Correlation (numeric columns)")
if len(num_cols) >= 2:
    corr = filtered[num_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax3)
    st.pyplot(fig3)
else:
    st.write("Not enough numeric columns for correlation matrix.")

# -------------------------
# Quick model: predict target using numeric features
# -------------------------
st.subheader("Quick model: predict price (toy example)")
if target_candidates and (target in filtered.columns):
    # choose numeric features automatically (exclude target)
    X_all = filtered.select_dtypes(include=[np.number]).copy()
    if target in X_all.columns:
        X_all = X_all.drop(columns=[target], errors="ignore")
    y_all = filtered[target].dropna()
    X_all = X_all.loc[y_all.index]  # align

    if X_all.shape[0] >= 50 and X_all.shape[1] >= 1:
        # allow user to adjust model params
        st.sidebar.header("Model options")
        n_estimators = st.sidebar.slider("RandomForest n_estimators", 10, 200, 50)
        test_size = st.sidebar.slider("Test set %", 10, 50, 20) / 100.0

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        with st.spinner("Training model..."):
            model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        st.write(f"Trained RandomForestRegressor. MAE: **{mae:.2f}** on test set ({X_test.shape[0]} rows).")

        # feature importances
        fi = pd.Series(model.feature_importances_, index=X_all.columns).sort_values(ascending=False).head(20)
        st.subheader("Feature importances (top 20)")
        st.bar_chart(fi)
    else:
        st.info("Not enough numeric data to train the example model (need >=50 rows and at least 1 numeric feature).")
else:
    st.info("No price/rent-like target detected; quick model disabled.")

st.markdown("---")
st.write("Tips: convert `.xls` to `.xlsx` or `.csv` to avoid needing an older `xlrd` package. Uploaded files live only for the session; persist elsewhere if needed.")
