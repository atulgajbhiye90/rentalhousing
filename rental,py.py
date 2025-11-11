# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Rental Property Analysis", layout="wide")

st.title("🏠 Rental Property — Quick Analysis")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("rental_property_dataset_india.xls", engine="xlrd")
    except Exception as e:
        st.warning("Could not read rental_property_dataset_india.xls with xlrd. Trying openpyxl or csv.")
        try:
            df = pd.read_excel("rental_property_dataset_india.xlsx", engine="openpyxl")
        except Exception:
            try:
                df = pd.read_csv("rental_property_dataset_india.csv")
            except Exception:
                st.error("Dataset not found in repo root. Please upload rental_property_dataset_india.xls or a CSV.")
                return pd.DataFrame()
    return df

df = load_data()
if df.empty:
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head(20))

# Basic info
st.write("Shape:", df.shape)
st.write("Columns:", df.columns.tolist())

# Example: quick numeric summary
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    st.subheader("Numeric summary")
    st.write(df[num_cols].describe())

# Example plot: distribution of price_per_sqft if it exists
target_candidates = [c for c in df.columns if "price" in c.lower() or "rent" in c.lower()]
if target_candidates:
    target = target_candidates[0]
    st.subheader(f"Distribution of `{target}`")
    fig, ax = plt.subplots()
    sns.histplot(df[target].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

# Simple interactive filters (if columns exist)
st.sidebar.header("Filters")
filters = {}
if "city" in df.columns:
    cities = ["All"] + sorted(df["city"].dropna().unique().tolist())
    sel_city = st.sidebar.selectbox("City", cities)
    if sel_city != "All":
        filters["city"] = sel_city

if "property_type" in df.columns:
    types = ["All"] + sorted(df["property_type"].dropna().unique().tolist())
    sel_type = st.sidebar.selectbox("Property type", types)
    if sel_type != "All":
        filters["property_type"] = sel_type

filtered = df.copy()
for k, v in filters.items():
    filtered = filtered[filtered[k] == v]

st.subheader("Filtered sample")
st.dataframe(filtered.head(50))

# Quick model (if meaningful numeric target and some features exist)
st.subheader("Quick model: predict price (toy example)")
if target_candidates:
    target = target_candidates[0]
    # choose numeric features automatically
    X = filtered.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = filtered[target].dropna()
    # align X and y
    X = X.loc[y.index]
    if X.shape[0] >= 50 and X.shape[1] >= 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        st.write("Model trained on numeric columns. MAE:", round(mae, 2))
        st.write("Feature importances (top 10)")
        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        st.bar_chart(fi)
    else:
        st.info("Not enough numeric data to train the example model (need >=50 rows and at least 1 numeric feature).")

st.write("— End of quick demo —")
