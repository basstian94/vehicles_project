"""
App Streamlit: Análisis Exploratorio de Datos (EDA)
Dataset: vehicles_us.csv  (anuncios de vehículos usados en EE.UU.)

Ejecutar:
    streamlit run app.py
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Page and style settings
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EDA · Used Vehicles",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def find_file(name: str, start: Path = Path.cwd(), max_up: int = 5) -> Path:
    current = start.resolve()
    for _ in range(max_up + 1):
        candidate = current / name
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        f"File {name!r} not found searching upwards from {start}")


@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df["is_4wd"] = df["is_4wd"].fillna(0).astype(bool)
    df["cylinders"] = df["cylinders"].astype("Int64")
    df["model_year"] = df["model_year"].astype("Int64")
    return df


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    return (
        pd.DataFrame({"nulls": miss, "pct": pct})
        .query("nulls > 0")
        .sort_values("pct", ascending=False)
    )


def iqr_outlier_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def fig_to_st(fig):
    st.pyplot(fig)
    plt.close(fig)


try:
    CSV_PATH = find_file("vehicles_us.csv")
except FileNotFoundError:
    st.error(
        "`vehicles_us.csv` not found. "
        "Place it in the same folder as `app.py` or adjust the path in the code."
    )
    st.stop()

df_raw = load_data(str(CSV_PATH))


st.sidebar.title("🚗 EDA · Used Vehicles")
st.sidebar.caption(f"File: `{CSV_PATH.name}`")

section = st.sidebar.radio(
    "Section",
    [
        "1. Overview",
        "2. Data quality",
        "3. Cleaning",
        "4. Univariate",
        "5. Bivariate",
        "6. Multivariate",
        "7. Conclusions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global filters")

min_price = int(df_raw["price"].min())
max_price = int(df_raw["price"].max())
price_range = st.sidebar.slider(
    "Price range (USD)",
    min_value=0,
    max_value=min(max_price, 100_000),
    value=(0, min(max_price, 100_000)),
    step=500,
)

available_types = sorted(df_raw["type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Vehicle types",
    options=available_types,
    default=available_types,
)

available_fuels = sorted(df_raw["fuel"].dropna().unique().tolist())
selected_fuels = st.sidebar.multiselect(
    "Fuel",
    options=available_fuels,
    default=available_fuels,
)

df = df_raw.query(
    "@price_range[0] <= price <= @price_range[1] "
    "and type in @selected_types "
    "and fuel in @selected_fuels"
).copy()

st.sidebar.markdown("---")
st.sidebar.metric("Rows after filters", f"{len(df):,}")
st.sidebar.metric("Total rows", f"{len(df_raw):,}")


# ---------------------------------------------------------------------------
# 1. General view
# ---------------------------------------------------------------------------
if section == "1. Overview":
    st.title("📊 Exploratory Analysis · Used Vehicles")
    st.markdown(
        "Analysis of the US used vehicle listings dataset. "
        "Explore the structure, quality, and main patterns."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df_raw):,}")
    col2.metric("Columns", df_raw.shape[1])
    col3.metric("Median price", f"${df_raw['price'].median():,.0f}")
    col4.metric("Days listed (med)", f"{df_raw['days_listed'].median():.0f}")

    st.subheader("First rows")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Descriptive statistics")
    st.dataframe(df_raw.describe(include="all").T, use_container_width=True)

    st.subheader("Data types")
    data_types = pd.DataFrame(
        {
            "type": df_raw.dtypes.astype(str),
            "non_nulls": df_raw.notna().sum(),
            "nulls": df_raw.isna().sum(),
            "unique": df_raw.nunique(),
        }
    )
    st.dataframe(data_types, use_container_width=True)


# ---------------------------------------------------------------------------
# 2. Data quality
# ---------------------------------------------------------------------------
elif section == "2. Data quality":
    st.title("🧹 Data quality")

    st.subheader("Null values by column")
    miss = missing_summary(df_raw)
    if miss.empty:
        st.success("No null values found.")
    else:
        st.dataframe(miss, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(miss))))
        sns.barplot(x=miss["pct"], y=miss.index, ax=ax, color="steelblue")
        ax.set_xlabel("% of nulls")
        ax.set_ylabel("")
        ax.set_title("Percentage of nulls by column")
        fig_to_st(fig)

    st.subheader("Duplicates")
    st.write(f"Exact duplicate rows: **{df_raw.duplicated().sum()}**")

    st.subheader("Cardinality (unique values)")
    card = pd.DataFrame(
        {
            "unique": df_raw.nunique(),
            "examples": [
                df_raw[c].dropna().unique()[:5].tolist() for c in df_raw.columns
            ],
        }
    )
    st.dataframe(card, use_container_width=True)

    st.subheader("Outliers detected (IQR rule)")
    for col in ["price", "odometer"]:
        low, high = iqr_outlier_bounds(df_raw[col].dropna())
        n_out = ((df_raw[col] < low) | (df_raw[col] > high)).sum()
        st.write(
            f"**{col}**: bounds = ({low:,.0f}, {high:,.0f}) · "
            f"outliers = **{n_out:,}** ({n_out / df_raw[col].notna().sum() * 100:.2f} %)"
        )


# ---------------------------------------------------------------------------
# 3. Cleanning
# ---------------------------------------------------------------------------
elif section == "3. Cleaning":
    st.title("🧽 Applied cleaning")
    st.markdown(
        """
        Decisions made:

        1. `date_posted` converted to `datetime`.
        2. `is_4wd`: `1.0 → True`, `NaN → False` (absence of flag is interpreted
           as "not 4x4").
        3. `cylinders` and `model_year` converted to nullable integers (`Int64`).
        4. Conservative outlier filter:
           - `0 < price < 100_000`
           - `0 < odometer < 500_000`
        """
    )

    df_analysis = df_raw.query(
        "0 < price < 100_000 and 0 < odometer < 500_000").copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Original rows", f"{len(df_raw):,}")
    col2.metric("Rows after cleaning", f"{len(df_analysis):,}")
    col3.metric(
        "% removed",
        f"{(1 - len(df_analysis) / len(df_raw)) * 100:.2f} %",
    )

    st.subheader("Before vs after price filter")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df_raw["price"], bins=60, ax=axes[0], color="indianred")
    axes[0].set_title("Price (original)")
    axes[0].set_xlabel("USD")

    sns.histplot(df_analysis["price"], bins=60, ax=axes[1], color="seagreen")
    axes[1].set_title("Price (after cleaning)")
    axes[1].set_xlabel("USD")

    plt.tight_layout()
    fig_to_st(fig)

    st.subheader("Before vs after odometer filter")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df_raw["odometer"].dropna(), bins=60,
                 ax=axes[0], color="indianred")
    axes[0].set_title("Odometer (original)")
    axes[0].set_xlabel("Miles")

    sns.histplot(df_analysis["odometer"], bins=60,
                 ax=axes[1], color="seagreen")
    axes[1].set_title("Odometer (after cleaning)")
    axes[1].set_xlabel("Miles")

    plt.tight_layout()
    fig_to_st(fig)


# ---------------------------------------------------------------------------
# 4. Univariate
# ---------------------------------------------------------------------------
elif section == "4. Univariate":
    st.title("📈 Univariate analysis")

    if df.empty:
        st.warning("Current filters leave no rows to analyze.")
        st.stop()

    # --- Price ---
    st.subheader("Price distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["price"], bins=50, kde=True, ax=axes[0])
    axes[0].set_title("Histogram + KDE")
    axes[0].set_xlabel("Price (USD)")

    sns.boxplot(x=df["price"], ax=axes[1])
    axes[1].set_title("Boxplot")
    axes[1].set_xlabel("Price (USD)")

    plt.tight_layout()
    fig_to_st(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Median", f"${df['price'].median():,.0f}")
    col2.metric("Mean", f"${df['price'].mean():,.0f}")
    col3.metric("Standard deviation", f"${df['price'].std():,.0f}")
    st.caption(
        "The distribution is right-skewed: the mean is higher than the median. "
        "Therefore, reporting the median is recommended."
    )

    # --- Odometer ---
    st.subheader("Odometer distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["odometer"].dropna(), bins=50, kde=True, ax=ax)
    ax.set_title("Miles driven")
    ax.set_xlabel("Miles")
    fig_to_st(fig)

    # --- Days listed ---
    st.subheader("Days listed")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["days_listed"], bins=50, kde=True, ax=ax)
    ax.set_title("Days listed")
    ax.set_xlabel("Days")
    fig_to_st(fig)

    # --- Categories ---
    st.subheader("Categorical variables")
    cat_cols = ["condition", "fuel", "transmission", "type", "cylinders"]
    cols = st.columns(2)

    for i, col in enumerate(cat_cols):
        with cols[i % 2]:
            counts = df[col].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(x=counts.values, y=counts.index,
                        ax=ax, color="steelblue")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel("Count")
            ax.set_ylabel("")
            plt.tight_layout()
            fig_to_st(fig)

    # --- Dates ---
    st.subheader("Listings published by month")
    ts = df.set_index("date_posted").resample("ME").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    ts.plot(ax=ax, marker="o")
    ax.set_title("Listings published by month")
    ax.set_xlabel("Date")
    ax.set_ylabel("No. of listings")
    fig_to_st(fig)


# ---------------------------------------------------------------------------
# 5. Bivariate
# ---------------------------------------------------------------------------
elif section == "5. Bivariate":
    st.title("🔎 Bivariate analysis")

    if df.empty:
        st.warning("Current filters leave no rows to analyze.")
        st.stop()

    # --- Price vs year ---
    st.subheader("Price vs Model Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="model_year", y="price", alpha=0.2, ax=ax)
    ax.set_title("Price vs Model Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price (USD)")
    fig_to_st(fig)
    st.caption(
        "Positive relationship: higher year (newer), higher price. "
        "Classic vehicles (very low years) break the trend."
    )

    # --- Price vs Odometer ---
    st.subheader("Price vs Odometer")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="odometer", y="price", alpha=0.2, ax=ax)
    ax.set_title("Price vs Odometer")
    ax.set_xlabel("Miles")
    ax.set_ylabel("Price (USD)")
    fig_to_st(fig)
    st.caption("Negative relationship: more miles → lower price.")

    # --- Price by Condition ---
    st.subheader("Price by Condition")
    order = ["salvage", "fair", "good", "excellent", "like new", "new"]
    order = [c for c in order if c in df["condition"].unique()]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="condition", y="price", order=order, ax=ax)
    ax.set_title("Price by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Price (USD)")
    plt.xticks(rotation=30)
    fig_to_st(fig)
    st.caption(
        "Median rises with condition, but there is significant overlap. "
        "Condition alone does not explain the price."
    )

    # --- Price by Vehicle Type ---
    st.subheader("Price by Vehicle Type")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="type", y="price", ax=ax)
    ax.set_title("Price by Vehicle Type")
    ax.set_xlabel("Type")
    ax.set_ylabel("Price (USD)")
    plt.xticks(rotation=30)
    fig_to_st(fig)


# ---------------------------------------------------------------------------
# 6. Multivariate
# ---------------------------------------------------------------------------
elif section == "6. Multivariate":
    st.title("🧩 Multivariate analysis")

    if df.empty:
        st.warning("Current filters leave no rows to analyze.")
        st.stop()

    # --- Correlation matrix ---
    st.subheader("Correlation matrix")
    num_cols = ["price", "model_year", "odometer", "cylinders", "days_listed"]
    num_cols = [c for c in num_cols if c in df.columns]

    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlations (Pearson)")
    fig_to_st(fig)

    # --- Median price by decade and vehicle type ---
    st.subheader("Median price by decade and vehicle type")
    pivot = (
        df.assign(decade=(df["model_year"] // 10) * 10)
        .groupby(["decade", "type"])["price"]
        .median()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(marker="o", ax=ax)
    ax.set_title("Median price by decade and type")
    ax.set_xlabel("Model decade")
    ax.set_ylabel("Median price (USD)")
    ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig_to_st(fig)

# ---------------------------------------------------------------------------
# 7. Conclusion
# ---------------------------------------------------------------------------
elif section == "7. Conclusions":
    st.title("✅ Findings and decisions")

    if df.empty:
        st.warning("Current filters leave no rows to analyze.")
        st.stop()

    st.markdown(
        f"""
        ### Analyzed dataset summary
        - **Original rows:** {len(df_raw):,}
        - **Rows after filters:** {len(df):,}
        - **Median price:** ${df['price'].median():,.0f}         - **Mean price:**${df['price'].mean():,.0f}
        - **Median odometer:** {df['odometer'].median():,.0f} miles
        - **Days listed (median):** {df['days_listed'].median():.0f} days
        """
    )

    st.subheader("Top 5 types by volume")
    st.dataframe(
        df["type"].value_counts().head(5).rename("count").to_frame(),
        use_container_width=True,
    )

    st.subheader("Top 5 makes by volume (first word of model)")
    makes = df["model"].str.split().str[0].value_counts().head(5)
    st.dataframe(makes.rename("count").to_frame(), use_container_width=True)

    st.subheader("Key findings")
    st.markdown(
        """
        1. **Price is right-skewed.** Report **median** and IQR,
           not just the mean.
        2. **Year and odometer** are the strongest predictors of price
           (~0.5 absolute correlations).
        3. **Condition alone does not discriminate well**: there is significant overlap.
           It is best combined with year and odometer.
        4. **`is_4wd` came as a `float` with ~50% nulls.** The absence of flag
           was interpreted as "not 4WD".
        5. **There are clearly erroneous price outliers** (`$1`, `$999999`) that
           should be handled before modeling.
        6. **The dataset covers ~1 year of listings** (late 2018 – mid-2019).
           Be careful when generalizing to other seasons.
        """
    )

    st.subheader("Next steps")
    st.markdown(
        """
        - **Feature engineering:** extract make from `model`, calculate vehicle age
          (`current_year - model_year`), normalize odometer.
        - **Modeling:** linear regression or gradient boosting to predict `price`.
        - **Reproducible report:** export this analysis to a report
          (`st.download_button` with a summary CSV or HTML).
        """
    )

    # Botón de descarga del dataset filtrado
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered dataset as CSV",
        data=csv_bytes,
        file_name=f"vehicles_filtered_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Active filters: price [{price_range[0]:,}–{price_range[1]:,}], "
    f"types {len(selected_types)}/{len(available_types)}, "
    f"fuels {len(selected_fuels)}/{len(available_fuels)}"
)
