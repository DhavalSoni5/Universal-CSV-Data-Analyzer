import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import warnings

warnings.filterwarnings(
    "ignore",
    message="Could not infer format",
    category=UserWarning
)


# ===============================
# STAGE 3: SESSION STATE INIT
# ===============================
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = False

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Universal CSV Analyzer", layout="wide")

st.title("📊 Universal CSV Data Analyzer")
st.write("Upload any CSV file. Filter, analyze, visualize, and explore time-series data.")

# ===============================
# SIDEBAR – FILE UPLOAD
# ===============================
st.sidebar.header("📂 Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type="csv",
    help="Upload any CSV file. The app supports messy data and duplicate columns."
)


if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()

# ===============================
# READ CSV
# ===============================
try:
    df = pd.read_csv(uploaded_file)
    # Save uploaded data to session
    st.session_state.uploaded_df = df.copy()

except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# ===============================
# FIX DUPLICATE COLUMN NAMES (SAFE)
# ===============================
new_cols = []
seen = {}

for col in df.columns:
    if col in seen:
        seen[col] += 1
        new_cols.append(f"{col}_{seen[col]}")
    else:
        seen[col] = 0
        new_cols.append(col)

df.columns = new_cols

# ===============================
# DATASET OVERVIEW
# ===============================
st.subheader("ℹ️ Dataset Overview")
c1, c2 = st.columns(2)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])

# ===============================
# SIDEBAR – ROW IDENTIFIER
# ===============================
st.sidebar.header("🆔 Row Settings")

row_id = st.sidebar.selectbox(
    "Select row identifier (optional)",
    ["None"] + list(df.columns),
    help="Choose a column to use as row index (optional). Useful for IDs."
)


if row_id != "None":
    df = df.set_index(row_id)

# ===============================
# SIDEBAR – FILTERS
# ===============================
st.sidebar.header("🔍 Filters")

filtered_df = df.copy()

# ---- Numeric Filter
numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

if numeric_cols:
    num_col = st.sidebar.selectbox(
    "Numeric filter column",
    numeric_cols,
    help="Filter rows based on a numeric column range."
)

    min_val = float(filtered_df[num_col].min())
    max_val = float(filtered_df[num_col].max())

    value_range = st.sidebar.slider(
        "Numeric range",
        min_val,
        max_val,
        (min_val, max_val)
    )

    filtered_df = filtered_df[
        (filtered_df[num_col] >= value_range[0]) &
        (filtered_df[num_col] <= value_range[1])
    ]

# ---- Categorical Filter
cat_cols = filtered_df.select_dtypes(include="object").columns.tolist()

if cat_cols:
    cat_col = st.sidebar.selectbox(
    "Category filter column",
    cat_cols,
    help="Filter rows by selecting one or more category values."
)

    categories = filtered_df[cat_col].dropna().unique().tolist()

    selected_categories = st.sidebar.multiselect(
        "Select categories",
        categories,
        default=categories
    )

    filtered_df = filtered_df[filtered_df[cat_col].isin(selected_categories)]
    # Mark filters applied
    st.session_state.filters_applied = True

# ===============================
# SHOW FILTERED DATA
# ===============================
st.subheader("📄 Filtered Data")
st.dataframe(filtered_df)

if filtered_df.empty:
    st.warning("No data available after filters.")
    st.stop()
  # ===============================
# SESSION STATUS
# ===============================
if st.session_state.uploaded_df is not None:
    st.success("✅ Dataset loaded and preserved in session")

if st.session_state.filters_applied:
    st.info("🔄 Filters are active and remembered")

# ===============================
# BASIC STATISTICS
# ===============================
st.subheader("📌 Statistics")

analysis_cols = filtered_df.select_dtypes(include="number").columns.tolist()
selected_col = st.selectbox(
    "Select numeric column for analysis",
    analysis_cols,
    help="This column will be used for statistics and charts."
)

selected_col = st.selectbox("Select numeric column for analysis", analysis_cols)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Mean", f"{filtered_df[selected_col].mean():.2f}")
s2.metric("Median", f"{filtered_df[selected_col].median():.2f}")
s3.metric("Min", f"{filtered_df[selected_col].min():.2f}")
s4.metric("Max", f"{filtered_df[selected_col].max():.2f}")

# ===============================
# HISTOGRAM (MEDIUM SIZE)
# ===============================
st.subheader("📈 Distribution")

fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.hist(filtered_df[selected_col].dropna(), bins=15)
ax1.set_xlabel(selected_col)
ax1.set_ylabel("Count")
ax1.set_title(f"Distribution of {selected_col}")
st.pyplot(fig1)

# ===============================
# SCATTER PLOT (MEDIUM SIZE)
# ===============================
st.subheader("🔵 Scatter Plot")

if len(analysis_cols) >= 2:
    x_col = st.selectbox("X-axis", analysis_cols, index=0)
    y_col = st.selectbox("Y-axis", analysis_cols, index=1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(filtered_df[x_col], filtered_df[y_col], alpha=0.7)
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    ax2.set_title(f"{x_col} vs {y_col}")
    st.pyplot(fig2)

# ===============================
# TIME-SERIES (FINAL FIX)
# ===============================
st.subheader("⏱️ Time-Series (Animated / Interactive)")

# IMPORTANT: reset index to avoid duplicate label issues
ts_base = filtered_df.reset_index()

date_columns = []
for col in ts_base.columns:
    try:
        pd.to_datetime(ts_base[col], errors="raise")
        date_columns.append(col)
    except:
        pass

if date_columns and analysis_cols:
    # ---- AUTO-DETECT BEST DATE COLUMN ----
    def detect_best_date_column(df, candidates):
     for col in candidates:
        parsed = pd.to_datetime(df[col], errors="coerce")
        # choose column with max valid dates
        if parsed.notna().sum() > len(df) * 0.6:
            return col
     return candidates[0]

    auto_date_col = detect_best_date_column(ts_base, date_columns)

    date_col = st.selectbox(
    "Select date/time column",
    date_columns,
    help="Select a column containing date or time values."
)

    value_col = st.selectbox(
    "Select numeric column",
    analysis_cols,
    help="This numeric column will be plotted over time."
)
# ---- VALIDATION: prevent same column selection ----
if date_col == value_col:
    st.warning(
        "⚠️ Please select DIFFERENT columns for date/time and numeric value."
    )
    st.stop()

    ts_df = ts_base.copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df = ts_df[[date_col, value_col]].dropna()
    ts_df = ts_df.loc[:, [date_col, value_col]].copy()
    # ---- BUILD TIME-SERIES DATAFRAME (DUPLICATE-SAFE) ----
    ts_df = ts_base[[date_col, value_col]].copy()

    # convert date column safely
    ts_df.iloc[:, 0] = pd.to_datetime(ts_df.iloc[:, 0], errors="coerce")

    # drop invalid rows
    ts_df = ts_df.dropna()

    
 # ---- FINAL DUPLICATE-SAFE SORT (ABSOLUTE FIX) ----
    ts_df = ts_df.assign(
    __sort_key__=ts_df.iloc[:, 0].values
    ).sort_values("__sort_key__").drop(columns="__sort_key__")


    chart = (
        alt.Chart(ts_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{date_col}:T", title=date_col),
            y=alt.Y(f"{value_col}:Q", title=value_col),
            tooltip=[date_col, value_col]
        )
        .properties(width=600, height=350)
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.info(
    "ℹ️ Time-series requires a date/time column. "
    "No valid date column was detected in this dataset."
)


# ===============================
# DOWNLOAD DATA
# ===============================
st.subheader("⬇️ Download Filtered Data")

csv_data = filtered_df.reset_index().to_csv(index=False)
st.download_button(
    "Download CSV",
    csv_data,
    "filtered_data.csv",
    "text/csv",
    key="download_tab_csv"
)

# ===============================
# PHASE 1 UI POLISH (TABS)
# ===============================
st.divider()
st.subheader("🧭 Dashboard View (Phase-1 UI)")

tab1, tab2, tab3 = st.tabs(["📄 Data", "📊 Charts", "⏱️ Time-Series"])

# ---- TAB 1 : DATA ----
with tab1:
    st.subheader("📄 Filtered Data")
    st.dataframe(filtered_df)

    st.subheader("⬇️ Download Filtered Data")
    csv_data = filtered_df.reset_index().to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv_data,
        "filtered_data.csv",
        "text/csv"
    )

# ---- TAB 2 : CHARTS ----
with tab2:
    st.subheader("📌 Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{filtered_df[selected_col].mean():.2f}")
    c2.metric("Median", f"{filtered_df[selected_col].median():.2f}")
    c3.metric("Min", f"{filtered_df[selected_col].min():.2f}")
    c4.metric("Max", f"{filtered_df[selected_col].max():.2f}")

    st.subheader("📈 Distribution")
    fig_t1, ax_t1 = plt.subplots(figsize=(6, 4))
    ax_t1.hist(filtered_df[selected_col].dropna(), bins=15)
    ax_t1.set_title(f"Distribution of {selected_col}")
    st.pyplot(fig_t1)

    if len(analysis_cols) >= 2:
        st.subheader("🔵 Scatter Plot")
        fig_t2, ax_t2 = plt.subplots(figsize=(6, 4))
        ax_t2.scatter(filtered_df[x_col], filtered_df[y_col], alpha=0.7)
        ax_t2.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig_t2)

# ---- TAB 3 : TIME-SERIES ----
with tab3:
    if 'ts_df' in locals():
        st.subheader("⏱️ Time-Series (Animated)")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Time-series not available for current selection.")
# ===============================
# STAGE 1: COLUMN INSIGHT PANEL
# ===============================
st.divider()
st.subheader("🔎 Column Insight Panel")

col_to_inspect = st.selectbox(
    "Select a column to inspect",
    filtered_df.columns
)

col_data = filtered_df[col_to_inspect]

c1, c2, c3, c4 = st.columns(4)

# ---- BASIC INFO ----
c1.metric("Data Type", str(col_data.dtype))
c2.metric("Total Rows", len(col_data))

# ---- MISSING VALUES ----
missing_count = col_data.isna().sum()
missing_pct = (missing_count / len(col_data)) * 100
c3.metric("Missing Values", f"{missing_count} ({missing_pct:.1f}%)")

# ---- UNIQUE VALUES ----
unique_count = col_data.nunique(dropna=True)
c4.metric("Unique Values", unique_count)

# ---- NUMERIC INSIGHTS ----
if pd.api.types.is_numeric_dtype(col_data):
    st.markdown("**📐 Numeric Summary**")
    n1, n2 = st.columns(2)
    n1.metric("Minimum", f"{col_data.min():.2f}")
    n2.metric("Maximum", f"{col_data.max():.2f}")
else:
    st.info("ℹ️ Selected column is not numeric. Numeric summary not available.")
# ===============================
# STAGE 2: CORRELATION HEATMAP
# ===============================
st.divider()
st.subheader("🔥 Correlation Heatmap (Numeric Columns)")

numeric_df = filtered_df.select_dtypes(include="number")

if numeric_df.shape[1] < 2:
    st.info("Correlation heatmap requires at least two numeric columns.")
else:
    corr_matrix = numeric_df.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
    cax = ax_corr.matshow(corr_matrix)

    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="left")
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

    fig_corr.colorbar(cax)
    ax_corr.set_title("Correlation Heatmap", pad=20)

    st.pyplot(fig_corr)
# ===============================
# STAGE 4: OUTLIER DETECTION (IQR)
# ===============================
st.divider()
st.subheader("🚨 Outlier Detection (IQR Method)")

numeric_cols_od = filtered_df.select_dtypes(include="number").columns.tolist()

if not numeric_cols_od:
    st.info("No numeric columns available for outlier detection.")
else:
    outlier_col = st.selectbox(
        "Select numeric column for outlier detection",
        numeric_cols_od
    )

    col_data = filtered_df[outlier_col].dropna()

    # ---- IQR CALCULATION ----
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_df = filtered_df[
        (filtered_df[outlier_col] < lower_bound) |
        (filtered_df[outlier_col] > upper_bound)
    ]

    # ---- METRICS ----
    c1, c2, c3 = st.columns(3)
    c1.metric("Lower Bound", f"{lower_bound:.2f}")
    c2.metric("Upper Bound", f"{upper_bound:.2f}")
    c3.metric("Outliers Found", len(outliers_df))

    # ---- SHOW OUTLIERS ----
    if outliers_df.empty:
        st.success("✅ No outliers detected for this column.")
    else:
        st.warning("⚠️ Outliers detected")
        st.dataframe(outliers_df)

    # ---- OPTIONAL VISUALIZATION ----
    st.subheader("📊 Outlier Visualization")

    fig_out, ax_out = plt.subplots(figsize=(6, 4))
    ax_out.boxplot(col_data, vert=False)
    ax_out.set_title(f"Boxplot for {outlier_col}")
    st.pyplot(fig_out)
# ===============================
# STAGE 5: TIME-SERIES ENHANCEMENTS
# ===============================
st.divider()
st.subheader("📉 Time-Series Enhancements")

# This stage depends on ts_df from your existing time-series logic
if 'ts_df' not in locals() or ts_df.empty:
    st.info("Time-series data not available for enhancements.")
else:
    # ---- USER CONTROLS ----
    show_ma = st.checkbox("Show Rolling Average")
    show_trend = st.checkbox("Show Trend Line")

    ma_window = 7
    if show_ma:
        ma_window = st.slider(
            "Rolling window size",
            min_value=3,
            max_value=30,
            value=7
        )

    enhanced_df = ts_df.copy()
    date_col_enh = enhanced_df.columns[0]
    value_col_enh = enhanced_df.columns[1]

    # ---- ROLLING AVERAGE ----
    if show_ma:
        enhanced_df["Rolling_Avg"] = (
            enhanced_df[value_col_enh]
            .rolling(window=ma_window)
            .mean()
        )

    # ---- BASE LINE ----
    base_chart = alt.Chart(enhanced_df).mark_line(point=True).encode(
        x=alt.X(f"{date_col_enh}:T", title=date_col_enh),
        y=alt.Y(f"{value_col_enh}:Q", title=value_col_enh),
        tooltip=[date_col_enh, value_col_enh]
    )

    charts = base_chart

    # ---- ROLLING AVERAGE LINE ----
    if show_ma:
        ma_chart = alt.Chart(enhanced_df).mark_line(
            color="orange"
        ).encode(
            x=alt.X(f"{date_col_enh}:T"),
            y=alt.Y("Rolling_Avg:Q", title=f"{ma_window}-period Rolling Avg")
        )
        charts = charts + ma_chart

    # ---- TREND LINE ----
    if show_trend:
        trend_chart = alt.Chart(enhanced_df).mark_line(
            color="red"
        ).encode(
            x=alt.X(f"{date_col_enh}:T"),
            y=alt.Y(f"{value_col_enh}:Q")
        ).transform_regression(
            date_col_enh,
            value_col_enh
        )
        charts = charts + trend_chart

    st.altair_chart(
        charts.properties(width=700, height=400),
        use_container_width=True
    )
# ===============================
# STAGE 6: PERFORMANCE & UX POLISH
# ===============================
st.divider()
st.subheader("⚙️ App Status & Performance")

# ---- DATASET SIZE WARNING ----
rows, cols = filtered_df.shape

c1, c2 = st.columns(2)
c1.metric("Current Rows", rows)
c2.metric("Current Columns", cols)

if rows > 100_000:
    st.warning(
        "⚠️ Large dataset detected. "
        "Performance may be slower. Consider applying filters."
    )
else:
    st.success("✅ Dataset size is within safe limits.")

# ---- LOADING SPINNER DEMO (REAL USE CASE) ----
with st.spinner("🔄 Preparing analytics..."):
    # lightweight no-op to show spinner during reruns
    pass

# ---- SESSION STATE STATUS ----
st.subheader("🧠 Session Status")

if "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None:
    st.success("📂 Dataset is loaded and preserved in session.")
else:
    st.info("ℹ️ No dataset stored in session yet.")

if "filters_applied" in st.session_state and st.session_state.filters_applied:
    st.info("🔍 Filters are active.")
else:
    st.info("ℹ️ No filters applied.")

# ---- RESET SESSION BUTTON ----
st.subheader("♻️ Reset Application")

if st.button("Reset Session & Reload App"):
    st.session_state.clear()
    st.rerun()
