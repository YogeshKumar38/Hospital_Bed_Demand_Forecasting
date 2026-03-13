import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Bed Demand Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Hospital Bed Demand Prediction Dashboard")

# ── Session state ─────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "demo"

if "patients" not in st.session_state:
    st.session_state.patients  = pd.read_csv("data/patients.csv")
    st.session_state.services  = pd.read_csv("data/services_weekly.csv")
    st.session_state.staff     = pd.read_csv("data/staff.csv")
    st.session_state.schedule  = pd.read_csv("data/staff_schedule.csv")

# ── Sidebar – upload ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your Dataset")
    st.caption(
        "Upload all four CSV files to replace the demo data.  "
        "Expected columns are shown in the info boxes below each uploader."
    )

    patients_file  = st.file_uploader("patients.csv",          type="csv")
    services_file  = st.file_uploader("services_weekly.csv",   type="csv")
    staff_file     = st.file_uploader("staff.csv",             type="csv")
    schedule_file  = st.file_uploader("staff_schedule.csv",    type="csv")

    if st.button("🔄 Load Dataset", use_container_width=True):
        if not (patients_file and services_file and staff_file and schedule_file):
            st.error("Please upload all 4 CSV files before loading.")
        else:
            st.session_state.patients  = pd.read_csv(patients_file)
            st.session_state.services  = pd.read_csv(services_file)
            st.session_state.staff     = pd.read_csv(staff_file)
            st.session_state.schedule  = pd.read_csv(schedule_file)
            st.session_state.mode      = "user"
            st.success("Dataset loaded successfully!")

    st.divider()
    st.header("🎛️ Filters")

# ── Mode banner ───────────────────────────────────────────────────────────────
if st.session_state.mode == "demo":
    st.info(
        "ℹ️ **Demo mode** — using the built-in sample dataset.  "
        "Upload your own CSVs via the sidebar to analyse your hospital's data.",
        icon="📊",
    )

# ── Raw data ──────────────────────────────────────────────────────────────────
patients  = st.session_state.patients.copy()
services  = st.session_state.services.copy()
staff     = st.session_state.staff.copy()
schedule  = st.session_state.schedule.copy()

# Normalise column names
for df in [patients, services, staff, schedule]:
    df.columns = df.columns.str.lower().str.strip()

# ── Build analysis-ready dataframe (FIX: no row explosion) ───────────────────
# 1. Aggregate schedule to one row per (week, service): count present staff
sched_agg = (
    schedule
    .groupby(["week", "service"])
    .agg(
        staff_present=("present", "sum"),
        staff_total=("present", "count"),
    )
    .reset_index()
)
sched_agg["attendance_rate"] = (
    sched_agg["staff_present"] / sched_agg["staff_total"]
)

# 2. Aggregate patients to one row per service: avg age, avg satisfaction
pat_agg = (
    patients
    .groupby("service")
    .agg(
        avg_patient_age=("age", "mean"),
        avg_patient_satisfaction=("satisfaction", "mean"),
        total_patients=("patient_id", "count"),
    )
    .reset_index()
)

# 3. Merge services ← schedule ← patients
df = services.merge(sched_agg, on=["week", "service"], how="left")
df = df.merge(pat_agg, on="service", how="left")

# 4. Derived features
df["month"] = ((df["week"] - 1) // 4) + 1
df["refusal_rate"] = df["patients_refused"] / df["patients_request"].replace(0, np.nan)
df["utilisation"]  = df["patients_admitted"] / df["available_beds"].replace(0, np.nan)

df = df.fillna(0)

# ── Sidebar filters (applied after df is built) ───────────────────────────────
all_services = sorted(df["service"].unique().tolist())
with st.sidebar:
    selected_services = st.multiselect(
        "Services", all_services, default=all_services
    )
    week_range = st.slider(
        "Week range",
        int(df["week"].min()), int(df["week"].max()),
        (int(df["week"].min()), int(df["week"].max())),
    )

mask = (
    df["service"].isin(selected_services) &
    df["week"].between(*week_range)
)
df_filtered = df[mask].copy()

# ── ML model (cached so it doesn't rerun on every filter change) ──────────────
FEATURES = [
    "month", "available_beds", "patients_admitted",
    "patients_refused", "patient_satisfaction", "staff_morale",
    "avg_patient_age", "avg_patient_satisfaction",
    "staff_present", "attendance_rate", "refusal_rate", "utilisation",
]

@st.cache_resource(show_spinner="Training model…")
def train_model(data_hash):
    """Train once; re-train only when underlying data changes."""
    X = df[FEATURES].fillna(0)
    y = df["patients_request"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    r2     = r2_score(y_test, preds)
    rmse   = np.sqrt(mean_squared_error(y_test, preds))
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    return model, r2, rmse, importances, y_test, preds

# Use a simple hash of the data shape + first values so cache invalidates
# if the user uploads new data.
data_hash = (df.shape, tuple(df["patients_request"].head(10)))
model, r2, rmse, importances, y_test, preds = train_model(data_hash)

# ── KMeans clustering ─────────────────────────────────────────────────────────
cluster_features = ["patients_request", "patients_admitted",
                    "patients_refused", "refusal_rate"]
scaler  = StandardScaler()
scaled  = scaler.fit_transform(df_filtered[cluster_features].fillna(0))
kmeans  = KMeans(n_clusters=3, random_state=42, n_init=10)
df_filtered = df_filtered.copy()
df_filtered["cluster"] = kmeans.fit_predict(scaled)
df_filtered["cluster"] = df_filtered["cluster"].astype(str)

# ── Layout: KPI metrics ───────────────────────────────────────────────────────
st.subheader("📈 Model Performance")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("R² Score",       f"{r2:.3f}")
k2.metric("RMSE",           f"{rmse:.1f} patients")
k3.metric("Weeks Analysed", df_filtered["week"].nunique())
k4.metric("Total Requests", int(df_filtered["patients_request"].sum()))
k5.metric("Avg Utilisation",f"{df_filtered['utilisation'].mean():.1%}")

st.divider()

# ── Row 1: Demand trend + Refusal rate ───────────────────────────────────────
st.subheader("📅 Weekly Patient Demand by Service")
col1, col2 = st.columns(2)

with col1:
    fig_trend = px.line(
        df_filtered.sort_values("week"),
        x="week", y="patients_request",
        color="service",
        markers=True,
        labels={"patients_request": "Patients Requesting",
                "week": "Week", "service": "Service"},
        title="Patient Demand Trend",
    )
    fig_trend.update_layout(legend_title_text="Service")
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    fig_refused = px.area(
        df_filtered.sort_values("week"),
        x="week", y="patients_refused",
        color="service",
        labels={"patients_refused": "Patients Refused",
                "week": "Week", "service": "Service"},
        title="Refused Patients Over Time",
    )
    st.plotly_chart(fig_refused, use_container_width=True)

# ── Row 2: Utilisation heatmap + Cluster scatter ──────────────────────────────
col3, col4 = st.columns(2)

with col3:
    pivot = (
        df_filtered
        .pivot_table(index="service", columns="month",
                     values="utilisation", aggfunc="mean")
        .round(2)
    )
    fig_heat = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        labels={"color": "Utilisation"},
        title="Avg Bed Utilisation by Service & Month",
        aspect="auto",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col4:
    fig_cluster = px.scatter(
        df_filtered,
        x="patients_request", y="patients_refused",
        color="cluster",
        size="available_beds",
        hover_data=["service", "week"],
        labels={"patients_request": "Patients Requesting",
                "patients_refused": "Patients Refused",
                "cluster": "Segment"},
        title="Hospital Demand Segmentation (K-Means)",
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

# ── Row 3: Feature importance + Actual vs Predicted ──────────────────────────
col5, col6 = st.columns(2)

with col5:
    imp_df = (
        importances
        .sort_values(ascending=True)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
    )
    fig_imp = px.bar(
        imp_df, x=importances.sort_values().values,
        y=importances.sort_values().index,
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        title="Feature Importance (Random Forest)",
        color=importances.sort_values().values,
        color_continuous_scale="Blues",
    )
    fig_imp.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

with col6:
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=list(range(len(y_test))),
        y=y_test.values,
        mode="lines", name="Actual",
        line=dict(color="#636EFA"),
    ))
    fig_pred.add_trace(go.Scatter(
        x=list(range(len(preds))),
        y=preds,
        mode="lines", name="Predicted",
        line=dict(color="#EF553B", dash="dash"),
    ))
    fig_pred.update_layout(
        title="Actual vs Predicted Patient Requests (Test Set)",
        xaxis_title="Sample Index",
        yaxis_title="Patients Requesting",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# ── Row 4: Morale vs Satisfaction + Staff attendance ─────────────────────────
col7, col8 = st.columns(2)

with col7:
    fig_scatter2 = px.scatter(
        df_filtered,
        x="staff_morale", y="patient_satisfaction",
        color="service",
        trendline="ols",
        labels={"staff_morale": "Staff Morale",
                "patient_satisfaction": "Patient Satisfaction"},
        title="Staff Morale vs Patient Satisfaction",
    )
    st.plotly_chart(fig_scatter2, use_container_width=True)

with col8:
    fig_attend = px.bar(
        df_filtered.groupby("service")["attendance_rate"].mean().reset_index(),
        x="service", y="attendance_rate",
        color="service",
        labels={"attendance_rate": "Avg Attendance Rate", "service": "Service"},
        title="Average Staff Attendance Rate by Service",
        text_auto=".1%",
    )
    fig_attend.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig_attend, use_container_width=True)

# ── Interactive Predictor ─────────────────────────────────────────────────────
st.divider()
st.subheader("🔮 Predict Patient Demand")
st.caption("Adjust the sliders below to simulate a scenario and get an instant demand forecast.")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    p_month       = st.slider("Month",               1,  12, 6)
    p_avail_beds  = st.slider("Available Beds",       5,  80, 30)
    p_admitted    = st.slider("Patients Admitted",    0,  80, 25)
with pc2:
    p_refused     = st.slider("Patients Refused",     0, 300, 50)
    p_pat_sat     = st.slider("Patient Satisfaction", 60, 99, 80)
    p_morale      = st.slider("Staff Morale",         40, 99, 75)
with pc3:
    p_avg_age     = st.slider("Avg Patient Age",      0,  90, 40)
    p_avg_psat    = st.slider("Avg Patient Sat.",     60, 99, 78)
    p_present     = st.slider("Staff Present",         1,  30, 15)
with pc4:
    p_attend_rate = st.slider("Attendance Rate",    0.0, 1.0, 0.8)
    p_refusal_rt  = st.slider("Refusal Rate",       0.0, 1.0, 0.3)
    p_util        = st.slider("Utilisation",        0.0, 1.0, 0.8)

input_vec = np.array([[
    p_month, p_avail_beds, p_admitted, p_refused,
    p_pat_sat, p_morale, p_avg_age, p_avg_psat,
    p_present, p_attend_rate, p_refusal_rt, p_util,
]])
predicted_demand = model.predict(input_vec)[0]

st.metric(
    "📊 Predicted Patient Requests",
    f"{predicted_demand:.0f} patients",
    help="Estimated number of patients who will request beds under these conditions.",
)

# ── Sidebar download ──────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.header("💾 Export")
    st.download_button(
        "Download Processed Data",
        df_filtered.to_csv(index=False),
        "processed_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
