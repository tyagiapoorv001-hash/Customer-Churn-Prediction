"""
Customer Churn Prediction Dashboard
Author: Apoorv Mudgal
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #2980B9;
    }
    .churn-high { color: #E74C3C; font-weight: bold; font-size: 1.2rem; }
    .churn-low  { color: #27AE60; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Assets ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_names.pkl")
    return model, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv("data/telecom_churn.csv")

model, scaler, feature_names = load_model()
df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔮 Predict Churn", "📈 EDA Insights"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Apoorv Mudgal**")
st.sidebar.markdown("B.Tech AI & Data Science | Batch 2028")
st.sidebar.markdown("[LinkedIn](https://linkedin.com/in/apoorv-mudgal) | [GitHub](https://github.com/tyagiapoorv001-hash)")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown("Telecom customer churn analysis on **7,000 records** using Random Forest + Gradient Boosting.")
    st.markdown("---")

    # KPI row
    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = total - churned
    churn_rate = churned / total * 100
    avg_tenure = df[df["Churn"] == "No"]["tenure"].mean()
    avg_charge = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned", f"{churned:,}", delta=f"{churn_rate:.1f}%", delta_color="inverse")
    c3.metric("Retained", f"{retained:,}")
    c4.metric("Avg Tenure (Retained)", f"{avg_tenure:.0f} mo")
    c5.metric("Avg Charge (Churned)", f"₹{avg_charge:.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ct = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
        ct.plot(kind="bar", ax=ax, color=["#27AE60", "#E74C3C"], edgecolor="white")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(["No Churn", "Churn"])
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Monthly Charges Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for label, color in [("No", "#27AE60"), ("Yes", "#E74C3C")]:
            subset = df[df["Churn"] == label]["MonthlyCharges"]
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label=f"{'No ' if label=='No' else ''}Churn")
        ax.set_xlabel("Monthly Charges (₹)")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Tenure vs Churn")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        df.boxplot(column="tenure", by="Churn", ax=ax,
                   boxprops=dict(color="#2980B9"),
                   medianprops=dict(color="#E74C3C", linewidth=2))
        ax.set_title("")
        plt.suptitle("")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Tenure (months)")
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Internet Service vs Churn Rate")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        isc = df.groupby("InternetService")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100)
        isc.plot(kind="bar", ax=ax, color="#9B59B6", edgecolor="white")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(isc):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
        st.pyplot(fig)
        plt.close()

    # Screenshots from training
    st.markdown("---")
    st.subheader("🤖 Model Performance")
    if os.path.exists("screenshots/model_performance.png"):
        st.image("screenshots/model_performance.png", use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Fill in customer details below to get a churn probability prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Customer Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 1, 72, 12)

    with col2:
        st.subheader("📡 Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_bkp = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_sup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_mv = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("💳 Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges (₹)", 18.0, 120.0, 65.0, step=0.5)
        total_charges = st.number_input("Total Charges (₹)", 18.0, 8500.0, monthly_charges * tenure, step=1.0)

    st.markdown("---")

    if st.button("🔮 Predict Churn Probability", type="primary"):
        # Build input dict
        input_dict = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp, "DeviceProtection": device_prot,
            "TechSupport": tech_sup, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mv, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
        }

        row = pd.DataFrame([input_dict])

        # Feature engineering (mirror train_model.py)
        row["AvgMonthlyCharge"] = row["TotalCharges"] / (row["tenure"] + 1)
        row["ChargeRatio"] = row["MonthlyCharges"] / (row["TotalCharges"] + 1)
        row["IsNewCustomer"] = (row["tenure"] < 12).astype(int)
        row["IsLongTermCustomer"] = (row["tenure"] > 48).astype(int)
        row["NumServices"] = (
            (row["PhoneService"] == "Yes").astype(int) +
            (row["MultipleLines"] == "Yes").astype(int) +
            (row["OnlineSecurity"] == "Yes").astype(int) +
            (row["OnlineBackup"] == "Yes").astype(int) +
            (row["DeviceProtection"] == "Yes").astype(int) +
            (row["TechSupport"] == "Yes").astype(int) +
            (row["StreamingTV"] == "Yes").astype(int) +
            (row["StreamingMovies"] == "Yes").astype(int)
        )
        row["HasInternetService"] = (row["InternetService"] != "No").astype(int)
        row["IsMonthToMonth"] = (row["Contract"] == "Month-to-month").astype(int)
        row["IsPaperless"] = (row["PaperlessBilling"] == "Yes").astype(int)
        row["IsElectronicPayment"] = (row["PaymentMethod"] == "Electronic check").astype(int)

        from sklearn.preprocessing import LabelEncoder
        for col in row.select_dtypes("object").columns:
            row[col] = LabelEncoder().fit_transform(row[col].astype(str))

        row = row[feature_names]
        prob = model.predict_proba(row)[0][1]
        pred = "Churn" if prob > 0.5 else "No Churn"

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob*100:.1f}%")
        c2.metric("Prediction", pred)
        c3.metric("Confidence", f"{'High' if abs(prob - 0.5) > 0.2 else 'Medium'}")

        if prob > 0.5:
            st.error(f"⚠️ **High Churn Risk ({prob*100:.1f}%)** — This customer is likely to churn. Consider offering a retention incentive.")
        else:
            st.success(f"✅ **Low Churn Risk ({prob*100:.1f}%)** — This customer is likely to stay.")

        # Gauge
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(["Churn Risk"], [prob], color="#E74C3C" if prob > 0.5 else "#27AE60",
                height=0.4)
        ax.barh(["Churn Risk"], [1 - prob], left=[prob], color="#ECF0F1", height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6, label="Threshold (0.5)")
        ax.legend(fontsize=8)
        ax.set_title(f"Churn Probability: {prob*100:.1f}%", fontweight="bold")
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA Insights":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Deep dive into the patterns driving customer churn.")

    if os.path.exists("screenshots/eda_overview.png"):
        st.image("screenshots/eda_overview.png", use_column_width=True)
    if os.path.exists("screenshots/correlation_heatmap.png"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image("screenshots/correlation_heatmap.png", use_column_width=True)
        with col2:
            st.subheader("Key Insights")
            st.markdown("""
- 📌 **Month-to-month contracts** have the highest churn rate (~40%)
- 📌 **Fiber optic internet** customers churn more than DSL users
- 📌 **New customers** (tenure < 12 months) are most at risk
- 📌 **Higher monthly charges** are positively correlated with churn
- 📌 **Electronic check** payment method shows highest churn rate
- 📌 Customers with **no online security** churn significantly more
- 📌 **Long-term customers** (tenure > 48 months) rarely churn
            """)

    if os.path.exists("screenshots/churn_drivers.png"):
        st.image("screenshots/churn_drivers.png", use_column_width=True)

    # Raw data preview
    st.markdown("---")
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"Showing 50 of {len(df):,} records")

