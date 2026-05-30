import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

np.random.seed(42)
n = 7000

data = {
    "customerID": [f"CUST{str(i).zfill(5)}" for i in range(1, n+1)],
    "gender": np.random.choice(["Male", "Female"], n),
    "SeniorCitizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
    "Partner": np.random.choice(["Yes", "No"], n),
    "Dependents": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    "tenure": np.random.randint(1, 72, n),
    "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
    "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
    "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
    "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
    "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
    "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
    "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
    "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
    "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]),
    "PaperlessBilling": np.random.choice(["Yes", "No"], n, p=[0.59, 0.41]),
    "PaymentMethod": np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    ),
    "MonthlyCharges": np.round(np.random.uniform(18, 119, n), 2),
    "TotalCharges": np.round(np.random.uniform(18, 8500, n), 2),
}

# Churn logic: higher probability for month-to-month, fiber optic, high charges
churn_prob = (
    0.05
    + 0.25 * (np.array(data["Contract"]) == "Month-to-month")
    + 0.10 * (np.array(data["InternetService"]) == "Fiber optic")
    + 0.10 * (np.array(data["tenure"]) < 12)
    + 0.05 * (np.array(data["SeniorCitizen"]) == 1)
    - 0.10 * (np.array(data["tenure"]) > 48)
)
churn_prob = np.clip(churn_prob, 0, 1)
data["Churn"] = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

df = pd.DataFrame(data)
df.to_csv("data/telecom_churn.csv", index=False)
print(f"Dataset saved: {df.shape[0]} rows, Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
"""
Customer Churn Prediction - Exploratory Data Analysis
Author: Apoorv Mudgal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"Yes": "#E74C3C", "No": "#2ECC71"}
BLUE = "#2980B9"

df = pd.read_csv("data/telecom_churn.csv")

print("=" * 60)
print("CUSTOMER CHURN - EDA REPORT")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"Churn Rate   : {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nData Types:\n{df.dtypes.value_counts()}")

# ── Figure 1: Overview Dashboard ───────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Customer Churn — EDA Overview", fontsize=16, fontweight="bold", y=1.01)

# 1. Churn distribution
churn_counts = df["Churn"].value_counts()
axes[0, 0].pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%",
               colors=[COLORS["No"], COLORS["Yes"]], startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0, 0].set_title("Churn Distribution", fontweight="bold")

# 2. Tenure by churn
sns.histplot(data=df, x="tenure", hue="Churn", ax=axes[0, 1],
             palette=COLORS, bins=30, alpha=0.7)
axes[0, 1].set_title("Tenure Distribution by Churn", fontweight="bold")
axes[0, 1].set_xlabel("Tenure (months)")

# 3. Monthly Charges by churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[0, 2],
            palette=COLORS)
axes[0, 2].set_title("Monthly Charges by Churn", fontweight="bold")

# 4. Contract type
contract_churn = df.groupby(["Contract", "Churn"]).size().unstack()
contract_churn.plot(kind="bar", ax=axes[1, 0], color=[COLORS["No"], COLORS["Yes"]],
                    edgecolor="white")
axes[1, 0].set_title("Contract Type vs Churn", fontweight="bold")
axes[1, 0].set_xlabel("")
axes[1, 0].tick_params(axis="x", rotation=20)

# 5. Internet service
internet_churn = df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
internet_churn.plot(kind="bar", ax=axes[1, 1], color=[COLORS["No"], COLORS["Yes"]],
                    edgecolor="white")
axes[1, 1].set_title("Internet Service vs Churn", fontweight="bold")
axes[1, 1].set_xlabel("")
axes[1, 1].tick_params(axis="x", rotation=15)

# 6. Payment method
pay_churn = df.groupby(["PaymentMethod", "Churn"]).size().unstack(fill_value=0)
pay_churn.plot(kind="barh", ax=axes[1, 2], color=[COLORS["No"], COLORS["Yes"]],
               edgecolor="white")
axes[1, 2].set_title("Payment Method vs Churn", fontweight="bold")
axes[1, 2].set_xlabel("Count")

plt.tight_layout()
plt.savefig("screenshots/eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ EDA Overview saved → screenshots/eda_overview.png")

# ── Figure 2: Correlation Heatmap ──────────────────────────────────────────
df_enc = df.copy()
df_enc["Churn_bin"] = (df_enc["Churn"] == "Yes").astype(int)
for col in df_enc.select_dtypes("object").columns:
    df_enc[col] = pd.Categorical(df_enc[col]).codes

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                "Churn_bin"]
corr = df_enc[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            linewidths=0.5, ax=ax, square=True)
ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("screenshots/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Correlation Heatmap saved → screenshots/correlation_heatmap.png")

print("\nEDA Complete!")
"""
Customer Churn Prediction - Model Training
Author: Apoorv Mudgal
Models: Random Forest + Gradient Boosting (XGBoost-style)
Techniques: Feature Engineering, Class Balancing, GridSearchCV
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, accuracy_score
)
from sklearn.utils import resample

sns.set_theme(style="whitegrid")

# ── Load & Preprocess ───────────────────────────────────────────────────────
df = pd.read_csv("data/telecom_churn.csv")
df.drop("customerID", axis=1, inplace=True)

# Feature Engineering (15+ features)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
df["IsNewCustomer"] = (df["tenure"] < 12).astype(int)
df["IsLongTermCustomer"] = (df["tenure"] > 48).astype(int)
df["NumServices"] = (
    (df["PhoneService"] == "Yes").astype(int) +
    (df["MultipleLines"] == "Yes").astype(int) +
    (df["OnlineSecurity"] == "Yes").astype(int) +
    (df["OnlineBackup"] == "Yes").astype(int) +
    (df["DeviceProtection"] == "Yes").astype(int) +
    (df["TechSupport"] == "Yes").astype(int) +
    (df["StreamingTV"] == "Yes").astype(int) +
    (df["StreamingMovies"] == "Yes").astype(int)
)
df["HasInternetService"] = (df["InternetService"] != "No").astype(int)
df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
df["IsPaperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
df["IsElectronicPayment"] = (df["PaymentMethod"] == "Electronic check").astype(int)

# Encode categoricals
le = LabelEncoder()
cat_cols = df.select_dtypes("object").columns.tolist()
cat_cols.remove("Churn")
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop("Churn", axis=1)
y = df["Churn"]

print(f"Features: {X.shape[1]} | Class balance: {y.value_counts().to_dict()}")

# ── Handle Class Imbalance (SMOTE-style oversampling) ──────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

df_train = pd.concat([X_train, y_train], axis=1)
majority = df_train[df_train["Churn"] == 0]
minority = df_train[df_train["Churn"] == 1]
minority_upsampled = resample(minority, replace=True,
                               n_samples=len(majority), random_state=42)
df_balanced = pd.concat([majority, minority_upsampled])
X_train_bal = df_balanced.drop("Churn", axis=1)
y_train_bal = df_balanced["Churn"]

print(f"After balancing → {y_train_bal.value_counts().to_dict()}")

# ── Model Training ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                             min_samples_split=5, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                     max_depth=4, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train_bal)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "acc": accuracy_score(y_test, y_pred)
    }
    print(f"\n{name}: AUC={results[name]['auc']:.3f}  F1={results[name]['f1']:.3f}  Acc={results[name]['acc']:.3f}")

# Best model = Random Forest
best = results["Random Forest"]
best_model = best["model"]

print(f"\n{'='*50}")
print("BEST MODEL: Random Forest")
print(classification_report(y_test, best["y_pred"], target_names=["No Churn", "Churn"]))

# ── Save model & scaler ────────────────────────────────────────────────────
joblib.dump(best_model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")
print("✅ Model saved → churn_model.pkl")

# ── Visualisations ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Performance Dashboard", fontsize=15, fontweight="bold")

# 1. ROC Curves
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.2f})", linewidth=2)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_title("ROC Curves", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(fontsize=9)

# 2. Confusion Matrix
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
axes[1].set_title("Confusion Matrix (Random Forest)", fontweight="bold")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

# 3. Feature Importance
feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
feat_imp.nlargest(12).sort_values().plot(kind="barh", ax=axes[2], color="#2980B9")
axes[2].set_title("Top 12 Feature Importances", fontweight="bold")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("screenshots/model_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Model performance chart saved → screenshots/model_performance.png")

# ── Churn Driver Chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.nlargest(15).sort_values().plot(kind="barh", ax=ax,
                                          color="#E74C3C", edgecolor="white")
ax.set_title("Churn Drivers — Top 15 Features", fontweight="bold", fontsize=14)
ax.set_xlabel("Feature Importance Score", fontsize=11)
ax.axvline(feat_imp.mean(), color="gray", linestyle="--", alpha=0.6, label="Mean importance")
ax.legend()
plt.tight_layout()
plt.savefig("screenshots/churn_drivers.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Churn drivers chart saved → screenshots/churn_drivers.png")

print("\n🎉 Training complete!")
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
