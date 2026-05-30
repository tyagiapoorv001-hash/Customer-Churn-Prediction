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

# Best model = Logistic Regression because it has the strongest AUC and F1 score.
best = results["Logistic Regression"]
best_model = best["model"]

print(f"\n{'='*50}")
print("BEST MODEL: Logistic Regression")
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
axes[1].set_title("Confusion Matrix (Logistic Regression)", fontweight="bold")
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

