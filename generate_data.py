"""Generate a synthetic telecom customer churn dataset."""

import os

import numpy as np
import pandas as pd


os.makedirs("data", exist_ok=True)
np.random.seed(42)

N_CUSTOMERS = 7000

data = {
    "customerID": [f"CUST{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)],
    "gender": np.random.choice(["Male", "Female"], N_CUSTOMERS),
    "SeniorCitizen": np.random.choice([0, 1], N_CUSTOMERS, p=[0.84, 0.16]),
    "Partner": np.random.choice(["Yes", "No"], N_CUSTOMERS),
    "Dependents": np.random.choice(["Yes", "No"], N_CUSTOMERS, p=[0.3, 0.7]),
    "tenure": np.random.randint(1, 72, N_CUSTOMERS),
    "PhoneService": np.random.choice(["Yes", "No"], N_CUSTOMERS, p=[0.9, 0.1]),
    "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], N_CUSTOMERS),
    "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], N_CUSTOMERS, p=[0.34, 0.44, 0.22]),
    "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], N_CUSTOMERS),
    "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], N_CUSTOMERS),
    "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], N_CUSTOMERS),
    "TechSupport": np.random.choice(["Yes", "No", "No internet service"], N_CUSTOMERS),
    "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], N_CUSTOMERS),
    "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], N_CUSTOMERS),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], N_CUSTOMERS, p=[0.55, 0.24, 0.21]),
    "PaperlessBilling": np.random.choice(["Yes", "No"], N_CUSTOMERS, p=[0.59, 0.41]),
    "PaymentMethod": np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        N_CUSTOMERS,
    ),
    "MonthlyCharges": np.round(np.random.uniform(18, 119, N_CUSTOMERS), 2),
    "TotalCharges": np.round(np.random.uniform(18, 8500, N_CUSTOMERS), 2),
}

churn_prob = (
    0.05
    + 0.25 * (np.array(data["Contract"]) == "Month-to-month")
    + 0.10 * (np.array(data["InternetService"]) == "Fiber optic")
    + 0.10 * (np.array(data["tenure"]) < 12)
    + 0.05 * (np.array(data["SeniorCitizen"]) == 1)
    - 0.10 * (np.array(data["tenure"]) > 48)
)
churn_prob = np.clip(churn_prob, 0, 1)
data["Churn"] = np.where(np.random.rand(N_CUSTOMERS) < churn_prob, "Yes", "No")

df = pd.DataFrame(data)
df.to_csv("data/telecom_churn.csv", index=False)
print(f"Dataset saved: {df.shape[0]} rows, churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
