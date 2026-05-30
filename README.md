# 📊 Customer Churn Prediction Dashboard

A complete end-to-end machine learning project to predict telecom customer churn, built with Python, Scikit-learn, and Streamlit.

> **Author:** Apoorv Mudgal — B.Tech AI & Data Science | Batch 2028

---

## 🚀 Live Demo

Run the dashboard locally:

```bash
streamlit run dashboard.py
```

Then open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── telecom_churn.csv        # Generated dataset (7,000 records)
│
├── screenshots/
│   ├── eda_overview.png          # EDA charts
│   ├── correlation_heatmap.png   # Feature correlation matrix
│   ├── model_performance.png     # ROC curves + confusion matrix
│   └── churn_drivers.png         # Top 15 feature importances
│
├── generate_data.py              # Synthetic dataset generator + EDA + model training
├── dashboard.py                  # Streamlit interactive dashboard
│
├── churn_model.pkl               # Saved Random Forest model
├── scaler.pkl                    # Saved StandardScaler
├── feature_names.pkl             # Feature names list
│
└── README.md
```

---

## 🧠 ML Pipeline

### 1. Data Generation
Synthetic telecom dataset with realistic churn logic:
- 7,000 customers, ~23% churn rate
- Churn driven by contract type, internet service, tenure, and senior citizen status

### 2. Feature Engineering (28 features)
| Feature | Description |
|---|---|
| `AvgMonthlyCharge` | TotalCharges / (tenure + 1) |
| `ChargeRatio` | MonthlyCharges / (TotalCharges + 1) |
| `IsNewCustomer` | tenure < 12 months |
| `IsLongTermCustomer` | tenure > 48 months |
| `NumServices` | Count of active services |
| `HasInternetService` | Internet ≠ "No" |
| `IsMonthToMonth` | Contract = Month-to-month |
| `IsPaperless` | Paperless billing flag |
| `IsElectronicPayment` | Electronic check payment flag |

### 3. Class Balancing
Upsampling minority class (churned customers) to match majority class size before training.

### 4. Models Trained

| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| Random Forest | 0.712 | 0.442 | 65.9% |
| Gradient Boosting | 0.704 | 0.456 | 65.0% |
| Logistic Regression | 0.731 | 0.485 | 65.1% |

---

## 📈 Key Insights

- 📌 **Month-to-month contracts** have the highest churn rate (~40%)
- 📌 **Fiber optic internet** customers churn more than DSL users
- 📌 **New customers** (tenure < 12 months) are the most at-risk segment
- 📌 **Higher monthly charges** positively correlate with churn
- 📌 **Electronic check** payment method shows the highest churn rate
- 📌 **Long-term customers** (tenure > 48 months) rarely churn

---

## 🖥️ Dashboard Pages

**📊 Dashboard** — KPI metrics, churn by contract/internet type, tenure analysis, model performance charts

**🔮 Predict Churn** — Fill in customer details and get a real-time churn probability with a visual gauge

**📈 EDA Insights** — Full exploratory data analysis with correlation heatmap, raw data preview, and key findings

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/tyagiapoorv001-hash/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit

# Generate data, run EDA, and train models
python generate_data.py

# Launch the dashboard
streamlit run dashboard.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Charts-blue)

- **Data:** Pandas, NumPy
- **ML:** Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Persistence:** Joblib

---

## 📬 Connect

**Apoorv Mudgal**
[LinkedIn](https://linkedin.com/in/apoorv-mudgal) · [GitHub](https://github.com/tyagiapoorv001-hash)
