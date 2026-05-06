# 🏥 Hospital Readmission Risk Analysis
### Diabetes 130-US Hospitals | UCI ML Repository | Python · Power BI · EDA

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-EDA-green?logo=pandas)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow?logo=powerbi)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project analyzes **101,766 diabetic patient encounters** from 130 US hospitals (1999–2008) to identify the key clinical and demographic drivers of **30-day hospital readmissions**. The goal is to produce actionable, data-backed insights that hospital administrators can use to target high-risk patients before discharge.

**Business Question:** *Which patient profiles are most likely to be readmitted within 30 days, and what clinical factors are the strongest predictors?*

---

## 📂 Project Structure

```
hospital-readmission-analysis/
│
├── data/
│   ├── diabetic_data.csv          # Raw dataset (UCI ML Repository)
│   └── IDS_mapping.csv            # Admission/discharge/source ID reference
│
├── src/
│   └── eda.py                     # Main analysis script (cleaning + EDA)
│
├── notebooks/
│   └── hospital_eda_notebook.ipynb  # Interactive Jupyter notebook
│
└── outputs/
    ├── 01_readmission_distribution.png
    ├── 02_readmission_by_age.png
    ├── 03_readmission_by_prior_visits.png
    ├── 04_readmission_by_los.png
    ├── 05_readmission_by_specialty.png
    ├── 06_insulin_impact.png
    ├── 07_high_utiliser_risk.png
    ├── 08_correlation_heatmap.png
    ├── cleaned_readmission_data.csv  # Cleaned output for downstream use
    └── hopspital data.pbix           # Power BI dashboard
```

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| **Source** | [UCI ML Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/) |
| **Records** | 101,766 patient encounters |
| **Features** | 50 attributes (demographics, diagnoses, medications, lab results) |
| **Target** | `readmitted` — whether patient was readmitted within 30 days |
| **Period** | 1999–2008, 130 US hospitals |

---

## 🔧 Methodology

### Phase 1 — Data Cleaning (6 documented decisions)

| Decision | Reason |
|---|---|
| Dropped `weight` (96.9% missing) | Imputing would introduce noise; noted as limitation |
| Dropped `payer_code` (39.6% missing) | Low clinical relevance for readmission |
| Imputed `medical_specialty` → `'Unknown'` | Specialty is still useful where present |
| Imputed `race` → `'Unknown'` | Only 2.2% missing |
| Kept first encounter per patient | Prevents data leakage from repeat encounters |
| Removed hospice / deceased discharges | These patients cannot be readmitted |

### Phase 2 — Feature Engineering

- **`readmitted_binary`** — Binary target: `1` if readmitted `<30` days, else `0`
- **`age_numeric`** — Numeric midpoint extracted from age bracket strings
- **`age_group`** — Binned into `<40`, `40-59`, `60-74`, `75+`
- **`total_medications_changed`** — Count of medication dosage changes at discharge
- **`on_insulin`** — Boolean flag for insulin prescription
- **`high_utiliser`** — Flag for patients with 3+ prior inpatient visits
- **`los_bucket`** — Length-of-stay category: Short / Medium / Long

### Phase 3 — Exploratory Data Analysis

8 charts generated covering readmission distribution, age risk, prior visit patterns, length of stay, medical specialties, insulin impact, high-utiliser risk, and a correlation heatmap.

---

## 📈 Key Findings

| # | Finding |
|---|---|
| 1 | **Overall 30-day readmission rate: ~11%** across all cleaned encounters |
| 2 | **75+ age group** has the highest readmission rate among all age bands |
| 3 | **Prior inpatient visits** is the strongest single predictor — patients with 3+ prior visits are significantly more likely to be readmitted |
| 4 | **High utilisers** (3+ prior visits) have a readmission rate nearly **2× the overall average** |
| 5 | **Insulin dosage changes** at discharge correlate with different readmission patterns — warrants protocol review |
| 6 | **Number of diagnoses** is positively correlated with readmission — comorbidities compound risk |

---

## 📉 Charts Preview

| Chart | Description |
|---|---|
| `01_readmission_distribution.png` | Overall breakdown: No readmission / >30 days / <30 days |
| `02_readmission_by_age.png` | Readmission rate by age group with overall average line |
| `03_readmission_by_prior_visits.png` | Risk curve as number of prior inpatient visits increases |
| `04_readmission_by_los.png` | Readmission rate by length-of-stay bucket |
| `05_readmission_by_specialty.png` | Top 10 medical specialties ranked by readmission rate |
| `06_insulin_impact.png` | Readmission rate by insulin prescription change status |
| `07_high_utiliser_risk.png` | High-utiliser vs standard patient comparison |
| `08_correlation_heatmap.png` | Correlation matrix across all numeric features |

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn
```

### Run the EDA script

```bash
# Clone the repository
git clone https://github.com/your-username/hospital-readmission-analysis.git
cd hospital-readmission-analysis

# Download dataset from UCI (link above) and place as:
#   data/diabetic_data.csv

# Run analysis
python src/eda.py
```

Charts will be saved to `outputs/`. The cleaned dataset will be exported as `outputs/cleaned_readmission_data.csv`.

### Run the Jupyter Notebook

```bash
jupyter notebook notebooks/hospital_eda_notebook.ipynb
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9 | Data processing and analysis |
| Pandas | Data loading, cleaning, feature engineering |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Chart generation |
| Power BI | Interactive dashboard (`.pbix` file) |
| Jupyter | Exploratory notebook |

---

## ⚠️ Limitations

- `weight` data is missing for 96.9% of patients and was excluded — this is a notable gap given its clinical relevance to diabetes outcomes
- Dataset covers 1999–2008; clinical protocols have evolved since then
- Analysis is descriptive/exploratory — no predictive model has been built in this phase
- ICD-9 diagnosis codes used without grouping into clinical categories (future work)

---

## 🔮 Next Steps (Planned)

- **Phase 3:** SQL-based analysis on cleaned dataset for deeper segment queries
- **Phase 4:** Machine learning model (Logistic Regression / XGBoost) to predict readmission probability
- **Phase 5:** Model explainability using SHAP values

---

## 📜 Data Source

Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International.

Dataset: [https://archive.ics.uci.edu/dataset/296/](https://archive.ics.uci.edu/dataset/296/)

---

## 👤 Author

**[Your Name]**
Data Analyst | Python · SQL · Power BI

📧 your.email@example.com
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://yourportfolio.com)

---

*This project is part of a data analytics portfolio demonstrating end-to-end EDA on real-world healthcare data.*
