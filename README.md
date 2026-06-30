<div align="center">

# 🏠 Ames Housing Price Prediction
### An End-to-End Linear Regression Project — From First Principles to Production

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success)

*A deep, ground-up exploration of every core linear regression concept — derived and explained, not just imported.*

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [Model Performance](#-model-performance)
- [How the Final Model Works](#-how-the-final-model-works-plain-english)
- [Installation & Usage](#-installation--usage)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🎯 Overview

This project predicts residential home sale prices in Ames, Iowa using **Linear Regression**, built deliberately from first principles rather than treated as a black box. Every major concept in linear regression is covered end-to-end on a single real dataset: simple and multiple regression derived from raw matrix algebra, gradient descent implemented by hand, all five LINE assumptions tested statistically, regularization (Ridge / Lasso / ElasticNet) tuned via cross-validation, and rigorous evaluation through K-Fold validation and learning curves.

**Goal:** predict `SalePrice` for a home from its physical, locational, and qualitative attributes — while being able to explain *why* the model predicts what it does, not just *what* it predicts.

---

## 📊 Dataset

**Source:** [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) (Kaggle) — compiled by Dean De Cock as a modern, richer alternative to the Boston Housing dataset.

| | |
|---|---|
| **Rows** | 2,930 individual home sales |
| **Columns** | 82 (79 features + `Order` + `PID` + target `SalePrice`) |
| **Location** | Ames, Iowa, USA — sales recorded 2006–2010 |
| **Target** | `SalePrice` — USD, ranging $12,789 – $755,000 (mean ≈ $180,796) |
| **Feature types** | Continuous, discrete, nominal categorical, and ordinal categorical |

---

## 🗂 Project Structure

```
house-price-prediction/
├── data/
│   ├── raw/                   # AmesHousing.csv — not committed, see Installation
│   └── processed/             # train.csv, test.csv, final_model.pkl
├── notebooks/
│   ├── 01_data_loading_and_inspection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_simple_linear_regression.ipynb
│   ├── 05_multiple_linear_regression.ipynb
│   ├── 06_assumption_testing.ipynb
│   ├── 07_feature_engineering.ipynb
│   ├── 08_regularization.ipynb
│   └── 09_evaluation_and_comparison.ipynb
├── reports/
│   └── figures/                # 24 saved plots referenced throughout this README
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Methodology

| Phase | Notebook | What Happens |
|---|---|---|
| 1 | Data Loading | Inspect raw data; classify every column as ID, numerical, nominal, or ordinal |
| 2 | EDA | Distribution analysis, missing-value classification, correlation discovery, outlier detection |
| 3 | Preprocessing | Drop dead-weight columns, impute missing values by *meaning* (not blindly), ordinal-encode, one-hot encode, log-transform, scale |
| 4 | Simple Linear Regression | Built from scratch two ways — closed-form OLS and Gradient Descent — verified against scikit-learn |
| 5 | Multiple Linear Regression | Matrix-form OLS (`β = (XᵀX)⁻¹Xᵀy`) from scratch, Adjusted R², full VIF multicollinearity audit |
| 6 | Assumption Testing | All 5 LINE assumptions tested statistically (Durbin-Watson, Shapiro-Wilk, Breusch-Pagan) |
| 7 | Feature Engineering & Selection | 8 new engineered features; 3 independent selection methods compared (correlation filter, RFE, Lasso) |
| 8 | Regularization | Ridge, Lasso, ElasticNet — hyperparameters tuned via `GridSearchCV` with 5-fold CV |
| 9 | Evaluation | Full model comparison, K-Fold cross-validation, learning curves for bias-variance diagnosis |
| 10 | Documentation | This README — findings, limitations, and reproducibility |

---

## 💡 Key Findings

**Quality alone explains nearly two-thirds of price variation.** `Overall Qual` — a single 1–10 rating — produces an R² of **0.644** on its own (Phase 4). Before factoring in size, location, or age, simply knowing how well-built a house is gets you most of the way to predicting its price.

**All features together explain 95% of price variation.** Multiple Linear Regression using all 188 encoded features reached a test R² of **0.9522** (RMSE ≈ 0.093 in log-space, roughly ±9% / ~$16–18k typical error in dollar terms) — confirming that price is genuinely a multi-factor outcome, not dominated by one or two variables.

**Engineered features beat raw ones.** A hand-built `Total SF` feature (basement + 1st floor + 2nd floor combined) correlated more strongly with price (**r = 0.832**) than any single original square-footage column — the model benefits from being handed the *sum* directly rather than discovering the relationship on its own.

**One-hot encoding manufactures multicollinearity.** After encoding categorical variables, **62 of 188 features** had a Variance Inflation Factor above 10; a few (`Exterior 1st_PreCast` / `Exterior 2nd_PreCast`) were mathematically infinite, caused by a single rare category appearing identically across two columns. This is precisely the failure mode Ridge regression is designed to absorb.

**Aggressive feature selection costs real accuracy.** Requiring three different selection methods (correlation filter, RFE, Lasso) to *all* agree left only 10 features, dropping R² from 0.952 to 0.868. Consensus filtering is safe but conservative — it discards features that matter through interaction even when they don't stand out individually.

> *`[ Add 1–2 more findings here once Phase 8/9 are complete — e.g. which regularized model won, and how stable it was across CV folds. ]`*

---

## 📈 Model Performance

| Model | Features Used | Test RMSE | Test R² | CV Mean R² |
|---|---:|---:|---:|---:|
| Simple LR (`Overall Qual` only) | 1 | — | 0.6441 | — |
| Multiple LR (all features) | 188 | 0.0926 | 0.9522 | — |
| Multiple LR — Adjusted R² | 188 | — | 0.9296 | — |
| Feature-Selected LR (consensus, 10 features) | 10 | — | 0.8684 | — |
| Ridge (tuned) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Lasso (tuned) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| ElasticNet (tuned) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| **🏆 Final Selected Model** | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |

> Run `09_evaluation_and_comparison.ipynb` end-to-end and copy Cells 5–9's output into the bracketed cells above before publishing this repo.

**Reading RMSE in dollars:** since `SalePrice` was log-transformed, a test RMSE of 0.0926 approximates a **±9% average prediction error** — roughly $16,000–$18,000 on a typical $180,000 home.

---

## 🧠 How the Final Model Works (Plain English)

Strip away the math, and the model does something close to what an experienced appraiser does intuitively: it starts from the average home price in Ames, then adjusts up or down based on dozens of signals — adding value for higher overall quality, more living space, a newer or recently-remodeled build, a larger garage, and a more desirable neighborhood, while subtracting value for the opposite. Regularization (Ridge/Lasso) prevents the model from over-trusting any single feature when several features say the same thing (like garage size and garage capacity), keeping the final price estimate stable rather than swinging wildly based on one column.

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Get AmesHousing.csv from Kaggle (link above) and place it at:
#   data/raw/AmesHousing.csv

# 5. Run the notebooks in order
jupyter notebook notebooks/01_data_loading_and_inspection.ipynb
```

Each notebook is numbered and self-contained — run them sequentially from `01` through `09` to reproduce the full pipeline end-to-end.

---

## ⚠️ Limitations

This model has boundaries worth stating honestly rather than hiding. The data covers a single market — Ames, Iowa, 2006–2010 — so coefficients learned here, especially neighborhood effects, won't transfer to other cities or time periods without retraining. Several LINE assumptions were only partially satisfied: mild heteroscedasticity and residual multicollinearity remain present even after regularization, so individual coefficients should be read as directional rather than precise. The model is also linear by construction — it cannot capture genuine non-linear interactions (a renovated kitchen mattering far more in an expensive neighborhood than a cheap one, for example) beyond the single interaction term (`Overall Score`) engineered in Phase 7.

---

## 🚀 Future Improvements

Next steps that would meaningfully extend this project: tree-based models (Random Forest, XGBoost) as a non-linear benchmark against this linear baseline; SHAP values for richer feature attribution beyond raw coefficients; a stacked ensemble combining Ridge with a gradient-boosted model; and external data (school district ratings, walkability scores) to capture location value more precisely than `Neighborhood` alone allows.

---

## 🛠 Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `matplotlib` · `seaborn` · `Jupyter Notebook`

---

## 👤 Author

**[Your Name]**
[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)

---

<div align="center">

*If this project helped you understand linear regression more deeply, consider giving it a ⭐*

</div>
