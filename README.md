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
- [LINE Assumption Results](#-line-assumption-results)
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

**Final result:** a cross-validated Ridge Regression model explaining **95.3%** of price variation, selected specifically for its stability across data splits, not just its raw accuracy.

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

**All features together explain over 95% of price variation.** Multiple Linear Regression using all 188 encoded features reached a test R² of **0.9523** (Phase 5) — confirming price is genuinely a multi-factor outcome, not dominated by one or two variables.

**Cross-validation told a more honest story than the single train/test split.** The unregularized model looked strong on the original 80/20 split (test R² = 0.9523, even slightly exceeding its own training score) — but proper 5-fold cross-validation revealed its *true* average performance was only **0.900**, swinging between 0.874 and 0.922 depending on the fold (std = 0.018). A single split had simply gotten lucky.

**Regularization didn't just improve accuracy — it cut prediction variance nearly in half.** Ridge's cross-validated mean R² rose to **0.9174** while its fold-to-fold standard deviation dropped to **0.011**, a ~40% reduction in variance versus the unregularized model. This is the strongest evidence in the whole project that regularization was worth doing: a more *stable* model, not just a marginally more accurate one.

**Engineered features beat raw ones.** A hand-built `Total SF` feature (basement + 1st floor + 2nd floor combined) correlated more strongly with price (**r = 0.832**) than any single original square-footage column — the model benefits from being handed the *sum* directly rather than discovering the relationship on its own.

**One-hot encoding manufactures multicollinearity.** After encoding categorical variables, **62 of 188 features** had a Variance Inflation Factor above 10; a few (`Exterior 1st_PreCast` / `Exterior 2nd_PreCast`) were mathematically infinite, caused by a single rare category appearing identically across two columns — precisely the failure mode Ridge is designed to absorb.

**The final model choice came down to a coin flip — and that's an honest finding too.** Ridge, Lasso, and ElasticNet converged to nearly identical cross-validated performance (0.9174, 0.9164, 0.9172 — all within 0.001 of each other). Ridge was selected by the smallest possible margin, but Lasso achieved almost the same score using **36 fewer features** (151 vs. 187) — a genuinely close call where interpretability, not raw accuracy, would decide the winner in a real deployment.

---

## 📈 Model Performance

| Model | Features Used | Test RMSE | Test R² | CV Mean R² | CV Std R² |
|---|---:|---:|---:|---:|---:|
| Simple LR (`Overall Qual` only) | 1 | — | 0.6441 | — | — |
| Multiple LR (all 188 features) | 188 | 0.0926 | 0.9523 | 0.9000 | 0.0178 |
| Feature-Selected LR (consensus, 10 features) | 10 | — | 0.8684 | — | — |
| Lasso (α=0.0001) | 151 | 0.0921 | 0.9527 | 0.9164 | 0.0107 |
| ElasticNet | 147 | 0.0920 | 0.9529 | 0.9172 | 0.0111 |
| **🏆 Final Model — Ridge (α=10)** | **187** | **0.0920** | **0.9529** | **0.9174** | **0.0114** |

**Reading RMSE in dollars:** since `SalePrice` was log-transformed, the final model's test RMSE of 0.0920 approximates a **±9–10% average prediction error** — roughly **$17,000** on a typical $180,000 home.

---

## ✅ LINE Assumption Results

| Assumption | Test Used | Result | Status |
|---|---|---:|---|
| **L**inearity | Residual vs. fitted plot (visual) | See `06_assumption_testing.ipynb`, Cell 3 | Visual inspection |
| **I**ndependence | Durbin-Watson | DW = 2.066 | ✅ Pass |
| **N**ormality | Shapiro-Wilk | p ≈ 0.0000 | ⚠️ Fails formally* |
| **E**qual Variance | Breusch-Pagan | p ≈ 0.0000 | ⚠️ Mild heteroscedasticity |
| No Multicollinearity | VIF | 62 / 188 features > 10 | ❌ Violated (mitigated by Ridge) |

*Shapiro-Wilk is extremely sensitive at n > 2,900 — even minor, practically harmless deviations from normality register as statistically significant. The Q-Q plot was used alongside the p-value to confirm this wasn't a severe violation.

---

## 🧠 How the Final Model Works (Plain English)

The final model — **Ridge Regression** — does something close to what an experienced appraiser does intuitively: it starts from the average home price in Ames, then adjusts up or down based on all 187 available signals at once — quality, living space, age, garage size, neighborhood, and more — without fully discarding any of them. Where Ridge differs from plain Linear Regression is restraint: when two features say almost the same thing (like garage capacity and garage square footage), Ridge spreads the credit between them instead of letting one swing wildly. That restraint is exactly why its cross-validated predictions stayed far more consistent across different slices of the data than the unregularized model's did.

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/feroz-crash/house-price-prediction.git
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

This model has boundaries worth stating honestly rather than hiding. The data covers a single market — Ames, Iowa, 2006–2010 — so coefficients learned here, especially neighborhood effects, won't transfer to other cities or time periods without retraining. Two LINE assumptions failed formal statistical tests: residuals were not perfectly normal (Shapiro-Wilk p≈0, though this is a known large-sample artifact rather than a severe distortion) and showed mild heteroscedasticity (Breusch-Pagan p≈0), meaning prediction intervals are likely narrower than ideal for the most expensive homes. Multicollinearity, while substantially reduced by Ridge, was not eliminated at the raw feature level — 62 of 188 features still exceed VIF=10 — so individual coefficients should be read as directional rather than precise causal effects. The model is also linear by construction and cannot capture genuine non-linear interactions beyond the single interaction term (`Overall Score`) engineered in Phase 7.

---

## 🚀 Future Improvements

Next steps that would meaningfully extend this project: tree-based models (Random Forest, XGBoost) as a non-linear benchmark against this linear baseline; SHAP values for richer feature attribution beyond raw coefficients; a stacked ensemble combining Ridge with a gradient-boosted model; and external data (school district ratings, walkability scores) to capture location value more precisely than `Neighborhood` alone allows.

---

## 🛠 Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `matplotlib` · `seaborn` · `Jupyter Notebook`

---

## 👤 Author

**Feroz Shaik**
[GitHub](https://github.com/feroz-crash) · [LinkedIn](https://www.linkedin.com/in/ferozshaik-tech/)

---

<div align="center">

*If this project helped you understand linear regression more deeply, consider giving it a ⭐*

</div>