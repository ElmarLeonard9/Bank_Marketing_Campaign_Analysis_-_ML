# Bank Marketing Campaign, Term Deposit Subscription Prediction

Predicting which bank customers are likely to subscribe to a term deposit, so marketing and sales teams can prioritize outreach instead of contacting the entire prospect list.

Built by **Elmar Leonard** & **Nadya Divia Go**.

---

## Problem

Banks run marketing campaigns (mostly by phone) to sell term deposits. Contacting every customer is time consuming and most people say no. This project builds a supervised binary classification model that scores each prospect's probability of subscribing, so campaigns can be targeted at high-propensity leads instead of the full database.

**Core business questions**

1. What customer characteristics indicate a higher tendency to open a term deposit?
2. Which factors most heavily influence a customer's decision?
3. How do past campaign history and contact method affect conversion?
4. Which customer segments should be prioritized?
5. How accurately can a model target high-potential customers?

## Dataset

- **File:** `data_bank_marketing_campaign.csv`
- **Size:** 7,813 records (7,805 after removing 8 duplicates), 11 columns
- **Target:** `deposit` (`yes` / `no`) a near-balanced ~52% / 48% split
- **Features:** `age`, `job`, `balance`, `housing`, `loan`, `contact`, `month`, `campaign`, `pdays`, `poutcome`

No explicit nulls, but `contact` (~21%) and `poutcome` (~74%) contain `'unknown'` as a meaningful placeholder for customers who were never previously contacted which treated as a valid category rather than dropped or imputed.

## Repository Structure

Notebooks follow the CRISP-DM workflow, one phase per file:

| Notebook                              | Phase                        | Contents                                                                                                                             |
| :------------------------------------ | :--------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| `1_Business_Understanding.Ipynb`    | Business Understanding       | Context, problem statements, goals, analytical approach, metric selection (ROC-AUC, Precision, Recall, F1), success criteria         |
| `2_Data_Understanding.Ipynb`        | Data Understanding           | Structural audit, feature dictionary, descriptive stats, missing/duplicate/outlier handling                                          |
| `3_Exploratory_Data_Analysis.ipynb` | EDA                          | Univariate/bivariate analysis, Cramér's V & correlation ratio, multivariate interactions, feature engineering ideas                 |
| `4_Data_Preparation.ipynb`          | Data Preparation             | Train/test split, feature engineering, one-hot encoding, Winsorization, scaling, feature selection                                   |
| `5_Machine_Learning_Modeling.ipynb` | Modeling                     | Model benchmarking, hyperparameter tuning, threshold optimization, calibration, SHAP/permutation importance, counterfactual analysis |
| `6_Business_Recommendation.Ipynb`   | Deployment & Recommendations | Implementation plan, model limitations, business cost-benefit calculation, final conclusions & recommendations                       |
| `data_bank_marketing_campaign.csv`  | -                            | Raw dataset                                                                                                                          |

## Methodology Summary

**Feature engineering** (Notebook 4): `pdays_contacted` (binary flag decoupled from raw `pdays`), `age_group` bins, `job_grouped` (rare categories collapsed), `balance_negative` flag + `balance_log` transform. Categorical features one-hot encoded (`drop="first"`, `handle_unknown="ignore"`); `balance`, `campaign`, and `pdays` Winsorized (IQR, fold=3) and scaled with `RobustScaler`.

**Modeling** (Notebook 5): Benchmarked linear models (Logistic Regression, Ridge, LinearSVC, SGD) and tree-based/ensemble models (Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost) via 5-fold stratified cross-validation on ROC-AUC.

- **Champion model:** `GradientBoostingClassifier`, tuned via `RandomizedSearchCV` (`learning_rate=0.1`, `max_depth=4`, `subsample=0.85`, `n_estimators=100`) resulted CV ROC-AUC **0.7762**. An exhaustive grid search over 3,240 candidates only improved this by +0.0017, confirming the random search had already found the performance plateau.
- **Decision threshold** was tuned on a held-out validation split (independent of the test set) to **0.351**, trading precision for recall in line with the business cost asymmetry between a wasted call (false positive) and a missed deposit (false negative).
- The model was cross-checked for calibration (`CalibratedClassifierCV` gave a negligible Brier score improvement which mean the raw model is already well-calibrated) and explained using impurity-based importance, permutation importance, SHAP values, and counterfactual probability sweeps. Conclude they all converging on the same three dominant drivers: `poutcome`, `month`, and `contact`.
- `duration` (call length) was intentionally excluded from the feature set to avoid information leakage, since it is only known *after* a call takes place.

## Results

**Test set performance (n = 1,561, threshold = 0.351):**

| Metric              | Target |     Achieved     | Status           |
| :------------------ | :-----: | :--------------: | :--------------- |
| ROC-AUC             | ≥ 0.75 | **0.796** | Passed           |
| F1-Score (`yes`)  | ≥ 0.65 |  **0.71**  | Passed           |
| Recall (`yes`)    | ≥ 70% |  **80%**  | Passed           |
| Precision (`yes`) | ≥ 65% |  **63%**  | Marginally short |
| Generalization gap  |  ≤ 5%  | **~2–3%** | Passed           |

**Key findings:**

- **Campaign history beats customer profile.** `poutcome` (outcome of the previous campaign) is by far the strongest predictor; `age`, `job`, `housing`, and `loan` carry near-zero model weight.
- **Timing matters enormously.** Off-peak months (`mar`, `sep`, `oct`, `dec`) convert at 85–92%, while the bank's highest-volume month (`may`) converts at only ~33%.
- **`unknown` contact logging correlates with poor conversion** (~22%), suggesting a CRM data-quality issue rather than a genuine behavioral signal.
- **Contact fatigue is real** , counterfactual analysis shows predicted conversion probability drops from 0.367 to 0.181 as contact attempts increase from 1 to 8.
- Against a fixed-capacity baseline, model-driven targeting delivers a **~36% relative lift in conversion rate** over random dialing.

Full limitations (cold-start risk on `poutcome = unknown` leads, an unresolved precision shortfall, a feature-representation ceiling, and unvalidated illustrative cost assumptions) and the full deployment roadmap are documented in `6_Business_Recommendation.Ipynb`.

## Tech Stack

- **Language:** Python
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Preprocessing:** `feature-engine` (Winsorizer)
- **Explainability:** `shap`, `sklearn.inspection.permutation_importance`
- **Stats testing:** `scipy.stats` (chi-square, ANOVA)

## Getting Started

```bash
git clone [https://github.com/](https://github.com/)ElmarLeonard9/hBank_Marketing_Campaign_Analysis_-_ML
cd Bank_Marketing_Campaign_Analysis_-_ML
pip install -r requirements.txt
```

Run the notebooks in numeric order (`1` → `6`); each stage exports the artifacts (processed data, fitted pipeline) consumed by the next.
