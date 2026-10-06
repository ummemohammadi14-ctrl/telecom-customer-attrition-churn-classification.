# Telecom Customer Attrition: Churn Classification and Risk Analysis

A Data Science capstone project analyzing customer attributes associated with a supplied telecom churn label and comparing classification models.

**Capstone dataset provided by StarAgile** for my Data Science course.

## Project highlights

- Audited a dataset of **3,333 records**, **10 customer attributes**, and a binary `Churn` label.
- Explored churn patterns across contract renewal, data plans, customer-service calls, usage, and charges.
- Compared a majority-class baseline, Logistic Regression, Random Forest, and XGBoost.
- Used stratified cross-validation, hyperparameter search, and probability-threshold analysis.
- Selected a tuned Random Forest with a **0.40 classification threshold** for the project’s reported evaluation.

## Project question

Can the customer attributes in this dataset distinguish records labeled as churned from records labeled as retained?

The target is `Churn`:

- `0` = labeled as no churn
- `1` = labeled as churn

The notebook analyzes the dataset’s supplied label. It does not establish when the customer attributes were recorded relative to churn.

## Dataset

The CSV is stored at `data/raw/telecom_churn.csv`. It contains **3,333 records** and **11 columns**: 10 predictors and the `Churn` target. Churn is the minority class: **483 records (14.5%)** are labeled as churned.

The predictors cover:

- **Account history:** `AccountWeeks`
- **Contract and plan:** `ContractRenewal`, `DataPlan`
- **Usage and charges:** `DataUsage`, `DayMins`, `DayCalls`, `MonthlyCharge`, `OverageFee`, `RoamMins`
- **Customer support:** `CustServCalls`

## Notebook workflow

1. **Audit the data.** Check its dimensions, types, missing values, duplicates, binary values, numerical ranges, and constant columns. This establishes what the analysis is working with.

2. **Explore customer attributes.** Review distributions and compare churn rates across customer groups. The notebook also examines correlations, mean and median differences, Mann–Whitney U tests, and usage bands to identify patterns for further investigation.

3. **Prepare the modeling data.** Separate predictors from the target, remove temporary EDA-only bands, and create a reproducible, stratified 80/20 split using `random_state=42`. The split check confirms there is no row overlap.

4. **Establish a baseline.** A majority-class Dummy Classifier provides a reference point. It achieves 85.46% accuracy by predicting no churn for everyone, but identifies no churners—showing why accuracy alone is insufficient here.

5. **Compare models.** Evaluate standardized Logistic Regression, Random Forest, and XGBoost using classification metrics and confusion matrices. This compares an interpretable linear model with two tree-based approaches.

6. **Assess performance across folds.** Use stratified five-fold cross-validation to compare mean scores and variation across folds. PR-AUC receives particular attention because churn is the smaller class.

7. **Tune models and review thresholds.** Use `RandomizedSearchCV` to tune Random Forest and XGBoost with average precision as the search metric. Then compare precision, recall, and F1 across thresholds from 0.05 to 0.95 using out-of-fold probabilities.

8. **Interpret and report results.** Review the selected model’s test-split metrics, confusion matrix, and feature importances, then connect the patterns to possible customer-review questions.

## Model selection and results

The tuned Random Forest had a cross-validation PR-AUC of **0.8197**, compared with **0.8011** for tuned XGBoost. The notebook selects Random Forest with a threshold of **0.40**.

| Metric | Reported score |
|---|---:|
| Accuracy | 92.20% |
| Precision | 77.11% |
| Recall | 65.98% |
| F1-score | 71.11% |
| ROC-AUC | 0.8638 |
| PR-AUC | 0.7454 |

At this threshold, the model identified **64 of the 97 records labeled as churned** in the evaluated split. Of the 83 records it flagged, 64 had the churn label.

**Evaluation context:** The test split was used for initial model comparisons, so these scores are exploratory results rather than an independent final estimate. The threshold is an F1-based development choice; the dataset does not provide campaign capacity or retention-offer costs for setting a business-optimal threshold.

## Selected findings

- Customers who did not renew had an observed churn rate of **42.4%**, compared with **11.5%** among customers who renewed.
- Customers without a data plan had an observed churn rate of **16.7%**, compared with **8.7%** among customers with a data plan.
- Churn rates were higher among customers with repeated service calls. The notebook notes that some high-call groups are small, so their rates need cautious interpretation.
- Churned customers had higher average daytime usage: **206.91 minutes**, compared with **175.18 minutes** for non-churned customers.
- The top three Random Forest feature importances were `DayMins` (**0.2319**), `MonthlyCharge` (**0.1784**), and `CustServCalls` (**0.1754**).

These are observed associations and model-based importance scores. They help identify areas for further investigation; they do not establish that a feature causes churn.

## Reproduce the analysis

1. Keep the CSV at `data/raw/telecom_churn.csv`.
2. Install the pinned dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Open `notebooks/telecom_churn_analysis.ipynb` and run the cells from top to bottom.

The notebook uses fixed random seeds for reproducible sampling, splitting, cross-validation, and model search. It creates the `outputs/` directories when saving results.

## Generated outputs

- `outputs/final_model_results.csv`
- `outputs/final_feature_importance.csv`
- `outputs/figures/final_confusion_matrix.png`
- `outputs/figures/final_feature_importance.png`

## Repository structure

```text
telecom-customer-attrition-churn-classification/
├── data/
│   └── raw/
│       └── telecom_churn.csv
├── notebooks/
│   └── telecom_churn_analysis.ipynb
├── outputs/                 # Generated when the notebook runs
├── .gitignore
├── README.md
└── requirements.txt
```

## Next validation step

The dataset documentation does not specify the feature measurement time or churn prediction horizon. A stronger future evaluation would use time-stamped customer data, define a prediction window, and assess performance on a later period. Retention actions could then be evaluated against observed customer outcomes.