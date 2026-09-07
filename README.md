# Customer Churn Prediction

Capstone task for InternSpark — end-to-end pipeline predicting which customers are likely to churn, with business takeaways.

## Files

- `Customer_Churn_Prediction.ipynb` — main notebook, run top to bottom in Colab
- `customer_churn.csv` — dataset (2000 rows)
- `churn_model.joblib` — saved model (generated after running the notebook)

## How to run

1. Open `Customer_Churn_Prediction.ipynb` in Google Colab
2. Upload `customer_churn.csv` to the Colab session (Files panel on the left)
3. Run all cells

No API keys or downloads needed, the CSV is loaded locally.

## Why this project

Went a bit past the Iris/House Price level on purpose — churn brings in a mix of numeric and categorical features, some real missing data, and a class imbalance (about 25% churn) that actually needs handling instead of just reporting accuracy. It's also a genuinely common business use case, so the end result has to say something useful, not just report a score.

## What's in the notebook

- EDA — churn by tenure, contract type, monthly charges, payment method
- Handled missing values in TotalCharges, encoded categoricals
- Trained and compared Logistic Regression, Random Forest, and Gradient Boosting, using balanced class weights where possible since churn is imbalanced
- Evaluated with accuracy, precision, recall, F1, and AUC, plus confusion matrices and ROC curves
- Pulled feature importances from Random Forest to see what's actually driving churn
- Saved the best model and closed with business-facing takeaways

Contract type turned out to be the single biggest driver, month-to-month customers churn a lot more than customers on annual contracts. Tenure and monthly charges matter too. On model choice, Logistic Regression with balanced class weights ended up winning on AUC and catches far more actual churners than the tree models, even though its precision is lower, which matters more here since missing a churner is worse than a false alarm.

## Running inference

```python
predict_churn(3, 95.5, 300, 0, "No", "No", "Month-to-month", "Fiber optic", "No", "No", "Yes", "Electronic check")
```

Loads the saved model and returns whether the customer is predicted to churn, plus the churn probability.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

Packages come pre-installed in Colab, so no extra setup needed there.
