# Customer Churn Analysis

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Losing customers increases acquisition costs and reduces revenue.

This project analyzes customer demographics, service usage, contract information, and billing behavior to identify factors associated with customer churn.

The project applies exploratory data analysis (EDA) and machine learning techniques to understand churn patterns, predict customers at risk of leaving, and provide actionable retention strategies.

---

## Business Problem

Subscription businesses need to understand why customers leave and identify high-risk customers before they churn.

The goal of this project is to:

- Analyze customer churn patterns.
- Identify the main factors influencing customer churn.
- Build an interpretable churn prediction model.
- Translate analytical findings into actionable retention strategies.

---

## Dataset

**Dataset Source:**

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Original Dataset Size:**

- 7,043 customers
- 21 columns

**Data Cleaning:**

The dataset contained missing values in the `TotalCharges` column. After removing incomplete records, 7,032 customers were used for exploratory analysis and model training.

**Target Variable:**

`Churn`

**Project Goal:**

Predict whether a customer is likely to leave the service and identify the factors associated with customer churn.

---

# Methodology

## 1. Data Preparation

Steps performed:

- Removed missing values from `TotalCharges`.
- Converted data types appropriately.
- Separated numerical and categorical features.
- Built a Scikit-learn preprocessing pipeline.
- Applied one-hot encoding for categorical variables.

---

## 2. Exploratory Data Analysis

Analyzed customer characteristics associated with churn:

- Contract type
- Customer tenure
- Payment method
- Monthly charges
- Technical support availability

Key patterns were identified to understand high-risk customer segments.

---

## 3. Machine Learning Models

The following models were evaluated:

1. Baseline Logistic Regression
2. Improved Logistic Regression
3. Random Forest Classifier

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# Key Findings

## Contract Type

Month-to-month customers have the highest churn rate (42.7%), while customers with two-year contracts have a much lower churn rate (2.8%).

Longer contract commitments are associated with stronger customer retention.

---

## Customer Tenure

Customers in their first 12 months have the highest churn risk, with a churn rate of 47.7%.

The early customer lifecycle appears to be a critical retention period.

---

## Payment Method

Customers using electronic checks have a higher churn rate (45.3%) compared with customers using automatic payment methods.

Payment behavior may indicate differences in customer engagement and billing experience.

---

## Monthly Charges

Customers who churn have higher average monthly charges ($74.44) compared with customers who stay ($61.31).

Higher-value customers may require additional attention to ensure they receive sufficient value.

---

## Technical Support

Customers without technical support have a higher churn rate (41.6%) compared with customers who have technical support (15.2%).

Support availability is strongly associated with customer retention.

---

# Model Results

Three machine learning models were compared:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 79.39% | 63.55% | 52.67% | 57.60% | 83.49% |
| Improved Logistic Regression | **80.38%** | **64.85%** | **57.22%** | **60.80%** | **83.59%** |
| Random Forest | 78.75% | 63.07% | 48.40% | 54.77% | 81.37% |

---

## Final Model Selection

The improved Logistic Regression model achieved the best overall performance.

The model improvement came from removing the engineered `tenure_group` feature, which duplicated information already captured by the original `tenure` variable.

The final model was selected because it provided:

- Strong predictive performance.
- Better recall for identifying churn-risk customers.
- Clear interpretation of factors influencing churn.

---

# Business Recommendations

Based on the analysis:

## 1. Encourage Longer Contracts

Month-to-month customers represent the highest-risk segment.

Retention strategies could include:

- Contract upgrade incentives.
- Loyalty discounts.
- Long-term commitment benefits.

---

## 2. Improve Early Customer Experience

Customers within their first year show the highest churn risk.

Businesses should focus on:

- Better onboarding.
- Customer education.
- Early engagement campaigns.

---

## 3. Promote Automatic Payment Methods

Customers using electronic checks show higher churn rates.

Possible actions:

- Encourage automatic payments.
- Simplify billing experiences.
- Reduce payment friction.

---

## 4. Increase Technical Support Adoption

Customers without technical support are more likely to churn.

Possible actions:

- Promote support services.
- Provide proactive assistance.
- Identify customers experiencing service issues.

---

## 5. Monitor High-Value Customers

Customers with higher monthly charges may have higher expectations.

Businesses should:

- Monitor satisfaction.
- Provide personalized offers.
- Improve perceived value.

---

# Project Structure

```
Customer-Churn-Analysis/

├── data/
├── notebooks/
│   └── churn_analysis.ipynb
├── reports/
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Future Improvements

Potential improvements for future iterations:

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Evaluate Gradient Boosting and XGBoost models.
- Apply cross-validation for more robust evaluation.
- Explore class imbalance techniques.
- Deploy the churn prediction model as an API.