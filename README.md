# 💳 Credit Card Fraud Detection with XGBoost

End-to-end machine learning pipeline for real-time fraud detection in financial transactions, built in partnership with EBAC and Semantix. The final model achieved a projected net balance of **+USD 866,004** on the test set.

---

## 🎯 Business Objective

The Brazilian digital payments market processes trillions of reais annually. Companies like PicPay, BoaVista, and Travelex face the daily challenge of identifying fraudulent transactions in real time — without blocking legitimate customers or disrupting the user experience.

Manual rule-based systems don't scale. A model that learns behavioral patterns directly from data is fundamentally superior. The core business problem: **every dollar matters on both sides of the equation** — missed fraud costs money, but so does a false positive that blocks a legitimate transaction.

---

## 📂 Dataset

| Attribute | Detail |
|---|---|
| Source | [Credit Card Transactions Fraud Detection — Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
| License | ODbL (Open Database License) |
| Training set | 1,296,675 transactions |
| Test set | 555,719 transactions |
| Class imbalance | 99.42% legitimate / 0.58% fraud (ratio 171:1) |

---

## 🗂️ Pipeline

```
Raw data
    │
    ├── EDA (distribution, transaction value, category, age group)
    │
    ├── Preprocessing
    │       ├── Removal of irrelevant columns
    │       ├── Feature engineering (age, hour, day_of_week, distance)
    │       └── Label Encoding for categorical variables
    │
    ├── Class Balancing
    │       ├── Strategy 1: SMOTE + Undersampling
    │       └── Strategy 2: scale_pos_weight (winner)
    │
    ├── Modeling
    │       ├── Decision Tree (baseline)
    │       ├── XGBoost + SMOTE
    │       └── XGBoost + scale_pos_weight
    │
    ├── Evaluation
    │       ├── Metrics (Precision, Recall, F1, AUC-ROC)
    │       ├── Confusion matrices
    │       ├── Precision-Recall curve and threshold analysis
    │       └── Financial impact (net balance)
    │
    ├── Cross-Validation (Stratified K-Fold, 5 folds)
    │
    └── Hyperparameter Optimization (RandomizedSearchCV)
```

---

## 📊 Results

### Model Comparison

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Decision Tree | 0.16 | 0.93 | 0.27 | 0.9842 |
| XGBoost SMOTE | 0.19 | 0.90 | 0.32 | 0.9892 |
| XGBoost scale_pos_weight | 0.28 | 0.95 | 0.43 | 0.9972 |
| **XGBoost SPW optimized** | **0.78** | **0.81** | **0.79** | **0.9931** |

### Financial Impact by Scenario

| Scenario | Fraud prevented | Legitimate blocked | Net balance |
|---|---|---|---|
| SMOTE, threshold 0.5 | USD 1,122,095 | USD 2,050,334 | **-USD 928,239** |
| SMOTE, threshold 0.97 | USD 988,132 | USD 329,820 | **+USD 658,312** |
| SPW, threshold 0.5 | USD 1,122,442 | USD 1,810,154 | **-USD 687,712** |
| SPW, threshold 0.98 | USD 998,614 | USD 183,964 | **+USD 814,651** |
| **SPW optimized, threshold 0.85** | **USD 1,009,028** | **USD 143,024** | **+USD 866,004** |

---

## 💡 Key Findings

- **Accuracy is the wrong metric.** With 99.42% legitimate transactions, a model that classifies everything as legitimate would achieve 99.42% accuracy — and catch zero fraud.
- **SMOTE produces a negative balance at default threshold.** Synthetic data generation creates more aggressive decision boundaries, blocking too many legitimate transactions.
- **scale_pos_weight outperforms SMOTE across all metrics.** Training on real data with asymmetric penalization produces more precise patterns and fewer false positives.
- **Threshold is a business decision.** The same model with threshold 0.5 yields -USD 688k; with threshold 0.85 it yields +USD 866k. A difference of USD 1.55 million from a single parameter.
- **`amt` accounts for 50% of model importance.** Fraudsters concentrate attacks on high-value transactions, especially in `shopping_net` and `misc_net` categories.

---

## 🏆 Final Recommended Model

**XGBoost with optimized scale_pos_weight, threshold 0.85**

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 8 |
| `learning_rate` | 0.3 |
| `subsample` | 0.8 |
| `colsample_bytree` | 1.0 |
| `min_child_weight` | 3 |
| `gamma` | 0.3 |
| `scale_pos_weight` | 171.8 |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Class Balancing | Imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## ▶️ How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyter

# Download the dataset
# Go to https://www.kaggle.com/datasets/kartik2112/fraud-detection
# Place fraudTrain.csv and fraudTest.csv in the project root

# Launch the notebook
jupyter notebook Semantix.ipynb
```

---

## 👤 Author

**João Alfredo de Sousa Siqueira**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-oporaxuao-blue)](https://linkedin.com/in/oporaxuao)
[![GitHub](https://img.shields.io/badge/GitHub-oporaxuao-black)](https://github.com/oporaxuao)
