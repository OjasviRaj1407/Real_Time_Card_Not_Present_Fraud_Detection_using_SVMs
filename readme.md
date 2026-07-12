# 💳 Credit Card Fraud Detection using Support Vector Machines

## 📌 Overview

This project implements a complete fraud detection pipeline using **Support Vector Machines (SVMs)** to identify fraudulent credit card transactions.

The notebook explores multiple SVM kernels, performs hyperparameter tuning using GridSearchCV, applies feature selection, and evaluates models using metrics suitable for highly imbalanced datasets.

This project was developed as part of a Machine Learning assignment focusing on rare-event classification.

---

## 🎯 Objectives

- Explore the characteristics of fraudulent transactions.
- Handle severe class imbalance.
- Compare Linear, Polynomial and RBF SVM kernels.
- Tune hyperparameters using GridSearchCV.
- Reduce dimensionality using feature selection.
- Evaluate models using Precision, Recall, F1-score and PR-AUC.

---

## 📂 Dataset

**Dataset Name**

Credit Card Fraud Detection

**Source**

OpenML (Version 1)

**Original Dataset**

Kaggle Credit Card Fraud Detection Dataset

**Features**

- PCA Features (V1–V28)
- Transaction Amount
- Fraud Label (IsFraud)

**Target**

- 0 → Legitimate
- 1 → Fraudulent

---

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenML

---

## 🧠 Machine Learning Pipeline

```
Dataset
      │
      ▼
EDA
      │
      ▼
Data Preprocessing
      │
      ▼
StandardScaler
      │
      ▼
Linear SVM
Polynomial SVM
RBF SVM
      │
      ▼
GridSearchCV
      │
      ▼
Feature Selection
      │
      ▼
Model Evaluation
```

---

## 📊 Evaluation Metrics

Since fraud detection is a highly imbalanced classification problem, the following metrics are used:

- Precision
- Recall
- F1-score
- Precision-Recall AUC
- Confusion Matrix

---

## 📈 Results

The notebook compares:

| Kernel | Precision | Recall | F1 | PR-AUC | Training Time |
|---------|----------:|-------:|---:|--------:|--------------:|
| Linear | ✓ | ✓ | ✓ | ✓ | ✓ |
| Polynomial | ✓ | ✓ | ✓ | ✓ | ✓ |
| RBF | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 🚀 Best Model

The tuned **RBF Support Vector Machine** achieved the best balance between precision and recall while maintaining the highest PR-AUC.

Feature selection reduced computational cost without significantly affecting predictive performance.

---

## 📁 Repository Structure

```
SVM-CreditCard-Fraud-Detection/
│
├── credit_card_fraud_detection.ipynb
├── README.md
├── requirements.txt
├── LICENSE
├── comparison_results.csv
├── svm_results.csv
└── images/
```

---

## ▶️ Running the Project

Clone the repository

```bash
git clone https://github.com/yourusername/SVM-CreditCard-Fraud-Detection.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Run all cells from top to bottom.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Ojasvi Raj**