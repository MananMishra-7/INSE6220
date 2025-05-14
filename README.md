# 🧠 Classification of Parkinson’s Disease Using PCA and Supervised Learning

![Python](https://img.shields.io/badge/Language-Python-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PCA](https://img.shields.io/badge/Technique-PCA%20%2B%20Supervised%20Learning-brightgreen)

## 📌 Overview

This project implements **Principal Component Analysis (PCA)** and **Supervised Machine Learning models** to classify individuals as Parkinson’s-positive or healthy using voice measurements. The aim is to reduce dimensionality and improve model performance and interpretability.

- PCA is used to reduce correlated features.
- Three classifiers are used: **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **Logistic Regression**.
- Results are compared before and after dimensionality reduction using metrics like **Accuracy**, **Precision**, **Recall**, and **AUC**.

## 🧪 Tools and Libraries

- Python
- [PyCaret](https://pycaret.org/)
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn

## 📊 Dataset

- **Source:** UCI Machine Learning Repository  
- **Features:** 24 total (voice measurements), 195 entries  
- **Target:** `status` — 0 (healthy), 1 (Parkinson's)  
- **Preprocessing:**
  - Removed the `name` feature
  - Selected key features based on correlation and medical relevance
  - Standardized data

## 🔍 Exploratory Data Analysis

- Checked class balance: 48 healthy, 147 Parkinson’s
- Detected and retained outliers due to medical significance
- Visualized feature distribution using box plots and heatmaps

## 🧠 PCA (Principal Component Analysis)

- Reduced data to **2 principal components** while retaining **79.8% of variance**
- Used scree plot and Pareto chart to determine optimal component count
- Visualized PCA components and contributions using **biplots**

## 🤖 Machine Learning Models

Three models were implemented under two conditions:

1. **Without PCA** – Trained on original dataset
2. **With PCA** – Trained on reduced dataset (2 principal components)

### Models Used

- **K-Nearest Neighbors (KNN)**
- **Decision Tree (DT)**
- **Logistic Regression (LR)**

Each model was tuned using **K-fold cross-validation** via PyCaret’s `tune_model()`.

### Performance Summary

| Model                | PCA  | Accuracy | Precision | Recall | AUC  |
|---------------------|------|----------|-----------|--------|------|
| K-Nearest Neighbors | No   | 0.86     | 0.89      | 0.94   | 0.77 |
| K-Nearest Neighbors | Yes  | 0.90     | 0.93      | 0.94   | 0.95 |
| Decision Tree       | No   | 0.87     | 0.92      | 0.91   | 0.81 |
| Decision Tree       | Yes  | 0.88     | 0.93      | 0.91   | 0.85 |
| Logistic Regression | No   | 0.88     | 0.91      | 0.93   | 0.70 |
| Logistic Regression | Yes  | 0.85     | 0.88      | 0.94   | 0.84 |

🔹 **Best Performance:** KNN with PCA (Accuracy: 0.90, AUC: 0.95)

## 🧠 Explainable AI (XAI)

Used **Shapley values** for model explainability:

- Visualized feature importance (SHAP summary plot)
- Used force plots to explain individual predictions

## 🎯 Conclusion

- **Dimensionality reduction** using PCA improved AUC across all models
- **K-Nearest Neighbors** outperformed other classifiers both with and without PCA
- Shapley values helped interpret model decisions

