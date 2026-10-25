# Loan Approval Prediction with Machine Learning

A collaborative data mining course project that compares classification models for predicting whether a loan application will be approved or rejected. The workflow covers exploratory data analysis, preprocessing experiments, feature engineering, model tuning, decision-threshold selection, and interpretation.

**Tools:** Python, pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, SHAP, and Jupyter.

## Project overview

The project investigates how applicant characteristics and credit history relate to historical loan approval decisions. Six classifiers are evaluated within preprocessing pipelines, with cross-validation used for tuning and stability comparisons.

The target is **historical approval status**, not repayment, default, or independently measured creditworthiness. Results describe performance on this dataset and should be interpreted within that scope.

The notebook and final report contain Persian explanations, with English code identifiers and many English plot labels.

## Dataset

| File | Rows | Columns | Role |
| --- | ---: | ---: | --- |
| `loan_train.csv` | 614 | 12 | 11 predictors and the labeled target `Status` |
| `loan_test.csv` | 367 | 11 | Unlabeled observations for final predictions |

The target mapping is `N = 0` (rejected) and `Y = 1` (approved). The approved class is more frequent, so accuracy is considered alongside class-sensitive and ranking metrics.

### Features

| Column | Description |
| --- | --- |
| `Gender` | Applicant gender |
| `Married` | Marital status |
| `Dependents` | Number of dependents |
| `Education` | Education category |
| `Self_Employed` | Self-employment indicator |
| `Applicant_Income` | Applicant income |
| `Coapplicant_Income` | Coapplicant income |
| `Loan_Amount` | Requested loan amount |
| `Term` | Loan term |
| `Credit_History` | Credit-history indicator |
| `Area` | Residential area category |
| `Status` | Approval/rejection label, available only in training data |

**Dataset source cited in the project proposal:** [Loan Data Set on Kaggle](https://www.kaggle.com/datasets/mirzahasnine/loan-data-set). The CSV schema above describes the supplied project copies; it does not imply that column names are unchanged from the source. Consult the source for data documentation and reuse terms.

## Analysis workflow

### 1. Data understanding and exploratory analysis

The notebook examines data types, duplicate records, invalid values, missingness, class balance, numeric distributions, outliers, correlations, and approval rates across applicant groups.

### 2. Training and validation design

The labeled data are split into **491 training observations** and **123 validation observations**, using a stratified 80/20 split with random seed **42**. The external 367-row file remains unlabeled and is used for final predictions rather than performance measurement.

Preprocessing transformations and resampling are included in fitted pipelines so that their fitted parameters are learned within training folds. The workflow uses **3-fold stratified cross-validation** for its main comparisons and tuning.

### 3. Preprocessing and feature engineering

Experiments compare missing-value treatments, scaling, transformations for skewness and outliers, and class-imbalance approaches. These include simple and KNN imputation, multiple scalers, logarithmic transformations, random oversampling, SMOTE, and class weighting.

The saved report identifies **mean imputation, MinMax scaling, logarithmic transformation, and no resampling** as the selected configuration. This is an empirical choice for this dataset; resampling is evaluated rather than assumed to improve performance.

### 4. Model comparison and hyperparameter tuning

| Classifier | Purpose in the comparison |
| --- | --- |
| Logistic Regression | Linear classification baseline |
| K-Nearest Neighbors | Distance-based classifier |
| Support Vector Machine | Margin-based classifier |
| Decision Tree | Rule-based nonlinear classifier |
| Random Forest | Ensemble of randomized trees |
| Gradient Boosting | Sequential boosted-tree ensemble |

Grid search and randomized search use cross-validation, with **F1 as the refit metric**. The four leading tuned models are compared using **5-fold stratified cross-validation repeated 3 times**. The final model is selected by repeated-CV mean F1, with balanced accuracy and ROC-AUC used as tie-breaking criteria in the code.

These comparisons reuse the available training sample for multiple modeling decisions; they are not a nested-CV estimate of the complete selection process.

### 5. Metrics and threshold analysis

Evaluation includes accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrices, and classification reports. Precision–recall and ROC plots supplement the scalar metrics.

Out-of-fold probability predictions are used to select an F1-oriented decision threshold. The notebook distinguishes the threshold estimated on the training split from the threshold subsequently estimated using all labeled data for final external predictions.

For the reported binary metrics, **approved applications are the positive class**. In this setting, a false positive means predicting approval for a historically rejected application; it does not establish a future loan default.

### 6. Interpretation and error analysis

The workflow includes permutation importance, SHAP explanations, partial dependence, a custom **LIME-style local surrogate**, error analysis, and subgroup performance comparisons. The local surrogate is implemented in the notebook and should not be described as a call to the official LIME package.

Credit history is the leading feature in the report's exploratory, permutation-importance, and SHAP analyses. These explanations characterize model behavior and associations, not causal effects.

## Selected reported results



**Selected model: SVM**

| Evaluation setting | Metric | Reported value |
| --- | --- | ---: |
| Repeated training cross-validation | Mean F1 | 0.8707 |
| Repeated training cross-validation | Standard deviation of F1 | 0.0213 |
| Held-out validation, default prediction rule | Accuracy | 0.8537 |
| Held-out validation, default prediction rule | Approved-class recall | 0.9882 |
| Held-out validation, default prediction rule | Approved-class F1 | 0.9032 |

The reported default-rule validation confusion matrix contains **84 true positives, 21 true negatives, 17 false positives, and 1 false negative**. High approved-class recall therefore coexists with weaker detection of rejected applications.

Gradient Boosting achieved a higher F1 on the single validation split, but SVM was selected using repeated cross-validation. Different validation criteria can produce different rankings.

### Final external predictions

After selection, the SVM pipeline is refitted using all **614 labeled observations**. A new full-data out-of-fold analysis selects a threshold of **0.265**, compared with **0.545** in the earlier training-split threshold analysis.

The report records **308 predicted approvals** and **59 predicted rejections** for the 367 unlabeled observations. These counts are predictions, not verified outcomes. Accuracy, F1, ROC-AUC, and confusion matrices cannot be computed for this external file without true labels.

## Repository files

| Path | Content |
| --- | --- |
| `README.md` | Project documentation |
| `DataMining_Project_Group1_Main.ipynb` | Complete analysis notebook |
| `loan_train.csv` | Labeled input data |
| `loan_test.csv` | Unlabeled input data |
| `reports/loan_approval_report.pdf` | Final Persian report |





## Academic context and team

Developed for **Fundamentals of Data Mining and Its Applications**, Sharif University of Technology, Spring 1405.

**Team:**

- Pardis Rahiminasr
- Sonia Dabbagh Tafreshi

**Instructor:** Dr. Maryam Bostanara.



