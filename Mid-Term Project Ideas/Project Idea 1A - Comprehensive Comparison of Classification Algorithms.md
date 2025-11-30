# Project Idea 1A: Comprehensive Comparison of Classification Algorithms

## Aim

To compare several core classification algorithms covered in the course—k-Nearest Neighbours (kNN), Naive Bayes (Gaussian), Logistic Regression, and Decision Tree classification—on a single standard classification dataset from scikit-learn (e.g. the Breast Cancer Wisconsin Diagnostic, Wine, or Iris datasets) in order to:

* identify which algorithm performs best under consistent conditions, and
* understand the trade-offs between accuracy, overfitting, regularisation, and interpretability.

At least one of these algorithms (likely kNN or Gaussian Naive Bayes) will be implemented **from scratch** in a separate pure-Python/NumPy implementation (using only Python and NumPy and no ML libraries), with the others possibly using **scikit-learn** for comparison.

---

## Datasets

This project will use a **single standard classification dataset** from scikit-learn, such as the **Breast Cancer Wisconsin Diagnostic** dataset (`sklearn.datasets.load_breast_cancer()`), the **Wine** dataset (`sklearn.datasets.load_wine()`), or the **Iris** dataset (`sklearn.datasets.load_iris()`).

The chosen dataset will be small and relatively clean, making it ideal for focusing on the algorithms themselves rather than extensive data preprocessing, while still allowing the use of rich evaluation metrics such as accuracy, precision, recall, and F1 score. Where the problem context makes particular types of errors more costly (for example, false negatives in a medical diagnosis setting), the analysis will pay special attention to the relevant class-specific metrics and the associated precision–recall trade-off.

---

## Algorithms

All algorithms are from the lecture list and may be implemented either:

* **from scratch** (NumPy + matplotlib only), or
* via **scikit-learn**,

with the constraint that the coursework requires *at least one* from-scratch implementation.

Planned algorithms:

* **k-Nearest Neighbours (kNN)**
* **Naive Bayes**

  * Gaussian NB (continuous features)
* **Logistic Regression** (with L2 regularisation; optimisation discussed in terms of gradient descent)
* **Decision Tree Classifier** (using the entropy criterion)

---

## Steps

### 1. Dataset Preparation

* Load the dataset from scikit-learn.
* Handle any missing values if present (or note if none).
* Split into **training (80%)** and **test (20%)** sets.
* Standardise features (e.g. with `StandardScaler`) for algorithms that are sensitive to scale (kNN, Logistic Regression).

### 2. Exploratory Data Analysis (EDA)

* Summarise basic statistics and visualise:

  * class distribution,
  * feature histograms/boxplots,
  * simple pair plots or scatter plots.
* Comment briefly on how separable the classes appear.

### 3. Algorithm Implementation & Baseline Training

* Implement at least one algorithm from scratch (e.g. kNN or Naive Bayes) using only Python + NumPy.
* Define a simple **naive baseline classifier** (e.g. always predicting the majority class) and evaluate it as a reference point for all other models.

### 4. Cross-Validation

* Because this dataset is of **small to moderate size**, perform **k-fold cross-validation (e.g. k = 5)** on the **training set** only.
* Use average metrics across folds to:

  * estimate generalisation performance,
  * compare algorithms more robustly than a single train/test split.
* Note: for **much larger datasets**, a single hold-out validation split might be preferred for computational efficiency, whereas for smaller datasets k-fold cross-validation helps make better use of limited data.

### 5. Hyperparameter Tuning

* Use grid search (or similar) with cross-validation on the **training set** to tune key hyperparameters:

  * kNN: `n_neighbors (k)`, distance metric.
  * Naive Bayes: usually few or no hyperparameters; use the standard Gaussian NB implementation with its default `var_smoothing` for numerical stability rather than treating it as a tuned hyperparameter.
  * Logistic Regression: regularisation strength `C`, penalty type (e.g. L2).
  * Decision Tree: `max_depth`, `min_samples_split` (with the criterion fixed to `entropy`).
* Select the best hyperparameters based on validation performance.
* Retrain each model with its best hyperparameters on the full training set and evaluate on the held-out test set using:

  * **Accuracy**, **Precision**, **Recall**, **F1 score**.
* Include **confusion matrices** where helpful for a more detailed error breakdown.

### 6. Model Complexity & Regularisation Analysis

* Investigate **overfitting vs underfitting**:

  * For kNN: vary `k` and observe train vs validation performance.
  * For Decision Tree: vary `max_depth`.
  * For Logistic Regression: vary **C** (regularisation strength).
* Plot performance vs model complexity to illustrate where overfitting occurs.

### 7. Feature Importance & Interpretability

* For **Logistic Regression**: interpret feature importance via the magnitude/sign of coefficients.
* For **Decision Tree**: use feature importance scores from the entropy-based tree (e.g. information gain) and visualise a small tree if feasible.
* Discuss which features appear most important and whether this matches intuition/EDA.

### 8. Model Comparison

* Summarise final tuned models and results on the **held-out test set** in a table, e.g.:

| Model               | Key Hyperparameters (tuned)                               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | --------------------------------------------------------- | -------- | --------- | ------ | -------- |
| kNN                 | k = …, metric = …                                         | …%       | …%        | …%     | …%       |
| Naive Bayes         | –                                                         | …%       | …%        | …%     | …%       |
| Logistic Regression | C = …, penalty = L2, solver = …                           | …%       | …%        | …%     | …%       |
| Decision Tree       | criterion = entropy, max_depth = …, min_samples_split = … | …%       | …%        | …%     | …%       |

* Discuss the trade-offs by:

  * Identifying which algorithm performs best overall based on the chosen test metrics (e.g. F1 score for the positive or most critical class and overall accuracy) and explaining *why* it performs best in terms of its assumptions and decision boundaries.
  * Assessing which algorithm is most robust to hyperparameters by examining how its performance varies across different hyperparameter settings (e.g. different values of k for kNN, max_depth for Decision Trees, and C for Logistic Regression) and summarising which model is least sensitive to these changes.
  * Comparing interpretability by discussing how easy it is to explain each model’s predictions to a non-technical audience (e.g. coefficients and odds ratios for Logistic Regression, decision paths for the Decision Tree, versus more opaque behaviour of kNN and Naive Bayes).

---

## Report Structure (Aligned with Coursework)

1. **Abstract**
   Concise summary of the aim, methods, key results, and main conclusion.

2. **Introduction**
   Context: supervised classification in machine learning.
   Motivation: why comparing multiple classifiers on the same problem is useful.
   Brief description of the dataset and target variable.

3. **Background**
   Explain each algorithm (kNN, Naive Bayes, Logistic Regression, Decision Tree) and the optimisation method **gradient descent**, which is typically used to train Logistic Regression:

   * main idea,
   * underlying assumptions,
   * typical strengths and weaknesses,
   * where overfitting and regularisation come in.

4. **Methodology**

   * Dataset preparation and EDA.
   * Train–test split and cross-validation strategy, with justification based on dataset size (moderately sized dataset ⇒ 80/20 split plus 5-fold cross-validation on the training set).
   * Implementation details (which algorithms are from scratch, which via scikit-learn).
   * Hyperparameter tuning procedure and evaluation metrics.

5. **Results**

   * Naive baseline vs tuned model performance.
   * Cross-validation results.
   * Confusion matrices and any plots of performance vs complexity.

6. **Evaluation**

   * Critical discussion of:

     * why some algorithms performed better than others,
     * overfitting/underfitting observations,
     * effect of regularisation and hyperparameters,
     * implications of false negatives (where relevant for the chosen dataset) and the importance of recall for the positive or most critical class and the precision–recall trade-off.
   * Limitations (e.g. small dataset, simple models).

7. **Conclusions**

   * Which classifier is recommended for this specific dataset and why.
   * Key lessons learned about each algorithm’s behaviour.

8. **References**

   * Textbook, lecture notes, and any external sources used.
