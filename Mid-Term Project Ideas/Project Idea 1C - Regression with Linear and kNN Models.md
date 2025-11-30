# Project Idea 1C: Regression with Linear and kNN Models

## Aim

To compare several core regression algorithms and concepts from the course—**Linear Regression** and **k-Nearest Neighbours (kNN) Regression**—on a single standard regression dataset from scikit-learn. The goals are to:

* understand how linear vs non-parametric (kNN) models behave on the same regression problem,
* analyse the effect of **L2 regularisation** and key training hyperparameters (such as learning rate) on linear models,
* study **model complexity** (k in kNN, regularisation strength) and its impact on overfitting and generalisation.

At least one, and preferably two, of these algorithms (most naturally **Linear Regression with gradient descent and L2 regularisation** and **kNN Regression**) will be implemented **from scratch** using only Python and NumPy (no ML libraries), with any remaining models (e.g. a closed-form linear regression solution or a scikit-learn regressor) used for comparison. This creates clear opportunities to go beyond the minimum coursework requirement in terms of implementation depth.

---

## Dataset

This project will use a **single standard regression dataset** from scikit-learn, such as the **Diabetes** dataset (`sklearn.datasets.load_diabetes()`) or the **California Housing** dataset (`sklearn.datasets.fetch_california_housing()`).

The chosen dataset will:

* involve a continuous target variable (e.g. disease progression or house price),
* be small to moderately sized so that **5-fold cross-validation on the training set** is practical,
* be relatively clean and numerical, requiring only light preprocessing (e.g. scaling, handling any missing values if needed).

This makes it ideal for focusing on regression algorithms themselves rather than extensive data cleaning.

---

## Algorithms

All algorithms are taken from the lecture list (linear regression, gradient descent, regularisation, kNN) and will be applied to the chosen regression dataset.

Planned algorithms:

* **Linear Regression (OLS)**

  * implemented with **gradient descent**, including an **L2 regularisation** term (ridge-style),
  * optionally compared with the closed-form normal equation (via NumPy) or scikit-learn's `LinearRegression`.

* **k-Nearest Neighbours (kNN) Regression**

  * using Euclidean distance on appropriately scaled features,
  * exploring different values of k to understand bias–variance trade-offs.

The intention is to implement **Linear Regression with gradient descent and L2 regularisation** and, if feasible, **kNN Regression** from scratch in Python/NumPy, while scikit-learn implementations (e.g. `LinearRegression`, `Ridge`, `KNeighborsRegressor`) are used for comparison and to support hyperparameter tuning experiments.

---

## Steps

### 1. Dataset Preparation

* Load the chosen regression dataset (e.g. Diabetes or California Housing) from scikit-learn.
* Inspect and clean the data:

  * handle missing values if present (or note if none),
  * remove clearly corrupted entries (if any),
  * separate features and target, and standardise/normalise features where appropriate.
* Split into **training (e.g. 80%)** and **test (20%)** sets.

### 2. Exploratory Data Analysis (EDA)

* Summarise basic statistics of each feature and the target.
* Visualise:

  * histograms of features and target,
  * scatter plots of selected features vs target to gauge linear vs non-linear relationships,
  * a simple correlation heatmap (if dimensionality is moderate).
* Comment on which features appear most strongly related to the target and whether relationships look roughly linear or more complex.

### 3. Baselines and From-Scratch Implementations

* Define a simple **naive baseline regressor**, such as always predicting the **mean** (or median) of the training target, and evaluate it as a reference point for all other models.
* Implement **Linear Regression with gradient descent and L2 regularisation from scratch**:

  * define the mean squared error (MSE) loss with an L2 penalty,
  * derive and implement gradient descent update rules for the weights and bias,
  * explore sensible choices of learning rate and number of iterations.
* Additionally, implement a simple **kNN Regression from scratch**:

  * use Euclidean distance on scaled features to find the k nearest neighbours of a query point,
  * predict by averaging the target values of the neighbours.
* Optionally verify the from-scratch implementations of **Linear Regression** and **kNN Regression** by comparing their predictions or metrics against the corresponding scikit-learn models (`LinearRegression`/`Ridge`, `KNeighborsRegressor`) on a subset of the data; the main experiments will then use the scikit-learn implementations for efficiency and more extensive hyperparameter searches.

### 4. Cross-Validation Strategy

* Because we will choose a **small to moderately sized** regression dataset, use **k-fold cross-validation (e.g. k = 5)** on the **training set**.
* Use average metrics across folds to:

  * estimate generalisation performance,
  * compare algorithms more robustly than a single train/test split,
  * select hyperparameters (regularisation strength, k in kNN).
* Note: for **much larger datasets**, a simpler hold-out validation split within the training data might be preferred for computational efficiency, whereas for very small tabular datasets with only a few hundred points k-fold cross-validation is especially important to make good use of limited data. Here we deliberately choose a small to moderately sized dataset where 5-fold cross-validation on the training set remains practical, and we will briefly justify this choice in the report.

### 5. Hyperparameter Tuning

* Use grid search or a small manual search over key hyperparameters:

  * Linear Regression: regularisation strength (e.g. L2 penalty parameter lambda) and, where necessary, learning rate for gradient descent.
  * kNN Regression: number of neighbours `k` (e.g. 1, 3, 5, 7, 9).
* Perform hyperparameter tuning using cross-validation on the **training set** only.
* Select the best hyperparameters based on validation performance (e.g. lowest average RMSE or MAE).
* Retrain each model with its best hyperparameters on the full training set and evaluate on the held-out test set using:

  * **MSE (Mean Squared Error)**,
  * **RMSE (Root Mean Squared Error)**,
  * **MAE (Mean Absolute Error)**,
  * and optionally **R² (coefficient of determination)**.
* Compare these metrics against the naive baseline to show absolute performance gains.

### 6. Model Complexity & Regularisation Analysis

* Investigate **overfitting vs underfitting** and the effect of model complexity:

  * For **Linear Regression with L2 regularisation**: vary the regularisation strength lambda and examine how performance and weight magnitudes change; discuss the bias–variance trade-off.
  * For **kNN Regression**: vary `k` and observe how performance changes (small k → low bias, high variance; large k → higher bias, lower variance).
* Plot performance vs model complexity/regularisation (e.g. validation RMSE vs lambda or k) to illustrate these effects.
* Plot performance vs model complexity/regularisation (e.g. validation RMSE vs degree or (\lambda)) to illustrate these effects.

### 7. Feature Importance & Interpretability

* For **Linear Regression**:

  * examine the fitted coefficients to understand which features (or polynomial terms) have the largest influence on the target,
  * discuss how regularisation (L2) shrinks coefficients and helps mitigate overfitting, particularly when multicollinearity or noisy features are present.
* For **kNN Regression**:

  * discuss interpretability in terms of local neighbourhoods (predictions as averages of nearby points), and
  * comment on how scaling and distance metric choices affect which neighbours are most influential.
* Relate the findings back to EDA (e.g. whether features that looked important in scatter plots also have large coefficients in the linear models).

### 8. Model Comparison

* Summarise final tuned models and results on the **held-out test set** in a table, e.g.:

| Model                           | Key Hyperparameters (tuned)                      | Test RMSE | Test MAE | Test R² |
| ------------------------------- | ------------------------------------------------ | --------- | -------- | ------- |
| Naive Baseline (mean predictor) | –                                                | …         | …        | …       |
| Linear Regression (GD + L2)     | learning rate = …, iterations = …, (\lambda) = … | …         | …        | …       |
| Polynomial Regression           | degree = …, (\lambda) = …                        | …         | …        | …       |
| kNN Regression                  | k = …                                            | …         | …        | …       |

* Discuss the trade-offs by:

  * Identifying which model performs best overall based on the chosen test metrics (e.g. lowest RMSE and MAE, highest R²) and explaining *why* it performs best in terms of assumptions (linear vs non-linear) and flexibility.
  * Assessing which model is most robust to hyperparameters by examining how its performance varies across different hyperparameter settings (e.g. degree, (\lambda), k) and summarising which model is least sensitive.
  * Comparing interpretability by discussing how easy it is to explain each model’s predictions (e.g. coefficients for linear/polynomial regression vs more opaque kNN behaviour).

---

## Report Structure (Aligned with Coursework)

1. **Abstract**
   Concisely summarise the aim (regression comparison using linear, polynomial, and kNN models), methods (datasets, from-scratch implementations, cross-validation, hyperparameter tuning), key results, and main conclusions.

2. **Introduction**

   * Introduce regression problems and why predicting a continuous target (e.g. disease progression or house price) is important.
   * State the chosen dataset and target variable.
   * Outline the aim: compare multiple regression algorithms (linear, polynomial, kNN) and understand the roles of gradient descent, regularisation, and model complexity.

3. **Background**

   * Describe Linear Regression and Polynomial Regression, including the concept of adding polynomial features and the role of **gradient descent** and **L2 regularisation**.
   * Explain k-Nearest Neighbours Regression and its reliance on distance metrics and local averaging.
   * Discuss overfitting/underfitting in regression and the bias–variance trade-off, relating these to polynomial degree, regularisation strength, and k.

4. **Methodology**

   * Dataset description and preprocessing (scaling, handling missing values, splitting).
   * From-scratch implementations of Linear Regression (with gradient descent and L2) and kNN Regression.
   * Train–test split and cross-validation strategy, with justification based on dataset size.
   * Hyperparameter tuning procedure (grids for degree, (\lambda), k, learning rate) and chosen evaluation metrics (MSE, RMSE, MAE, R²).

5. **Results**

   * Performance tables for each model (baseline, linear, polynomial, kNN).
   * Cross-validation results for key hyperparameters.
   * Plots of validation performance vs model complexity/regularisation (degree, (\lambda), k).
   * Explicit comparison of tuned models against the naive baseline regressor to show absolute performance gains.

6. **Evaluation**

   * Critical discussion of:

     * which models performed best and why,
     * evidence of overfitting/underfitting (e.g. high-degree polynomials vs regularised models),
     * sensitivity to hyperparameters and learning rates,
     * trade-offs between performance, complexity, and interpretability.
   * Limitations (e.g. choice of dataset, simplicity of feature engineering).

7. **Conclusions**

   * Summarise the main findings about linear vs polynomial vs kNN regression and the impact of regularisation and model complexity.
   * Highlight which approach you would recommend for this type of regression problem and why.

8. **References**

   * Textbook, lecture notes, and any external sources (papers, tutorials) used for algorithms, regularisation, or datasets.
