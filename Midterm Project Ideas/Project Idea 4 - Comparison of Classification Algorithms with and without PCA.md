# Project Idea 4: Comparison of Classification Algorithms with and without PCA

## Aim

To investigate how applying **Principal Component Analysis (PCA)** as a dimensionality-reduction step affects the performance of several core classification algorithms from the course on a single standard dataset from scikit-learn.

The goals are to:

* compare classifier performance on the **original feature space** vs a **PCA-reduced feature space**,
* understand how PCA interacts with different classifier types (distance-based, probabilistic, linear, tree-based),
* analyse trade-offs between **performance, overfitting, computational cost, and interpretability** when PCA is used.

The intention is to implement **PCA** and at least one classifier (most naturally **Gaussian Naive Bayes** or **k-Nearest Neighbours (kNN)**) **from scratch** using only Python and NumPy (no ML libraries). Logistic Regression and Decision Trees (and, if desired, scikit-learn versions of kNN / Naive Bayes) will be used via **scikit-learn** for comparison and to support cross-validation and hyperparameter tuning. PCA dimensionality will be chosen once using a simple variance-based rule and then kept fixed; cross-validation will be used only to optimise classifier hyperparameters.

---

## Dataset

This project will use a **single standard classification dataset** from scikit-learn, such as:

* **Iris** (`sklearn.datasets.load_iris()`),
* **Wine** (`sklearn.datasets.load_wine()`), or
* **Breast Cancer Wisconsin Diagnostic** (`sklearn.datasets.load_breast_cancer()`).

The chosen dataset will:

* have a **small to moderate number of features** (e.g. 4–30) so that PCA is meaningful but not overwhelming,
* be small to moderately sized so that **5-fold cross-validation on the training set** is practical,
* be relatively clean and numerical, requiring only light preprocessing (standardisation; handling any missing values if needed).

If the dataset is class-imbalanced (e.g. Breast Cancer), the analysis will pay particular attention to **recall and F1** for the minority / most critical class; for more balanced datasets (e.g. Iris, Wine), **accuracy and macro F1** will be emphasised.

---

## Algorithms

All classifiers are taken from the lecture list and applied to the same dataset.

Planned classifiers:

* **k-Nearest Neighbours (kNN) Classifier**

  * distance metric: Euclidean (and optionally others, e.g. Minkowski),
  * key hyperparameter: number of neighbours `k`.

* **Gaussian Naive Bayes**

  * assumes normally distributed features within each class,
  * simple probabilistic model that reacts strongly to scaling and correlations (hence interesting under PCA).

* **Logistic Regression**

  * with **L2 regularisation** (ridge-style),
  * implemented via scikit-learn (`LogisticRegression`) but discussed in terms of gradient descent and regularisation.

* **Decision Tree Classifier**

  * using a fixed impurity criterion (e.g. `entropy`),
  * key hyperparameters: `max_depth`, `min_samples_split`.

From-scratch implementation plan:

- In line with the Aim, **PCA** and at least one classifier (most naturally **Gaussian Naive Bayes** or **kNN**) will be implemented **from scratch** in pure Python + NumPy, and, if feasible, both classifiers.  
- Compare these from-scratch implementations against scikit-learn counterparts (e.g. `GaussianNB`, `KNeighborsClassifier`, `PCA`) on a subset of the data to validate correctness.  
- Use the scikit-learn implementations in the main experiments to run more extensive cross-validation and hyperparameter tuning efficiently.

---

## PCA (From Scratch and Choice of Dimensionality)

PCA will be implemented from scratch as follows:

1. Standardise the training data to have zero mean (and optionally unit variance) per feature.
2. Compute the **covariance matrix** of the standardised training data.
3. Compute the **eigenvalues and eigenvectors** of the covariance matrix.
4. Sort eigenvalues in descending order and form the corresponding ordered eigenvectors.
5. Choose a number of components `d` based on the training data using a simple rule:

   * either **cumulative explained variance** (choose the smallest `d` that captures, for example, 90–95% of the total variance), or
   * the **Kaiser rule** (keep components with eigenvalues greater than 1, assuming features are standardised).
6. Construct a projection matrix from the top `d` eigenvectors and project the standardised data onto these `d` principal components to obtain a lower-dimensional representation.
7. To transform new data (validation/test), apply the same standardisation (using training means/stds) and the same projection matrix.

The PCA dimensionality `d` is thus determined once, using the training data. In practice, we will primarily use a cumulative explained variance threshold (e.g. 90–95%) to choose `d`, with the Kaiser rule serving mainly as a sanity check. Once chosen, `d` is kept fixed for all experiments. Cross-validation will not tune `d`; it will only tune classifier hyperparameters.

---

## Steps

### 1. Dataset Preparation

* Load the chosen dataset from scikit-learn.
* Inspect and clean the data:

  * handle missing values if present (or report that there are none),
  * check for obvious outliers or corrupted entries (if any).
* Split into **training (e.g. 80%)** and **test (20%)** sets, stratified by class to preserve class proportions.
* Standardise features (e.g. using `StandardScaler` or equivalent) based on the training data; apply the same transformation to the test data.

### 2. Exploratory Data Analysis (EDA)

* Summarise class distribution (check for imbalance).
* Visualise:

  * histograms / boxplots of selected features,
  * a correlation heatmap to identify redundant or highly correlated features,
  * if suitable, a 2D projection of the data (e.g. onto the first two principal components) to provide a rough visual sense of class separability.
* Comment on:

  * which features appear most discriminative,
  * any signs of multicollinearity,
  * whether the data looks linearly separable or more complex.

### 3. Baselines and From-Scratch Implementations

* Define a simple **naive baseline classifier** (e.g. always predicting the majority class) and evaluate it as a reference point.
* Implement **Gaussian Naive Bayes** and/or **kNN classifier from scratch**:

  * for NB: estimate class priors and per-class feature means/variances; predict via class posteriors (using log-probabilities),
  * for kNN: compute distances to training samples and use majority vote among the `k` nearest neighbours.
* Implement **PCA from scratch** as described above.
* Optionally verify the from-scratch PCA and classifier implementations by comparing their predictions/metrics to the corresponding scikit-learn implementations on a subset of the data.

### 4. Cross-Validation Strategy

* Use **k-fold cross-validation (e.g. k = 5)** on the **training set**.
* After `d` has been chosen from the training data (via cumulative explained variance or the Kaiser rule):

  * For each fold:

    * split the training data into a **fold-specific training subset** and **validation subset**,
    * for the PCA experiments, fit PCA **only on the fold’s training subset** (using the fixed number of components `d`) and transform both the fold’s training and validation data,
    * train each classifier on:

      * the original (non-PCA) features, and
      * the PCA-transformed features (with fixed `d`),
        using the fold’s training subset,
    * evaluate on the fold’s validation subset.
* Use average metrics across folds to:

  * estimate generalisation performance on original vs PCA-reduced feature spaces,
  * compare classifiers and decide which hyperparameters work best.

The PCA dimensionality `d` is **not** treated as a hyperparameter in the CV grid; it is fixed by the variance-based rule above. PCA is always fit **within each training fold** and applied to that fold’s validation data to avoid data leakage.

### 5. Hyperparameter Tuning

* Use a small grid search (or systematic manual search) over key classifier hyperparameters:

  * **kNN**: number of neighbours `k` (e.g. {1, 3, 5, 7, 9}) and optionally the distance metric.
  * **Gaussian Naive Bayes**: typically default parameters; optionally explore a small range of `var_smoothing` for the scikit-learn version if needed for numerical stability.
  * **Logistic Regression**: regularisation strength `C` (e.g. {0.01, 0.1, 1, 10}) with L2 penalty and an appropriate solver.
  * **Decision Tree**: `max_depth` (e.g. {3, 5, 7, None}) and `min_samples_split` (e.g. {2, 5}).

* Perform hyperparameter tuning using cross-validation on the **training set only**, separately for:

  * models trained on original features, and
  * models trained on PCA-transformed features (with fixed `d`).

* Select the best hyperparameters for each classifier based on appropriate CV metrics:

  * **Accuracy + macro F1** overall,
  * and, where relevant, **class-specific recall / F1** for the most critical class.

### 6. Training and Evaluation

* Retrain each classifier with its **best hyperparameters** under each representation (original vs PCA) on the **full training set**.
* Evaluate on the **held-out test set** using:

  * Accuracy,
  * Precision, Recall, F1 (macro and/or per-class),
  * Confusion matrices for selected models (especially for the best-performing variants).
* Compare results against the naive baseline to show absolute performance gains.

### 7. Model Comparison (Original vs PCA)

* Summarise final test-set performance in a table such as:

| Model               | Features    | Key Hyperparameters                  | Accuracy | Macro F1 | Precision (pos) | Recall (pos) |
| ------------------- | ----------- | ------------------------------------ | -------- | -------- | --------------- | ------------ |
| kNN                 | Original    | k = …, metric = Euclidean            | …%       | …        | …               | …            |
| kNN                 | PCA-reduced | k = …, metric = Euclidean            | …%       | …        | …               | …            |
| Naive Bayes         | Original    | –                                    | …%       | …        | …               | …            |
| Naive Bayes         | PCA-reduced | –                                    | …%       | …        | …               | …            |
| Logistic Regression | Original    | C = …, penalty = L2                  | …%       | …        | …               | …            |
| Logistic Regression | PCA-reduced | C = …, penalty = L2                  | …%       | …        | …               | …            |
| Decision Tree       | Original    | max_depth = …, min_samples_split = … | …%       | …        | …               | …            |
| Decision Tree       | PCA-reduced | max_depth = …, min_samples_split = … | …%       | …        | …               | …            |

For binary problems such as Breast Cancer, “Precision (pos)” and “Recall (pos)” refer to the positive (typically more critical) class. For multi-class datasets such as Iris or Wine, precision and recall will instead be reported as macro-averaged scores across classes.

* Discuss trade-offs by:

  * identifying which **model + feature representation** (original vs PCA) performs best and why,
  * analysing whether PCA tends to help or hurt each classifier type (e.g. kNN may benefit from de-correlated features; trees may not gain much),
  * evaluating whether dimensionality reduction leads to noticeable changes in training time, overfitting behaviour, or robustness.

### 8. Feature Importance & Interpretability

* **Logistic Regression**:

  * on the original feature space, use the magnitude and sign of coefficients to interpret which features increase or decrease the odds of belonging to a particular class,
  * discuss how interpretability changes when working in PCA space (principal components are linear combinations of original features).

* **Decision Tree**:

  * on the original features, use tree-based feature importances to see which features are most influential,
  * compare this with any patterns seen in the PCA loadings (which original features contribute most to the principal components).

* **PCA itself**:

  * inspect explained variance per component,
  * examine which original features load most heavily on the most important principal components.

Relate these findings back to EDA (e.g. whether features that appeared important visually also play a large role in classifiers and PCA components).

### 9. Model Complexity Analysis

* Investigate how model complexity affects performance under original vs PCA-reduced features:

  * For kNN: vary `k` and observe validation/test performance; check whether the optimal `k` differs between original and PCA features.
  * For Decision Tree: vary `max_depth` and monitor overfitting/underfitting in both feature spaces.
  * For Logistic Regression: vary `C` (regularisation strength) and examine how performance and coefficient magnitudes change.

* Plot validation performance vs complexity (e.g. macro F1 vs k, macro F1 vs max_depth, macro F1 vs C) for original vs PCA features to visualise differences in bias–variance behaviour.

---

## Report Structure (Aligned with Coursework)

1. **Abstract**
   Briefly summarise the aim (comparison of multiple classifiers on original vs PCA-reduced features), dataset, methods (from-scratch PCA and at least one classifier, cross-validation, hyperparameter tuning), key results, and main conclusions.

2. **Introduction**

   * Motivate the importance of dimensionality reduction (e.g. noise reduction, speed, dealing with correlated features) in classification tasks.
   * Introduce the chosen dataset and classification task.
   * State the aim: compare kNN, Naive Bayes, Logistic Regression, and Decision Trees on original vs PCA-reduced features and assess the impact on performance and interpretability.

3. **Background**

   * Explain each classifier: kNN, Gaussian Naive Bayes, Logistic Regression (with L2 regularisation), Decision Tree.
   * Introduce PCA as a linear, unsupervised dimensionality-reduction technique based on variance and eigen-decomposition.
   * Discuss potential benefits and drawbacks of using PCA before classification (e.g. reduced dimensionality vs loss of interpretability).

4. **Methodology**

   * Dataset description and preprocessing (scaling, handling missing values, train/test split).
   * From-scratch implementations of PCA and at least one classifier (NB or kNN).
   * Cross-validation and hyperparameter tuning procedures, including how PCA is fit within each training fold (with fixed `d`).
   * Details of the evaluation metrics and original vs PCA-reduced setups.

5. **Results**

   * Performance tables for each classifier on original and PCA-reduced features.
   * Confusion matrices for selected models (e.g. best model in each feature space).
   * Plots of performance vs number of PCA components (for exploratory analysis) and vs hyperparameters (k, C, max_depth).
   * Explicit comparison of tuned models against the naive baseline.

6. **Evaluation**

   * Critical analysis of:

     * which classifiers benefitted most/least from PCA,
     * evidence of overfitting/underfitting in original vs PCA-reduced spaces,
     * sensitivity to hyperparameters and to the choice of number of components `d`,
     * trade-offs between performance, computational cost, and interpretability.
   * Limitations (e.g. choice of dataset, focus on PCA only rather than other feature-selection methods).

7. **Conclusions**

   * Summarise the main findings regarding the impact of PCA on different classifiers.
   * Indicate which combinations of classifier and feature representation you would recommend for similar problems and why.
   * Briefly suggest future extensions (e.g. comparing PCA with supervised feature selection such as SelectKBest).

8. **References**

   * Textbook, lecture notes, and external sources used for PCA, classifiers, and evaluation methods.
