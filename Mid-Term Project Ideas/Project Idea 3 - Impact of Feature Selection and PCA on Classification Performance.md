# Project Idea 3: Impact of Feature Selection and PCA on Classification Performance

## Aim

To investigate how different **feature selection and dimensionality reduction techniques** affect the performance of several core classification algorithms from the course on a single standard dataset from scikit-learn.

The goals are to:

* compare classifier performance **with vs without** feature selection/dimensionality reduction,
* study how different techniques (e.g. **PCA from scratch** vs **supervised feature selection** such as SelectKBest / mutual information) interact with different classifiers,
* analyse trade-offs between **performance, overfitting, computational cost, and interpretability**.

The intention is to implement **PCA** and at least one classifier (most naturally **Gaussian Naive Bayes** or **kNN**) **from scratch** using only Python and NumPy (no ML libraries). The remaining algorithms (e.g. Logistic Regression and Decision Tree) and feature-selection utilities (e.g. SelectKBest) will use scikit-learn for comparison. This creates clear opportunities to go beyond the minimum coursework requirement in terms of implementation depth.

---

## Dataset

This project will use a **single standard classification dataset** from scikit-learn, such as:

* **Wine** (`sklearn.datasets.load_wine()`),
* **Breast Cancer Wisconsin Diagnostic** (`sklearn.datasets.load_breast_cancer()`), or
* **Digits** (`sklearn.datasets.load_digits()`).

The chosen dataset will:

* have a **moderate number of features** (e.g. 10–60) so that feature selection/dimensionality reduction are meaningful,
* be small to moderately sized so that **5-fold cross-validation on the training set** is practical,
* be relatively clean and numerical, requiring only light preprocessing (standardisation, handling any missing values if needed).

If the dataset is class-imbalanced (e.g. Breast Cancer), the analysis will pay particular attention to **recall and F1** for the minority / most critical class; for more balanced or multi-class datasets, **accuracy and macro F1** will be emphasised.

---

## Algorithms

All classifiers are drawn from the lecture list and applied to the same dataset.

Planned classifiers:

* **k-Nearest Neighbours (kNN) Classifier**

  * distance metric: Euclidean (and optionally others, e.g. Minkowski),
  * key hyperparameter: number of neighbours `k`.

* **Gaussian Naive Bayes**

  * assumes normally distributed features within each class,
  * simple baseline that often reacts strongly to feature selection / scaling.

* **Logistic Regression**

  * with **L2 regularisation** (ridge-style),
  * implemented via scikit-learn (`LogisticRegression`) but discussed in terms of gradient descent and regularisation.

* **Decision Tree Classifier**

  * using a fixed impurity criterion (e.g. `entropy`),
  * key hyperparameters: `max_depth`, `min_samples_split`.

From-scratch implementation plan:

* In line with the Aim, **PCA** and at least one classifier (most naturally **Gaussian Naive Bayes** or **kNN**) will be implemented from scratch in pure Python + NumPy, and, if feasible, both classifiers:

  * **Gaussian Naive Bayes**,
  * **kNN classifier**,
  * **PCA** (Principal Component Analysis) for dimensionality reduction.

These from-scratch implementations will then be compared against scikit-learn counterparts to validate correctness and understand the algorithms more deeply.

---

## Feature Selection & Dimensionality Reduction Techniques

The project will compare multiple feature selection / reduction strategies:

1. **No feature selection (All features)**

   * Baseline: use all original features.

2. **PCA (Principal Component Analysis)**

   * **From scratch**:

     * standardise the training data,
     * compute the covariance matrix,
     * compute eigenvalues/eigenvectors,
     * sort by eigenvalue magnitude,
     * project onto the top `d` principal components.
   * Optionally confirm results against scikit-learn’s `PCA` on a subset.
   * `d` (number of components) will be treated as a hyperparameter, chosen from a small set of candidate values (e.g. 2, 5, 10, 15) and informed by standard rules such as retaining a target proportion of cumulative explained variance (e.g. 90–95%) or the Kaiser rule (eigenvalues greater than 1 for standardised features).

3. **Univariate Feature Selection (SelectKBest)**

   * Use scikit-learn’s `SelectKBest` with a chosen scoring function, e.g. **ANOVA F-score** or **mutual information**.
   * `k` (number of selected features) will be a hyperparameter.

In practice, to keep the scope manageable, the core comparison will focus on a subset of classifier–feature-selection combinations, for example:

* **kNN** with **No selection**, **PCA**, and **SelectKBest**,
* **Naive Bayes** with **No selection** and **PCA**,
* **Logistic Regression** and **Decision Tree** primarily with **SelectKBest**,

**Important:** For all techniques, feature selection / PCA will be **fitted only on the training folds** within cross-validation, then applied to validation/test folds, to avoid data leakage.

---

## Steps

### 1. Dataset Preparation

* Load the chosen dataset from scikit-learn.
* Inspect and clean the data:

  * handle missing values if present (or report that there are none),
  * check for obvious outliers or corrupted entries (if any).
* Split into **training (e.g. 80%)** and **test (20%)** sets, stratified by class to preserve class proportions.
* Standardise features (e.g. `StandardScaler`) where appropriate, especially before PCA and kNN.

### 2. Exploratory Data Analysis (EDA)

* Summarise class distribution (check for imbalance).
* Visualise:

  * histograms / boxplots for selected features,
  * a correlation heatmap to identify redundant features,
  * if suitable, 2D projections (e.g. first two PCA components) for a rough visual sense of class separability.
* Comment on:

  * which features look most discriminative,
  * any signs of multicollinearity or clusters.

### 3. Baselines and From-Scratch Implementations

* Define a simple **naive baseline classifier** (e.g. always predict the majority class) and evaluate it as a reference point.
* Implement **Gaussian Naive Bayes** and/or **kNN classifier from scratch**:

  * for NB: estimate class priors and per-class feature means/variances; predict via class posteriors (using log-probabilities),
  * for kNN: compute distances to training samples and use majority vote among the `k` nearest neighbours.
* Implement **PCA from scratch** as described above.
* Optionally verify the from-scratch implementations (Naive Bayes, kNN, PCA) by comparing their outputs / metrics against scikit-learn (`GaussianNB`, `KNeighborsClassifier`, `PCA`) on a subset of the data; use scikit-learn versions in the main experiments for efficiency.

### 4. Cross-Validation Strategy

* Use **k-fold cross-validation (e.g. k = 5)** on the **training set**.

* For each fold:

  * fit the feature selection / PCA **only on the fold’s training portion**,
  * transform the training and validation portions,
  * train the classifier on the transformed training data and evaluate on the transformed validation data.

* Use average metrics across folds to:

  * estimate generalisation performance,
  * compare feature selection strategies and classifiers,
  * select both **model hyperparameters** and **feature-selection hyperparameters** (e.g. number of components/features).

* Note: for this project, we deliberately choose a dataset size where 5-fold CV is practical; for much larger datasets, a simple hold-out scheme would be more efficient.

### 5. Hyperparameter & Feature-Selection Tuning

* Use a small grid search (or systematic manual search) over key hyperparameters, under each feature-selection strategy:

  * **kNN**: `n_neighbors` (e.g. {1, 3, 5, 7, 9}).
  * **Gaussian Naive Bayes**: typically default parameters; optionally check a small range of `var_smoothing` if needed (for the scikit-learn version).
  * **Logistic Regression**: regularisation strength `C` (e.g. {0.01, 0.1, 1, 10}), with L2 penalty and a suitable solver.
  * **Decision Tree**: `max_depth` (e.g. {3, 5, 7, None}), `min_samples_split` (e.g. {2, 5}).
  * **PCA**: number of components `d`, selected from a small set of values (e.g. {2, 5, 10, 15}) based on cumulative explained variance or, as a sanity check, the Kaiser rule; the final choice of `d` will be validated by cross-validation performance.
  * **SelectKBest**: number of selected features `k` (e.g. {5, 10, 15}).

* Perform hyperparameter tuning using cross-validation on the **training set only**.

* Select the best hyperparameters per **classifier + feature-selection combination** based on a suitable CV metric:

  * **Accuracy + macro F1** overall,
  * and, where relevant, **class-specific recall / F1** for the most critical class.

### 6. Training and Evaluation

* Retrain each classifier with its **best hyperparameters** under each feature-selection strategy on the **full training set**.
* Evaluate on the **held-out test set** using:

  * Accuracy,
  * Precision, Recall, F1 (macro and/or per-class),
  * Confusion matrices for key models (especially for the chosen “best” model per strategy).
* Compare results against the naive baseline to show absolute performance gains.

### 7. Model & Feature-Selection Strategy Comparison

* Summarise final test-set performance in a table like:

| Model               | Feature Selection | Key Hyperparameters                  | Accuracy | Macro F1 | Precision (pos) | Recall (pos) |
| ------------------- | ----------------- | ------------------------------------ | -------- | -------- | --------------- | ------------ |
| kNN                 | None              | k = …                                | …%       | …        | …               | …            |
| kNN                 | PCA               | k = …, d = …                         | …%       | …        | …               | …            |
| kNN                 | SelectKBest       | k = …                                | …%       | …        | …               | …            |
| Naive Bayes         | None              | –                                    | …%       | …        | …               | …            |
| Naive Bayes         | PCA               | d = …                                | …%       | …        | …               | …            |
| Logistic Regression | SelectKBest       | C = …, penalty = L2                  | …%       | …        | …               | …            |
| Decision Tree       | SelectKBest       | max_depth = …, min_samples_split = … | …%       | …        | …               | …            |

* Discuss trade-offs by:

  * identifying which **classifier + feature-selection** combinations perform best and why,
  * analysing whether PCA (unsupervised) helps or hurts compared to supervised feature selection,
  * examining whether simpler models (e.g. Naive Bayes with selected features) can match or beat more complex models with all features.

### 8. Feature Importance & Interpretability

* **Decision Tree**:

  * use tree-based feature importances to see which original features are most influential,
  * discuss how these change (if at all) when feature selection is applied beforehand.

* **Logistic Regression**:

  * use the magnitude and sign of coefficients (on selected features) to interpret which features increase or decrease the odds of belonging to a particular class.

* **PCA**:

  * inspect explained variance ratios per component,
  * examine which original features load most heavily on the most important PCs, and how this relates to EDA and supervised feature selection results.

* Relate findings back to EDA and to the selection scores from SelectKBest.

### 9. Model Complexity Analysis

* Investigate how model complexity affects performance under different feature-selection strategies:

  * For kNN: vary `k` and observe validation/test performance (small vs large k).
  * For Decision Tree: vary `max_depth` and monitor overfitting/underfitting.
  * For Logistic Regression: vary `C` (regularisation strength) and examine performance and coefficient magnitudes.

* Plot validation performance vs complexity (e.g. macro F1 vs k, macro F1 vs max_depth, macro F1 vs C) under a chosen feature-selection strategy (e.g. with SelectKBest) to visualise bias–variance behaviour.

---

## Report Structure (Aligned with Coursework)

1. **Abstract**
   Briefly summarise the aim (impact of feature selection/PCA on multiple classifiers), dataset, methods (from-scratch PCA and at least one classifier, CV, tuning), key results, and main conclusions.

2. **Introduction**

   * Motivate the importance of **feature selection** and **dimensionality reduction** in classification tasks (performance, overfitting, speed, interpretability).
   * Introduce the chosen dataset and classification task.
   * State the project aim: compare several classifiers under different feature-selection strategies and assess their impact on performance and interpretability.

3. **Background**

   * Explain each classifier: kNN, Gaussian Naive Bayes, Logistic Regression (with L2 regularisation), Decision Tree.
   * Introduce feature selection vs dimensionality reduction:

     * PCA (unsupervised, variance-based, linear projection),
     * supervised methods (SelectKBest with F-score or mutual information).
   * Discuss the potential benefits and drawbacks of each technique and how they might interact with different classifiers.

4. **Methodology**

   * Dataset description and preprocessing (scaling, splitting, handling missing values).
   * From-scratch implementations (PCA and at least one classifier: NB or kNN).
   * Cross-validation and hyperparameter tuning strategy, including how feature selection is applied **within** CV folds.
   * Exact feature-selection techniques and hyperparameter grids used.

5. **Results**

   * Performance tables for each classifier under different feature-selection strategies (on CV and test set).
   * Confusion matrices for selected models.
   * Plots of performance vs model complexity and, where relevant, vs the number of selected features/components.
   * Explicit comparison of tuned models against the naive baseline.

6. **Evaluation**

   * Critical analysis of:

     * which classifier–feature-selection combinations performed best and why,
     * evidence of overfitting/underfitting and the effect of reducing dimensionality,
     * sensitivity to hyperparameters (k, max_depth, C, number of features/components),
     * trade-offs between performance, computational cost, and interpretability.
   * Reflect on how well PCA vs supervised feature selection worked for this dataset.

7. **Conclusions**

   * Summarise the main findings about the **impact of feature selection techniques** (including PCA) on classification performance.
   * Indicate which combinations you would recommend for similar problems and why.
   * Briefly discuss limitations (dataset choice, number of feature-selection techniques, implementation constraints) and possible future extensions.

8. **References**

   * Textbook, lecture notes, and external sources used for PCA, feature selection methods, and classifiers.
