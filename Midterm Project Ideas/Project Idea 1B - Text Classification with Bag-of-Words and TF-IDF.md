# Project Idea 1B: Text Classification with Bag-of-Words and TF-IDF

## Aim

To compare several core classification algorithms from the course—k-Nearest Neighbours (kNN), Multinomial Naive Bayes, and Logistic Regression (with L2 regularisation)—on a **single text classification task**, using both **Bag-of-Words (BoW)** and **TF-IDF** representations. The goal is to:

* understand how different algorithms behave on high-dimensional sparse text features,
* compare the impact of BoW vs TF-IDF representations on performance,
* analyse trade-offs between accuracy, robustness to hyperparameters, and interpretability.

At least one, and preferably two, of these algorithms (most naturally **Multinomial Naive Bayes** and **k-Nearest Neighbours**) will be implemented **from scratch** (using only Python and NumPy), with Logistic Regression and any remaining models possibly using **scikit-learn** for comparison. This creates clear opportunities to go beyond the minimum coursework requirement in terms of implementation depth.

---

## Dataset

Use a **single publicly available labelled text dataset** for a binary classification task, such as:

* **Sentiment analysis** (e.g. positive vs negative reviews), or
* **Fake news detection** (e.g. real vs fake news articles or headlines).

The exact dataset will be chosen from a standard open dataset (for example, a movie review sentiment dataset or a fake vs real news dataset), with the following properties:

* Each instance consists of a short or medium-length piece of text (sentence, headline, or document) and a binary label.
* The dataset is large enough to benefit from train/test splits and cross-validation, but small enough to be processed on a standard laptop and to allow 5-fold cross-validation on the training set.

The project will focus on **converting raw text into numerical features** via Bag-of-Words and TF-IDF so that standard classification algorithms from the course can be applied.

---

## Text Representation

Two main feature representations will be used:

* **Bag-of-Words (BoW)** using token counts for each word in a vocabulary.
* **TF-IDF (Term Frequency–Inverse Document Frequency)** features to downweight common words and emphasise more informative ones.

Using scikit-learn tools such as `CountVectorizer` and `TfidfVectorizer`, with simple preprocessing steps:

* lowercasing,
* tokenisation based on whitespace and basic punctuation,
* optional removal of stopwords,
* limiting the vocabulary size via `max_features` or `min_df` to control dimensionality.

Where appropriate, small grids over representation hyperparameters (e.g. unigram vs bigram features, vocabulary size) may be explored, but the main focus will remain on the classification algorithms themselves.

---

## Algorithms

All algorithms are drawn from the lecture list and will be applied to both BoW and TF-IDF feature spaces.

Planned algorithms:

* **k-Nearest Neighbours (kNN)** (with Euclidean and cosine distance metrics; cosine distance is particularly natural for TF-IDF vectors)
* **Naive Bayes**

  * Multinomial NB (for discrete word counts; primary focus for text)
  * Gaussian NB may be briefly discussed or used as a comparison if appropriate
* **Logistic Regression** (with L2 regularisation; optimisation discussed in terms of gradient descent)

The intention is to implement **Multinomial Naive Bayes** and, if feasible, **k-Nearest Neighbours** from scratch in Python/NumPy, while Logistic Regression (and any additional variants) are used via scikit-learn for convenience and to focus effort on the most informative implementations.

---

## Steps

### 1. Dataset Preparation

* Load the chosen text dataset (e.g. sentiment or fake news) from a public source.
* Inspect and clean the data:

  * remove clearly corrupted entries (if any),
  * handle missing texts or labels,
  * standardise label encoding (e.g. 0/1 for negative/positive or real/fake).
* Split into **training (e.g. 80%)** and **test (20%)** sets, stratified by label to preserve class proportions.

### 2. Text Preprocessing and Feature Extraction

* Apply basic text preprocessing:

  * lowercasing,
  * tokenisation,
  * optional stopword removal.
* Construct two separate feature sets:

  * **BoW features** via `CountVectorizer`,
  * **TF-IDF features** via `TfidfVectorizer`.
* Optionally limit vocabulary size (`max_features`) and optionally include bigrams to balance expressiveness and computational cost.

### 3. Baselines and From-Scratch Implementation

* Define a simple **naive baseline classifier** (e.g. always predicting the majority class) and evaluate it as a reference point on the training/test split.
* Implement **Multinomial Naive Bayes from scratch** using word count features and Laplace smoothing:

  * compute class prior probabilities,
  * compute conditional probabilities of words given each class,
  * represent predictions using log-probabilities to avoid underflow.
* Additionally, implement a simple **kNN classifier from scratch** (e.g. using cosine distance on TF-IDF vectors) to further demonstrate implementational skill and provide a direct comparison with the scikit-learn implementation.
* Verify the from-scratch Multinomial NB by comparing its predictions or metrics with scikit-learn’s `MultinomialNB` on a subset of the data, and similarly sanity-check the from-scratch kNN implementation against scikit-learn’s `KNeighborsClassifier`. The main experiments will then use the scikit-learn implementations for efficiency.

### 4. Cross-Validation Strategy

* Because we will choose a **moderately sized** dataset (e.g. tens of thousands of documents), we will use **5-fold cross-validation (k = 5)** on the **training set**.
* Use average metrics across folds to:

  * estimate generalisation performance,
  * compare algorithms more robustly than a single train/test split.
* Note: for much larger text datasets, a simpler hold-out validation split within the training data is often preferred for efficiency. In contrast, for small tabular datasets with only a few hundred points (such as Iris or Wine), k-fold cross-validation is especially important to make good use of limited data. Here we deliberately choose a moderately sized text dataset where 5-fold cross-validation on the training set remains practical, and we will briefly justify this choice in the report.

### 5. Hyperparameter Tuning

* Use grid search or a small manual search over key hyperparameters, separately for BoW and TF-IDF features:

  * kNN: `n_neighbors (k)` and distance metric (e.g. Euclidean vs cosine).
  * Multinomial Naive Bayes: smoothing parameter `alpha` (Laplace/Lidstone smoothing).
  * Logistic Regression: regularisation strength `C` (L2), and possibly penalty type.
* Select the best hyperparameters based on cross-validation performance (e.g. macro F1 score).
* Retrain each model with its best hyperparameters on the full training set and evaluate on the held-out test set using:

  * **Accuracy**, **Precision**, **Recall**, **F1 score** (macro and/or per-class),
  * optionally **ROC-AUC** if appropriate for the dataset.
* Include **confusion matrices** for key models to inspect common types of misclassification (e.g. fake labelled as real, or negative sentiment labelled as positive).

### 6. Model Complexity & Regularisation Analysis

* Investigate **overfitting vs underfitting** and robustness to hyperparameters:

  * For kNN: vary `k` and compare Euclidean vs cosine distance, observing validation performance.
  * For Logistic Regression: vary `C` (L2 regularisation strength) and examine its effect on performance and weight magnitudes.
* Compare how sensitive each algorithm is to hyperparameter changes when using BoW vs TF-IDF features.

### 7. Effect of Representation and Interpretability

* Compare algorithms under **two representations**:

  * BoW + each classifier,
  * TF-IDF + each classifier.
* For **Logistic Regression**:

  * Examine the largest positive and negative coefficients to identify which words most strongly indicate each class,
  * discuss interpretability of feature weights.
* For **Multinomial Naive Bayes**:

  * Examine the most probable words per class,
  * relate them to intuitive sentiment or fake/real indicators.
* Discuss which representation (BoW or TF-IDF) appears more effective and why.

### 8. Model Comparison

* Summarise final tuned models and results on the **held-out test set** in tables, separated by representation, e.g.:

| Model               | Representation | Key Hyperparameters (tuned) | Accuracy | Macro F1 | Precision (pos) | Recall (pos) |
| ------------------- | -------------- | --------------------------- | -------- | -------- | --------------- | ------------ |
| kNN                 | BoW            | k = …                       | …%       | …        | …               | …            |
| kNN                 | TF-IDF         | k = …                       | …%       | …        | …               | …            |
| Multinomial NB      | BoW            | alpha = …                   | …%       | …        | …               | …            |
| Multinomial NB      | TF-IDF         | alpha = …                   | …%       | …        | …               | …            |
| Logistic Regression | BoW            | C = …, penalty = L2         | …%       | …        | …               | …            |
| Logistic Regression | TF-IDF         | C = …, penalty = L2         | …%       | …        | …               | …            |

* Discuss the trade-offs by:

  * Identifying which algorithm and representation combination performs best overall (e.g. Logistic Regression + TF-IDF) based on the chosen test metrics, and explaining *why* this might be the case in terms of the representation and model assumptions.
  * Assessing which algorithm is most robust to hyperparameters by examining how its performance varies across different hyperparameter settings for both BoW and TF-IDF, and summarising which model is least sensitive to these changes.
  * Comparing interpretability by discussing how easy it is to explain each model’s predictions to a non-technical audience (e.g. word weights for Logistic Regression/Naive Bayes versus less intuitive kNN behaviour in high-dimensional sparse spaces).

---

## Report Structure (Aligned with Coursework)

1. **Abstract**
   Briefly summarise the goal (text classification with BoW/TF-IDF and multiple algorithms), methods, key results, and the main conclusions about which combinations worked best.

2. **Introduction**

   * Introduce text classification tasks (e.g. sentiment analysis or fake news detection) and why they are important.
   * State the chosen dataset and task (binary classification).
   * Outline the aim: compare multiple classifiers and representations (BoW vs TF-IDF).

3. **Background**

   * Describe Bag-of-Words and TF-IDF representations and their properties (sparsity, high dimensionality).
   * Explain each algorithm (kNN, Multinomial Naive Bayes, Logistic Regression) and the optimisation method **gradient descent**, which is typically used to train Logistic Regression.
   * Discuss expected behaviour of these algorithms on text data (e.g. why linear models and Naive Bayes are often effective).

4. **Methodology**

   * Dataset description and preprocessing (cleaning, tokenisation, splitting).
   * Feature construction with BoW and TF-IDF, including any representation hyperparameters (e.g. n-grams, vocabulary size).
   * Train–test split and cross-validation strategy, with justification based on dataset size.
   * Implementation details (which algorithms are from scratch, which via scikit-learn).
   * Hyperparameter tuning procedure and evaluation metrics.

5. **Results**

   * Performance tables for each algorithm under BoW and TF-IDF.
   * Cross-validation results for key hyperparameters.
   * Explicit comparison of tuned models against the naive baseline classifier to show absolute performance gains.
   * Confusion matrices and, where appropriate, ROC curves or PR curves.

6. **Evaluation**

   * Critical discussion of:

     * which algorithm–representation combinations performed best and why,
     * overfitting/underfitting and regularisation effects,
     * sensitivity to hyperparameters,
     * trade-offs between performance and interpretability.
   * Limitations (e.g. choice of dataset, feature engineering simplicity).

7. **Conclusions**

   * Summarise the main findings about BoW vs TF-IDF and the comparative performance of the classifiers.
   * Highlight which approach you would recommend for this type of text classification problem and why.

8. **References**

   * Textbook, lecture notes, and any external sources (papers, tutorials) used for algorithms, representations, or datasets.
