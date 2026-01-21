# Project Idea 1: Sentiment Analysis with TF-IDF and Dense Networks

## Aim

To classify text sentiment (positive/negative/neutral) using TF-IDF vectorisation and dense neural networks, systematically comparing:

- Single Layer Perceptron (SLP) baseline
- Multi-layer perceptrons with regularisation
- Different architecture variants (wider, deeper, narrower)

This project demonstrates the Universal ML Workflow (Chollet, 2021, Chapter 4.5) on an NLP classification task where Dense layers are effective because TF-IDF ignores word order.

---

## Constraints

All projects must adhere to DLWP Part 1 (Chapters 1-4) constraints:

| Allowed | Not Allowed |
|---------|-------------|
| Dense layers | CNNs, RNNs, Transformers |
| Dropout layers | Early Stopping* |
| L1/L2 regularisation | Pre-trained models |
| Adam, SGD optimisers | BatchNormalisation |

*Check your specific assignment for Early Stopping restrictions.

---

## Datasets

Choose one of the following sentiment analysis datasets:

| Dataset | Classes | Samples | Source |
|---------|---------|---------|--------|
| **Twitter US Airline Sentiment** | 3 | 14,640 | [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) |
| **IMDB Movie Reviews** | 2 | 50,000 | `tensorflow.keras.datasets.imdb` |
| **Amazon Product Reviews** | 2-5 | Varies | [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) |

**Recommendation:** Twitter US Airline Sentiment is ideal—3-class imbalanced data provides opportunities to demonstrate class weights and F1-Score evaluation.

---

## Steps

### 1. Data Loading and Preprocessing

- Load dataset and explore class distribution
- Clean text data (remove URLs, handles, special characters)
- Apply TF-IDF vectorisation:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
  X_tfidf = tfidf.fit_transform(X_train).toarray()
  ```
- Split into train (80%) and test (20%) with stratification

### 2. Choosing a Measure of Success

- Identify class imbalance and compute class weights:
  ```python
  from sklearn.utils.class_weight import compute_class_weight
  weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
  class_weight = dict(enumerate(weights))
  ```
- Select metrics: **F1-Score (primary)**, Accuracy, AUC
- Define naive baseline (majority class prediction)

### 3. Deciding on an Evaluation Protocol

- For >10,000 samples: **Hold-Out validation** (Kohavi, 1995)
- For <10,000 samples: **K-Fold cross-validation** (K=5)
- Use stratified splits to maintain class proportions:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, stratify=y, random_state=42
  )
  ```

### 4. Developing a Model Better than Baseline

- Build Single Layer Perceptron (SLP):
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense

  model = Sequential([
      Dense(num_classes, activation='softmax', input_shape=(num_features,))
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
  ```
- Train with class weights
- Verify it beats naive baseline

### 5. Scaling Up: Developing a Model that Overfits

- Add hidden layer(s) to increase capacity:
  ```python
  model = Sequential([
      Dense(64, activation='relu', input_shape=(num_features,)),
      Dense(num_classes, activation='softmax')
  ])
  ```
- Train for enough epochs (150-200) to observe overfitting
- Document the overfitting pattern: training loss ↓, validation loss ↑

### 6. Regularising and Tuning Hyperparameters

- Add Dropout (Srivastava et al., 2014) and L2 regularisation (Krogh & Hertz, 1992):
  ```python
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.regularizers import l2

  model = Sequential([
      Dense(64, activation='relu', kernel_regularizer=l2(0.001),
            input_shape=(num_features,)),
      Dropout(0.3),
      Dense(num_classes, activation='softmax')
  ])
  ```
- Use Hyperband (Li et al., 2018) or Grid Search to tune:
  - Learning rate (1e-4 to 1e-2)
  - Dropout rate (0.0 to 0.5)
  - L2 strength (1e-5 to 1e-2)
- Retrain with best hyperparameters

### 7. Architecture Exploration (Optional, for Additional Credit)

- Compare architecture variants:
  - **Wider:** 128 neurons (1 hidden layer)
  - **Deeper:** 64 neurons × 2 hidden layers
  - **Narrower:** 32 neurons (1 hidden layer)
- Document that regularisation matters more than architecture changes

---

## Report Structure

### 1. Introduction
- Problem: Sentiment analysis for customer feedback
- Motivation: Business value of automated sentiment classification
- Dataset description and class distribution

### 2. Methodology
- Data preprocessing and TF-IDF vectorisation
- Evaluation protocol justification (Hold-Out vs K-Fold)
- Model architectures and hyperparameter search strategy

### 3. Results
- Training curves for each model stage
- Performance comparison table:

| Model | Architecture | Accuracy | F1-Score | AUC |
|-------|--------------|----------|----------|-----|
| Naive Baseline | — | 0.63 | 0.26 | 0.50 |
| SLP | 0 hidden layers | — | — | — |
| DNN (overfit) | 64 neurons | — | — | — |
| DNN (regularised) | 64 + Dropout + L2 | — | — | — |

- Confusion matrix and per-class metrics

### 4. Analysis
- Why regularisation works better than adding capacity
- Effect of class weights on minority class performance
- Interpretation of misclassified examples

### 5. Conclusions
- Key findings and best model recommendation
- Limitations (TF-IDF loses word order)
- Future work (embeddings, attention mechanisms)

### 6. Code Attribution & References

---

## Common Pitfalls to Avoid

1. **Starting with data visualisation** — No marks for EDA; start with methodology
2. **Skipping the baseline** — Always establish SLP baseline first
3. **Not showing overfitting** — Section 5 requires demonstrating overfitting
4. **Over-engineering** — Regularisation > complex architectures
5. **Forgetting code attribution** — Required per coursework instructions

---

## References

- Chollet, F. (2021) *Deep Learning with Python*. 2nd edn. Shelter Island, NY: Manning Publications.
- Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', *IJCAI*, 14(2), pp. 1137–1145.
- Srivastava, N. et al. (2014) 'Dropout: A simple way to prevent neural networks from overfitting', *JMLR*, 15(1), pp. 1929–1958.
- Krogh, A. and Hertz, J.A. (1992) 'A simple weight decay can improve generalization', *NeurIPS*, 4, pp. 950–957.
- Li, L. et al. (2018) 'Hyperband: A novel bandit-based approach to hyperparameter optimization', *JMLR*, 18(1), pp. 6765–6816.
- He, H. and Garcia, E.A. (2009) 'Learning from imbalanced data', *IEEE TKDE*, 21(9), pp. 1263–1284.
