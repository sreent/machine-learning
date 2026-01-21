# Project Idea 4: Multi-Class Classification with Tabular Data

## Aim

To perform multi-class classification on structured tabular data using dense neural networks, demonstrating:

- Feature preprocessing for mixed data types
- Handling of moderate class imbalance
- Systematic hyperparameter tuning
- The full Universal ML Workflow (Chollet, 2021, Chapter 4.5)

This project showcases Dense networks on their natural domain—tabular/structured data—where they often perform comparably to gradient boosting methods.

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

| Dataset | Classes | Features | Samples | Source |
|---------|---------|----------|---------|--------|
| **Wine Quality** | 6-7 | 11 | 6,497 | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| **Covertype** | 7 | 54 | 581,012 | `sklearn.datasets.fetch_covtype()` |
| **Letter Recognition** | 26 | 16 | 20,000 | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/letter+recognition) |
| **Dry Bean** | 7 | 16 | 13,611 | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset) |

**Recommendation:** Wine Quality is manageable size with interesting feature relationships. Consider merging quality scores into fewer classes (e.g., Low/Medium/High) for better class balance.

---

## Steps

### 1. Data Loading and Preprocessing

- Load dataset:
  ```python
  import pandas as pd
  # Wine Quality (red and white combined)
  red = pd.read_csv('winequality-red.csv', sep=';')
  white = pd.read_csv('winequality-white.csv', sep=';')
  df = pd.concat([red, white], ignore_index=True)
  ```
- Explore class distribution:
  ```python
  print(df['quality'].value_counts().sort_index())
  # Typically imbalanced: few wines rated 3, 4, 8, 9
  ```
- Optional: Bin quality scores for better balance:
  ```python
  def categorise_quality(q):
      if q <= 4: return 'Low'
      elif q <= 6: return 'Medium'
      else: return 'High'
  df['quality_cat'] = df['quality'].apply(categorise_quality)
  ```
- Feature scaling (critical for Dense networks):
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```

### 2. Choosing a Measure of Success

- Check class balance:
  ```python
  print(y_train.value_counts(normalize=True))
  ```
- Select metrics based on balance:
  - **Balanced:** Accuracy
  - **Imbalanced:** F1-Score (macro), consider class weights
- Compute class weights if needed:
  ```python
  from sklearn.utils.class_weight import compute_class_weight
  weights = compute_class_weight('balanced',
                                  classes=np.unique(y_train),
                                  y=y_train)
  class_weight = dict(enumerate(weights))
  ```
- Define naive baseline:
  ```python
  from sklearn.dummy import DummyClassifier
  dummy = DummyClassifier(strategy='most_frequent')
  dummy.fit(X_train, y_train)
  baseline_acc = dummy.score(X_test, y_test)
  ```

### 3. Deciding on an Evaluation Protocol

- **Dataset size determines method:**
  - Wine Quality (~6,500): K-Fold (K=5) or Hold-Out
  - Covertype (~580,000): Hold-Out only
  - Letter Recognition (~20,000): Hold-Out

- Example K-Fold setup:
  ```python
  from sklearn.model_selection import StratifiedKFold

  kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  for train_idx, val_idx in kfold.split(X, y):
      X_train_fold, X_val_fold = X[train_idx], X[val_idx]
      y_train_fold, y_val_fold = y[train_idx], y[val_idx]
      # Train and evaluate...
  ```

### 4. Developing a Model Better than Baseline

- Build Single Layer Perceptron:
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense

  model = Sequential([
      Dense(num_classes, activation='softmax', input_shape=(num_features,))
  ])
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  ```
- Train and compare to baseline:
  ```python
  history = model.fit(X_train_scaled, y_train_cat,
                      validation_split=0.1,
                      epochs=100, batch_size=32,
                      class_weight=class_weight)
  ```

### 5. Scaling Up: Developing a Model that Overfits

- Add hidden layers:
  ```python
  model = Sequential([
      Dense(64, activation='relu', input_shape=(num_features,)),
      Dense(32, activation='relu'),
      Dense(num_classes, activation='softmax')
  ])
  ```
- Train longer to observe overfitting
- Document the gap between training and validation accuracy

### 6. Regularising and Tuning Hyperparameters

- Add regularisation:
  ```python
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.regularizers import l2

  model = Sequential([
      Dense(64, activation='relu', kernel_regularizer=l2(0.01),
            input_shape=(num_features,)),
      Dropout(0.3),
      Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
      Dropout(0.3),
      Dense(num_classes, activation='softmax')
  ])
  ```
- Hyperparameters to tune:
  - Learning rate: 0.0001 - 0.01
  - Dropout: 0.0 - 0.5
  - L2 strength: 0.001 - 0.1
  - Neurons per layer: 16, 32, 64, 128

### 7. Architecture Exploration (Optional, for Additional Credit)

- Systematic comparison:
  - **Shallow wide:** 1 layer × 128 neurons
  - **Deep narrow:** 3 layers × 32 neurons each
  - **Balanced:** 2 layers × 64 neurons each
- Document which architecture works best for this dataset size

---

## Report Structure

### 1. Introduction
- Problem: Predicting categorical outcomes from features
- Motivation: Quality control, automated classification
- Dataset description and feature overview

### 2. Methodology
- Data preprocessing and feature scaling
- Class imbalance handling strategy
- Evaluation protocol justification
- Model architectures explored

### 3. Results
- Training curves for each model
- Performance comparison table:

| Model | Architecture | Accuracy | F1-Score |
|-------|--------------|----------|----------|
| Naive Baseline | Majority class | — | — |
| SLP | Linear | — | — |
| DNN (overfit) | 64-32 neurons | — | — |
| DNN (regularised) | + Dropout + L2 | — | — |

- Multi-class confusion matrix
- Per-class precision, recall, F1

### 4. Analysis
- Feature importance (coefficient analysis for SLP)
- Which classes are hardest to distinguish
- Effect of class weights on minority classes
- Comparison with traditional ML (optional)

### 5. Conclusions
- Best model recommendation
- Key insights about tabular data with DNNs
- Limitations and future work

### 6. Code Attribution & References

---

## Common Pitfalls to Avoid

1. **Not scaling features** — Dense networks are sensitive to feature scales
2. **Ignoring class imbalance** — Use class weights or stratified sampling
3. **Too many neurons for small datasets** — Risk overfitting with high parameter count
4. **Forgetting to stratify splits** — Important for imbalanced multi-class

---

## Feature Preprocessing Tips

### Numerical Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X_numerical)
```

### Categorical Features
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_categorical = encoder.fit_transform(X_categorical)
```

### Combining Features
```python
import numpy as np
X_combined = np.concatenate([X_numerical, X_categorical], axis=1)
```

---

## References

- Chollet, F. (2021) *Deep Learning with Python*. 2nd edn. Shelter Island, NY: Manning Publications.
- Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', *IJCAI*, 14(2), pp. 1137–1145.
- He, H. and Garcia, E.A. (2009) 'Learning from imbalanced data', *IEEE TKDE*, 21(9), pp. 1263–1284.
- Cortez, P. et al. (2009) 'Modeling wine preferences by data mining from physicochemical properties', *Decision Support Systems*, 47(4), pp. 547–553.
