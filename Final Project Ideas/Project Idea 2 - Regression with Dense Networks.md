# Project Idea 2: Regression with Dense Networks

## Aim

To predict a continuous target variable using dense neural networks, demonstrating the Universal ML Workflow (Chollet, 2021, Chapter 4.5) on a regression task:

- Establish baseline with linear model (SLP)
- Build capacity to overfit
- Regularise to generalise
- Compare architectures systematically

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

| Dataset | Features | Samples | Target | Source |
|---------|----------|---------|--------|--------|
| **Bike Sharing Demand** | 12 | 17,379 | Hourly bike rentals | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) |
| **California Housing** | 8 | 20,640 | Median house value | `sklearn.datasets.fetch_california_housing()` |
| **Boston Housing** | 13 | 506 | Median house value | `tensorflow.keras.datasets.boston_housing` |

**Recommendation:** Bike Sharing Demand is ideal—large enough for Hold-Out validation, clear temporal patterns, and intuitive features.

---

## Steps

### 1. Data Loading and Preprocessing

- Load dataset and handle missing values:
  ```python
  import pandas as pd
  df = pd.read_csv('hour.csv')
  print(df.isnull().sum())  # Check for missing values
  ```
- Feature engineering (e.g., cyclical encoding for hour/month):
  ```python
  import numpy as np
  df['hour_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
  df['hour_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
  ```
- Normalise features using StandardScaler:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```
- Split into train/test (80/20)

### 2. Choosing a Measure of Success

- **Primary metric:** MAE (interpretable in original units)
- **Secondary metric:** R² (proportion of variance explained)
- Define naive baseline:
  ```python
  # Naive baseline: predict mean of target
  naive_prediction = np.mean(y_train)
  naive_mae = np.mean(np.abs(y_test - naive_prediction))
  print(f"Naive Baseline MAE: {naive_mae:.2f}")
  ```
- **Loss function:** MSE (for training), MAE (for evaluation)

### 3. Deciding on an Evaluation Protocol

- **Hold-Out validation** for large datasets (>10,000 samples)
- **K-Fold cross-validation** for smaller datasets (e.g., Boston Housing with 506 samples)
- Split strategy:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  ```

### 4. Developing a Model Better than Baseline

- Build linear model (SLP with no hidden layers):
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense

  model = Sequential([
      Dense(1, activation='linear', input_shape=(num_features,))
  ])
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  ```
- Train and evaluate:
  ```python
  history = model.fit(X_train, y_train, validation_split=0.1,
                      epochs=100, batch_size=64, verbose=0)
  mae = model.evaluate(X_test, y_test)[1]
  print(f"SLP MAE: {mae:.2f}")
  ```
- Verify MAE is lower than naive baseline

### 5. Scaling Up: Developing a Model that Overfits

- Add hidden layers:
  ```python
  model = Sequential([
      Dense(64, activation='relu', input_shape=(num_features,)),
      Dense(1, activation='linear')
  ])
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  ```
- Train for enough epochs (150-200) to observe overfitting
- Document overfitting pattern: training loss ↓, validation loss ↑

### 6. Regularising and Tuning Hyperparameters

- Add Dropout and L2 regularisation:
  ```python
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.regularizers import l2

  model = Sequential([
      Dense(64, activation='relu', kernel_regularizer=l2(0.001),
            input_shape=(num_features,)),
      Dropout(0.3),
      Dense(1, activation='linear')
  ])
  ```
- Use Hyperband to tune:
  ```python
  import keras_tuner as kt

  def build_model(hp):
      model = Sequential()
      l2_reg = hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')
      model.add(Dense(64, activation='relu',
                      kernel_regularizer=l2(l2_reg),
                      input_shape=(num_features,)))
      dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
      model.add(Dropout(dropout))
      model.add(Dense(1, activation='linear'))
      lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
      model.compile(optimizer=Adam(learning_rate=lr),
                    loss='mse', metrics=['mae'])
      return model

  tuner = kt.Hyperband(build_model, objective='val_mae', max_epochs=20)
  tuner.search(X_train, y_train, validation_split=0.1)
  ```

### 7. Architecture Exploration (Optional, for Additional Credit)

- Compare architecture variants:
  - **Wider:** 128 neurons
  - **Deeper:** 64 × 2 hidden layers
  - **Narrower:** 32 neurons
- Document diminishing returns from added complexity

---

## Report Structure

### 1. Introduction
- Problem: Predicting demand/prices from features
- Motivation: Resource planning, pricing strategies
- Dataset description and feature overview

### 2. Methodology
- Data preprocessing and feature engineering
- Evaluation protocol justification
- Model architectures and hyperparameter search

### 3. Results
- Training curves for each model stage
- Performance comparison table:

| Model | Architecture | MAE | R² |
|-------|--------------|-----|-----|
| Naive Baseline | Mean prediction | — | 0.00 |
| SLP | 0 hidden layers | — | — |
| DNN (overfit) | 64 neurons | — | — |
| DNN (regularised) | 64 + Dropout + L2 | — | — |

- Residual plots and error distribution

### 4. Analysis
- Feature importance analysis
- Comparison of MSE (training loss) vs MAE (evaluation metric)
- Why regularisation improves generalisation

### 5. Conclusions
- Key findings and best model recommendation
- Limitations and future work

### 6. Code Attribution & References

---

## Common Pitfalls to Avoid

1. **Not scaling features** — Critical for Dense networks; use StandardScaler
2. **Using MAE as loss** — Use MSE for training (smoother gradients), MAE for evaluation
3. **Forgetting the linear output** — No activation on output layer for regression
4. **Skipping R²** — Include both MAE and R² for complete picture

---

## References

- Chollet, F. (2021) *Deep Learning with Python*. 2nd edn. Shelter Island, NY: Manning Publications.
- Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', *IJCAI*, 14(2), pp. 1137–1145.
- Srivastava, N. et al. (2014) 'Dropout: A simple way to prevent neural networks from overfitting', *JMLR*, 15(1), pp. 1929–1958.
- Krogh, A. and Hertz, J.A. (1992) 'A simple weight decay can improve generalization', *NeurIPS*, 4, pp. 950–957.
