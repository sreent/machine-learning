# Final Coursework Project Ideas: Deep Neural Networks

> This guide provides structured project ideas for the Final DNN Coursework, following the Universal ML Workflow (Chollet, 2021, Chapter 4.5). Each idea includes dataset suggestions, methodology, and report structure aligned with the coursework requirements.

---

## Constraints Reminder

All projects must adhere to DLWP Part 1 (Chapters 1-4) constraints:

| Allowed | Not Allowed |
|---------|-------------|
| Dense layers | CNNs, RNNs, Transformers |
| Dropout layers | Early Stopping* |
| L1/L2 regularisation | Pre-trained models |
| Adam, SGD optimisers | BatchNormalisation |

*Check your specific assignment for Early Stopping restrictions.

---

## Project Idea 1: Sentiment Analysis with TF-IDF and Dense Networks

### Aim

To classify text sentiment (positive/negative/neutral) using TF-IDF vectorisation and dense neural networks, systematically comparing:
- Single Layer Perceptron (SLP) baseline
- Multi-layer perceptrons with regularisation
- Different architecture variants (wider, deeper, narrower)

This project demonstrates the Universal ML Workflow on an NLP classification task where Dense layers are effective because TF-IDF ignores word order.

---

### Datasets

Choose one of the following sentiment analysis datasets:

| Dataset | Classes | Samples | Source |
|---------|---------|---------|--------|
| **Twitter US Airline Sentiment** | 3 | 14,640 | [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) |
| **IMDB Movie Reviews** | 2 | 50,000 | `tensorflow.keras.datasets.imdb` |
| **Amazon Product Reviews** | 2-5 | Varies | [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) |

**Recommendation:** Twitter US Airline Sentiment is ideal—3-class imbalanced data provides opportunities to demonstrate class weights and F1-Score evaluation.

---

### Steps

#### 1. Data Loading and Preprocessing
- Load dataset and explore class distribution
- Clean text data (remove URLs, handles, special characters)
- Apply TF-IDF vectorisation:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
  X_tfidf = tfidf.fit_transform(X_train).toarray()
  ```
- Split into train (80%) and test (20%) with stratification

#### 2. Choosing a Measure of Success
- Identify class imbalance and compute class weights
- Select metrics: F1-Score (primary), Accuracy, AUC
- Define naive baseline (majority class prediction)

#### 3. Deciding on an Evaluation Protocol
- For >10,000 samples: Hold-Out validation (Kohavi, 1995)
- For <10,000 samples: K-Fold cross-validation (K=5)
- Use stratified splits to maintain class proportions

#### 4. Developing a Model Better than Baseline
- Build Single Layer Perceptron (SLP):
  ```python
  model = Sequential([
      Dense(num_classes, activation='softmax', input_shape=(num_features,))
  ])
  ```
- Train with class weights
- Verify it beats naive baseline

#### 5. Scaling Up: Developing a Model that Overfits
- Add hidden layer(s) to increase capacity:
  ```python
  model = Sequential([
      Dense(64, activation='relu', input_shape=(num_features,)),
      Dense(num_classes, activation='softmax')
  ])
  ```
- Train for enough epochs to observe overfitting
- Document the overfitting pattern in training curves

#### 6. Regularising and Tuning Hyperparameters
- Add Dropout and L2 regularisation (Srivastava et al., 2014; Krogh & Hertz, 1992)
- Use Hyperband or Grid Search to tune:
  - Learning rate (1e-4 to 1e-2)
  - Dropout rate (0.0 to 0.5)
  - L2 strength (1e-5 to 1e-2)
- Retrain with best hyperparameters

#### 7. Architecture Exploration (Optional, for Additional Credit)
- Compare wider (128 neurons), deeper (2 layers), narrower (32 neurons) variants
- Document that regularisation matters more than architecture changes

---

### Report Structure

1. **Introduction**
   - Problem: Sentiment analysis for customer feedback
   - Motivation: Business value of automated sentiment classification
   - Dataset description and class distribution

2. **Methodology**
   - Data preprocessing and TF-IDF vectorisation
   - Evaluation protocol justification (Hold-Out vs K-Fold)
   - Model architectures and hyperparameter search strategy

3. **Results**
   - Training curves for each model stage
   - Performance comparison table
   - Confusion matrix and per-class metrics

4. **Analysis**
   - Why regularisation works better than adding capacity
   - Effect of class weights on minority class performance
   - Interpretation of misclassified examples

5. **Conclusions**
   - Key findings and best model recommendation
   - Limitations (TF-IDF loses word order)
   - Future work (embeddings, attention mechanisms)

6. **Code Attribution & References**

---

## Project Idea 2: Regression with Dense Networks (Bike Sharing or Housing)

### Aim

To predict a continuous target variable using dense neural networks, demonstrating the Universal ML Workflow on a regression task:
- Establish baseline with linear model (SLP)
- Build capacity to overfit
- Regularise to generalise
- Compare architectures systematically

---

### Datasets

| Dataset | Features | Samples | Target |
|---------|----------|---------|--------|
| **Bike Sharing Demand** | 12 | 17,379 | Hourly bike rentals |
| **California Housing** | 8 | 20,640 | Median house value |
| **Boston Housing** | 13 | 506 | Median house value |

**Recommendation:** Bike Sharing Demand is ideal—large enough for Hold-Out validation, clear temporal patterns, and intuitive features.

---

### Steps

#### 1. Data Loading and Preprocessing
- Load dataset and handle missing values
- Feature engineering (e.g., cyclical encoding for hour/month)
- Normalise features using StandardScaler
- Split into train/test (80/20)

#### 2. Choosing a Measure of Success
- Primary metric: MAE (interpretable in original units)
- Secondary metric: R² (proportion of variance explained)
- Naive baseline: Predict mean of target

#### 3. Deciding on an Evaluation Protocol
- Hold-Out validation for large datasets (>10,000)
- K-Fold for smaller datasets (Boston Housing)

#### 4. Developing a Model Better than Baseline
- Build linear model (SLP with no hidden layers):
  ```python
  model = Sequential([
      Dense(1, activation='linear', input_shape=(num_features,))
  ])
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  ```
- Verify MAE is lower than naive baseline

#### 5. Scaling Up: Developing a Model that Overfits
- Add hidden layers:
  ```python
  model = Sequential([
      Dense(64, activation='relu', input_shape=(num_features,)),
      Dense(1, activation='linear')
  ])
  ```
- Train until overfitting is visible

#### 6. Regularising and Tuning Hyperparameters
- Add Dropout and L2 regularisation
- Tune hyperparameters using Hyperband
- Compare training vs validation MAE

#### 7. Architecture Exploration
- Test wider, deeper, narrower variants
- Document diminishing returns from added complexity

---

### Report Structure

Similar to Project Idea 1, with regression-specific analysis:
- Residual plots and error distribution
- Feature importance analysis
- Comparison of MSE (training loss) vs MAE (evaluation metric)

---

## Project Idea 3: Image Classification with Flattened Dense Networks

### Aim

To classify simple images using flattened pixel inputs and dense neural networks, understanding the limitations of Dense layers for image data:
- Demonstrate that Dense networks can work on simple images
- Show why CNNs would be better (but are outside scope)
- Apply the full Universal ML Workflow

---

### Datasets

| Dataset | Classes | Image Size | Samples |
|---------|---------|------------|---------|
| **MNIST** | 10 | 28×28 | 70,000 |
| **Fashion MNIST** | 10 | 28×28 | 70,000 |
| **CIFAR-10** (challenging) | 10 | 32×32×3 | 60,000 |

**Recommendation:** Fashion MNIST is ideal—more challenging than MNIST but still achievable with Dense layers.

---

### Steps

#### 1. Data Loading and Preprocessing
- Load dataset from Keras:
  ```python
  from tensorflow.keras.datasets import fashion_mnist
  (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
  ```
- Flatten images: `X_train = X_train.reshape(-1, 784)`
- Normalise to [0, 1]: `X_train = X_train / 255.0`
- One-hot encode labels

#### 2. Choosing a Measure of Success
- Accuracy (balanced classes)
- Confusion matrix for per-class analysis
- Naive baseline: ~10% (random guessing)

#### 3-7. Follow Universal ML Workflow
- Same progression: SLP → Overfit → Regularise → Compare architectures
- Note: Dense networks will plateau around 88-90% on Fashion MNIST
- Discuss why spatial features (CNNs) would help

---

### Report Structure

Include a section discussing:
- Why flattening loses spatial information
- Comparison with published CNN results
- When Dense networks are sufficient vs when CNNs are needed

---

## Project Idea 4: Multi-Class Classification with Tabular Data

### Aim

To perform multi-class classification on structured tabular data, demonstrating:
- Feature preprocessing for mixed data types
- Handling of moderate class imbalance
- Systematic hyperparameter tuning

---

### Datasets

| Dataset | Classes | Features | Samples |
|---------|---------|----------|---------|
| **Wine Quality** | 6-7 | 11 | 6,497 |
| **Covertype** | 7 | 54 | 581,012 |
| **Letter Recognition** | 26 | 16 | 20,000 |

**Recommendation:** Wine Quality is manageable and has interesting feature relationships.

---

### Steps

Follow the Universal ML Workflow with emphasis on:
- Feature scaling (critical for Dense networks)
- Class weight handling for imbalanced classes
- Careful interpretation of multi-class confusion matrix

---

## Choosing Your Project

| If you want... | Choose... |
|----------------|-----------|
| NLP experience | Project 1 (Sentiment) |
| Regression practice | Project 2 (Bike/Housing) |
| Computer vision intro | Project 3 (Images) |
| Tabular data focus | Project 4 (Wine/Covertype) |

---

## Common Pitfalls to Avoid

1. **Starting with data visualisation** — No marks for EDA; start with methodology
2. **Skipping the baseline** — Always establish SLP baseline first
3. **Not showing overfitting** — Section 5 requires demonstrating overfitting
4. **Over-engineering** — Regularisation > complex architectures
5. **Forgetting code attribution** — Required per coursework instructions
6. **Using disallowed techniques** — No CNNs, RNNs, Early Stopping

---

## References

- Chollet, F. (2021) *Deep Learning with Python*. 2nd edn. Manning Publications.
- Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', *IJCAI*, 14(2), pp. 1137–1145.
- Srivastava, N. et al. (2014) 'Dropout: A simple way to prevent neural networks from overfitting', *JMLR*, 15(1), pp. 1929–1958.
- Krogh, A. and Hertz, J.A. (1992) 'A simple weight decay can improve generalization', *NeurIPS*, 4, pp. 950–957.

---

*Last updated: January 2025*
