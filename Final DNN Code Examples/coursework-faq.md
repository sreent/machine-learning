# Deep Neural Network Coursework - Frequently Asked Questions

> **Note:** This FAQ accompanies the Final DNN Code Examples and follows the Universal ML Workflow from *Deep Learning with Python* (Chollet, 2021).

---

## Table of Contents

1. [Dataset Selection](#1-dataset-selection)
2. [Network Architecture](#2-network-architecture)
3. [Training Issues](#3-training-issues)
4. [Universal ML Workflow](#4-universal-ml-workflow)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Common Questions](#6-common-questions)

---

## 1. Dataset Selection

### What kind of datasets work well for this coursework?

Since we are limited to **Dense and Dropout layers only**, certain dataset types are more suitable:

| Dataset Type | Suitability | Examples |
|--------------|-------------|----------|
| **Structured/Tabular (CSV)** | Excellent | Boston Housing, Bike Sharing |
| **Sentiment Analysis (NLP)** | Good | Twitter Sentiment, IMDB Reviews |
| **Simple Images** | Acceptable | MNIST, Fashion MNIST |
| **Complex Images** | Poor | CIFAR-100, ImageNet |
| **Temporal/Sequential** | Poor | Time series, Language modelling |

**Recommended datasets:**

- **Structured data** - CSV format works best with Dense layers
- **Sentiment analysis** - Ignore word order, use Bag-of-Words or TF-IDF vectorisation
- **MNIST-type images** - Low variety, easy to classify (considered "toy datasets")
- **Balanced classes** - Easier to train; imbalanced data requires extra techniques

**Datasets to avoid:**

- High-variety images (require CNNs)
- Sequential/temporal data (require RNNs/LSTMs)
- Severely imbalanced classes (hard to beat baseline with Dense networks)

### Why do large datasets (>10,000 samples) train more stably?

With large datasets:
- Training is very stable with minimal fluctuation
- Validation metrics are consistent
- Hold-out validation is sufficient (no need for K-Fold)

---

## 2. Network Architecture

### How many hidden layers should I use?

With Dense and Dropout layers only (Multi-Layer Perceptrons):

| Data Type | Recommended Hidden Layers |
|-----------|---------------------------|
| Structured/Tabular | 0-1 |
| Sentiment Analysis | 0-1 |
| Simple Images | 1-2 |

**Key points:**
- Rarely need more than 2 hidden layers
- Going beyond 2 requires massive amounts of data
- For small datasets, 0 or 1 hidden layers gives more stable performance
- Sometimes 0 hidden layers (SLP) gives the best result!

### How many neurons should I use in hidden layers?

**Rules of thumb:**

1. Number of neurons should be **between input size and output size**
2. Approximately **2/3 of input size + output size**
3. **Less than twice** the input size
4. **More than** the number of output classes

**Practical advice:**

- Use powers of 2: 4, 8, 16, 32, 64, 128, 256...
- For 10-class classification → at least 16 neurons
- For 64 input features → try 16 or 32 neurons
- Start small, scale up only if needed

**Example:**
```
Input: 64 features
Output: 10 classes
Hidden neurons: 16-32 (between 10 and 64)
```

---

## 3. Training Issues

### What if the validation loss doesn't decrease?

**Symptom:** Validation loss stays flat or fluctuates wildly.

**Most likely cause:** Learning rate too high.

**Solution:** Lower the learning rate.

```python
# Too high (common mistake)
optimizer = Adam(learning_rate=0.01)

# Better starting point
optimizer = Adam(learning_rate=0.001)

# If still unstable
optimizer = Adam(learning_rate=0.0001)
```

### What if both training and validation loss decrease in a straight line?

**Symptom:** Loss keeps decreasing but never reaches a minimum.

**Solutions:**
1. **Increase learning rate** (training is too slow)
2. **Increase epochs** until early stopping triggers

### Why is my validation loss lower than training loss?

This can happen when:

1. **Validation set is "easier"** - Contains less diverse/exotic examples
2. **Dropout effect** - Dropout is active during training but not validation
3. **Data distribution** - Validation data is closer to the "centre" of the distribution

**Example analogy:** Training set has exotic cats that look like dogs, but validation set has only typical-looking cats and dogs.

### How many epochs do I need?

- **Baseline/SLP:** 50-100 epochs (converges quickly)
- **Overfitting model:** 150-200 epochs (to clearly show overfitting)
- **Regularised model:** 100-150 epochs (needs time to converge with dropout)

**Key principle:** Set epochs long enough for early stopping to trigger naturally.

---

## 4. Universal ML Workflow

### Step 4: Developing a Model that Does Better than a Baseline

**What is the baseline?**

| Problem Type | Baseline |
|--------------|----------|
| Balanced classification | 1/num_classes (e.g., 0.5 for binary, 0.33 for 3-class) |
| Imbalanced classification | Largest class proportion (e.g., 0.7 if 70% is one class) |
| Regression | Mean or median of target values |

**First model to try:**

Start with a **Single Layer Perceptron (SLP)** - zero hidden layers:

```python
model = Sequential([
    Dense(num_classes, activation='softmax', input_shape=(num_features,))
])
```

**Why SLP?**
- Mathematically equivalent to Logistic/Linear Regression
- Very likely to beat the naive baseline
- With good features, can perform surprisingly well
- Provides a strong foundation for comparison

### Step 5: Scaling Up - Developing a Model that Overfits

**Purpose:** Confirm you have enough model capacity.

**How to create an overfitting model:**

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),
    Dense(num_classes, activation='softmax')
])
```

**What to look for:**
- Training loss keeps decreasing
- Validation loss starts increasing (classic overfitting pattern)
- This confirms: "We have enough capacity"

**Skinny Jean Analogy:**
> Start with an oversized jean, then shrink it to fit perfectly.

In ML terms: Start with a model that can overfit, then regularise it down.

### Step 6: Regularising Your Model

**Two approaches:**

| Approach | Description |
|----------|-------------|
| **Fix architecture, vary settings** | Keep layers/neurons fixed, tune dropout, learning rate, batch size |
| **Vary both** | Explore different architectures AND settings |

**Regularisation techniques for Dense networks:**

| Technique | Effect |
|-----------|--------|
| **Dropout** | Randomly drops neurons during training |
| **L2 (Weight Decay)** | Penalises large weights |
| **L1 (Lasso)** | Drives some weights to exactly zero |

---

## 5. Hyperparameter Tuning

### What is the purpose of K-Fold and Grid Search?

**Important distinction:**
- Grid Search + K-Fold → Finds optimal **hyperparameters**
- Retraining with those hyperparameters → Gives optimal **model**

**Process:**
1. Use K-Fold + Grid Search on **training data only**
2. Find best hyperparameters (no validation set used!)
3. Retrain model with best hyperparameters on full training data
4. Use validation set **only** for early stopping
5. Evaluate final model on test set

**Why this matters:**
- No information leak from validation to training
- Validation and test performance will be more consistent

### Should I combine train and validation sets for final training?

**No.** You still need validation data for early stopping.

If you combine them:
- No way to know when to stop training
- Model will overfit

### When to use Hold-Out vs K-Fold?

| Dataset Size | Recommended | Rationale |
|--------------|-------------|-----------|
| < 1,000 | K-Fold (K=5 or 10) | Small hold-out sets have high variance |
| 1,000 - 10,000 | Either | K-Fold more robust |
| > 10,000 | Hold-Out | Sufficient data; K-Fold too expensive |

---

## 6. Common Questions

### Why shouldn't I use too many neurons?

**The equation analogy:**

```
2 equations, 4 unknowns → Infinite solutions (overfitting)
2 equations, 2 unknowns → Exactly one solution (good)
Many equations, 2 unknowns → Robust solution (best)
```

**Example:**
- 500 data points, 30 features
- 1 hidden layer with 512 neurons
- Total parameters: ~16,385

This is **32× more parameters than data points**. The model will memorise, not learn.

**Rule:** Keep parameters << data points.

### What models should I explore for the coursework?

**Recommended progression:**

| Model | Architecture | Purpose |
|-------|--------------|---------|
| **Baseline (SLP)** | 0 hidden layers | Beat naive baseline |
| **Overfitting** | 1 hidden layer, 64 neurons | Prove sufficient capacity |
| **Regularised** | 1 hidden layer, 64 neurons + Dropout + L2 | Optimal model |
| **Wider** | 1 hidden layer, 128 neurons + Dropout + L2 | Architecture exploration |
| **Deeper** | 2 hidden layers, 64 each + Dropout + L2 | Architecture exploration |
| **Narrower** | 1 hidden layer, 32 neurons + Dropout + L2 | Architecture exploration |

### What are the review criteria looking for?

1. **Report structure and quality** - Clear headings, tables, narrative
2. **Adherence to the deep learning workflow** - Follow Steps 1-6/7
3. **Systematic investigation** - Not just one model
4. **Interpretation of results** - Explain *why*, not just *what*

**For additional credit:**
- Extensive experimentation
- Understanding beyond the syllabus

---

## Quick Reference

### Recommended Starting Configuration

```python
# Architecture
hidden_layers = 1
hidden_neurons = 64
dropout_rate = 0.3

# Training
batch_size = 32 or 64
learning_rate = 0.001
epochs = 150

# Regularisation
l2_strength = 0.001
```

### Hyperparameter Search Ranges

| Hyperparameter | Range |
|----------------|-------|
| Learning rate | 1e-4 to 1e-2 |
| Dropout | 0.0 to 0.5 |
| L2 regularisation | 1e-5 to 1e-2 |
| Batch size | 32, 64, 128, 256, 512 |
| Hidden neurons | 16, 32, 64, 128 |

---

## References

- Chollet, F. (2021) *Deep Learning with Python*. 2nd edn. Manning Publications.
- Heaton, J. (2008) *Introduction to Neural Networks with Java*. Heaton Research.

---

*Last updated: 2024*
