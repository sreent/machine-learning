# Deep Neural Network Coursework - Frequently Asked Questions

> This FAQ accompanies the Final DNN Code Examples and follows the Universal ML Workflow from *Deep Learning with Python* (Chollet, 2021).

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Dataset Selection](#2-dataset-selection)
3. [Model Architecture](#3-model-architecture)
4. [Training Problems](#4-training-problems)
5. [Metrics & Evaluation](#5-metrics--evaluation)
6. [Class Imbalance](#6-class-imbalance)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Common Errors](#8-common-errors)
9. [Write-up & Submission](#9-write-up--submission)
10. [Quick Reference](#10-quick-reference)

---

## 1. Getting Started

### What are the constraints for this coursework?

You must use **only** techniques from DLWP Chapters 1-4:

| Allowed | Not Allowed |
|---------|-------------|
| Dense layers | CNNs (Conv2D) |
| Dropout layers | RNNs (LSTM, GRU) |
| L1/L2 regularisation | Transformers |
| Adam, SGD optimisers | Early Stopping* |
| Softmax, ReLU, Sigmoid | Pre-trained models |

*Early Stopping may be restricted depending on your specific assignment.

### What should my notebook look like?

Your notebook should read as a **report**, not just code:

```
✓ Introduction (problem, motivation, dataset)
✓ Methodology sections with explanations
✓ Tables and visualisations
✓ Analysis and interpretation
✓ Conclusions
✓ Code attribution
✓ References
```

### Which workflow should I follow?

Follow the **Universal ML Workflow** (Chollet, 2021, Chapter 4.5):

1. Define the problem & assemble dataset
2. Choose a measure of success
3. Decide on an evaluation protocol
4. Prepare your data
5. Develop a model better than baseline
6. Scale up: develop a model that overfits
7. Regularise and tune hyperparameters

---

## 2. Dataset Selection

### What datasets work well with Dense layers?

| Dataset Type | Suitability | Why |
|--------------|-------------|-----|
| **Tabular/CSV** | Excellent | Dense layers designed for this |
| **Sentiment Analysis** | Good | TF-IDF ignores word order |
| **MNIST-type images** | Good | Simple, low variety |
| **Complex images** | Poor | Need CNNs for spatial features |
| **Time series** | Poor | Need RNNs for temporal patterns |

### What if my dataset is imbalanced?

Imbalanced data (e.g., 90% class A, 10% class B) is challenging but manageable:

1. **Use class weights** - Most important technique
2. **Use F1-Score** - Not accuracy (see [Section 6](#6-class-imbalance))
3. **Check confusion matrix** - See per-class performance

### How much data do I need?

| Dataset Size | What to Expect |
|--------------|----------------|
| < 1,000 | High variance, use K-Fold, keep model simple |
| 1,000 - 10,000 | Moderate, either K-Fold or Hold-Out |
| > 10,000 | Stable training, Hold-Out is fine |

**Rule of thumb:** You need at least 10× more samples than model parameters.

---

## 3. Model Architecture

### How many hidden layers do I need?

**Short answer:** Usually 0-2 for this coursework.

| Scenario | Recommended |
|----------|-------------|
| Simple tabular data | 0-1 hidden layers |
| Sentiment analysis | 0-1 hidden layers |
| Image classification | 1-2 hidden layers |

**Why not more?**
- Dense networks don't benefit much from depth
- More layers = harder to train, need more data
- Overfitting risk increases

### How many neurons per layer?

**Guidelines:**

1. Start with 32-64 neurons
2. Must be ≥ number of output classes
3. Usually ≤ number of input features
4. Use powers of 2: 16, 32, 64, 128...

**Example:**
```python
# 100 input features, 3 output classes
# Try: 32 or 64 neurons

model = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(3, activation='softmax')
])
```

### Which activation function should I use?

| Layer | Activation | When |
|-------|------------|------|
| Hidden layers | `relu` | Default choice, works well |
| Output (multi-class) | `softmax` | Classes are mutually exclusive |
| Output (binary) | `sigmoid` | Two classes |
| Output (regression) | `linear` or none | Predicting continuous values |

### Which loss function should I use?

| Problem | Loss Function | Output Activation |
|---------|---------------|-------------------|
| Multi-class (one-hot labels) | `categorical_crossentropy` | `softmax` |
| Multi-class (integer labels) | `sparse_categorical_crossentropy` | `softmax` |
| Binary classification | `binary_crossentropy` | `sigmoid` |
| Regression | `mse` or `mae` | `linear`/none |

---

## 4. Training Problems

### My loss is not decreasing at all

**Possible causes and solutions:**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss stuck at high value | Learning rate too high | Reduce to 0.0001 |
| Loss stuck at random-guess level | Model too simple | Add neurons/layers |
| Loss is NaN | Learning rate way too high | Use 0.0001, check for data issues |
| Loss jumps around wildly | Learning rate too high | Reduce learning rate |

```python
# If loss won't decrease, try:
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### My loss decreases but very slowly (straight line)

**The model is learning, but too slowly.**

Solutions:
1. **Increase learning rate** (e.g., 0.001 → 0.01)
2. **Increase epochs** (e.g., 100 → 300)
3. **Reduce batch size** (more gradient updates per epoch)

### Training loss decreases but validation loss increases

**This is overfitting!** Your model is memorising, not learning.

Solutions:
1. Add **Dropout** (0.2-0.5) — randomly drops neurons during training (Srivastava et al., 2014)
2. Add **L2 regularisation** (0.001-0.01) — penalises large weights (Krogh & Hertz, 1992)
3. **Reduce** model size (fewer neurons)
4. Get **more data** (if possible)

```python
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
```

### Validation loss is lower than training loss - is this wrong?

**No, this is normal!** Common reasons:

1. **Dropout** - Active during training, disabled during validation
2. **Easier validation set** - By chance, validation samples are easier
3. **Regularisation** - Adds penalty to training loss only

This is usually fine. Focus on whether validation loss is stable.

### How many epochs should I use?

| Model Type | Epochs | Rationale |
|------------|--------|-----------|
| Baseline (SLP) | 50-100 | Simple model, converges quickly |
| Overfitting model | 150-200 | Need to clearly see overfitting |
| Regularised model | 100-200 | Regularisation slows learning |

**Tip:** Watch the validation loss plot. Stop when it plateaus or increases.

---

## 5. Metrics & Evaluation

Choosing the right evaluation metric is crucial for assessing model performance (Sokolova & Lapalme, 2009).

### Which metric should I use?

| Situation | Primary Metric | Why |
|-----------|----------------|-----|
| Balanced classes | Accuracy | All classes equally important |
| Imbalanced classes | F1-Score (macro) | Accounts for minority classes |
| Ranking/threshold tuning | AUC | Threshold-independent |
| Regression | MAE or R² | Interpretable error measure |

### What's the difference between accuracy and F1-Score?

**Example:** Dataset with 95% negative, 5% positive

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Predicts all negative | 95% (looks great!) | 0.00 (terrible!) |
| Actually learns | 90% | 0.70 (much better) |

**Lesson:** For imbalanced data, accuracy is misleading. Use F1-Score.

### How do I read a confusion matrix?

```
                 Predicted
              Neg    Pos
Actual  Neg   TN     FP    ← FP = False alarms
        Pos   FN     TP    ← FN = Missed cases
              ↑
           FN = Missed
```

**Good model:** High numbers on diagonal (TN, TP), low off-diagonal (FP, FN)

### What is the naive baseline?

The simplest possible prediction:

| Problem Type | Naive Baseline | Example |
|--------------|----------------|---------|
| Balanced 2-class | 50% accuracy | Random guessing |
| Balanced 3-class | 33% accuracy | Random guessing |
| Imbalanced (70-30) | 70% accuracy | Always predict majority |
| Regression | Mean of target | Predict average value |

**Your model must beat this to be useful!**

---

## 6. Class Imbalance

Class imbalance is a common challenge in machine learning where some classes have significantly more samples than others (He & Garcia, 2009).

### How do I know if my data is imbalanced?

```python
# Check class distribution
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True))  # As percentages
```

| Ratio | Severity |
|-------|----------|
| 60-40 | Mild - usually fine |
| 70-30 | Moderate - use class weights |
| 90-10 | Severe - definitely use class weights + F1 |
| 99-1 | Extreme - consider resampling |

### How do I use class weights?

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute weights
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = dict(zip(classes, weights))

print(f"Class weights: {class_weight}")
# Example output: {0: 0.52, 1: 1.89, 2: 3.21}

# Use in training
model.fit(X_train, y_train, class_weight=class_weight, ...)
```

**What this does:** Makes the model pay more attention to minority classes.

### Should I use oversampling or undersampling?

For this coursework, **class weights are usually sufficient** and simpler.

If you want to try resampling:
- **Oversampling** (SMOTE): Better when you have limited data (Chawla et al., 2002)
- **Undersampling**: Risk losing information from majority class

---

## 7. Hyperparameter Tuning

### What hyperparameters should I tune?

**Priority order:**

1. **Learning rate** - Most impactful (try: 0.0001, 0.001, 0.01)
2. **Dropout rate** - For regularisation (try: 0.0, 0.2, 0.3, 0.5)
3. **L2 strength** - For regularisation (try: 0.0001, 0.001, 0.01)
4. **Number of neurons** - Architecture (try: 32, 64, 128)
5. **Batch size** - Training dynamics (try: 32, 64, 128, 256)

### Hold-Out vs K-Fold: When to use which?

The choice depends on dataset size (Kohavi, 1995):

| Dataset Size | Method | Why |
|--------------|--------|-----|
| < 1,000 | K-Fold (K=5) | Small test sets are unreliable |
| 1,000 - 10,000 | Either | K-Fold more robust |
| > 10,000 | Hold-Out | Sufficient data, K-Fold too slow |

### What is Hyperband and when should I use it?

**Hyperband** (Li et al., 2018) is an efficient hyperparameter search method:

- Trains many configurations for few epochs
- Eliminates poor performers early
- Focuses resources on promising configurations

**Use Hyperband when:**
- You have many hyperparameters to tune
- Training is slow
- Dataset is large (>10,000 samples)

```python
import keras_tuner as kt

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3
)
tuner.search(X_train, y_train, validation_split=0.1)
```

---

## 8. Common Errors

### ValueError: Shapes (X,) and (Y,) are incompatible

**Cause:** Mismatch between model output and labels.

**Solutions:**

```python
# If using categorical_crossentropy, labels must be one-hot:
from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train)

# OR use sparse_categorical_crossentropy with integer labels:
model.compile(loss='sparse_categorical_crossentropy', ...)
```

### ValueError: Input 0 is incompatible with layer

**Cause:** Input shape doesn't match your data.

```python
# Check your data shape
print(f"X_train shape: {X_train.shape}")  # e.g., (1000, 64)

# Model input_shape should match (exclude batch dimension)
model = Sequential([
    Dense(32, activation='relu', input_shape=(64,)),  # Not (1000, 64)!
    ...
])
```

### ResourceExhaustedError: OOM (Out of Memory)

**Cause:** Model or batch too large for GPU memory.

**Solutions:**
1. Reduce batch size: `batch_size=32` instead of 512
2. Reduce model size (fewer neurons)
3. Use CPU: `import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

### My Colab session keeps disconnecting

**Tips:**
- Save checkpoints: `ModelCheckpoint('model.h5')`
- Save to Google Drive regularly
- Keep browser tab active
- Use smaller batch sizes (faster epochs)

---

## 9. Write-up & Submission

### What should I include in my report?

| Section | Contents |
|---------|----------|
| **Introduction** | Problem definition, motivation, dataset description |
| **Methodology** | Data preprocessing, evaluation protocol, model architecture |
| **Results** | Training curves, performance metrics, comparison table |
| **Analysis** | Interpretation of results, why models performed as they did |
| **Conclusions** | Key findings, limitations, future work |
| **Code Attribution** | What code was adapted vs original |
| **References** | Chollet (2021), dataset source, other citations |

### How do I cite/attribute code?

The coursework instruction says: *"reference all code that is not original"*

**Create a Code Attribution table:**

| Component | Source | Adaptation |
|-----------|--------|------------|
| TF-IDF vectorisation | scikit-learn | Standard usage |
| Model architecture | Chollet (2021) Ch.4 | Applied to my dataset |
| Hyperband tuning | keras-tuner docs | Customised for this problem |
| Training loop | Course materials | Modified for class weights |

**Mark original contributions:**
- Novel problem framing
- Custom analysis
- Unique experiments

### What format should I submit?

Per the instruction: **Export to HTML only**

```python
# In Colab:
# 1. Download .ipynb
# 2. Open in Jupyter
# 3. File → Export as → HTML

# Or use command line:
# jupyter nbconvert --to html notebook.ipynb
```

### What are markers looking for?

| Criterion | What They Want |
|-----------|----------------|
| Report structure | Clear sections, tables, professional presentation |
| Workflow adherence | Follow the 7 steps systematically |
| Systematic investigation | Multiple models compared fairly |
| Interpretation | Explain *why*, not just *what* |
| Code attribution | Honest about what's original vs adapted |

**For additional credit:**
- Extensive experimentation beyond the minimum
- Insights that show deep understanding

---

## 10. Quick Reference

### Recommended Starting Point

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

model = Sequential([
    Dense(64, activation='relu',
          kernel_regularizer=l2(0.001),
          input_shape=(num_features,)),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=150,
    batch_size=64,
    class_weight=class_weight,  # If imbalanced
    verbose=1
)
```

### Hyperparameter Cheat Sheet

| Parameter | Start With | Range to Try |
|-----------|------------|--------------|
| Learning rate | 0.001 | 0.0001 - 0.01 |
| Dropout | 0.3 | 0.0 - 0.5 |
| L2 regularisation | 0.001 | 0.00001 - 0.01 |
| Batch size | 64 | 32 - 512 |
| Hidden neurons | 64 | 16 - 128 |
| Hidden layers | 1 | 0 - 2 |

### Decision Flowchart

```
Is model learning? (training loss decreasing)
├── No → Lower learning rate OR simplify model
└── Yes → Is it overfitting? (val loss increasing)
    ├── Yes → Add dropout/L2 OR reduce model size
    └── No → Is performance good enough?
        ├── No → Try different architecture
        └── Yes → Done! Evaluate on test set
```

---

## References

### Primary Text

- Chollet, F. (2021) *Deep Learning with Python*. 2nd edn. Shelter Island, NY: Manning Publications.
  - Chapter 4.5: The Universal Workflow of Machine Learning
  - Chapter 4.4: Overfitting and Underfitting
  - Chapter 4.3: Evaluating Machine Learning Models

### Validation & Cross-Validation

- Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', *IJCAI*, 14(2), pp. 1137–1145.

- Bergstra, J. and Bengio, Y. (2012) 'Random search for hyper-parameter optimization', *Journal of Machine Learning Research*, 13(1), pp. 281–305.

### Class Imbalance

- He, H. and Garcia, E.A. (2009) 'Learning from imbalanced data', *IEEE Transactions on Knowledge and Data Engineering*, 21(9), pp. 1263–1284.

- Chawla, N.V. et al. (2002) 'SMOTE: Synthetic minority over-sampling technique', *Journal of Artificial Intelligence Research*, 16, pp. 321–357.

### Regularisation

- Srivastava, N. et al. (2014) 'Dropout: A simple way to prevent neural networks from overfitting', *Journal of Machine Learning Research*, 15(1), pp. 1929–1958.

- Krogh, A. and Hertz, J.A. (1992) 'A simple weight decay can improve generalization', *Advances in Neural Information Processing Systems*, 4, pp. 950–957.

### Hyperparameter Tuning

- Li, L. et al. (2018) 'Hyperband: A novel bandit-based approach to hyperparameter optimization', *Journal of Machine Learning Research*, 18(1), pp. 6765–6816.

### Evaluation Metrics

- Sokolova, M. and Lapalme, G. (2009) 'A systematic analysis of performance measures for classification tasks', *Information Processing & Management*, 45(4), pp. 427–437.

### Software Documentation

- Keras Documentation. Available at: https://keras.io/ (Accessed: January 2025).

- scikit-learn Documentation. Available at: https://scikit-learn.org/ (Accessed: January 2025).

- O'Malley, T. et al. (2019) *Keras Tuner*. Available at: https://github.com/keras-team/keras-tuner (Accessed: January 2025).

---

*Last updated: January 2025*
