# Project Idea 3: Image Classification with Flattened Dense Networks

## Aim

To classify simple images using flattened pixel inputs and dense neural networks, demonstrating:

- That Dense networks can work on simple image datasets
- The limitations of Dense layers compared to CNNs (outside scope)
- The full Universal ML Workflow (Chollet, 2021, Chapter 4.5)

This project helps understand why spatial features matter for images, even though we cannot use CNNs for this coursework.

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

| Dataset | Classes | Image Size | Samples | Difficulty |
|---------|---------|------------|---------|------------|
| **MNIST** | 10 (digits) | 28×28 | 70,000 | Easy |
| **Fashion MNIST** | 10 (clothing) | 28×28 | 70,000 | Medium |
| **CIFAR-10** | 10 (objects) | 32×32×3 | 60,000 | Hard |

**Recommendation:** Fashion MNIST is ideal—more challenging than MNIST but still achievable with Dense layers. Expect ~88-90% accuracy ceiling.

---

## Steps

### 1. Data Loading and Preprocessing

- Load dataset from Keras:
  ```python
  from tensorflow.keras.datasets import fashion_mnist
  (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
  print(f"Training: {X_train.shape}, Test: {X_test.shape}")
  # Training: (60000, 28, 28), Test: (10000, 28, 28)
  ```
- Flatten images (28×28 → 784):
  ```python
  X_train_flat = X_train.reshape(-1, 784)
  X_test_flat = X_test.reshape(-1, 784)
  ```
- Normalise to [0, 1]:
  ```python
  X_train_flat = X_train_flat / 255.0
  X_test_flat = X_test_flat / 255.0
  ```
- One-hot encode labels:
  ```python
  from tensorflow.keras.utils import to_categorical
  y_train_cat = to_categorical(y_train, 10)
  y_test_cat = to_categorical(y_test, 10)
  ```

### 2. Choosing a Measure of Success

- **Primary metric:** Accuracy (balanced classes)
- **Secondary:** Confusion matrix for per-class analysis
- Define naive baseline:
  ```python
  # Random guessing for 10 classes
  naive_accuracy = 1 / 10  # 10%
  print(f"Naive Baseline: {naive_accuracy:.1%}")
  ```
- Class names for Fashion MNIST:
  ```python
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  ```

### 3. Deciding on an Evaluation Protocol

- **Hold-Out validation** — Dataset has 60,000 training samples (>10,000 threshold)
- Pre-defined train/test split is already provided
- Create validation split from training data:
  ```python
  from sklearn.model_selection import train_test_split
  X_train_split, X_val, y_train_split, y_val = train_test_split(
      X_train_flat, y_train_cat, test_size=0.1, random_state=42
  )
  ```

### 4. Developing a Model Better than Baseline

- Build Single Layer Perceptron (SLP):
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense

  model = Sequential([
      Dense(10, activation='softmax', input_shape=(784,))
  ])
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  ```
- Train and evaluate:
  ```python
  history = model.fit(X_train_flat, y_train_cat,
                      validation_split=0.1,
                      epochs=50, batch_size=128, verbose=0)
  accuracy = model.evaluate(X_test_flat, y_test_cat)[1]
  print(f"SLP Accuracy: {accuracy:.2%}")
  # Expected: ~84-85%
  ```

### 5. Scaling Up: Developing a Model that Overfits

- Add hidden layers:
  ```python
  model = Sequential([
      Dense(256, activation='relu', input_shape=(784,)),
      Dense(128, activation='relu'),
      Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  ```
- Train for enough epochs to observe overfitting:
  ```python
  history = model.fit(X_train_flat, y_train_cat,
                      validation_split=0.1,
                      epochs=100, batch_size=128, verbose=0)
  ```
- Document: training accuracy ↑ (approaching 100%), validation accuracy plateaus

### 6. Regularising and Tuning Hyperparameters

- Add Dropout and L2 regularisation:
  ```python
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.regularizers import l2

  model = Sequential([
      Dense(256, activation='relu', kernel_regularizer=l2(0.001),
            input_shape=(784,)),
      Dropout(0.4),
      Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
      Dropout(0.4),
      Dense(10, activation='softmax')
  ])
  ```
- Tune hyperparameters:
  - Learning rate: 0.0001 - 0.01
  - Dropout: 0.2 - 0.5
  - L2 strength: 0.0001 - 0.01
  - Hidden neurons: 64, 128, 256

### 7. Architecture Exploration (Optional, for Additional Credit)

- Compare variants:
  - **Wider:** 512 neurons per layer
  - **Deeper:** 3 hidden layers
  - **Narrower:** 64 neurons per layer
- Note the accuracy ceiling (~88-90%) due to Dense layer limitations

---

## Report Structure

### 1. Introduction
- Problem: Image classification without CNNs
- Motivation: Understanding Dense network limitations
- Dataset description (Fashion MNIST categories)

### 2. Methodology
- Image preprocessing (flattening, normalisation)
- Why flattening loses spatial information
- Model architectures and hyperparameter search

### 3. Results
- Training curves showing overfitting progression
- Performance comparison table:

| Model | Architecture | Accuracy |
|-------|--------------|----------|
| Naive Baseline | Random | 10.0% |
| SLP | 784 → 10 | ~84% |
| DNN (overfit) | 784 → 256 → 128 → 10 | ~90%* |
| DNN (regularised) | + Dropout + L2 | ~88% |

*Training accuracy; validation accuracy lower

- Confusion matrix highlighting commonly confused classes

### 4. Analysis
- Which classes are hardest to distinguish (e.g., Shirt vs T-shirt/top)
- Why Dense networks plateau around 88-90%
- Comparison with published CNN results (~93%+)
- Discussion: when are Dense networks sufficient?

### 5. Conclusions
- Key findings about Dense network limitations
- When Dense networks are appropriate for images
- Future work: CNNs for spatial feature extraction

### 6. Code Attribution & References

---

## Common Pitfalls to Avoid

1. **Forgetting to flatten** — Dense layers expect 1D input
2. **Not normalising** — Pixel values 0-255 should be scaled to 0-1
3. **Expecting CNN-level accuracy** — Dense networks plateau around 88-90%
4. **Using too many neurons** — More neurons ≠ better for flattened images

---

## Key Discussion Points

### Why Dense Networks Have Limitations for Images

1. **No spatial awareness:** Flattening destroys 2D structure
2. **No translation invariance:** A shifted digit looks completely different
3. **Too many parameters:** 784 inputs × 256 neurons = 200,704 parameters (just first layer!)

### When Dense Networks Are Sufficient

- Very simple images (MNIST digits)
- When spatial relationships don't matter
- As a baseline before trying CNNs

---

## References

- Chollet, F. (2021) *Deep Learning with Python*. 2nd edn. Shelter Island, NY: Manning Publications.
- LeCun, Y. et al. (1998) 'Gradient-based learning applied to document recognition', *Proceedings of the IEEE*, 86(11), pp. 2278–2324.
- Xiao, H., Rasul, K. and Vollgraf, R. (2017) 'Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms', arXiv:1708.07747.
