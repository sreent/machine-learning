# DNN Code Examples - Use Case Summary

This document summarizes the 10 Deep Neural Network code examples in this folder. Each notebook is a **standalone learning resource** that follows the **Universal ML Workflow** and can be used independently.

---

## Universal ML Workflow

All notebooks follow the same 7-step workflow from Chapter 4.5 of *Deep Learning with Python* (Chollet, 2021):

1. **Defining the problem and assembling a dataset**
2. **Choosing a measure of success**
3. **Deciding on an evaluation protocol**
4. **Preparing your data**
5. **Developing a model that does better than a baseline**
6. **Scaling up: developing a model that overfits**
7. **Regularising your model and tuning your hyperparameters**

---

## Design Principles

All notebooks follow consistent, data-driven design principles:

### Batch Size Selection

| Dataset Size | Batch Size | Rationale |
|--------------|------------|-----------|
| > 10,000 samples | **512** | Efficient GPU utilisation |
| < 10,000 samples | **32-64** | Better gradient estimates for small data |

### Validation Protocol

| Dataset Size | Protocol | Rationale |
|--------------|----------|-----------|
| > 10,000 samples | **Hold-Out** (10%) | Sufficient data for reliable estimates |
| < 10,000 samples | **K-Fold** (5 folds) | Reduces variance in small datasets |

*Reference: Kohavi (1995)*

### Primary Metric Selection

| Class Imbalance | Primary Metric | Rationale |
|-----------------|----------------|-----------|
| < 3:1 ratio | **Accuracy** | Classes roughly balanced |
| > 3:1 ratio | **F1-Score** | Accuracy becomes misleading |

*Reference: He and Garcia (2009)*

### Image Preprocessing

All image notebooks use: **Resize to 32×32 → Grayscale → Flatten → Normalise**

### Architecture

All notebooks use **1 hidden layer with 64 neurons** and demonstrate:
- Why this architecture is sufficient (capacity to overfit)
- Why more layers aren't needed (regularise, don't expand)

### Regularisation

All notebooks use **Dropout + L2** without early stopping, and explain:
- Why regularised models train longer (150 vs 100 epochs)
- *"Regularisation buys you the freedom to train longer."*

---

## Quick Reference Table

| # | Use Case | Problem Type | Data Type | Samples | Batch | Validation | Metric |
|---|----------|--------------|-----------|---------|-------|------------|--------|
| 1 | Twitter US Airline | 3-class NLP | Text | 14,640 | 512 | Hold-Out | F1 |
| 2 | Twitter Entity | 4-class NLP | Text | 74,682 | 512 | Hold-Out | F1 |
| 3 | Movie Review | Binary NLP | Text | 65,000 | 512 | Hold-Out | Accuracy |
| 4 | Rain in Australia | Binary | Mixed Tabular | 142,193 | 512 | Hold-Out | F1 |
| 5 | German Credit | Binary + SMOTE | Mixed Tabular | 1,000 | 32 | K-Fold | AUC |
| 6 | Bike Sharing | Regression | Mixed Tabular | 731 | 32 | K-Fold | MAE |
| 7 | Fashion MNIST | 10-class Image | Grayscale | 70,000 | 512 | Hold-Out | Accuracy |
| 8 | Imagenette | 10-class Image | Colour | 13,000 | 512 | Hold-Out | Accuracy |
| 9 | CatVsDog | Binary Image | Colour | 23,000 | 512 | Hold-Out | Accuracy |
| 10 | ASL Sign Language | 3-class Image | Colour | 9,000 | 64 | K-Fold | Accuracy |

---

## Detailed Use Case Descriptions

### 1. Twitter US Airline Sentiment - NLP Example (Benchmark)

📁 **Folder:** `Twitter US Airline Sentiment/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class Classification (3 classes: Positive, Negative, Neutral) |
| **Data Balance** | Imbalanced (3.88:1 ratio) |
| **Data Type** | Unstructured Text (Tweets) |
| **Samples** | 14,640 |
| **Input Features** | TF-IDF Vectors (5,000 features, bigrams) |
| **Primary Metric** | F1-Score (imbalanced) |
| **Imbalance Handling** | Class Weights |
| **Special Features** | Benchmark notebook; TF-IDF design rationale tables |

---

### 2. Twitter Entity Sentiment - NLP Example

📁 **Folder:** `Twitter Entity Sentiment/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class Classification (4 classes: Positive, Negative, Neutral, Irrelevant) |
| **Data Balance** | Mild Imbalance (1.7:1 ratio) |
| **Data Type** | Unstructured Text (Tweets) |
| **Samples** | 74,682 |
| **Input Features** | TF-IDF Vectors (5,000 features, bigrams) |
| **Primary Metric** | F1-Score |
| **Imbalance Handling** | Class Weights |
| **Special Features** | Entity-level sentiment; 4-class softmax |

---

### 3. Movie Review - NLP Binary Classification Example

📁 **Folder:** `Movie Review/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Classification (Positive/Negative) |
| **Data Balance** | Nearly Balanced (~51:49) |
| **Data Type** | Unstructured Text (Movie Reviews) |
| **Samples** | ~65,000 |
| **Input Features** | TF-IDF Vectors (5,000 features, bigrams) |
| **Primary Metric** | Accuracy (balanced) |
| **Output Layer** | 1 neuron, sigmoid activation |
| **Special Features** | Binary vs multi-class comparison; threshold at 0.5 |

---

### 4. Rain in Australia - Mixed Feature Type & Missing Value Example

📁 **Folder:** `Rain in Australia/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Classification (Rain Tomorrow: Yes/No) |
| **Data Balance** | Imbalanced (3.51:1 ratio) |
| **Data Type** | Structured Tabular (Mixed Categorical & Numerical) |
| **Samples** | 142,193 |
| **Missing Data** | kNN Imputation (Numerical), "Unknown" category (Categorical) |
| **Input Features** | One-Hot Encoding + StandardScaler via ColumnTransformer |
| **Primary Metric** | F1-Score (imbalanced) |
| **Imbalance Handling** | Class Weights |
| **Special Features** | ColumnTransformer pipeline; missing value strategies |

---

### 5. German Credit Data - SMOTE Example

📁 **Folder:** `German Credit Data/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Classification (Credit Risk: Good/Bad) |
| **Data Balance** | Imbalanced (2.33:1 ratio) |
| **Data Type** | Structured Tabular (Mixed Categorical & Numerical) |
| **Samples** | 1,000 |
| **Input Features** | One-Hot Encoding + StandardScaler |
| **Primary Metric** | AUC (industry standard for credit scoring) |
| **Imbalance Handling** | SMOTE (applied only to training folds) |
| **Validation** | 5-Fold Cross-Validation (small dataset) |
| **Special Features** | Lift Curves; SMOTE methodology; AUC vs F1 rationale |

---

### 6. Bike Sharing - Regression Example

📁 **Folder:** `Bike Sharing/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Regression (Predict bike rental count) |
| **Data Type** | Structured Tabular (Mixed Categorical & Numerical) |
| **Samples** | 731 |
| **Target Range** | 22 to 8,714 rentals/day |
| **Input Features** | One-Hot Encoding + StandardScaler |
| **Primary Metric** | MAE (interpretable in original units) |
| **Output Layer** | 1 neuron, linear activation |
| **Validation** | 5-Fold Cross-Validation (small dataset) |
| **Special Features** | Regression vs classification comparison; MAE/RMSE/R² metrics |

---

### 7. Fashion MNIST - TFDS Grayscale Image Example

📁 **Folder:** `Fashion MNIST/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class Classification (10 clothing categories) |
| **Data Balance** | Balanced |
| **Data Type** | Unstructured Images (Grayscale) |
| **Dataset Source** | TensorFlow Datasets: `fashion_mnist` |
| **Samples** | 70,000 |
| **Preprocessing** | Resize 28×28 → 32×32 → Flatten (1,024 features) |
| **Primary Metric** | Accuracy (balanced) |
| **Special Features** | Top-K accuracy; grayscale simplifies processing |

---

### 8. Imagenette - TFDS Colour Image Example

📁 **Folder:** `Imagenette/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class Classification (10 object categories) |
| **Data Balance** | Nearly Balanced |
| **Data Type** | Unstructured Images (Colour) |
| **Dataset Source** | TensorFlow Datasets: `imagenette/160px` |
| **Samples** | ~13,000 |
| **Preprocessing** | Resize 160×160 → 32×32 → Grayscale → Flatten (1,024 features) |
| **Primary Metric** | Accuracy |
| **Special Features** | Real-world photos; harder than synthetic datasets; ImageNet subset |

---

### 9. CatVsDog - TFDS Colour Binary Image Example

📁 **Folder:** `CatVsDog/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Image Classification (Cat vs Dog) |
| **Data Balance** | Balanced (~50:50) |
| **Data Type** | Unstructured Images (Colour) |
| **Dataset Source** | TensorFlow Datasets: `cats_vs_dogs` |
| **Samples** | ~23,000 |
| **Preprocessing** | Resize → 32×32 → Grayscale → Flatten (1,024 features) |
| **Output Layer** | 1 neuron, sigmoid activation |
| **Primary Metric** | Accuracy (balanced) |
| **Special Features** | Binary vs multi-class image classification comparison |

---

### 10. ASL Sign Language - Image Classification Example

📁 **Folder:** `ASL Sign Language/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class Classification (3 letters: A, B, C) |
| **Data Balance** | Perfectly Balanced (3,000 per class) |
| **Data Type** | Unstructured Images (Colour) |
| **Samples** | 9,000 |
| **Preprocessing** | Resize → 32×32 → Grayscale → Flatten (1,024 features) |
| **Primary Metric** | Accuracy (balanced) |
| **Validation** | 5-Fold Cross-Validation (below 10k threshold) |
| **Special Features** | Custom image loading from zip; K-Fold demonstration |

---

## Notebooks by Category

### NLP (Text Classification)
| Notebook | Classes | Key Learning |
|----------|---------|--------------|
| Twitter US Airline | 3 | TF-IDF design rationale; benchmark quality |
| Twitter Entity | 4 | 4-class softmax; entity-level sentiment |
| Movie Review | 2 (binary) | Binary vs multi-class NLP comparison |

### Tabular Data
| Notebook | Problem | Key Learning |
|----------|---------|--------------|
| Rain in Australia | Binary | ColumnTransformer; missing values; kNN imputation |
| German Credit | Binary | SMOTE; Lift Curves; AUC for credit scoring |
| Bike Sharing | Regression | MAE/RMSE/R²; linear output; K-Fold for small data |

### Image Classification
| Notebook | Classes | Key Learning |
|----------|---------|--------------|
| Fashion MNIST | 10 | Grayscale preprocessing; Top-K accuracy |
| Imagenette | 10 | Real-world images; TFDS; harder than synthetic |
| CatVsDog | 2 (binary) | Binary image classification; sigmoid output |
| ASL Sign Language | 3 | Custom data loading; K-Fold for images |

---

## References

- Chollet, F. (2021) *Deep learning with Python*. 2nd edn. Shelter Island, NY: Manning Publications.

- He, H. and Garcia, E.A. (2009) 'Learning from imbalanced data', *IEEE Transactions on Knowledge and Data Engineering*, 21(9), pp. 1263–1284.

- Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', *IJCAI*, 2, pp. 1137–1145.
