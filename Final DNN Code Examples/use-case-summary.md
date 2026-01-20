# DNN Code Examples - Use Case Summary

This document summarizes the Deep Neural Network code examples, organized by use case type. Each example follows the **Universal ML Workflow** from Chapter 4.5 of "Deep Learning with Python".

## Universal ML Workflow Steps

1. **Defining the problem and assembling a dataset**
2. **Choosing a measure of success**
3. **Deciding on an evaluation protocol**
4. **Preparing your data**
5. **Developing a model that does better than a baseline**
6. **Scaling up: developing a model that overfits**
7. **Regularizing your model and tuning your hyperparameters**

---

## 1. Twitter US Airline Sentiment - NLP Example

📁 **Folder:** `Twitter US Airline Sentiment/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class (3) Classification |
| **Data Balance** | Imbalanced |
| **Data Type** | Unstructured (Tweets/Text) |
| **Input Features** | TF-IDF Vectors converted from Tweets |
| **Output** | Multi-Class Probabilities |
| **Imbalance Handling** | Class Weights during Training |
| **Data File** | `Tweets.csv` |

---

## 2. Rain in Australia - Mixed Feature Type & Missing Value Example

📁 **Folder:** `Rain in Australia/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Classification |
| **Data Balance** | Imbalanced with Missing Values |
| **Data Type** | Structured (Mixed Categorical & Numerical) |
| **Missing Data Handling** | kNN Imputing (Numerical), Fill with "Unknown" (Categorical) |
| **Input Features** | One-Hot Encoding (Categorical) + Standardisation (Numerical) |
| **Output** | Binary |
| **Imbalance Handling** | Class Weights during Training |
| **Data File** | `weatherAUS.csv` |

---

## 3. German Credit Data - SMOTE Example

📁 **Folder:** `German Credit Data/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Classification |
| **Data Balance** | Imbalanced |
| **Data Type** | Structured (Mixed Categorical & Numerical) |
| **Input Features** | One-Hot Encoding (Categorical) + Standardisation (Numerical) |
| **Output** | Binary |
| **Imbalance Handling** | SMOTE to up-sample minority class; train on balanced data, validate on imbalanced data |

---

## 4. ASL Sign Language - Image Classification Example

📁 **Folder:** `ASL Sign Language/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class (3) Classification |
| **Data Balance** | Balanced |
| **Data Type** | Unstructured (Images) |
| **Input Features** | Flattened Gray-Scale Images |
| **Preprocessing** | Color Image (3D) → Gray-Scale → 2D Array → 1D Array |
| **Output** | Multi-Class Probabilities |

---

## 5. Bike Sharing - Regression Example

📁 **Folder:** `Bike Sharing/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Regression |
| **Data Type** | Structured (Mixed Categorical & Numerical) |
| **Input Features** | One-Hot Encoding (Categorical) + Standardisation (Numerical) |
| **Output** | Real Value (Predicted Bike Demand) |
| **Data File** | `Bike Sharing.csv` |

---

## 6. Imagenette - TFDS Color Image Example

📁 **Folder:** `Imagenette/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class (10) Image Classification |
| **Data Balance** | Balanced (minimally off-balanced) |
| **Data Type** | Unstructured (Color Images) |
| **Dataset** | TensorFlow Dataset: `imagenette/160px` |
| **Input Features** | Flattened Gray-Scale Images |
| **Preprocessing** | Color Image (3D) → Gray-Scale → 2D Array → 1D Array |
| **Output** | Multi-Class Probabilities, Top-N Accuracy |

---

## 7. Movie Review - NLP Binary Classification Example

📁 **Folder:** `Movie Review/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Classification |
| **Data Balance** | Nearly Balanced (~51:49) |
| **Data Type** | Unstructured (Text/Reviews) |
| **Input Features** | TF-IDF Vectors converted from Text |
| **Output** | Probabilities |
| **Imbalance Handling** | Class Weights (optional - works fine without due to near-balance) |
| **Data File** | `movie_review.csv` |

> **Note**: Class weights are included in the code for reusability with other imbalanced datasets.

---

## 8. CatVsDog - TFDS Color Image Binary Classification Example

📁 **Folder:** `CatVsDog/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Image Classification |
| **Data Balance** | Nearly Balanced |
| **Data Type** | Unstructured (Color Images) |
| **Dataset** | TensorFlow Dataset: `cats_vs_dogs` |
| **Input Features** | Flattened Gray-Scale Images |
| **Preprocessing** | Color Image (3D) → Gray-Scale → 2D Array → 1D Array |
| **Output** | Binary Probabilities |

---

## 9. Fashion MNIST - TFDS Gray-Scaled Image Example

📁 **Folder:** `Fashion MNIST/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class (10) Image Classification |
| **Data Balance** | Balanced |
| **Data Type** | Unstructured (Gray-Scale Images) |
| **Dataset** | TensorFlow Dataset: `fashion_mnist` |
| **Input Features** | Flattened Gray-Scale Images (28×28 → 784) |
| **Output** | Multi-Class Probabilities |

---

## 10. Twitter Sentiment Analysis - Grid Search Example

📁 **Folder:** `Twitter Sentiment Grid Search/`

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class Classification |
| **Data Type** | Unstructured (Tweets/Text) |
| **Input Features** | TF-IDF Vectors |
| **Output** | Multi-Class Probabilities |
| **Special Focus** | Comprehensive Grid Search for Hyperparameter Tuning |
| **Data File** | `twitter.csv` |

---

## Quick Reference Table

| # | Example | Problem Type | Data Type | Imbalance Handling | Folder |
|---|---------|--------------|-----------|-------------------|--------|
| 1 | Twitter US Airline | Multi-Class (3) | Text (NLP) | Class Weights | `Twitter US Airline Sentiment/` |
| 2 | Rain in Australia | Binary | Mixed Structured | Class Weights | `Rain in Australia/` |
| 3 | German Credit | Binary | Mixed Structured | SMOTE | `German Credit Data/` |
| 4 | ASL Sign Language | Multi-Class (3) | Images | N/A (Balanced) | `ASL Sign Language/` |
| 5 | Bike Sharing | Regression | Mixed Structured | N/A | `Bike Sharing/` |
| 6 | Imagenette | Multi-Class (10) | Color Images (TFDS) | N/A (Balanced) | `Imagenette/` |
| 7 | Movie Review | Binary | Text (NLP) | Class Weights (Optional) | `Movie Review/` |
| 8 | CatVsDog | Binary | Color Images (TFDS) | N/A (Balanced) | `CatVsDog/` |
| 9 | Fashion MNIST | Multi-Class (10) | Gray Images (TFDS) | N/A (Balanced) | `Fashion MNIST/` |
| 10 | Twitter Grid Search | Multi-Class | Text (NLP) | Grid Search Focus | `Twitter Sentiment Grid Search/` |
