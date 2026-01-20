# DNN Code Examples - Use Case Summary

This document summarizes the Deep Neural Network code examples, organized by use case type.

---

## 1. Twitter US Airline Sentiment - NLP Example + Cross-Validation

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Multi-Class (3) Classification |
| **Data Balance** | Imbalanced |
| **Data Type** | Unstructured (Tweets/Text) |
| **Input Features** | TF-IDF Vectors converted from Tweets |
| **Output** | Multi-Class Probabilities |
| **Imbalance Handling** | Class Weights during Training |

---

## 2. Rain in Australia - Mixed Feature Type & Missing Value Example

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Classification |
| **Data Balance** | Imbalanced with Missing Values |
| **Data Type** | Structured (Mixed Categorical & Numerical) |
| **Missing Data Handling** | kNN Imputing (Numerical), Fill with "Unknown" (Categorical) |
| **Input Features** | One-Hot Encoding (Categorical) + Standardisation (Numerical) |
| **Output** | Binary |
| **Imbalance Handling** | Class Weights during Training |

---

## 3. German Credit Data - SMOTE Example

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

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Regression |
| **Data Type** | Structured (Mixed Categorical & Numerical) |
| **Input Features** | One-Hot Encoding (Categorical) + Standardisation (Numerical) |
| **Output** | Real Value (Predicted Bike Demand) |

---

## 6. Imagenette - TFDS Color Image Example

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

| Attribute | Description |
|-----------|-------------|
| **Problem Type** | Binary Classification |
| **Data Balance** | Nearly Balanced (~51:49) |
| **Data Type** | Unstructured (Text/Reviews) |
| **Input Features** | TF-IDF Vectors converted from Text |
| **Output** | Probabilities |
| **Imbalance Handling** | Class Weights (optional - works fine without due to near-balance) |

> **Note**: Class weights are included in the code for reusability with other imbalanced datasets.

---

## 8. CatVsDog - TFDS Color Image Binary Classification Example

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

## Quick Reference Table

| Example | Problem Type | Data Type | Imbalance Handling |
|---------|--------------|-----------|-------------------|
| Twitter US Airline | Multi-Class (3) | Text (NLP) | Class Weights |
| Rain in Australia | Binary | Mixed Structured | Class Weights |
| German Credit | Binary | Mixed Structured | SMOTE |
| ASL Sign Language | Multi-Class (3) | Images | N/A (Balanced) |
| Bike Sharing | Regression | Mixed Structured | N/A |
| Imagenette | Multi-Class (10) | Color Images | N/A (Balanced) |
| Movie Review | Binary | Text (NLP) | Class Weights (Optional) |
| CatVsDog | Binary | Color Images | N/A (Balanced) |
