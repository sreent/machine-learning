# Style Guide: Final DNN Code Examples

This style guide establishes quality standards for all notebooks in the `Final DNN Code Examples` folder. It is based on the **Twitter US Airline Sentiment** notebook, which serves as the reference implementation.

---

## Table of Contents

1. [Core Principles](#1-core-principles)
2. [Notebook Structure](#2-notebook-structure)
3. [Narrative Standards](#3-narrative-standards)
4. [Code Standards](#4-code-standards)
5. [Design Decision Documentation](#5-design-decision-documentation)
6. [Data-Driven Decisions](#6-data-driven-decisions)
7. [British English Conventions](#7-british-english-conventions)
8. [Reference Formatting](#8-reference-formatting)
9. [Problem-Type Adaptations](#9-problem-type-adaptations)
10. [Quality Checklist](#10-quality-checklist)

---

## 1. Core Principles

### 1.1 Technique Scope

All notebooks use **only techniques from Chapters 1–4** of *Deep Learning with Python* (Chollet, 2021):

| Technique | Status | Chapter |
|-----------|--------|---------|
| Dense layers (MLP/DNN) | ✓ Allowed | Ch. 3-4 |
| Dropout | ✓ Allowed | Ch. 4 |
| L2 regularisation | ✓ Allowed | Ch. 4 |
| Early stopping | ✗ Not allowed | Ch. 7 |
| CNN | ✗ Not allowed | Ch. 8 |
| RNN/LSTM | ✗ Not allowed | Ch. 10 |
| Word embeddings | ✗ Not allowed | Ch. 11 |

**Rationale:** Demonstrates that fundamental techniques can achieve strong results before introducing advanced architectures.

### 1.2 Universal ML Workflow

All notebooks follow the **8-step Universal ML Workflow**:

1. Defining the problem and assembling a dataset
2. Choosing a measure of success
3. Deciding on an evaluation protocol
4. Preparing your data
5. Developing a model that does better than a baseline
6. Scaling up: developing a model that overfits
7. Regularising your model and tuning hyperparameters
8. Results summary and key takeaways

### 1.3 Terminology

| Avoid | Use Instead |
|-------|-------------|
| MLP | DNN (Deep Neural Network) |
| Regularization | Regularisation |
| Hidden layer network | DNN |
| Frozen (for fixed values) | Fixed |

---

## 2. Notebook Structure

### 2.1 Required Sections

Every notebook must include these sections in order:

```
1. Title & Colab Badge
2. Learning Objectives
3. Dataset Overview (table)
4. Technique Scope (table)
5. Section 1: Defining the Problem
6. Section 2: Choosing a Measure of Success
7. Section 3: Deciding on an Evaluation Protocol
8. Section 4: Preparing Your Data
9. Section 5: Developing a Model Better Than Baseline
10. Section 6: Scaling Up (Overfitting Demo)
11. Section 7: Regularising and Tuning
12. Section 8: Results Summary
13. Section 9: Key Takeaways
14. Appendix: Modular Helper Functions (optional)
```

### 2.2 Opening Section Template

```markdown
<Colab Badge>

# [Dataset Name] - [Problem Type] Example

This notebook demonstrates the **Universal ML Workflow** applied to [problem description].

## Learning Objectives

By the end of this notebook, you will be able to:
- [Objective 1]
- [Objective 2]
- ...

---

## Dataset Overview

| Attribute | Description |
|-----------|-------------|
| **Source** | [Link to dataset] |
| **Problem Type** | [Classification/Regression] |
| **Data Balance** | [Balanced/Imbalanced with ratio] |
| **Data Type** | [Tabular/Text/Image] |
| **Input Features** | [Description] |
| **Output** | [Target variable] |
| **Special Handling** | [Class weights/SMOTE/None] |

---

## Technique Scope

This notebook uses only techniques from **Chapters 1–4** of *Deep Learning with Python* (Chollet, 2021).

| Technique | Status | Rationale |
|-----------|--------|-----------|
| **Dense layers (DNN)** | ✓ Used | Core building block (Ch. 3-4) |
| **Dropout** | ✓ Used | Regularisation technique (Ch. 4) |
| **L2 regularisation** | ✓ Used | Weight penalty (Ch. 4) |
| **Early stopping** | ✗ Not used | Introduced in Ch. 7 |
| **CNN** | ✗ Not used | Introduced in Ch. 8 |
| **RNN/LSTM** | ✗ Not used | Introduced in Ch. 10 |

---
```

### 2.3 Key Takeaways Section Template

```markdown
## 9. Key Takeaways

### Decision Framework Summary

| Decision | Threshold | This Dataset | Choice | Reference |
|----------|-----------|--------------|--------|-----------|
| **[Decision 1]** | [Threshold] | [Value] | [Choice] | [Citation] |
| **[Decision 2]** | [Threshold] | [Value] | [Choice] | [Citation] |

### Lessons Learned

1. **[Lesson title]:** [Description]
2. **[Lesson title]:** [Description]
...

### References

- [Harvard-style references]
```

---

## 3. Narrative Standards

### 3.1 Explanation Depth

Every significant choice must be explained with:
- **What:** What we're doing
- **Why:** Why this approach (not just "it's common practice")
- **Trade-offs:** What we gain and lose

### 3.2 Formatting Conventions

| Element | Format |
|---------|--------|
| Key terms (first use) | **Bold** |
| Code/parameters | `backticks` |
| Book titles | *Italics* |
| Key insights | Blockquote (`>`) |
| Comparisons | Tables |
| Step-by-step processes | Numbered lists |

### 3.3 Blockquote Usage

Use blockquotes for memorable insights:

```markdown
> *"Regularisation adds noise and constraints that slow down learning. In exchange for protection against overfitting, the model needs more iterations to converge."*
```

### 3.4 Section Transitions

Each section should:
- Start with a brief explanation of what we're doing and why
- End with a clear outcome or observation
- Connect to the next step in the workflow

---

## 4. Code Standards

### 4.1 Constants and Configuration

Define all constants at the top of the relevant section:

```python
# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

BATCH_SIZE = 512
EPOCHS_BASELINE = 100      # SLP and DNN (no regularisation)
EPOCHS_REGULARIZED = 150   # DNN with Dropout + L2
```

### 4.2 Comment Standards

```python
# Good: Explains WHY
# Using EPOCHS_REGULARIZED: Regularisation slows learning, so we need more epochs

# Bad: Explains WHAT (obvious from code)
# Set epochs to 150
```

### 4.3 Magic Number Guidelines

| Type | Guideline |
|------|-----------|
| Thresholds | Define as constants with citations |
| Hyperparameters | Explain the choice in narrative |
| Random seeds | Always set and document |

```python
SEED = 204
HOLDOUT_THRESHOLD = 10000  # Use hold-out if samples > 10,000 (Kohavi, 1995)
IMBALANCE_THRESHOLD = 3.0  # Use F1-Score if ratio > 3.0 (He & Garcia, 2009)
```

### 4.4 Output Formatting

```python
# Use clear headers for important outputs
print("=" * 60)
print("DATA-DRIVEN CONFIGURATION")
print("=" * 60)

# Mark primary metrics clearly
print(f'F1-Score (Validation): {f1_score:.2f}  ← Primary Metric')
```

### 4.5 Package Installation

Use `%pip` instead of `!pip` for Colab compatibility:

```python
# Good
%pip install -q -U keras-tuner

# Avoid
!pip install -q -U keras-tuner
```

### 4.6 Prediction Consistency

Always use `argmax` for storing predictions (not probabilities):

```python
# Consistent approach
preds_val = model.predict(X_val, verbose=0).argmax(axis=1)
preds_test = model.predict(X_test, verbose=0).argmax(axis=1)
```

---

## 5. Design Decision Documentation

### 5.1 Required Design Decisions

Document these choices in every notebook:

| Category | Decisions to Document |
|----------|----------------------|
| **Data** | Preprocessing, feature engineering, encoding |
| **Metrics** | Primary metric selection, why not alternatives |
| **Evaluation** | Hold-out vs K-fold, split ratios |
| **Architecture** | Number of layers, neurons, activations |
| **Regularisation** | Dropout vs L2 vs both, why not early stopping |
| **Training** | Batch size, epochs, optimiser |
| **Imbalance** | Class weights vs SMOTE vs undersampling |

### 5.2 Comparison Table Format

```markdown
#### Why [Choice A] Instead of [Choice B]?

| Approach | Pros | Cons |
|----------|------|------|
| **[Choice A]** | [Pro 1], [Pro 2] | [Con 1] |
| **[Choice B]** | [Pro 1] | [Con 1], [Con 2] |

We use **[Choice A]** because:
1. [Reason 1]
2. [Reason 2]
3. [Reason 3]
```

### 5.3 Architecture Decision Template

```markdown
#### Why [N] neurons in the hidden layer?

This is a practical starting point that balances capacity and efficiency:
- **Too few (e.g., [smaller]):** [Problem]
- **Too many (e.g., [larger]):** [Problem]
- **[N] neurons:** [Justification]

#### Why only 1 hidden layer instead of 2-3?

Per the **Universal ML Workflow**, the goal is to demonstrate that the model *can* overfit. Once overfitting is observed:
1. **Capacity is proven sufficient**
2. **No need for more depth**
3. **Regularise, don't expand**

*"The right question is not 'How many layers?' but 'Can it overfit?' If yes, regularise. If no, add capacity."*
```

---

## 6. Data-Driven Decisions

### 6.1 Standard Thresholds

Use these thresholds with citations:

| Decision | Threshold | Citation |
|----------|-----------|----------|
| Hold-out vs K-Fold | > 10,000 samples | Kohavi (1995); Chollet (2021) |
| Accuracy vs F1-Score | > 3:1 imbalance ratio | He and Garcia (2009) |
| Severe imbalance | > 10:1 ratio | Branco et al. (2016) |

### 6.2 Data-Driven Code Block

```python
# =============================================================================
# DATA-DRIVEN ANALYSIS: Dataset Size & Imbalance
# =============================================================================

n_samples = len(data)
HOLDOUT_THRESHOLD = 10000  # Kohavi (1995); Chollet (2021)

imbalance_ratio = majority_class / minority_class
IMBALANCE_THRESHOLD = 3.0  # He & Garcia (2009)

use_holdout = n_samples > HOLDOUT_THRESHOLD
use_f1 = imbalance_ratio > IMBALANCE_THRESHOLD

print(f"Dataset size: {n_samples:,} → {'Hold-Out' if use_holdout else 'K-Fold'}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1 → {'F1-Score' if use_f1 else 'Accuracy'}")
```

---

## 7. British English Conventions

### 7.1 Spelling

| American (Avoid) | British (Use) |
|------------------|---------------|
| regularization | regularisation |
| optimization | optimisation |
| modeling | modelling |
| labeled | labelled |
| behavior | behaviour |
| organization | organisation |
| visualization | visualisation |
| parallelizable | parallelisable |
| color | colour |

### 7.2 Exceptions

Keep American spelling for:
- **API/library names:** `optimizer=`, `regularizers.l2()`, `OPTIMIZER`
- **Function names:** `TfidfVectorizer`
- **Error messages from libraries**

### 7.3 Narrative Examples

```markdown
# Good
The model uses **Dropout + L2 regularisation** to prevent overfitting.
We use the Adam optimiser for training.

# Bad (American spelling in narrative)
The model uses **Dropout + L2 regularization** to prevent overfitting.
```

---

## 8. Reference Formatting

### 8.1 Harvard Style

Use Harvard referencing for all citations:

```markdown
### References

- Chollet, F. (2021) *Deep learning with Python*. 2nd edn. Shelter Island, NY: Manning Publications.

- He, H. and Garcia, E.A. (2009) 'Learning from imbalanced data', *IEEE Transactions on Knowledge and Data Engineering*, 21(9), pp. 1263–1284.

- Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', *IJCAI*, 2, pp. 1137–1145.
```

### 8.2 In-Text Citations

```markdown
# Parenthetical
This threshold is a practical guideline (He and Garcia, 2009).

# Narrative
He and Garcia (2009) suggest using F1-Score for imbalanced data.
```

### 8.3 Required References

Every notebook should cite:
- Chollet (2021) - for technique scope and workflow
- Problem-specific references for metric/method choices

---

## 9. Problem-Type Adaptations

### 9.1 Classification (Multi-Class)

| Element | Standard |
|---------|----------|
| Primary metric | F1-Score (macro) if imbalanced, Accuracy if balanced |
| Loss function | `categorical_crossentropy` |
| Output activation | `softmax` |
| Label encoding | One-hot (`to_categorical`) |

### 9.2 Classification (Binary)

| Element | Standard |
|---------|----------|
| Primary metric | F1-Score if imbalanced, Accuracy if balanced |
| Loss function | `binary_crossentropy` |
| Output activation | `sigmoid` |
| Label encoding | Single column (0/1) |

### 9.3 Regression

| Element | Standard |
|---------|----------|
| Primary metric | RMSE or MAE (justify choice) |
| Loss function | `mse` or `mae` |
| Output activation | `linear` (none) |
| Evaluation | No class imbalance; consider target distribution |

### 9.4 Imbalanced Data

| Imbalance Ratio | Handling |
|-----------------|----------|
| ≤ 3:1 | No special handling needed |
| 3:1 – 10:1 | Class weights |
| > 10:1 | Class weights + consider SMOTE |

### 9.5 Image Data (Flattened)

| Element | Standard |
|---------|----------|
| Preprocessing | Flatten to 1D, normalise to [0, 1] |
| Input dimension | height × width × channels |
| Note in narrative | Explain why not using CNN (Ch. 1-4 scope) |

### 9.6 Text Data

| Element | Standard |
|---------|----------|
| Vectorisation | TF-IDF (word embeddings outside scope) |
| Preprocessing | Minimal (explain why) |
| Features | 5000 is a reasonable default |

---

## 10. Quality Checklist

Use this checklist before finalising any notebook:

### Structure
- [ ] Follows Universal ML Workflow (8 sections)
- [ ] Has Colab badge at top
- [ ] Has Learning Objectives
- [ ] Has Dataset Overview table
- [ ] Has Technique Scope table
- [ ] Has Key Takeaways section
- [ ] Has References section

### Narrative
- [ ] Every significant choice is explained (what, why, trade-offs)
- [ ] Design decisions use comparison tables
- [ ] Key insights use blockquotes
- [ ] British English throughout (except API names)
- [ ] No unexplained magic numbers
- [ ] Harvard-style citations

### Code
- [ ] Constants defined at top with comments
- [ ] Comments explain WHY, not WHAT
- [ ] Uses `%pip` not `!pip`
- [ ] Predictions use consistent `argmax` approach
- [ ] Random seed is set
- [ ] Output is clearly formatted

### Technical
- [ ] Only Ch. 1-4 techniques used
- [ ] Data-driven metric selection with thresholds
- [ ] Data-driven evaluation protocol selection
- [ ] Baseline model established before complex models
- [ ] Overfitting demonstrated before regularisation
- [ ] Regularisation compared to non-regularised model

### Consistency
- [ ] Terminology matches style guide (DNN, regularisation, etc.)
- [ ] Tables use consistent formatting
- [ ] Code style is consistent throughout

---

## Reference Implementation

**Twitter US Airline Sentiment - NLP Example** serves as the reference implementation. When in doubt, consult that notebook for examples of:
- Section structure and flow
- Design decision documentation
- Data-driven analysis code
- Narrative tone and depth
- British English usage

---

*Last updated: January 2025*
