# Decision Tree

## 1 Introduction  

Imagine you’re trying to **diagnose a problem** or **make a decision** by asking a series of questions. Each answer **narrows down the possibilities**, much like a flowchart or a game of **20 Questions**. A **Decision Tree** is simply this idea carried out in a formal, data-driven manner.

In a decision tree:

1. It begins with an initial query at the **root** and branches based on the answer.  
2. Each subsequent question (an **internal node**) further splits the data.  
3. Eventually, we reach a **leaf node**, where the tree outputs a class.  

These hierarchical questions mimic how we might reason through a decision ourselves, but they do so using data-driven criteria: the tree systematically asks the *most informative* questions first and splits accordingly.

**Café‐Pastry Example**  

Let’s bring this to life with a lighthearted scenario. Suppose we want to predict which pastry a customer will buy (*Muffin*, *Cake*, or *Cookie*) using two features:

- **Seating** – Indoor vs Outdoor  
- **Drink** – Coffee vs Tea  

Picture a café by the park, where customers who sit outside might be more likely to grab a cookie, while those inside might favor cake or muffins. Our job is to figure out how best to split on these features. Two possible strategies might be:

| Strategy              | Description                                                                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Drink → Seating**   | Root split on **Drink**. Each branch then asks **Seating** (Indoor / Outdoor) to decide the pastry. Both branches still need a second question because drink alone doesn’t yield pure groups. |
| **Seating → Drink**   | Root split on **Seating**. Outdoor customers jump straight to *Cookie* (pure). Indoor customers need one more split on **Drink** (Coffee → Muffin, Tea → Cake). This ordering yields purer groups earlier. |

Both approaches ultimately produce the same leaves (*Muffin*, *Cake*, *Cookie*). However, the *Seating-first* tree requires **one fewer question** on the Outdoor branch, illustrating how decision trees aim to **reduce uncertainty as quickly as possible**.

---

## 2 Entropy and Information Gain  

### 2.1 Entropy $H$  

To quantify “uncertainty,” we use a measure called **entropy**. Formally:

$$
H(S) = -\sum_{c \in \text{classes}} p(c)\,\log_2 p(c),
$$

where $p(c)$ is the proportion of class $c$ in set $S$. High entropy indicates a mixture of classes (uncertainty), whereas low entropy signifies mostly one class (pure node).

For instance:

- *50% Cookies / 50% Muffins → higher entropy.*  
- *100% Muffins → entropy 0.*

### 2.2 Information Gain (IG)  

A decision tree reduces this entropy by splitting on features that best separate the classes. The measure of that separation is **information gain**:

$$
\text{IG}(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v).
$$

This calculation starts with the entropy of the parent node ($H(S)$) and subtracts the weighted entropies of the child nodes. The split that **maximizes** this IG is chosen at each step.

**Revisiting the café example**: Splitting first on **Seating** sends Outdoor customers (100% Cookie) into a perfectly pure node, giving high IG. Splitting on **Drink** first, on the other hand, leaves both children mixed, and the resulting IG is lower—hence the preference for **Seating** as the top-level question.

---

## 3 Handling Numeric vs Categorical Features  

Decision trees handle different feature types in slightly different ways. Here’s a concise overview:

| Feature type           | How the tree splits                                                                                                                                                                                                                                                                       | Practical note                                                                                                                                                                                                          |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Numeric / continuous** | The algorithm sorts the unique values of the feature, then tests threshold candidates at each midpoint, e.g. $(v_i + v_{i+1})/2$. Whichever threshold yields the **highest information gain** is chosen for the binary split $x_j \le \theta$.                                                                                   | Works out-of-the-box for any real-valued column.                                                                                                                                                                        |
| **Categorical**        | A simple approach is **label encoding**: map each unique category to an integer (e.g. Red → 0, Green → 1, Blue → 2). The tree then treats the column like a numeric feature and searches for the best threshold. <br><br>*Alternative:* explicitly branch on each category (“IF color = Red → …”). | In scikit-learn, `LabelEncoder` (or `OrdinalEncoder`) is the quickest way to feed categories into a single decision tree. One-hot encoding is *not* required, because trees handle integer codes naturally. |

> **Key point:** After label encoding, **the same threshold-search routine used for continuous features handles categorical ones as well**.

In a practical setting, imagine a dataset for predicting whether a person will subscribe to a streaming service. You might have a numeric feature like *age* (split using a threshold such as 30.5) and a categorical feature like *favorite genre*, which can be turned into integers (e.g., Action → 0, Comedy → 1, Drama → 2, etc.) and then treated similarly by the algorithm.

---

## 4 Training a Decision Tree: Algorithm & Stopping Rules  

Building a decision tree can be broken into a sequence of steps, from selecting the root question to deciding when to stop:

1. **Root node**: Start with the entire training set.  
2. **Evaluate each feature / threshold**: Compute IG.  
3. **Split** on the feature with **max IG**, creating child nodes.  
4. **Partition** data into these child nodes.  
5. **Recurse** on each child:  
   - Repeat steps 1–4 for its subset of the data.  
6. **Stop** when:  
   - (a) a node is pure (entropy 0), **or**  
   - (b) no features remain, **or**  
   - (c) a **pre-pruning rule** triggers (e.g., `max_depth`, `min_samples_split`, `min_samples_leaf`).  

**Overfitting vs Underfitting**  

While following these steps, you must watch for:

| Scenario         | Symptoms                         | Cause                          | Remedy                                                                                           |
|------------------|----------------------------------|--------------------------------|--------------------------------------------------------------------------------------------------|
| **Overfitting**  | Train accuracy ≈ 100%, validation low | Tree too deep or many tiny leaves   | Increase `min_samples_split` or `min_samples_leaf`, reduce `max_depth`, or prune the tree.        |
| **Underfitting** | Train & validation both low      | Tree too shallow               | Allow deeper splits, reduce `min_samples_split`.                                                 |

In other words, you want the decision tree to balance capturing enough detail (without memorizing noise) and staying general enough to work on new data. **Cross-validation** helps you tune these hyperparameters by testing how well the model generalizes; final performance is then confirmed on a **held-out test set**.

---

## 5 Decision Boundaries & Interpretability  

Once you’ve trained a decision tree, its decision boundaries and interpretability often become big selling points for using it:

- **Axis-aligned splits**: Each rule $x_j \le t$ or $x_j > t$ creates a vertical or horizontal boundary (in 2D) or a hyperplane (in higher dimensions).  
- **Rectangular regions**: The feature space gets chopped up into rectangles (or hyper-rectangles), where each region corresponds to a leaf node predicting a specific class.  
- **Transparent rules**: You can read off the *if–then* path from root to leaf to see exactly why a given decision was made.  
  - *For example, “IF Seating = Outdoor → Cookie; ELSE IF Drink = Coffee → Muffin; ELSE → Cake.”*

Thanks to these straightforward splits, **stakeholders**—whether they’re business partners, doctors, or domain experts—can trace the reasoning behind a classification. This transparency can be critical in high-stakes fields where understanding the rationale behind a prediction is just as important as the prediction’s accuracy.

**Feature (Variable) Importance**  

Another interpretability advantage is that decision trees naturally provide **feature importance** scores. By summing the **information gain** a feature contributes across all the splits in which it appears, you get a measure of how pivotal that feature is overall. Features near the top of the tree often have the largest impact on predictions.

---

## 6 No Need for Feature Scaling  

Unlike some other algorithms, decision trees compare features **individually** with simple threshold checks. They do **not** rely on distances or dot products. As a result, **normalization and standardization** of features are unnecessary for tree-based methods.

If you’re used to scaling features for neural networks, k-Nearest Neighbors, or SVMs, this is one less step to worry about when setting up your data for a decision tree.

---

## 7 Summary  

A decision tree essentially asks a hierarchy of data-driven questions. Each node splits on the feature that most reduces entropy, measured by **information gain**. Along the way, the algorithm checks for stopping criteria or applies pruning to avoid overfitting. Here are the key takeaways:

- **Entropy** measures node impurity; **information gain** captures how much a split reduces that impurity.  
- Numeric features are split on an optimal **threshold**; categorical features can be **label-encoded** so the same threshold strategy applies.  
- Hyper-parameters (`max_depth`, `min_samples_split`, `min_samples_leaf`, etc.) help control a tree’s complexity.  
- **Cross-validation** is crucial for tuning those parameters and avoiding over- or underfitting.  
- Decision trees yield **transparent rules**, axis-aligned decision boundaries, and clear feature-importance scores.  
- They require **no feature scaling** and serve as the foundation for ensemble methods like Random Forests and Gradient Boosted Trees.

Armed with this understanding, we can now deploy decision trees with a good sense of when they shine (interpretability, minimal preprocessing) and where we’ll need caution (their tendency to overfit if not pruned).
