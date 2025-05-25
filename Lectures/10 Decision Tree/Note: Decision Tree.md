# Decision Tree Classification

## 1 Introduction  

Imagine you’re trying to diagnose a problem or make a decision by asking a series of questions. Each answer **narrows down the possibilities**, much like a flowchart or a game of **20 Questions**. A **Decision Tree** is this process in a formal, data-driven way.  

* It starts with an initial query at the **root** and branches based on the answer.  
* Each subsequent question (an **internal node**) further splits the data.  
* Eventually we reach a **leaf node**, where the tree outputs a class.  

A decision tree thus **systematically splits data** into sub-groups by asking the *most informative questions first*, homing in on a prediction.


### Café‐Pastry Example  

Suppose we want to predict which pastry a customer will buy (*Muffin*, *Cake*, or *Cookie*) using two features:

* **Seating** – Indoor vs Outdoor  
* **Drink** – Coffee vs Tea  

Two possible strategies:

| Strategy | Description |
|----------|-------------|
| **Drink → Seating** | Root split on **Drink**. Each branch then asks **Seating** (Indoor / Outdoor) to decide the pastry. Both branches still need a second question because drink alone doesn’t yield pure groups. |
| **Seating → Drink** | Root split on **Seating**. Outdoor customers jump straight to *Cookie* (pure). Indoor customers need one more split on **Drink** (Coffee → Muffin, Tea → Cake). This ordering yields purer groups earlier. |

Both trees reach the same three leaves, but the *Seating-first* tree does it with **one less question** on the Outdoor branch. This shows the tree’s goal: **ask the question that reduces uncertainty the most**. Next we formalize “uncertainty” with entropy and “reduction” with information gain.


## 2 Entropy and Information Gain  

### 2.1 Entropy ($H$)

$H(S) = -\sum_{c \in \text{classes}} p(c)\,\log_2 p(c)$

* $p(c)$ = proportion of class $c$ in set $S$.  
* **High $H$** → classes evenly mixed (uncertain).  
* **Low $H$** → node is pure (all one class).  

Example:  
*50 % Cookies / 50 % Muffins → higher entropy.* *100 % Muffins → entropy 0.*


### 2.2 Information Gain (IG)

When we split node $S$ on feature $A$:

$\text{IG}(S, A) = H(S) -\sum_{v \in \text{values}(A)}\frac{|S_v|}{|S|}H(S_v)$

*Parent entropy minus weighted child entropies.* The **split with the highest IG** is chosen. *(Gini impurity is an alternative, but we focus on entropy as in the lecture.)*

#### Café example revisited  
* **Seating-first** split creates an *Outdoor* branch that’s 100 % Cookie → entropy 0 → large IG.  
* **Drink-first** leaves both children mixed → smaller IG.  

Hence the algorithm prefers **Seating** as the root split.


## 3 Handling Numeric vs Categorical Features  

| Feature type | How the tree splits | Practical note |
|--------------|--------------------|----------------|
| **Numeric / continuous** | The algorithm sorts the unique values of the feature, then tests threshold candidates at each midpoint, e.g. $(v_i+v_{i+1})/2$. The threshold that yields the **highest information gain** becomes the binary split $x_j \le \theta$? | Works out-of-the-box for any real-valued column. |
| **Categorical** | A simple approach is **label encoding**: map each unique category to an integer (e.g. Red → 0, Green → 1, Blue → 2). The tree then treats the column like a numeric feature and still searches for the best threshold. <br><br>*Alternative:* explicitly branch on each category (“IF color = Red → …”). | In scikit-learn, `LabelEncoder` (or `OrdinalEncoder`) is the quickest way to feed categories into a single decision tree. One-hot encoding is *not* required, because trees handle integer codes naturally. |

> **Key point:** After label encoding, **the same threshold-search routine used for continuous features handles categorical ones as well**.


## 4 Training a Decision Tree: Algorithm & Stopping Rules  

1. **Root node:** start with entire training set.  
2. **Evaluate each feature / threshold:** compute IG.  
3. **Split** on the feature with **max IG** and create child nodes.  
4. **Partition** data to children.  
5. **Recurse** on each child:  
   * repeat steps 1–4 on its subset.  
6. **Stop** when  
   * (a) node is pure (entropy 0), **or**  
   * (b) no features remain, **or**  
   * (c) a **pre-pruning rule** triggers:  
     * **`max_depth`** – limit tree height.  
     * **`min_samples_split`** – require ≥ *N* samples to split.  
     * **`min_samples_leaf`** – require ≥ *N* samples in each leaf.


### Overfitting vs Underfitting  

| Scenario | Symptoms | Cause | Remedy |
|----------|----------|-------|--------|
| **Overfitting** | Train accuracy ≈ 100 %, validation low | Tree too deep / many tiny leaves | Increase `min_samples_split`, `min_samples_leaf`, reduce `max_depth`, or prune. |
| **Underfitting** | Train & validation both low | Tree too shallow | Allow deeper splits, lower `min_samples_split`. |

Use **cross-validation** to sweep hyper-parameters, picking the model that maximizes validation accuracy (best bias–variance balance).  
Final performance is reported on a **held-out test set**.


## 5 Decision Boundaries & Interpretability  

* Splits are **axis-aligned**: each rule $x_j \le t$? draws a vertical/horizontal line (or hyperplane) in feature space.  
* The space is carved into **rectangular regions**; each region predicts a constant class.  
* **Model explanation:** list the IF–THEN rules along the path from root to leaf.  
  *Example: “IF Seating = Outdoor → Cookie; ELSE IF Drink = Coffee → Muffin; ELSE → Cake.”*

### Feature (Variable) Importance  

Sum the **information gain** a feature contributes across all its splits ⇒ importance score. Features used near the top with large IG dominate. Most libraries (e.g., scikit-learn) output normalized importance values.


## 6 No Need for Feature Scaling  

Decision trees compare features **individually** with simple threshold tests. They do **not** rely on distances, dot products, or weight magnitudes. Therefore **normalization / standardization is unnecessary** for tree models.


## 7 Summary  

* A decision tree asks a hierarchy of questions, **greedily choosing the split that maximizes information gain** at each node.  
* **Entropy** measures node impurity; **information gain** quantifies how much a split reduces that impurity.
* Numeric features are split on optimal **thresholds**; categorical features can be **label-encoded** so the same threshold search applies.  
* Hyper-parameters such as `max_depth`, `min_samples_split`, and `min_samples_leaf` control complexity, preventing over- and underfitting.  
* **Cross-validation** tunes these parameters, safeguarding the model’s generalization.  
* Trees yield **transparent rules**, axis-aligned decision boundaries, and clear feature-importance scores.  
* They require **no feature scaling** and form the building blocks of powerful ensembles like Random Forests and Gradient Boosted Trees.

With a solid grasp of decision trees, we’re ready to deploy them responsibly and appreciate both their strengths (interpretability) and limitations (tendency to overfit without pruning).
