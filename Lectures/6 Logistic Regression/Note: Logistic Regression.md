## **Logistic Regression**

**Predicting Customer Churn: Focusing Resources with Precision**

Imagine you're working for an automotive brand and need to address customer retention. Research shows that car owners tend to leave dealer services after around three years. With a limited budget, your goal is to focus retention efforts on those most likely to churn. By calculating a **churn probability score** for each owner, you can prioritize high-risk individuals, maximizing the impact of your retention budget. Logistic regression helps us calculate this probability score, making it a powerful tool for tasks where we need confidence levels to guide decisions.

**What Makes Logistic Regression Different?**

Logistic regression is a classification technique designed to estimate the probability of a binary outcome—such as whether a customer will churn (1) or stay (0). Unlike linear regression, which predicts a continuous outcome, logistic regression provides a **probability score** between 0 and 1, making it ideal for binary classification tasks. Linear regression would not work well here because it could give probabilities outside this range, leading to impractical interpretations.

**Mapping Predictions to Probabilities with the Sigmoid Function**

To convert a linear combination of input features (like vehicle age and service frequency) into a probability, logistic regression uses the **sigmoid function**. This function is central to logistic regression as it “squeezes” values into a range between 0 and 1, making them interpretable as probabilities. The sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where $ z = w \cdot x + b $:
- $w$ represents the weights for each feature,
- $x$ represents the input features (e.g., vehicle age, service frequency),
- $b$ is the bias term.

This transformation allows us to interpret high $z$ values as probabilities close to 1 (indicating likely churn) and low $z$ values as probabilities close to 0.

**Finding Optimal Weights through Maximum Likelihood Estimation**

To determine the best weights $w$, logistic regression uses **Maximum Likelihood Estimation (MLE)**. This method seeks the parameters that make the observed data most probable under the model. The goal is to maximize the **log-likelihood function**, which expresses how well our model explains the actual data. For binary outcomes, the log-likelihood function is:

$$
\log L(w) = \sum_{i=1}^{N} \left( y^{(i)} \log P(y=1|x^{(i)}) + (1 - y^{(i)}) \log (1 - P(y=1|x^{(i)})) \right)
$$

where $y^{(i)}$ is the actual label (1 for churn, 0 for stay), and $P(y=1|x^{(i)})$ is the predicted probability. To find the weights that maximize this function, we use **gradient descent**. This iterative method adjusts the weights in small steps to increase the likelihood, moving closer to a model that accurately reflects the data.

**Evaluating with Precision, Recall, and Cross-Validation**

Logistic regression’s effectiveness can be measured using metrics like **precision** and **recall**, especially in cases where the data is imbalanced or the cost of false positives and false negatives varies:
- **Precision** measures how many of our predicted positives (high churn probability) are actual churn cases. This is crucial if false positives are costly.
- **Recall** measures how many actual churn cases we correctly identified, which is important if missing true positives (actual churners) is costly.

**K-Fold Cross Validation** is commonly used to assess how well our model generalizes to new data. It involves splitting the data into $K$ parts, training on $K-1$ parts, and validating on the remaining part. This process repeats $K$ times, with each part serving as the validation set once. Cross-validation ensures our model is robust and minimizes overfitting, making it especially valuable when data is limited.

**Summary**

Logistic regression is a powerful tool for binary classification, helping predict the probability of an outcome, such as customer churn. By mapping a linear combination of input features to probabilities through the **sigmoid function**, it provides interpretable scores between 0 and 1. Logistic regression uses **Maximum Likelihood Estimation (MLE)** to find the best-fitting model parameters, often optimized through **gradient descent**.

In practice, logistic regression models are evaluated with metrics like **precision** and **recall** to handle imbalanced data effectively. **K-Fold Cross Validation** further ensures that the model generalizes well, making it reliable even when data is limited. This combination of probabilistic predictions, robust evaluation, and interpretability makes logistic regression a valuable choice for applications like customer retention, fraud detection, and medical diagnostics.

