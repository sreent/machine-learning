## **Naïve Bayes — a short story about quick verdicts**

> *“If ten independent experts all whisper the same name, you don’t need a courtroom drama — the verdict is obvious.”*

Imagine you are a detective with a panel of hyper-specialised colleagues.
One knows every brand of glove, another memorises getaway playlists, a third can identify tyre tracks at a glance.
Each colleague inspects **only the clue they understand best** and quietly hands you a likelihood:
“*If Suspect A did it, the red hatchback is 80 % likely; under Suspect B only 5 %.*”
You multiply those private likelihoods, mix in how common each suspect is, and—click—out pops a posterior probability.
No debate, no waiting: the naive-but-dead-fast way to solve a case.
That, in a nutshell, is **Naïve Bayes** (NB).&#x20;

---

### 1  Why the model is called “naïve”

Bayes’ rule says

$$
P(y\mid\mathbf x)\;=\;\frac{P(y)\,P(\mathbf x\mid y)}{\sum_{y'}P(y')P(\mathbf x\mid y')}.
$$

Estimating the full joint likelihood $P(\mathbf x\mid y)$ is hopeless when $\mathbf x$ has thousands of coordinates: you would need elephant-size tables of probabilities.
NB waves a magic wand: **assume every feature is conditionally independent given the class**.
Now the joint factorises into a neat product $\prod_i P(x^{(i)}\mid y)$.
We’ve traded realism for tractability — but, like our expert panel, the math becomes a row of single-dimension look-ups and one big multiplication.
That is the “naïve” move, and it turns an impossible density-estimation task into a statistics-on-a-napkin exercise.&#x20;

---

### 2  Two flavours that cover most menus

* **Multinomial NB** assumes each document is a bag of token *counts*.
  Counting words in spam filtering is the classic showcase: a single appearance of “VIAGRA” may tilt the odds by a thousand, and ten appearances tilt it even further.
  Laplace (add-α) smoothing sprinkles a tiny probability on unseen words so you never multiply by zero.

* **Gaussian NB** replaces counts with a simple Normal curve for every $(\text{feature},\ \text{class})$ pair.
  Need a real-valued diagnostic? Just memorise μ and σ² per class, no gradient descent required.
  The catch is obvious: one outlier glucose reading can drag μ and σ wildly — the model’s speed is purchased with a tender sensitivity to outliers.

Both flavours finish training in a single pass: count, or compute mean/variance — that’s it.&#x20;

---

### 3  Log-probabilities and the underflow monster

NB multiplies hundreds or thousands of probabilities < 1; in raw space you soon hit numbers that underflow to zero.
The remedy is to **take logs**: products turn into sums, and the biggest log-sum wins.
A lecturer’s favourite demo is typing `0.1**1000` into Python and watching the result collapse to 0, then rerunning the same calculation in log-space with no drama.
Every industrial NB implementation works in log-space; the denominator is just a *log-sum-exp* that rescales everything back to finite probabilities.&#x20;

---

### 4  Bias, variance, and why NB wins small-data races

Those independence (or Gaussian) assumptions inject a **lot** of bias, but they also smash variance.
NB can squeeze useful posteriors from a few dozen examples because the number of parameters is tiny — one count or one mean per feature, not a full weight vector that must be learned by optimisation.
Compare the learning curves in the slides: NB hits respectable accuracy after 50 labelled emails; logistic regression needs ten times more to overtake.
So NB is the “high-bias, low-variance” detective: quick to form an opinion, a bit blind to subtle interactions, but hard to overfit.&#x20;

---

### 5  Representations that love (or hate) NB

* **Raw token counts** let Multinomial NB exploit *frequency* as well as presence — perfect for short, repetitive spam where “FREE” appears ten times.
* **TF-IDF weighting** can help when common words are just noise, yet it sometimes *hurts* NB by damping precisely those class-defining words; your SMS-spam study (TF 0.949 vs TF-IDF 0.944 F1) is a textbook example.
* **N-grams** patch NB’s biggest linguistic blind spot: context.  The bigram “not good” or “very bad” becomes its own feature, flipping or amplifying sentiment without abandoning the model.
* **Negation/intensifier tagging** goes one step further: turn every word between “not” and the next punctuation into `NEG_word`, or merge “very good” into `very_good`, so the independence trick still works on enriched tokens.

The guiding rule: give NB features whose meaning really is close to independent; then its naive multiplication becomes shockingly effective.&#x20;

---

### **Take-home intuition**

> *A panel of independent experts, each whispering a likelihood, can reach a solid consensus in the blink of an eye.*

That is Naïve Bayes: multiply a handful of one-dimensional clues, divide by a normalising constant, and you are done.
Its assumptions are heroic, yet on sparse, high-dimensional, low-sample problems NB often **wins the sprint** while heavier models are still tying their shoelaces.
Just remember the fine print: correlated clues and wild outliers can fool the panel — but as long as the evidence really *is* roughly independent, few methods deliver more accuracy per microsecond.

