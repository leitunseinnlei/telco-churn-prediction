# Math background: logistic regression and random forest

This note explains how the two models in this project learn. Each section starts with a plain-English idea, then shows the formula. The examples use numbers from our own Telco churn training data (5,634 customers, 1,495 of whom left).

You don't need to memorise the formulas. The goal is to understand what each part does.

---

## Part 1: Logistic regression

### The idea in one sentence

Logistic regression gives every customer a **score** by adding up points for each feature, then squeezes that score into a **chance of leaving** between 0% and 100%.

### Step 1: Add up a score

Each feature gets a weight, called a **coefficient**. A positive coefficient pushes the customer toward leaving; a negative one pushes toward staying.

$$
z = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n
$$

| Symbol | Meaning | In our project |
|---|---|---|
| $z$ | The total score | Can be any number, like −3.2 or 1.5 |
| $b_0$ | The starting score (intercept) | −1.584 |
| $x_1, x_2, \dots$ | The features of one customer | tenure, Contract_Two year, … |
| $b_1, b_2, \dots$ | The coefficients the model learns | One per feature (30 in total) |

Some of the coefficients our model learned:

| Feature | Coefficient | Meaning |
|---|---|---|
| InternetService_Fiber optic | +1.181 | Fibre customers are more likely to leave |
| PaymentMethod_Electronic check | +0.393 | Electronic-check payers are more likely to leave |
| Contract_Two year | −1.314 | Two-year contracts strongly protect against leaving |
| tenure (scaled) | −1.257 | Longer customers are much less likely to leave |

This is the same idea as linear regression from our earlier lesson: a starting value plus "coefficient × feature".

### Step 2: Turn the score into a chance (the sigmoid function)

A score like −1.7 isn't a chance. The **sigmoid** function squeezes any number into the range 0 to 1:

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

| Score $z$ | Chance $p$ |
|---|---|
| −4 | 2% |
| −2 | 12% |
| 0 | 50% |
| +2 | 88% |
| +4 | 98% |

A very negative score becomes a chance close to 0%, a score of 0 becomes exactly 50%, and a very positive score becomes close to 100%.

**A simplified example from our model.** Imagine a customer whose other features are all at their baseline, so only the intercept and one or two features count:

- Fibre internet, month-to-month contract: $z = -1.584 + 1.181 = -0.403$, so $p = 40\%$
- Fibre internet, two-year contract: $z = -1.584 + 1.181 - 1.314 = -1.717$, so $p = 15\%$

The two-year contract cuts the chance of leaving from 40% to 15%.

### Step 3: Make a decision (the threshold)

The model outputs a chance. To get a yes/no answer, we compare it with a **threshold**, by default 50%:

$$
\hat{y} = \begin{cases} 1 \text{ (will leave)} & \text{if } p \ge 0.5 \\ 0 \text{ (will stay)} & \text{if } p < 0.5 \end{cases}
$$

The business can move this threshold, as we saw in the fraud lesson.

### Step 4: Measure how wrong the model is (the cost function)

To learn, the model needs a way to measure its mistakes. Logistic regression uses **log loss** (also called cross-entropy).

**The idea:** a confident wrong answer is punished much more than an unsure wrong answer.

For one customer with true answer $y$ (1 = left, 0 = stayed) and predicted chance $p$:

$$
\text{loss} = -\big[\, y \log(p) + (1 - y)\log(1 - p) \,\big]
$$

Only one half of the formula is active at a time:

- If the customer **left** ($y = 1$): loss $= -\log(p)$
- If the customer **stayed** ($y = 0$): loss $= -\log(1 - p)$

Examples for a customer who **left**:

| Model's chance of leaving | Loss | Comment |
|---|---|---|
| 90% | 0.105 | Confident and right: tiny penalty |
| 50% | 0.693 | Unsure: medium penalty |
| 10% | 2.303 | Confident and wrong: big penalty |

The **cost function** is the average loss over all $n$ training customers:

$$
J(b) = -\frac{1}{n} \sum_{i=1}^{n} \Big[\, y_i \log(p_i) + (1 - y_i)\log(1 - p_i) \,\Big]
$$

Training means finding the coefficients $b$ that make $J$ as small as possible.

### Step 5: How the model learns (gradient descent)

**The idea:** imagine standing on a foggy hill in the dark, trying to reach the lowest valley. You can't see far, but you can feel which way the ground slopes under your feet. So you take a small step downhill, feel again, and repeat. Each step lowers the cost a little.

1. Start with all coefficients at 0 (every customer gets 50%).
2. Calculate how much the cost would change if each coefficient moved a little. This is the **gradient**, the slope of the hill.
3. Move each coefficient a small step in the downhill direction.
4. Repeat until the cost stops going down.

For logistic regression, the slope for coefficient $b_j$ has a simple form:

$$
\frac{\partial J}{\partial b_j} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)\, x_{ij}
$$

Read it as: "for each customer, take the error (predicted chance minus true answer), multiply it by that customer's feature value, and average." If customers with a high value of feature $j$ are often predicted too high, the slope is positive, so the coefficient goes down.

The update rule, with a step size $\alpha$ (the **learning rate**):

$$
b_j \leftarrow b_j - \alpha \cdot \frac{\partial J}{\partial b_j}
$$

Note: scikit-learn uses a smarter version of this idea (a method called L-BFGS) that chooses its own step sizes. `max_iter=1000` is the maximum number of steps it may take.

### Step 6: Keeping coefficients sensible (regularisation)

If a coefficient becomes very large, the model may be memorising the training data. **Regularisation** adds a small penalty for large coefficients, which keeps the model simpler. scikit-learn uses this by default (called L2 regularisation):

$$
J_{\text{total}}(b) = \underbrace{C \sum_{i=1}^{n} \text{loss}_i}_{\text{fit the data}} + \underbrace{\frac{1}{2} \sum_{j=1}^{m} b_j^2}_{\text{keep coefficients small}}
$$

Here $n$ is the number of customers and $m$ is the number of features (30 in our project).

- A **large C** means "fit the data closely" (risk of overfitting).
- A **small C** means "keep it simple" (risk of underfitting).
- The default is C = 1.0.

### Reading a coefficient as a business statement

Because of the sigmoid, a coefficient changes the **odds** of leaving (odds = chance of leaving ÷ chance of staying) by a factor of $e^{b}$:

- Contract_Two year: $e^{-1.314} = 0.27$, so a two-year contract multiplies the odds of leaving by 0.27, which means **73% lower odds**, all else equal.
- InternetService_Fiber optic: $e^{1.181} = 3.26$, so fibre customers have about **3.3 times the odds** of leaving.

This is why logistic regression is popular in business: each coefficient can be turned into a sentence a manager understands.

### Important parameters in scikit-learn

| Parameter | What it controls | Our setting |
|---|---|---|
| `C` | Strength of regularisation (smaller = simpler model) | 1.0 (default) |
| `max_iter` | Maximum number of learning steps | 1000 |
| `penalty` | Type of regularisation | L2 (default) |
| `class_weight` | Give more importance to the rarer class | None (default) |

---

## Part 2: Decision trees (the building block of a random forest)

A random forest is made of decision trees, so we start with one tree.

### The idea in one sentence

A decision tree repeatedly splits the customers into two groups, each time choosing the yes/no question that makes the groups as **pure** as possible (mostly leavers, or mostly stayers).

### Measuring purity: Gini impurity

**The idea:** pick two customers at random from a group. Gini impurity is the chance they have **different** answers. A pure group (everyone stayed) has Gini 0. A perfectly mixed 50/50 group has Gini 0.5, the worst possible for two classes.

$$
G = 1 - \sum_{k} p_k^2 = 1 - p_{\text{left}}^2 - p_{\text{stayed}}^2
$$

### Worked example with our data

**Before any split:** 5,634 training customers, 26.5% left.

$$
G = 1 - 0.265^2 - 0.735^2 = 0.390
$$

**Try the question "Is the contract month-to-month?"**

| Group | Customers | Left | Churn rate | Gini |
|---|---|---|---|---|
| Month-to-month | 3,102 | 1,326 | 42.7% | 0.489 |
| One or two years | 2,532 | 169 | 6.7% | 0.125 |

The month-to-month group is more mixed than before, but the other group is now very pure. To judge the split, we take the **weighted average** of the two groups' Gini, weighted by group size:

$$
G_{\text{after}} = \frac{3102}{5634} \times 0.489 + \frac{2532}{5634} \times 0.125 = 0.325
$$

$$
\text{Gini decrease} = 0.390 - 0.325 = 0.065
$$

### How a tree learns

1. Try every possible question on every feature (for example "tenure ≤ 12?", "fibre internet?").
2. Pick the question with the **biggest Gini decrease**.
3. Split the customers into two groups and repeat steps 1–2 inside each group.
4. Stop when a rule says so, for example when a group would have fewer than `min_samples_leaf` customers.

The final groups are called **leaves**. A new customer follows the questions down to a leaf, and the tree's chance of leaving is the share of leavers in that leaf.

There is no single cost function for the whole tree, as in logistic regression. Instead, each split greedily minimises Gini impurity at that moment.

### The weakness

A tree that keeps splitting can end up with leaves of one or two customers. It then memorises the training data (**overfitting**). That's why our default random forest scored 99.8% on the training data but less on the test data.

---

## Part 3: Random forest

### The idea in one sentence

Train many different decision trees and let them **vote**; their individual mistakes cancel out.

### How the trees are made different

If all trees saw the same data, they would all make the same mistakes. A random forest uses two sources of randomness:

1. **Random rows (bootstrap sampling).** Each tree gets its own sample of the training customers, drawn **with replacement**. Some customers appear twice, others not at all. On average, each tree sees about 63% of the different customers.
2. **Random columns.** At each split, a tree may only choose from a random subset of the features. For classification, scikit-learn uses $\sqrt{\text{number of features}}$ by default: $\sqrt{30} \approx 5$ features per split.

The second trick matters a lot. Without it, every tree would start with the same strongest question (for example, contract type), and the trees would look alike.

### How the forest predicts

Each tree $t$ gives a chance of leaving $p_t$. The forest averages them over all $T$ trees:

$$
p_{\text{forest}} = \frac{1}{T} \sum_{t=1}^{T} p_t
$$

Then the same 50% threshold turns the chance into a yes/no answer.

### Why averaging helps

**The idea:** one tree is like one person guessing the number of sweets in a jar. Their guess may be far off. The average of 300 guesses is usually much closer, because the too-high and too-low guesses cancel out.

In maths: if each tree's prediction has variance $\sigma^2$ (how much it jumps around), and two trees are correlated by $\rho$ (how similar their mistakes are), the variance of the average of $T$ trees is:

$$
\text{Var}(p_{\text{forest}}) = \rho\,\sigma^2 + \frac{1 - \rho}{T}\,\sigma^2
$$

- More trees ($T$ larger) make the second part shrink toward 0.
- Less similar trees ($\rho$ smaller) make the first part smaller.

This is exactly why the forest uses random rows and random columns: they make the trees less similar.

### Feature importance

A random forest can tell us which features mattered most. For each feature, it adds up the **Gini decrease** from every split that used that feature, across all trees, and then scales the totals so they add up to 1. A feature used often, for big Gini decreases, gets a high importance.

### Important parameters in scikit-learn

| Parameter | What it controls | Our setting |
|---|---|---|
| `n_estimators` | Number of trees | 300 |
| `min_samples_leaf` | Minimum customers in a leaf (higher = simpler trees) | 10 |
| `max_depth` | Maximum number of questions in a row | None (no limit) |
| `max_features` | Features to choose from at each split | "sqrt" (default), about 5 |
| `random_state` | Makes the randomness repeatable | 42 |

---

## Part 4: The two models side by side

| | Logistic regression | Random forest |
|---|---|---|
| Basic idea | Weighted score, squeezed into a chance | Many trees vote |
| Shape of the rule | A straight line (one weight per feature) | Many yes/no splits, can capture combinations |
| What it minimises | Log loss for the whole model | Gini impurity, one split at a time |
| How it learns | Gradient descent: small steps downhill | Greedy splitting, repeated in 300 trees |
| Needs scaled numbers? | Yes, so all features get a fair start | No, splits don't depend on size |
| Explaining it | Easy: coefficients and odds | Harder: use feature importance |
| Main risk | Misses combinations of features | Overfits if trees grow too deep |

## Quick glossary

| Word | Meaning |
|---|---|
| Coefficient | The weight a feature gets in the score |
| Sigmoid | The function that turns any score into a chance between 0 and 1 |
| Log loss | The cost function that punishes confident wrong answers |
| Gradient | The slope: which way, and how steeply, the cost changes |
| Learning rate | The size of each step downhill |
| Regularisation | A penalty that keeps the model simple |
| Gini impurity | How mixed a group is (0 = pure) |
| Bootstrap sample | A random sample drawn with replacement |
| Variance | How much a prediction would change with different training data |
