# Math Study Guide: Logistic Regression and Random Forest

A beginner-friendly guide to how the two models in this project learn and make predictions. Every idea is explained in the same order:

1. **The idea** in plain words, with an everyday example
2. **The formula**, with every symbol explained
3. **A worked example**, using real numbers from this project's data
4. **The algorithm**: the exact steps the computer follows

You don't need to memorise the formulas. The goal is to understand what each one does. All numbers come from the training data (5,634 customers) and test data (1,409 customers) in this project, and were checked with Python.

For the more formal version, see [`math_background.md`](math_background.md).

---

## Contents

**Part A: Before you start**
1. [Maths you need first](#1-maths-you-need-first)
2. [Clues, answers and the data the models see](#2-clues-answers-and-the-data-the-models-see)

**Part B: Logistic regression**

3. [Logistic regression in 9 steps (start here)](#3-logistic-regression-in-9-steps-start-here)
4. [Step 1: Adding up the points](#4-step-1-adding-up-the-points)
5. [Step 2: Turning points into a chance (the sigmoid)](#5-step-2-turning-points-into-a-chance-the-sigmoid)
6. [Step 3: Deciding yes or no (the threshold)](#6-step-3-deciding-yes-or-no-the-threshold)
7. [Where the points come from](#7-where-the-points-come-from)
8. [Measuring mistakes: the penalty (log loss)](#8-measuring-mistakes-the-penalty-log-loss)
9. [Learning the points: training round by round](#9-learning-the-points-training-round-by-round)
10. [Understanding the slope](#10-understanding-the-slope)
11. [A shortcut for one clue: odds](#11-a-shortcut-for-one-clue-odds)
12. [Extras: regularisation and odds in business language](#12-extras-regularisation-and-odds-in-business-language)
13. [The logistic regression algorithm](#13-the-logistic-regression-algorithm)

**Part C: Random forest**

14. [The big idea: many trees voting](#14-the-big-idea-many-trees-voting)
15. [One decision tree: a game of questions](#15-one-decision-tree-a-game-of-questions)
16. [Measuring how mixed a group is (Gini impurity)](#16-measuring-how-mixed-a-group-is-gini-impurity)
17. [Choosing the best question, with real data](#17-choosing-the-best-question-with-real-data)
18. [Building the whole tree](#18-building-the-whole-tree)
19. [The problem with one tree: overfitting](#19-the-problem-with-one-tree-overfitting)
20. [Making 300 different trees](#20-making-300-different-trees)
21. [Combining the trees: voting and averaging](#21-combining-the-trees-voting-and-averaging)
22. [Why averaging works](#22-why-averaging-works)
23. [Feature importance](#23-feature-importance)
24. [The decision tree and random forest algorithms](#24-the-decision-tree-and-random-forest-algorithms)

**Part D: Putting it together**

25. [Why scaling matters for one model but not the other](#25-why-scaling-matters-for-one-model-but-not-the-other)
26. [The math behind the evaluation metrics](#26-the-math-behind-the-evaluation-metrics)
27. [Formula sheet](#27-formula-sheet)
28. [Check yourself](#28-check-yourself)
29. [Glossary](#29-glossary)

---

# Part A: Before you start

## 1. Maths you need first

Only five small ideas.

### 1.1 Multiplying by 0 or 1 is a switch

Many columns in this project are 0 (no) or 1 (yes).

```
1.181 × 1 = 1.181   → the points are switched ON
1.181 × 0 = 0       → the points are switched OFF
```

### 1.2 Squaring means multiplying a number by itself

```
3²     = 3 × 3       = 9
0.5²   = 0.5 × 0.5   = 0.25
(−2)²  = (−2) × (−2) = 4    ← always positive
```

### 1.3 The special number e

`e ≈ 2.718` is a special number, like `π ≈ 3.14`. It is written as `e^z`, meaning "e to the power of z".

| z  | e^z (about) |
|----|-------------|
| −2 | 0.14        |
| 0  | 1           |
| 1  | 2.72        |
| 2  | 7.39        |

### 1.4 ln is the opposite of e

`ln` (the natural logarithm) answers: "e to the power of **what** gives this number?" The one thing to remember: **the closer a number is to 0, the more negative its ln.**

| Number | ln (about) |
|--------|------------|
| 1      | 0          |
| 0.9    | −0.11      |
| 0.5    | −0.69      |
| 0.1    | −2.30      |

### 1.5 Using a calculator

On an iPhone, turn the phone sideways for the scientific calculator. On a Mac, open Calculator and press **Cmd + 2**.

| Button  | What it does                                         |
|---------|------------------------------------------------------|
| `+/−`   | Changes a number to negative or positive             |
| `eˣ`    | e to the power of the number on the screen           |
| `ln`    | Natural log of the number on the screen              |
| `1/x`   | Divides 1 by the number on the screen                |

⚠️ Use `ln`, not `log`. They are different buttons.

In Python, `math.exp(z)` is the `eˣ` button and `math.log(x)` is the `ln` button.

---

## 2. Clues, answers and the data the models see

### 2.1 Clues and answers

Each row of the data is one customer. Each column plays one of two roles:

| Role            | Meaning                              | Symbol | In this project                    |
|-----------------|--------------------------------------|--------|------------------------------------|
| **Clue** (feature) | Information the model learns from | `x`    | tenure, contract, internet type, … |
| **Answer** (target) | What the model must predict      | `y`    | Churn: 1 = left, 0 = stayed        |

### 2.2 How the clues were selected

A clue is simply a column. The models were given **every column except two**:

| Column       | Used as a clue? | Why                                                      |
|--------------|-----------------|----------------------------------------------------------|
| `customerID` | No              | Just a name tag; it can't predict anything               |
| `Churn`      | No              | It is the answer; the model must not see it as a clue    |
| All others   | Yes             | Information about the customer                           |

After one-hot encoding, those columns became **30 clue columns**. We decide which columns the model may look at; **the model decides how much each one matters**.

### 2.3 From real customers to numbers

Here are real customers from the data:

| customerID | InternetService | Churn |
|------------|-----------------|-------|
| 7590-VHVEG | DSL             | No    |
| 3668-QPYBK | DSL             | Yes   |
| 9237-HQITU | Fiber optic     | Yes   |
| 9305-CDSKC | Fiber optic     | Yes   |
| 1452-KIOVK | Fiber optic     | No    |

After cleaning, `InternetService` becomes a 0/1 column called `InternetService_Fiber optic`, and `Churn` becomes 0/1:

| customerID | Fibre? (x) | Left? (y) |
|------------|------------|-----------|
| 7590-VHVEG | 0          | 0         |
| 3668-QPYBK | 0          | 1         |
| 9237-HQITU | 1          | 1         |
| 9305-CDSKC | 1          | 1         |
| 1452-KIOVK | 1          | 0         |

The columns are the clue and the answer; the 1s and 0s are each customer's values.

### 2.4 Why the answers are already in the data

The data is **past** data. The company already recorded who left. The model needs these known answers to **learn** (training) and to **check itself** (testing), so that later it can predict for **new** customers, whose answers are not known yet.

| | Past customers | New customers |
|---|---|---|
| Do we know if they left? | Yes, it was recorded | No, it hasn't happened yet |
| What we do | The model learns from them | The model predicts for them |

---

# Part B: Logistic regression

## 3. Logistic regression in 9 steps (start here)

> Give each customer **points** for each clue, turn the points into a **chance** of leaving, and **learn** the best points from past customers.

This section tells the whole story in 9 short steps, using this project's real data. It is written so it can be presented directly: each step has the calculation, shown in full, and a **Say** line you can use when presenting. Sections 4 to 13 explain each idea in more detail.

All calculations can be followed with a calculator (see section 1.5). Small differences in the last decimal place come from rounding.

### Before you start: five small calculation rules

**Rule 1: Plus and minus when multiplying**

```
plus  × plus  = plus        2 × 3   = 6
minus × minus = plus       (−2) × (−3) = 6
plus  × minus = minus       2 × (−3) = −6
minus × plus  = minus      (−2) × 3   = −6
```

**Rule 2: Subtracting a minus number is the same as adding**

```
5 − (−2) = 5 + 2 = 7
−0.0402 − (−0.0136) = −0.0402 + 0.0136 = −0.0266
```

**Rule 3: Any number multiplied by 0 is 0, and by 1 stays the same**

```
1.473 × 0 = 0        1.473 × 1 = 1.473
```

**Rule 4: Turning points (z) into a chance (p): four calculator steps**

```
1. Type z and press +/−          (this gives −z)
2. Press eˣ                      (this gives e^(−z))
3. Press + 1 =                   (this gives 1 + e^(−z))
4. Press 1/x                     (this gives p)
```

Example with z = −0.319: type `0.319` (because −z = +0.319) → `eˣ` → shows 1.376 → `+ 1 =` → shows 2.376 → `1/x` → shows **0.421**.

**Rule 5: Working out a penalty: three calculator steps**

```
1. Type the chance given to what really happened
2. Press ln                       (gives a minus number)
3. Press +/−                      (makes it positive)
```

Example: chance 0.5 → `ln` → shows −0.6931 → `+/−` → shows **0.6931**.

| Step | What happens                              | Detail in section |
|------|-------------------------------------------|-------------------|
| 1    | Ask the question                          | 2                 |
| 2    | Turn the customers into numbers           | 2                 |
| 3    | Write the model's recipe                  | 4, 5, 6           |
| 4    | Start knowing nothing                     | 9                 |
| 5    | Measure how wrong the guesses are         | 8                 |
| 6    | Decide which way to adjust, and adjust    | 9, 10             |
| 7    | Repeat until the guesses are right        | 9, 11             |
| 8    | Use the trained model                     | 5, 6              |
| 9    | From 1 clue to all 30 clues               | 7, 13             |

To keep steps 1 to 8 easy, the model uses **only one clue**: fibre internet. Step 9 shows the real model with all 30 clues.

### Step 1: Ask the question

> **Will this customer leave? Yes or no.**

The model learns from 5,634 past customers whose answers are already known.

**Say:** *"Logistic regression answers a yes/no question: will this customer leave? It learns from 5,634 past customers whose answers we already know."*

### Step 2: Turn the customers into numbers

Each customer gets two numbers:

| customerID | InternetService | Churn | Fibre? (x) | Left? (y) |
|------------|-----------------|-------|------------|-----------|
| 7590-VHVEG | DSL             | No    | 0          | 0         |
| 3668-QPYBK | DSL             | Yes   | 0          | 1         |
| 9237-HQITU | Fiber optic     | Yes   | 1          | 1         |
| 1452-KIOVK | Fiber optic     | No    | 1          | 0         |

Counting all training customers:

| Group             | Customers | Left  | Stayed | Churn rate |
|-------------------|-----------|-------|--------|------------|
| Fibre (x = 1)     | 2,483     | 1,045 | 1,438  | **42.1%**  |
| No fibre (x = 0)  | 3,151     | 450   | 2,701  | **14.3%**  |

**How these numbers are calculated:**

```
Fibre:     stayed      = 2,483 − 1,045 = 1,438
           churn rate  = left ÷ customers = 1,045 ÷ 2,483 = 0.421  →  42.1%

No fibre:  stayed      = 3,151 − 450 = 2,701
           churn rate  = 450 ÷ 3,151 = 0.143  →  14.3%

All:       customers   = 2,483 + 3,151 = 5,634
           left        = 1,045 + 450   = 1,495
           churn rate  = 1,495 ÷ 5,634 = 0.265  →  26.5%
```

The model does **not** see this summary table. It only sees each customer's x and y. Its job is to discover these churn rates by itself.

**Say:** *"Each customer becomes two numbers: fibre or not, and left or not. Fibre customers really left 42.1% of the time, and the others 14.3%. The model has to discover this on its own."*

### Step 3: Write the model's recipe

The recipe has two parts:

```
Part A. Points:   z = b₀ + b₁ × x
Part B. Chance:   p = 1 / (1 + e^(−z))
```

| Symbol | Meaning                                          |
|--------|--------------------------------------------------|
| `x`    | The customer's clue: 1 = fibre, 0 = no fibre     |
| `b₀`   | Starting points, which everyone gets             |
| `b₁`   | Fibre points, which only fibre customers get     |
| `z`    | The customer's total points                      |
| `p`    | The chance of leaving, between 0 and 1           |

The model only has to learn **two numbers**: b₀ and b₁.

**How the recipe treats the two groups** (rule 3):

```
Fibre customer (x = 1):     z = b₀ + b₁ × 1 = b₀ + b₁     ← gets both numbers
No-fibre customer (x = 0):  z = b₀ + b₁ × 0 = b₀          ← gets only the starting points
```

So the fibre points b₁ are the **extra** points a fibre customer gets.

**Say:** *"The recipe is simple: add up the points, then turn the points into a chance. The model only needs to learn two numbers: the starting points and the fibre points."*

### Step 4: Start knowing nothing

At the start, both numbers are **0**. So every customer gets the same points and the same chance:

```
Points:   z = b₀ + b₁ × x = 0 + 0 × x = 0

Chance:   1.  −z = 0
          2.  e^0 = 1
          3.  1 + 1 = 2
          4.  p = 1 ÷ 2 = 0.5   →  50%
```

Every customer gets a **50%** chance. The model can't tell anyone apart yet.

**Say:** *"At the start, the model knows nothing. Both numbers are zero, so it guesses 50% for every customer."*

### Step 5: Measure how wrong the guesses are (round 1)

**5a. Compare the guesses with reality.**

| Group     | Guess | Reality | The guess is…       |
|-----------|-------|---------|---------------------|
| Fibre     | 50%   | 42.1%   | too high            |
| No fibre  | 50%   | 14.3%   | **much** too high   |

**5b. Give each customer a penalty.** The rule (section 8): take the chance the model gave to **what really happened**, then calculate −ln of it.

```
If the customer LEFT:     penalty = −ln(p)
If the customer STAYED:   penalty = −ln(1 − p)
```

There are only four types of customers, so we calculate one penalty per type, then multiply by how many customers there are:

| Type              | Customers | Chance given to the real outcome | Penalty each      | Total            |
|-------------------|-----------|----------------------------------|-------------------|------------------|
| Fibre, left       | 1,045     | p = 0.5                          | −ln(0.5) = 0.6931 | 1,045 × 0.6931 = 724.3  |
| Fibre, stayed     | 1,438     | 1 − p = 0.5                      | −ln(0.5) = 0.6931 | 1,438 × 0.6931 = 996.7  |
| No fibre, left    | 450       | p = 0.5                          | −ln(0.5) = 0.6931 | 450 × 0.6931 = 311.9    |
| No fibre, stayed  | 2,701     | 1 − p = 0.5                      | −ln(0.5) = 0.6931 | 2,701 × 0.6931 = 1,872.2 |
| **All**           | **5,634** |                                  |                   | **3,905.2**      |

**5c. Take the average.**

```
Average penalty = 3,905.2 ÷ 5,634 = 0.693
```

**Say:** *"Then it checks its guesses against the real answers. Both groups were guessed too high, especially the no-fibre group. The model measures this with a penalty: 0.693 for every customer, so the average penalty starts at 0.693. Training will try to make this number smaller."*

### Step 6: Decide which way to adjust, and adjust (round 1)

The penalty says **how wrong** the model is. Now the model needs to know **which way** to change each number. For that, it uses each customer's **error**, then a **slope** for each number.

**6a. Calculate each customer's error.**

```
error = guess − real answer        (real answer: 1 = left, 0 = stayed)
```

| Type              | Customers | Guess | Real | Error each        | Customers × error         |
|-------------------|-----------|-------|------|-------------------|---------------------------|
| Fibre, left       | 1,045     | 0.5   | 1    | 0.5 − 1 = −0.5    | 1,045 × (−0.5) = −522.5   |
| Fibre, stayed     | 1,438     | 0.5   | 0    | 0.5 − 0 = +0.5    | 1,438 × (+0.5) = +719.0   |
| No fibre, left    | 450       | 0.5   | 1    | −0.5              | 450 × (−0.5) = −225.0     |
| No fibre, stayed  | 2,701     | 0.5   | 0    | +0.5              | 2,701 × (+0.5) = +1,350.5 |

A negative error means "guessed too low"; a positive error means "guessed too high".

**6b. The slope for the starting points (b₀): use ALL customers.** Everyone gets the starting points, so everyone's error counts.

```
Sum of errors = −522.5 + 719.0 − 225.0 + 1,350.5 = 1,322.0
Slope for b₀  = 1,322.0 ÷ 5,634 = +0.2346
```

**6c. The slope for the fibre points (b₁): use only FIBRE customers.** Only fibre customers get the fibre points. For everyone else x = 0, and error × 0 = 0.

```
Sum of fibre errors = −522.5 + 719.0 = 196.5
Slope for b₁        = 196.5 ÷ 5,634 = +0.0349
```

**6d. Read the slopes.**

| Number | Slope    | Meaning                                         | So it goes… |
|--------|----------|-------------------------------------------------|-------------|
| b₀     | +0.2346  | guesses too high overall (50% vs real 26.5%)    | **down**    |
| b₁     | +0.0349  | fibre guesses too high too (50% vs real 42.1%)  | **down**    |

**6e. Adjust.** The rule is: new number = old number − slope.

```
b₀ = 0 − 0.2346 = −0.2346
b₁ = 0 − 0.0349 = −0.0349
```

Like a shower that is too hot: you turn it down a little. A big slope means a big turn; a small slope means a small turn.

**How the penalty and the slope fit together:** the penalty is how high up a hill you are; the slope tells you which way is downhill. The error formula above is exactly the slope of the penalty, so every adjustment makes the penalty a little smaller.

**Say:** *"For each number, the model adds up the errors to work out whether it should go up or down. This is the slope. The starting points use every customer's error; the fibre points only use the fibre customers' errors. Both slopes are positive, meaning 'too high', so both numbers go down a little. That's one round of training."*

### Step 7: Repeat until the guesses are right

The model repeats steps 5 and 6 with the new numbers, round after round.

**Round 2, with every calculation**

*Guess (b₀ = −0.2346, b₁ = −0.0349):*

```
Fibre (x = 1):     z = −0.2346 + (−0.0349) × 1 = −0.2695
                   −z = 0.2695  →  e^0.2695 = 1.3093  →  1 + 1.3093 = 2.3093  →  p = 1 ÷ 2.3093 = 0.4330

No fibre (x = 0):  z = −0.2346 + (−0.0349) × 0 = −0.2346
                   −z = 0.2346  →  e^0.2346 = 1.2645  →  1 + 1.2645 = 2.2645  →  p = 1 ÷ 2.2645 = 0.4416
```

*Penalty:*

| Type              | Customers | Chance given to real outcome | Penalty each           | Total    |
|-------------------|-----------|------------------------------|------------------------|----------|
| Fibre, left       | 1,045     | 0.4330                       | −ln(0.4330) = 0.8370   | 874.6    |
| Fibre, stayed     | 1,438     | 1 − 0.4330 = 0.5670          | −ln(0.5670) = 0.5674   | 816.0    |
| No fibre, left    | 450       | 0.4416                       | −ln(0.4416) = 0.8173   | 367.8    |
| No fibre, stayed  | 2,701     | 1 − 0.4416 = 0.5584          | −ln(0.5584) = 0.5827   | 1,573.8  |
| **All**           | **5,634** |                              |                        | **3,632.3** |

```
Average penalty = 3,632.3 ÷ 5,634 = 0.645      ← down from 0.693 ✓
```

*Errors and slopes:*

| Type              | Customers | Error each               | Customers × error |
|-------------------|-----------|--------------------------|-------------------|
| Fibre, left       | 1,045     | 0.4330 − 1 = −0.5670     | −592.5            |
| Fibre, stayed     | 1,438     | 0.4330 − 0 = +0.4330     | +622.7            |
| No fibre, left    | 450       | 0.4416 − 1 = −0.5584     | −251.3            |
| No fibre, stayed  | 2,701     | 0.4416 − 0 = +0.4416     | +1,192.8          |

```
Slope for b₀ = (−592.5 + 622.7 − 251.3 + 1,192.8) ÷ 5,634 = 971.7 ÷ 5,634 = +0.1725   → still too high
Slope for b₁ = (−592.5 + 622.7) ÷ 5,634              = 30.2 ÷ 5,634  = +0.0054   → a little too high
```

*Adjust:*

```
b₀ = −0.2346 − 0.1725 = −0.4071
b₁ = −0.0349 − 0.0054 = −0.0402
```

**Round 3: the fibre slope turns around**

With b₀ = −0.4071 and b₁ = −0.0402, the fibre guess becomes **39.0%**, now **below** the real 42.1%:

```
Fibre errors:  1,045 × (0.3900 − 1) = −637.5
               1,438 × (0.3900 − 0) = +560.8
Slope for b₁ = (−637.5 + 560.8) ÷ 5,634 = −76.7 ÷ 5,634 = −0.0136   → NEGATIVE → too low → go UP

b₁ = −0.0402 − (−0.0136) = −0.0266      ← the fibre points start rising
```

**All the rounds**

| After round | b₀       | b₁         | Fibre guess | No-fibre guess | Penalty   |
|-------------|----------|------------|-------------|----------------|-----------|
| 0 (start)   | 0        | 0          | 50%         | 50%            | 0.693     |
| 1           | −0.235   | −0.035     | 43.3%       | 44.2%          | 0.645     |
| 2           | −0.407   | −0.040     | 39.0%       | 40.0%          | 0.619     |
| 3           | −0.537   | −0.027     | 36.3%       | 36.9%          | 0.603     |
| 10          | −0.963   | +0.237     | 32.6%       | 27.6%          | 0.566     |
| 50          | −1.565   | +1.152     | 39.8%       | 17.3%          | 0.532     |
| End         | **−1.792** | **+1.473** | **42.1%** | **14.3%**      | **0.529** |

- Both guesses first fall, because both were too high.
- From round 3, the fibre guess is **below** 42.1%, so the fibre slope is negative and the fibre points go **up**.
- The penalty goes down every round.

**When training stops: the errors cancel out.** At the end, the guesses are 42.1% (fibre) and 14.3% (no fibre):

```
Fibre:     1,045 × (0.421 − 1) = −605.2
           1,438 × (0.421 − 0) = +605.2      →  sum = 0  →  slope for b₁ = 0

No fibre:    450 × (0.143 − 1) = −385.7
           2,701 × (0.143 − 0) = +385.7      →  sum = 0

All:       −605.2 + 605.2 − 385.7 + 385.7 = 0  →  slope for b₀ = 0
```

Both slopes are 0, so "new number = old number − 0" leaves the numbers unchanged. **Training is finished.**

**The final penalty** (same method as step 5):

| Type              | Customers | Chance given to real outcome | Penalty each           | Total    |
|-------------------|-----------|------------------------------|------------------------|----------|
| Fibre, left       | 1,045     | 0.421                        | −ln(0.421) = 0.865     | 904.3    |
| Fibre, stayed     | 1,438     | 0.579                        | −ln(0.579) = 0.546     | 785.6    |
| No fibre, left    | 450       | 0.143                        | −ln(0.143) = 1.946     | 875.8    |
| No fibre, stayed  | 2,701     | 0.857                        | −ln(0.857) = 0.154     | 416.2    |
| **All**           | **5,634** |                              |                        | **2,981.9** |

```
Average penalty = 2,981.9 ÷ 5,634 = 0.529      ← the smallest it can be
```

**Say:** *"The model repeats this again and again. In round 2, the penalty drops from 0.693 to 0.645. In round 3, the fibre guess falls below the real rate, so the fibre slope turns negative and the fibre points start going up. At the end, the model guesses 42.1% for fibre customers and 14.3% for the others, exactly the real rates. Then the errors cancel out, both slopes are zero, and training stops. The final numbers are −1.792 and +1.473: having fibre makes leaving more likely."*

### Step 8: Use the trained model

The trained numbers are b₀ = −1.792 and b₁ = +1.473.

**A new customer with fibre (x = 1):**

```
Points:   z = −1.792 + 1.473 × 1 = −0.319

Chance:   1.  −z = 0.319
          2.  e^0.319 = 1.376
          3.  1 + 1.376 = 2.376
          4.  p = 1 ÷ 2.376 = 0.421   →  42.1%

Decide:   42.1% < 50%   →  "will stay"
```

**A new customer without fibre (x = 0):**

```
Points:   z = −1.792 + 1.473 × 0 = −1.792

Chance:   1.  −z = 1.792
          2.  e^1.792 = 6.001
          3.  1 + 6.001 = 7.001
          4.  p = 1 ÷ 7.001 = 0.143   →  14.3%

Decide:   14.3% < 50%   →  "will stay"
```

A model with only one clue can only ever say 42.1% or 14.3%. That is not enough to tell customers apart, which is why the real model uses all 30 clues.

**Say:** *"Now we can use it. A new fibre customer gets 42.1%, so the model says 'will stay'. But with only one clue, every fibre customer gets the same answer. That's why the real model uses all 30 clues."*

### Step 9: From 1 clue to all 30 clues

The real model uses **exactly the same recipe and the same training loop**, only with 30 clues:

```
z = b₀ + b₁x₁ + b₂x₂ + … + b₃₀x₃₀
```

It learned 31 numbers (section 7 lists them all), and needed only 40 rounds.

**A real test customer: 8375-DKEBR**
1 month as a customer, month-to-month, fibre, electronic check, 69.60 per month. **Really left.**

| Clue                         | Points × customer's value | Contribution |
|------------------------------|---------------------------|--------------|
| *Starting points*            |                           | −1.584       |
| tenure (1 month)             | −1.257 × (−1.282)         | **+1.611**   |
| Fibre internet               | +1.181 × 1                | **+1.181**   |
| Electronic check             | +0.393 × 1                | +0.393       |
| Paperless billing            | +0.374 × 1                | +0.374       |
| Senior citizen               | +0.148 × 1                | +0.148       |
| MonthlyCharges (69.60)       | −0.467 × 0.155            | −0.072       |
| TotalCharges (69.60)         | +0.528 × (−0.978)         | −0.516       |
| Phone service                | −0.521 × 1                | −0.521       |
| **Total points z**           |                           | **+1.014**   |

The other 22 clues are 0 for this customer (for example, no two-year contract), so they add nothing.

**Where the scaled numbers come from** (section 25: (value − average) ÷ spread, using the training averages):

```
tenure:          (1 − 32.49)    ÷ 24.57   = −1.282
MonthlyCharges:  (69.60 − 64.93) ÷ 30.14  = +0.155
TotalCharges:    (69.60 − 2,299.33) ÷ 2,279.20 = −0.978
```

**Adding up the points, one by one:**

```
Start:                    −1.584
+ tenure        +1.611  →  0.027
+ fibre         +1.181  →  1.208
+ e-check       +0.393  →  1.601
+ paperless     +0.374  →  1.975
+ senior        +0.148  →  2.123
+ monthly       −0.072  →  2.051
+ total         −0.516  →  1.535
+ phone         −0.521  →  1.014
```

**Turning the points into a chance:**

```
1.  −z = −1.014
2.  e^(−1.014) = 0.363
3.  1 + 0.363 = 1.363
4.  p = 1 ÷ 1.363 = 0.734   →  73.4%

Decide:  73.4% ≥ 50%   →  "will leave"   ✓ (the customer really left)
```

Why does a **short** tenure **add** points? Tenure has negative points (−1.257). After scaling, 1 month becomes −1.282 (far below average). A minus times a minus is a plus: **+1.611**. So a very new customer gets a lot of extra points toward leaving.

**For comparison, test customer 4376-KFVRS:** 72 months, two-year contract, fibre. Fibre still adds +1.181, but long tenure (−2.022) and the two-year contract (−1.314) pull the total down to **z = −3.023**, so p = **4.6%** → "will stay". This customer **really stayed** ✓.

**Say:** *"The real model works exactly the same way, just with 30 clues. Here's a real test customer: a brand-new, month-to-month fibre customer. The short tenure and the fibre add the most points. The total is +1.014, which becomes a 73% chance, so the model says 'will leave', and the customer really did leave. A different fibre customer with 72 months and a two-year contract only gets 4.6%. The model combines all the clues, not just one."*

### The 9 steps on one page

```
1. Question:   will this customer leave?
2. Numbers:    each customer → clue x and answer y
3. Recipe:     points z = b₀ + b₁x   →   chance p = 1/(1+e^(−z))
4. Start:      b₀ = b₁ = 0  →  everyone 50%
5. Measure:    penalty = −ln(chance given to what happened); average 0.693
6. Adjust:     error = guess − real; slope = sum of errors ÷ 5,634;
               new number = old number − slope   (positive → down, negative → up)
7. Repeat:     until the errors cancel out and the slopes are 0
               (guesses 42.1% and 14.3%, penalty 0.529)
8. Use:        new customer → points → chance → yes/no
9. Real model: same recipe and loop, with 30 clues
```

---

## 4. Step 1: Adding up the points

### The idea

Every customer starts with some points. Each clue adds points (more likely to leave) or takes points away (more likely to stay).

### The formula

```
z = b₀ + b₁·x₁ + b₂·x₂ + … + bₙ·xₙ
```

| Symbol      | Meaning                                         | In this project      |
|-------------|-------------------------------------------------|----------------------|
| `z`         | Total points of one customer                    | e.g. −0.403          |
| `b₀`        | Starting points everyone gets (**intercept**)   | −1.584               |
| `x₁, x₂, …` | The customer's clue values                      | fibre = 1 or 0       |
| `b₁, b₂, …` | Points for each clue (**coefficients**)         | fibre = +1.181       |
| `n`         | Number of clues                                 | 30                   |

**In words:** starting points + (points for clue 1 × clue 1) + (points for clue 2 × clue 2) + …

### Worked example

Imagine only two clues count: fibre internet (+1.181) and a two-year contract (−1.314).

**Anna** has fibre (x₁ = 1) and no two-year contract (x₂ = 0).

```
Step 1. Write the formula:      z = b₀ + b₁·x₁ + b₂·x₂
Step 2. Put in the numbers:     z = −1.584 + 1.181 × 1 + (−1.314) × 0
Step 3. Multiply first:         1.181 × 1 = 1.181        (fibre is ON)
                                −1.314 × 0 = 0           (two-year is OFF)
Step 4. Add:                    z = −1.584 + 1.181 + 0 = −0.403
```

| Customer | Fibre | Two-year | Calculation                    | z          |
|----------|-------|----------|--------------------------------|------------|
| Anna     | 1     | 0        | −1.584 + 1.181                 | **−0.403** |
| Ben      | 1     | 1        | −1.584 + 1.181 − 1.314         | **−1.717** |
| Carl     | 0     | 0        | −1.584                         | **−1.584** |

The real model works the same way, but adds up all 30 clues for each customer.

---

## 5. Step 2: Turning points into a chance (the sigmoid)

### The idea

Points like −0.403 are hard to read. The **sigmoid** is a squeezing machine: any number goes in, and a chance between 0 and 1 comes out.

### The formula

```
p = 1 / (1 + e^(−z))
```

| Symbol | Meaning                           |
|--------|-----------------------------------|
| `p`    | The chance of leaving (0 to 1)    |
| `z`    | The total points                  |
| `e`    | The special number 2.718          |

### It is only four small steps

| Step | What to do                         | Calculator          |
|------|------------------------------------|---------------------|
| 1    | Flip the sign of z (make −z)       | `+/−`               |
| 2    | Calculate e to that power          | `eˣ`                |
| 3    | Add 1                              | `+ 1 =`             |
| 4    | Divide 1 by the result             | `1/x`               |

### Worked example: Anna (z = −0.403)

```
Step 1.  −z = 0.403
Step 2.  e^0.403 = 1.496
Step 3.  1 + 1.496 = 2.496
Step 4.  p = 1 ÷ 2.496 = 0.401   →  about 40%
```

### Worked example: Ben (z = −1.717)

| Step | Calculation      | Result   |
|------|------------------|----------|
| 1    | −(−1.717)        | 1.717    |
| 2    | e^1.717          | 5.567    |
| 3    | 1 + 5.567        | 6.567    |
| 4    | 1 ÷ 6.567        | **0.152 → 15%** |

The two-year contract lowers the chance of leaving from 40% to 15%.

### What the sigmoid does

| Points z | Chance p |
|----------|----------|
| −4       | 2%       |
| −2       | 12%      |
| **0**    | **50%**  |
| +2       | 88%      |
| +4       | 98%      |

```
100% |                     ___________
     |                  _/
 50% |----------------/
     |             _/
  0% |____________/
     +----------------------------------
        −4    −2    0    +2    +4     points (z)
```

**Why it always stays between 0 and 1:** e to any power is always positive, so `1 + e^(−z)` is always bigger than 1. And 1 divided by something bigger than 1 is always between 0 and 1.

---

## 6. Step 3: Deciding yes or no (the threshold)

```
If p ≥ 0.5  →  "will leave"
If p < 0.5  →  "will stay"
```

The 0.5 line is the **threshold**. Anna (40%) and Ben (15%) are both predicted to stay. A business can move the threshold: a lower one catches more leavers, but also gives more false alarms.

---

## 7. Where the points come from

The points are **not** chosen by a person. The model **learns** them during training, in one line of `train_models.py`:

```python
model.fit(X_train, y_train)
```

After training, the model stores them:

| Inside the model    | What it holds                   | Official name  |
|---------------------|---------------------------------|----------------|
| `model.intercept_`  | The starting points (b₀)        | Intercept      |
| `model.coef_`       | The points for all 30 clues     | Coefficients   |

You can print them with `python src/show_points.py`:

```python
import joblib
import pandas as pd

model = joblib.load("models/logistic_regression.joblib")
X_train = pd.read_csv("data/processed/X_train.csv")

print(f"Starting points (intercept): {model.intercept_[0]:.3f}")
points = pd.Series(model.coef_[0], index=X_train.columns).sort_values()
print(points.round(3).to_string())
```

The points this project's model learned:

| Clue                                   | Points  |
|----------------------------------------|---------|
| *(starting points, intercept)*         | −1.584  |
| Contract_Two year                      | −1.314  |
| tenure                                 | −1.257  |
| Contract_One year                      | −0.684  |
| PhoneService_Yes                       | −0.521  |
| MonthlyCharges                         | −0.467  |
| OnlineSecurity_Yes                     | −0.349  |
| TechSupport_Yes                        | −0.299  |
| MultipleLines_No phone service         | −0.269  |
| Dependents_Yes                         | −0.226  |
| 7 "No internet service" columns        | −0.173 each |
| OnlineBackup_Yes                       | −0.095  |
| PaymentMethod_Credit card (automatic)  | −0.021  |
| gender_Male                            | +0.020  |
| Partner_Yes                            | +0.022  |
| DeviceProtection_Yes                   | +0.033  |
| PaymentMethod_Mailed check             | +0.082  |
| SeniorCitizen                          | +0.148  |
| MultipleLines_Yes                      | +0.362  |
| StreamingMovies_Yes                    | +0.373  |
| PaperlessBilling_Yes                   | +0.374  |
| StreamingTV_Yes                        | +0.375  |
| PaymentMethod_Electronic check         | +0.393  |
| TotalCharges                           | +0.528  |
| InternetService_Fiber optic            | +1.181  |

**How to read it:**

| Part of the list    | Meaning                          | Examples                                  |
|---------------------|----------------------------------|-------------------------------------------|
| Big minus           | Strongly linked to **staying**   | Two-year contract, long tenure            |
| Close to 0          | Hardly matters                   | gender_Male (+0.020), Partner_Yes (+0.022)|
| Big plus            | Strongly linked to **leaving**   | Fibre internet, electronic check          |

The seven "No internet service" columns all have −0.173 because they always switch on together: a customer with no internet has no online security, no streaming, and so on. The model shares the points equally between them.

**Compared with what?** Because of one-hot encoding (`drop_first=True`), each clue is compared with the category that was removed. The contract points are compared with **month-to-month**; the fibre points are compared with **DSL**.

The rest of Part B explains **how** the model finds these numbers.

---

## 8. Measuring mistakes: the penalty (log loss)

### The idea

To learn, the model needs to measure how wrong its guesses are. Imagine a strict teacher who gives penalty points:

| The guess was…          | Penalty   |
|-------------------------|-----------|
| Right and confident     | tiny      |
| Unsure (around 50%)     | medium    |
| Wrong and confident     | huge      |

### The formula

For one customer:

| If the customer…   | Penalty         |
|--------------------|-----------------|
| left (y = 1)       | −ln(p)          |
| stayed (y = 0)     | −ln(1 − p)      |

**In words:** take the chance the model gave to **what really happened**, and calculate −ln of it.

The official formula combines both rows into one line. When y = 1, the second half becomes 0; when y = 0, the first half becomes 0.

```
loss = −[ y·ln(p) + (1 − y)·ln(1 − p) ]
```

**On the calculator:** type the chance → `ln` → `+/−`

### How the penalty grows

| Chance given to the real outcome | Penalty | In words       |
|----------------------------------|---------|----------------|
| 0.99                             | 0.01    | almost perfect |
| 0.9                              | 0.11    | very good      |
| 0.7                              | 0.36    | good           |
| 0.5                              | 0.69    | unsure         |
| 0.3                              | 1.20    | bad            |
| 0.1                              | 2.30    | very bad       |
| 0.01                             | 4.61    | terrible       |

### The total penalty (the cost)

```
J = (loss₁ + loss₂ + … + lossₙ) ÷ n
```

`J` is the **average penalty** over all training customers. Training finds the points that make `J` as small as possible.

### Worked example with real customers

This example uses a simple model with **one clue only** (fibre), trained on the real training data. (Section 9 shows how it was trained.)

**At the start of training,** the model guesses 50% for everyone:

```
Customer 9237-HQITU (fibre, LEFT):       −ln(0.5)     = 0.693
Customer 7590-VHVEG (no fibre, STAYED):  −ln(1 − 0.5) = 0.693
```

Everyone gets 0.693, so the average penalty is **0.693**.

**At the end of training,** the model guesses 42.1% for fibre customers and 14.3% for the others:

```
Customer 9237-HQITU (fibre, LEFT):       −ln(0.421)     = 0.865
Customer 7590-VHVEG (no fibre, STAYED):  −ln(1 − 0.143) = −ln(0.857) = 0.154
```

The second customer's penalty dropped a lot: the model now expects no-fibre customers to stay. The first customer's penalty went up: the model can't be perfect for every customer; it aims to be right **on average**.

**The average penalty for all 5,634 customers.** There are only four types of customers, so we calculate one penalty per type:

| Type              | How many | Chance given to real outcome | Penalty each | Total      |
|-------------------|----------|------------------------------|--------------|------------|
| Fibre, left       | 1,045    | p = 0.421                    | 0.865        | 904.3      |
| Fibre, stayed     | 1,438    | 1 − p = 0.579                | 0.546        | 785.6      |
| No fibre, left    | 450      | p = 0.143                    | 1.946        | 875.8      |
| No fibre, stayed  | 2,701    | 1 − p = 0.857                | 0.154        | 416.2      |
| **All**           | **5,634**|                              |              | **2,981.9**|

```
Average penalty J = 2,981.9 ÷ 5,634 = 0.529
```

The biggest penalty per customer (1.946) is for "no fibre, left": customers who looked safe but left anyway. These are the hardest customers for the model.

---

## 9. Learning the points: training round by round

### The idea

Training is a loop. Each round, the model does four actions:

```
1. GUESS      →  Use the current points to guess each customer's chance
2. COMPARE    →  For each customer: error = guess − real answer
3. DIRECTION  →  Use the errors to decide: should each number go up or down? (the slope)
4. ADJUST     →  Move each number a small step in that direction
```

Then it repeats with the new numbers, until they stop changing.

### The formulas

```
error for customer i     =  pᵢ − yᵢ
slope for b₀             =  (sum of all errors) ÷ n
slope for b₁             =  (sum of errors × x₁ for each customer) ÷ n
new number               =  old number − α × slope
```

| Symbol | Meaning                                                      |
|--------|--------------------------------------------------------------|
| `pᵢ`   | The model's guess for customer i                             |
| `yᵢ`   | The real answer for customer i (1 or 0)                      |
| `n`    | Number of training customers (5,634)                         |
| `α`    | Step size ("alpha"), also called the **learning rate** (1 here) |

The slope for b₁ only counts **fibre customers**, because for everyone else x₁ = 0, and error × 0 = 0. That makes sense: only fibre customers get the fibre points, so only their errors should change them.

### The real data for this example

| Group             | Customers | Left  | Stayed | Churn rate |
|-------------------|-----------|-------|--------|------------|
| Fibre (x = 1)     | 2,483     | 1,045 | 1,438  | **42.1%**  |
| No fibre (x = 0)  | 3,151     | 450   | 2,701  | **14.3%**  |
| All               | 5,634     | 1,495 | 4,139  | 26.5%      |

This is the truth the model should discover. It must learn two numbers: **b₀** (starting points) and **b₁** (fibre points).

### Round 1, in full

**Start:** b₀ = 0 and b₁ = 0.

**Action 1: Guess.** Everyone gets z = 0, so everyone gets a chance of **50%**.

**Action 2: Compare.**

| Customers         | How many | Guess | Real | Error each |
|-------------------|----------|-------|------|------------|
| Fibre, left       | 1,045    | 0.5   | 1    | −0.5       |
| Fibre, stayed     | 1,438    | 0.5   | 0    | +0.5       |
| No fibre, left    | 450      | 0.5   | 1    | −0.5       |
| No fibre, stayed  | 2,701    | 0.5   | 0    | +0.5       |

A negative error means the guess was too low; a positive error means too high.

**Action 3: Direction.**

Slope for b₀ (all 5,634 customers):

```
Left:    1,495 × (−0.5) = −747.5
Stayed:  4,139 × (+0.5) = +2,069.5
Total = 1,322        Average = 1,322 ÷ 5,634 = +0.2346
```

Positive: the guesses are **too high** overall (50% guessed, 26.5% real). So b₀ goes **down**.

Slope for b₁ (only the 2,483 fibre customers):

```
Fibre, left:    1,045 × (−0.5) = −522.5
Fibre, stayed:  1,438 × (+0.5) = +719
Total = 196.5        Average = 196.5 ÷ 5,634 = +0.0349
```

Also positive: even fibre customers are guessed too high (50% guessed, 42.1% real). So b₁ also goes **down** at first.

**Action 4: Adjust.**

```
b₀ = 0 − 0.2346 = −0.2346
b₁ = 0 − 0.0349 = −0.0349
```

### Round 2

**Guess:**

| Group     | z = b₀ + b₁ × x                    | Chance   |
|-----------|------------------------------------|----------|
| Fibre     | −0.2346 + (−0.0349) = −0.2695      | 43.3%    |
| No fibre  | −0.2346                            | 44.2%    |

**Direction:** slope for b₀ = +0.1725 (still too high), slope for b₁ = +0.0054 (fibre still a little too high).

**Adjust:**

```
b₀ = −0.2346 − 0.1725 = −0.4071
b₁ = −0.0349 − 0.0054 = −0.0402
```

### All the rounds

| After round | b₀      | b₁ (fibre) | Fibre guess | No-fibre guess | Penalty |
|-------------|---------|------------|-------------|----------------|---------|
| 0 (start)   | 0       | 0          | 50%         | 50%            | 0.693   |
| 1           | −0.235  | −0.035     | 43.3%       | 44.2%          | 0.645   |
| 2           | −0.407  | −0.040     | 39.0%       | 40.0%          | 0.619   |
| 3           | −0.537  | −0.027     | 36.3%       | 36.9%          | 0.603   |
| 10          | −0.963  | +0.237     | 32.6%       | 27.6%          | 0.566   |
| 50          | −1.565  | +1.152     | 39.8%       | 17.3%          | 0.532   |
| 200         | −1.790  | +1.469     | 42.1%       | 14.3%          | 0.529   |
| End         | **−1.792** | **+1.473** | **42.1%** | **14.3%**     | **0.529** |

**What happened:**

1. **First, both guesses fell.** 50% was too high for both groups, so the model lowered everything, including the fibre points.
2. **Then the groups separated.** From about round 3, the fibre guess fell below the real 42.1%. The fibre slope turned negative, and the fibre points started climbing.
3. **Finally, the guesses matched reality** exactly: 42.1% and 14.3%.
4. **The penalty went down every round**, from 0.693 to 0.529.

### When training stops

When the guesses equal the real churn rates, the errors cancel out. For the fibre group at 42.1%:

```
1,045 who left:   1,045 × (0.421 − 1) = −605
1,438 who stayed: 1,438 × (0.421 − 0) = +605
Total ≈ 0
```

Both slopes become 0, so the adjustments become 0. The numbers stop changing. **Training is finished.**

### Why fibre is +1.181 in the real model, not +1.473

This one-clue model gives fibre **all the credit** for the higher churn. The real model uses 30 clues together. Many fibre customers also pay higher bills, are on month-to-month contracts, or pay by electronic check, so those clues take part of the credit. Like a group project: when all team members are named, the credit is shared.

### In the real project

| | One-clue example | Real project |
|---|---|---|
| Customers | 5,634 | 5,634 |
| Numbers to learn | 2 | 31 (b₀ and 30 clues) |
| Rounds needed | about 200 | **40** |

The real model needed only 40 rounds because scikit-learn uses a smarter way to choose the step size (a method called **L-BFGS**), which also looks at how the slope is changing. The idea is the same: guess, compare, find the direction, adjust, repeat. The setting `max_iter=1000` only sets the **maximum** number of rounds allowed.

---

## 10. Understanding the slope

### The idea: a shower

When you adjust a shower:

- too hot → turn it **down**
- too cold → turn it **up**
- much too hot → turn it a lot; a little too hot → turn it a little

The **slope** is your hand feeling the water. It tells you two things:

| The slope…           | means…                  | so the number…        |
|----------------------|-------------------------|-----------------------|
| is positive (+)      | the guess is too high   | goes **down**         |
| is negative (−)      | the guess is too low    | goes **up**           |
| is big               | far from the truth      | changes a lot         |
| is small             | close to the truth      | changes a little      |
| is 0                 | exactly right           | stops changing        |

### A simple way to calculate it

For one yes/no clue, the slope for its group is:

```
slope ≈ (guess − real churn rate) × group size ÷ all customers
```

**Example: the fibre slope in round 1**

```
Step 1. How far off?       0.50 − 0.421 = 0.079      (too high)
Step 2. × group size:      0.079 × 2,483 = 196.5
Step 3. ÷ all customers:   196.5 ÷ 5,634 = 0.0349    → positive → go down
```

### Why the fibre points went down first, then up

| After round | Fibre guess | Reality | Fibre slope        | Fibre points   |
|-------------|-------------|---------|--------------------|----------------|
| 1           | 43.3%       | 42.1%   | + (too high)       | go down        |
| 3           | 36.3%       | 42.1%   | − (too low)        | go up          |
| 10          | 32.6%       | 42.1%   | − (too low)        | go up          |
| 50          | 39.8%       | 42.1%   | − (a little low)   | go up a little |
| End         | 42.1%       | 42.1%   | 0                  | stop           |

The model never "thinks" that fibre customers leave less. It only reacts to whether its current guess is too high or too low.

### The slope and the penalty

The **penalty** is how high up the hill you are. The **slope** tells you which way is downhill. Walking downhill step by step, using the slope, is called **gradient descent**. ("Gradient" is another word for slope; "descent" means going down.)

---

## 11. A shortcut for one clue: odds

With only **one yes/no clue**, the best points can be calculated directly, without rounds.

**Odds** = chance of leaving ÷ chance of staying.

```
b₀ = ln(odds of the group WITHOUT the clue)
b₁ = ln(odds WITH the clue) − ln(odds WITHOUT the clue)
```

**Worked example with the real training data:**

| Group     | Chance of leaving | Odds                      | ln(odds) |
|-----------|-------------------|---------------------------|----------|
| No fibre  | 0.143             | 0.143 ÷ 0.857 = 0.167     | −1.792   |
| Fibre     | 0.421             | 0.421 ÷ 0.579 = 0.727     | −0.319   |

```
b₀ = −1.792
b₁ = −0.319 − (−1.792) = 1.473
```

Exactly the same as the end of the training rounds in section 9.

**Check:** a fibre customer gets z = −1.792 + 1.473 = −0.319, and sigmoid(−0.319) = 42.1%, the real churn rate.

The shortcut only works with **one** clue. With 30 clues working together, there is no shortcut, so the computer uses the rounds.

---

## 12. Extras: regularisation and odds in business language

### Regularisation: keeping the points sensible

If some points become very large, the model may be memorising the training customers. **Regularisation** adds a small extra penalty for big points:

```
total penalty = penalty for mistakes + penalty for big points
```

The setting `C` controls the balance:

| C        | Meaning                                              |
|----------|------------------------------------------------------|
| large C  | fit the data closely (risk: memorising)              |
| small C  | keep the points small and simple (risk: too simple)  |
| C = 1.0  | the default, used in this project                    |

### Turning points into a business sentence

Each clue **multiplies** the odds of leaving by `e^(points)`.

| Clue              | Points  | e^points | Meaning                                |
|-------------------|---------|----------|----------------------------------------|
| Two-year contract | −1.314  | 0.27     | odds of leaving **73% lower** (1 − 0.27) |
| Fibre internet    | +1.181  | 3.26     | odds of leaving about **3.3× higher**  |

This is why logistic regression is popular in business: each point becomes a clear sentence.

---

## 13. The logistic regression algorithm

### Training

```
INPUT:   training clues X (n customers × m clues)
         training answers y (1 = left, 0 = stayed)
         step size α, maximum number of rounds

1. Set all numbers to zero:  b₀ = 0, b₁ = 0, …, bₘ = 0

2. Repeat until the numbers stop changing (or the maximum rounds is reached):

   a. GUESS:      for each customer i:
                      zᵢ = b₀ + b₁·xᵢ₁ + … + bₘ·xᵢₘ
                      pᵢ = 1 / (1 + e^(−zᵢ))

   b. COMPARE:    for each customer i:
                      errorᵢ = pᵢ − yᵢ

   c. DIRECTION:  slope₀ = (error₁ + error₂ + … + errorₙ) ÷ n
                  for each clue j:
                      slopeⱼ = (error₁·x₁ⱼ + error₂·x₂ⱼ + … + errorₙ·xₙⱼ) ÷ n

   d. ADJUST:     b₀ ← b₀ − α · slope₀
                  for each clue j:
                      bⱼ ← bⱼ − α · slopeⱼ

   e. STOP CHECK: if every slope is (almost) 0, stop

OUTPUT:  the learned numbers b₀, b₁, …, bₘ
```

### Prediction for a new customer

```
1. z = b₀ + b₁·x₁ + … + bₘ·xₘ
2. p = 1 / (1 + e^(−z))
3. if p ≥ 0.5 → "will leave", otherwise → "will stay"
```

### What scikit-learn does differently

- It uses **L-BFGS**, a smarter version of step 2d that chooses its own step sizes. This is why the real model needed only 40 rounds.
- It adds **L2 regularisation** (section 12), controlled by `C = 1.0`.

### Logistic regression in one picture

```
Clues → z = b₀ + b₁x₁ + … → p = 1/(1+e^(−z)) → p ≥ 0.5 ? → Leave / Stay
        (add up points)     (squeeze to chance)  (threshold)

Learning: guess → measure penalty (log loss) → step downhill (gradient descent) → repeat
```

---

# Part C: Random forest

## 14. The big idea: many trees voting

> Build **300 different decision trees**. Each tree asks yes/no questions to sort customers into leavers and stayers. Then let all the trees **vote**, so the mistakes of single trees get outvoted.

```
1. Build one tree:   ask the questions that best separate leavers from stayers
2. Build 300 trees:  each from a random sample of customers and random clues
3. Combine:          average the 300 trees' answers
```

---

## 15. One decision tree: a game of questions

### The idea

A decision tree plays "20 Questions". Someone thinks of an animal, and you ask: "Does it fly?" "Does it live in water?" Each answer narrows the possibilities, until you can guess.

A tree does the same with customers. Here is a small tree built **by the computer** from this project's training data:

```
Has the customer been here 16 months or less?
│
├── YES (2,042 customers, 45.2% left)
│     └── Fibre internet?
│           ├── YES → 593 left, 275 stayed   → 68% left → LEAVES
│           └── NO  → 329 left, 845 stayed   → 28% left → STAYS
│
└── NO  (3,592 customers, 16.0% left)
      └── Fibre internet?
            ├── YES → 452 left, 1,163 stayed → 28% left → STAYS
            └── NO  → 121 left, 1,856 stayed →  6% left → STAYS
```

### Words to know

| Word          | Meaning                                                   |
|---------------|-----------------------------------------------------------|
| **Node**      | A point in the tree where a question is asked             |
| **Root**      | The first question, at the top                            |
| **Split**     | Dividing a group into two by a yes/no question            |
| **Leaf**      | A final group at the end, where no more questions are asked |
| **Depth**     | How many questions in a row, from the root to a leaf      |

### How a tree makes a prediction

A customer answers the questions and goes down the tree to a leaf. The chance of leaving is the **share of leavers in that leaf**:

```
chance of leaving = leavers in the leaf ÷ all customers in the leaf
```

**Example:** a new customer, here 5 months, with fibre.

```
16 months or less?  YES  →  Fibre?  YES  →  leaf: 593 left, 275 stayed
chance = 593 ÷ (593 + 275) = 593 ÷ 868 = 0.68  →  68%  →  "will leave"
```

The rest of this part explains **how the computer chooses the questions**.

---

## 16. Measuring how mixed a group is (Gini impurity)

### The idea

Imagine a bag of marbles: **red = left**, **blue = stayed**. A good question sorts the marbles into bags that are as **pure** as possible: almost all red, or almost all blue.

**Gini impurity** measures how mixed a bag is. Close your eyes and pick two marbles: Gini is the chance that they are **different colours**.

- only blue marbles → you can never pick two different colours → Gini = **0** (pure)
- half red, half blue → very likely different → Gini = **0.5** (most mixed)

### The formula

```
Gini = 1 − (share of red)² − (share of blue)²
```

### Worked examples: four bags of 10 marbles

| Bag            | Red share | Blue share | Calculation       | Gini              |
|----------------|-----------|------------|-------------------|-------------------|
| 10 blue        | 0         | 1          | 1 − 0² − 1²       | **0** (pure)      |
| 1 red, 9 blue  | 0.1       | 0.9        | 1 − 0.01 − 0.81   | **0.18**          |
| 3 red, 7 blue  | 0.3       | 0.7        | 1 − 0.09 − 0.49   | **0.42**          |
| 5 red, 5 blue  | 0.5       | 0.5        | 1 − 0.25 − 0.25   | **0.50** (mixed)  |

**Lower Gini means a purer group.**

### Step by step: the Gini of all training customers

5,634 customers, 1,495 left.

```
Step 1. Share who left:     1,495 ÷ 5,634 = 0.265
Step 2. Share who stayed:   1 − 0.265 = 0.735
Step 3. Square each:        0.265² = 0.070       0.735² = 0.540
Step 4. Subtract from 1:    1 − 0.070 − 0.540 = 0.390
```

**Gini of all training customers = 0.390.** The tree's job is to ask questions that bring this down.

---

## 17. Choosing the best question, with real data

### The idea

A question splits one group into two. A good question makes both new groups purer. To score a question:

1. Calculate the Gini of each new group.
2. Take a **weighted average**: bigger groups count more.
3. See how much the Gini **went down**. This is the **improvement**.

### The formulas

```
Gini after   = (size of group A ÷ total) × Gini A + (size of group B ÷ total) × Gini B
Improvement  = Gini before − Gini after
```

### Worked example: the question "Fibre internet?"

```
Group YES (fibre):     2,483 customers, 1,045 left → share 0.421
    Gini = 1 − 0.421² − 0.579² = 1 − 0.177 − 0.335 = 0.487

Group NO (no fibre):   3,151 customers, 450 left   → share 0.143
    Gini = 1 − 0.143² − 0.857² = 1 − 0.020 − 0.734 = 0.245

Gini after   = (2,483 ÷ 5,634) × 0.487 + (3,151 ÷ 5,634) × 0.245
             = 0.441 × 0.487 + 0.559 × 0.245
             = 0.215 + 0.137
             = 0.352

Improvement  = 0.390 − 0.352 = 0.038
```

### The tree tries many questions

The computer calculates the improvement for **every possible question**, then picks the best one. Here are some of the real candidates for the first question:

| Question                      | Group YES: customers (% left) | Group NO: customers (% left) | Gini after | Improvement |
|-------------------------------|-------------------------------|------------------------------|------------|-------------|
| **Tenure 16 months or less?** | 2,042 (45.2%)                 | 3,592 (16.0%)                | 0.3505     | **0.0394** ✓ |
| Tenure 12 months or less?     | 1,737 (47.3%)                 | 3,897 (17.3%)                | 0.3514     | 0.0385      |
| Fibre internet?               | 2,483 (42.1%)                 | 3,151 (14.3%)                | 0.3518     | 0.0381      |
| Pays by electronic check?     | 1,891 (45.7%)                 | 3,743 (16.8%)                | 0.3526     | 0.0373      |
| Two-year contract?            | 1,359 (2.9%)                  | 4,275 (34.1%)                | 0.3543     | 0.0356      |
| One-year contract?            | 1,173 (11.1%)                 | 4,461 (30.6%)                | 0.3773     | 0.0126      |
| Has online security?          | 1,623 (14.4%)                 | 4,011 (31.4%)                | 0.3780     | 0.0119      |
| Male?                         | 2,833 (26.4%)                 | 2,801 (26.6%)                | 0.3899     | **0.0000**  |

**"Tenure 16 months or less?" wins**, because it lowers the Gini the most. Notice "Male?": it doesn't separate leavers from stayers at all (26.4% vs 26.6%), so its improvement is 0. The tree would never choose it.

**How number columns are handled:** for a number column like tenure, the tree tries **every possible cut-off** (0 or less, 1 or less, 2 or less, …, 72 or less) and keeps the best one. For tenure, the best cut-off was 16 months.

**One question = one column.** A tree can only ask about one column at a time. After one-hot encoding, there is no single "month-to-month" column, so "month-to-month?" takes two questions: "two-year contract? no" and "one-year contract? no".

### The second question

The tree now repeats the same search **inside each new group**. For the 2,042 customers with 16 months or less (Gini 0.495):

| Question                   | Improvement |
|----------------------------|-------------|
| **Fibre internet?**        | **0.079** ✓ |
| Pays by electronic check?  | 0.040       |
| One-year contract?         | 0.020       |
| Has online security?       | 0.005       |

"Fibre internet?" wins, which gives the small tree in section 15.

---

## 18. Building the whole tree

### The idea

The tree keeps asking the best question inside each group, again and again, until a **stop rule** says "stop here and make a leaf".

### The steps

1. Start with all training customers in one group.
2. Try every possible question, and calculate each one's improvement.
3. Pick the question with the biggest improvement, and split the group in two.
4. Repeat steps 2 and 3 inside each new group.
5. Stop splitting a group when:
   - the group is pure (everyone left, or everyone stayed), or
   - a split would create a group smaller than `min_samples_leaf` (10 in this project), or
   - no question improves the Gini.
6. Each group that stops becomes a leaf. Its chance of leaving = leavers ÷ customers in the leaf.

The first tree in this project's forest grew to a **depth of 14** questions, with **168 leaves**. The small tree in section 15 shows only its first two levels.

### There is no single penalty for the whole tree

Logistic regression minimises one penalty (log loss) for the whole model. A tree works differently: it is **greedy**. At each step, it picks the best question **for that moment**, without planning ahead. This is fast and works well in practice.

---

## 19. The problem with one tree: overfitting

### The idea

Without a stop rule, a tree keeps asking questions until every leaf holds only one or two customers. It **memorises** the training customers, like a student who memorises last year's exam answers instead of understanding the subject.

### What happened in this project

| Forest                                | Tree size (first tree)       | Correct on training | Correct on test |
|---------------------------------------|------------------------------|---------------------|-----------------|
| No stop rule (default settings)       | depth 26, 1,161 leaves       | **99.8%**           | 79.5%           |
| With `min_samples_leaf=10`            | depth 14, 168 leaves         | 83.5%               | **80.3%**       |

Without the stop rule, the trees memorised the training customers (99.8%), but did **worse** on new customers. With `min_samples_leaf=10`, each leaf must hold at least 10 customers, so the trees learn **general patterns** instead of facts about single people.

**A very high training score can be a warning sign, not good news.**

---

## 20. Making 300 different trees

### The idea

One tree can learn strange rules by chance, for example "customers who pay exactly 53.85 always leave". If we build many trees and they all make the **same** mistake, nothing improves. So the forest makes every tree **different**, using two kinds of randomness.

### Randomness 1: a random sample of customers (bootstrap)

Each tree gets its own sample of 5,634 customers, picked **with replacement**: after picking a customer, they are "put back", so they can be picked again, and some are not picked at all.

**Example with 5 customers (A, B, C, D, E):**

| Tree   | Its sample (5 picks) | Missing |
|--------|----------------------|---------|
| Tree 1 | A, A, C, D, E        | B       |
| Tree 2 | B, C, C, D, D        | A, E    |
| Tree 3 | A, B, E, E, E        | C, D    |

**How many different customers does each tree see?** The chance that a customer is **never** picked in 5,634 picks is:

```
(1 − 1/5,634)^5,634  ≈  0.368
```

So each tree sees about **1 − 0.368 = 63.2%** of the different customers, some of them more than once.

### Randomness 2: a random set of clues at each question

At each question, a tree may only choose from a few **random** clues, not all 30:

```
clues per question = √(number of clues) = √30 = 5.48  →  5 clues
```

(`√` is the square root: √25 = 5, because 5 × 5 = 25.)

So at the first question, one tree might only be allowed to look at contract, payment method, gender, partner and streaming TV. Another might see tenure, fibre, tech support, senior citizen and phone service. This makes the trees start with **different questions**, even though tenure is the best question overall.

### The result

Different customers + different clues → different trees → **different mistakes**.

---

## 21. Combining the trees: voting and averaging

### The idea

Each of the 300 trees gives its own chance that a customer will leave. The forest **averages** them.

### The formula

```
p_forest = (p₁ + p₂ + … + p_T) ÷ T
```

| Symbol      | Meaning                                    |
|-------------|--------------------------------------------|
| `p_forest`  | The forest's chance of leaving             |
| `p₁, p₂, …` | Each tree's chance (from its leaf)         |
| `T`         | Number of trees (300 in this project)      |

Then the same threshold decides: `p_forest ≥ 0.5` → "will leave".

### Worked example with 5 trees

| Tree | Chance of leaving         |
|------|---------------------------|
| 1    | 80%                       |
| 2    | 70%                       |
| 3    | 20% ← a strange tree      |
| 4    | 75%                       |
| 5    | 65%                       |

```
p_forest = (80 + 70 + 20 + 75 + 65) ÷ 5 = 310 ÷ 5 = 62%  →  "will leave"
```

The strange tree was outweighed by the others.

### Real customers from the test data

| Customer    | Tenure | Contract       | Internet | Trees saying "leave" | Forest's chance | Forest says | Reality |
|-------------|--------|----------------|----------|----------------------|-----------------|-------------|---------|
| 8375-DKEBR  | 1      | Month-to-month | Fibre    | 300 of 300           | 88.0%           | Leaves      | Left ✓  |
| 4376-KFVRS  | 72     | Two year       | Fibre    | 0 of 300             | 1.4%            | Stays       | Stayed ✓|
| 4686-UXDML  | 21     | Month-to-month | Fibre    | 165 of 300           | 52.7%           | Leaves      | Stayed ✗|

- **8375-DKEBR:** a brand-new, month-to-month, fibre customer. Every tree agreed.
- **4376-KFVRS:** also has fibre, but 72 months and a two-year contract. Every tree said "stays". The trees combined several clues, not just fibre.
- **4686-UXDML:** the trees were split almost evenly. The forest said "leaves", but the customer stayed. When trees disagree, the customer is hard to predict.

"Trees saying leave" counts each tree's yes/no answer. The forest's chance is the average of each tree's chance from its leaf. The two usually agree.

---

## 22. Why averaging works

### The idea

300 people guess how many sweets are in a jar. Some guess too high, some too low. The **average** is usually closer to the truth than most single guesses, because the too-high and too-low mistakes **cancel out**.

This only works if the mistakes are **different**. If everyone copied one person, they would all make the same mistake, and averaging would not help.

### The formula

How much a forest's answer jumps around (its **variance**) depends on two things:

```
Variance of the forest = ρ × σ²  +  (1 − ρ) ÷ T × σ²
```

| Symbol | Meaning                                                       |
|--------|---------------------------------------------------------------|
| `σ²`   | How much one tree's answer jumps around                       |
| `ρ`    | How similar the trees' mistakes are (0 = different, 1 = same) |
| `T`    | Number of trees                                               |

**In words:**

- **More trees** (T bigger) → the second part shrinks toward 0.
- **More different trees** (ρ smaller) → the first part gets smaller.

That is exactly why the forest uses 300 trees, random samples and random clues.

---

## 23. Feature importance

### The idea

The forest can tell us which clues helped most. Every time a tree uses a clue for a question, that question lowers the Gini. The forest adds up all these improvements for each clue, across all 300 trees.

### The formula

For each clue:

```
importance = sum, over every question that used this clue, of:
             (customers at that question ÷ all customers) × improvement
```

Then all importances are scaled so they add up to 1.

A clue used often, high in the trees (where groups are big), for big improvements, gets a high importance.

### The real result

| Clue                           | Importance |
|--------------------------------|------------|
| tenure                         | 0.204      |
| TotalCharges                   | 0.149      |
| MonthlyCharges                 | 0.103      |
| InternetService_Fiber optic    | 0.085      |
| PaymentMethod_Electronic check | 0.070      |
| Contract_Two year              | 0.069      |
| Contract_One year              | 0.036      |
| OnlineSecurity_Yes             | 0.036      |

**Tenure** is the most important clue: about 20% of all the improvement in the forest came from questions about tenure.

Number columns like tenure and the charges tend to score higher, partly because they offer many possible cut-offs. So importance shows which clues the forest **used** most, which is helpful, but not a perfect measure of cause and effect.

---

## 24. The decision tree and random forest algorithms

### Building one tree

```
INPUT:   a group of customers (clues X, answers y)
         min_samples_leaf (10 in this project)
         clues allowed per question (all clues for a single tree; 5 in the forest)

BUILD_TREE(group):

1. If the group is pure, or too small to split into two groups of
   at least min_samples_leaf customers:
       make a LEAF with  chance = leavers ÷ customers in the group
       stop

2. Choose which clues may be used (in a forest: pick 5 at random)

3. For each allowed clue:
       for each possible cut-off of that clue:
           split the group into YES and NO
           if either side has fewer than min_samples_leaf customers: skip
           Gini after  = (size YES ÷ size group) × Gini(YES)
                       + (size NO  ÷ size group) × Gini(NO)
           improvement = Gini(group) − Gini after

4. If no question has an improvement above 0:
       make a LEAF, stop

5. Keep the question with the biggest improvement

6. BUILD_TREE(YES group)     ← repeat inside each new group
   BUILD_TREE(NO group)

where  Gini(group) = 1 − (share who left)² − (share who stayed)²
```

### Building the forest

```
INPUT:   training customers (n customers, m clues), number of trees T (300)

For t = 1 to T:
    1. Draw n customers at random WITH replacement      (bootstrap sample)
    2. Build a tree on this sample with BUILD_TREE,
       allowing only √m random clues at each question  (5 of 30)
    3. Save the tree

OUTPUT:  T trees
```

### Prediction for a new customer

```
1. For each tree t:
       follow the questions down to a leaf
       pₜ = leavers ÷ customers in that leaf

2. p_forest = (p₁ + p₂ + … + p_T) ÷ T

3. if p_forest ≥ 0.5 → "will leave", otherwise → "will stay"
```

### Random forest in one picture

```
Training data
  ├── random sample 1   → Tree 1   (5 random clues per question) → p₁
  ├── random sample 2   → Tree 2   (5 random clues per question) → p₂
  ├── …
  └── random sample 300 → Tree 300                               → p₃₀₀
                                   ↓
              Average of all chances → p ≥ 0.5 ? → Leave / Stay
```

---

# Part D: Putting it together

## 25. Why scaling matters for one model but not the other

### The problem

The three number columns use very different sizes:

| Column         | Typical value | Biggest value |
|----------------|---------------|---------------|
| tenure         | 32            | 72            |
| MonthlyCharges | 65            | 119           |
| TotalCharges   | 2,299         | 8,685         |

TotalCharges looks about 100 times bigger than tenure, only because of its unit (money in total vs months).

### The idea: "normal, high or low?"

Standard scaling rewrites each value as a score: 0 = average, + = above average, − = below average.

```
scaled value = (value − average) ÷ spread
```

### Worked example with tenure (average 32.49, spread 24.57)

| Tenure     | Calculation                   | Scaled |
|------------|-------------------------------|--------|
| 32.49      | (32.49 − 32.49) ÷ 24.57       | 0      |
| 57 months  | (57 − 32.49) ÷ 24.57          | +1.0   |
| 8 months   | (8 − 32.49) ÷ 24.57           | −1.0   |
| 0 months   | (0 − 32.49) ÷ 24.57           | −1.32  |

The average and spread are calculated from the **training data only**, then used for the test data, to avoid **data leakage**.

### How the average and spread are calculated

Example with 8 numbers: 2, 4, 4, 4, 5, 5, 7, 9

```
Average:   (2 + 4 + 4 + 4 + 5 + 5 + 7 + 9) ÷ 8 = 40 ÷ 8 = 5

Spread (standard deviation):
  1. Distance from the average:  −3, −1, −1, −1, 0, 0, +2, +4
  2. Square each:                 9,  1,  1,  1, 0, 0,  4, 16
  3. Average of the squares:      32 ÷ 8 = 4
  4. Square root:                 √4 = 2
```

### Does it matter for each model?

| Model               | Matters? | Why | What happened in this project |
|---------------------|----------|-----|-------------------------------|
| Logistic regression | **Yes**  | It multiplies and adds the numbers, and learns by walking downhill. Different sizes make the hill a long, narrow valley. | Without scaling: still not finished after 1,000 rounds. With scaling: finished in 40 rounds. |
| Random forest       | **No**   | It only asks questions like "tenure 16 or less?". Scaling changes the numbers but not their order, so the same split is found. | Identical results with and without scaling. |

**Rule:** if a model **calculates** with the numbers (logistic regression, KNN, K-means), scale. If it only **asks questions** (trees, forests), scaling is not needed.

---

## 26. The math behind the evaluation metrics

### The four boxes

Every test prediction lands in one of four boxes:

| Box | Name           | Meaning                               |
|-----|----------------|---------------------------------------|
| TP  | True positive  | Left, and the model caught it         |
| TN  | True negative  | Stayed, and the model knew            |
| FP  | False positive | Stayed, but the model raised an alarm |
| FN  | False negative | Left, but the model missed it         |

The real results on the 1,409 test customers:

| Model               | TN  | FP  | FN  | TP  |
|---------------------|-----|-----|-----|-----|
| Logistic regression | 927 | 108 | 165 | 209 |
| Random forest       | 939 | 96  | 182 | 192 |

### The formulas, with logistic regression's numbers

| Metric    | Formula                                         | Question it answers                              | Calculation              | Result |
|-----------|-------------------------------------------------|--------------------------------------------------|--------------------------|--------|
| Accuracy  | (TP + TN) ÷ all                                 | How many guesses were right?                     | (209 + 927) ÷ 1,409      | 80.6%  |
| Precision | TP ÷ (TP + FP)                                  | When the model warns, how often is it right?     | 209 ÷ 317                | 65.9%  |
| Recall    | TP ÷ (TP + FN)                                  | Of everyone who left, how many were caught?      | 209 ÷ 374                | 55.9%  |
| F1 score  | 2 × precision × recall ÷ (precision + recall)   | One score that balances precision and recall     | 2 × 0.659 × 0.559 ÷ 1.218 | 0.605 |

**ROC AUC:** pick one customer who left and one who stayed, at random. How often does the model give the leaver the higher chance? 0.5 = guessing, 1.0 = perfect. Logistic regression: **0.842**; random forest: **0.845**.

### The lazy baseline

A pretend model that always says "stays":

```
Accuracy = 1,035 stayers ÷ 1,409 = 73.5%
Recall   = 0 ÷ 374 = 0%
```

It looks accurate, but catches nobody. This is why accuracy alone can be misleading when one answer is much more common.

### The final comparison

|                     | Accuracy | Precision | Recall    | F1        | ROC AUC   |
|---------------------|----------|-----------|-----------|-----------|-----------|
| Lazy baseline       | 73.5%    | 0%        | 0%        | 0         | 0.500     |
| Logistic regression | **80.6%**| 65.9%     | **55.9%** | **0.605** | 0.842     |
| Random forest       | 80.3%    | **66.7%** | 51.3%     | 0.580     | **0.845** |

---

## 27. Formula sheet

### Logistic regression

| What                       | Formula                                     |
|----------------------------|---------------------------------------------|
| Total points               | z = b₀ + b₁x₁ + … + bₙxₙ                    |
| Chance (sigmoid)           | p = 1 / (1 + e^(−z))                        |
| Decision                   | leave if p ≥ 0.5                            |
| Penalty, one customer      | −[y·ln(p) + (1−y)·ln(1−p)]                  |
| Total penalty (cost)       | J = average of all penalties                |
| Error, one customer        | p − y                                       |
| Slope for b₀               | average of errors                           |
| Slope for bⱼ               | average of (error × xⱼ)                     |
| Learning step              | b ← b − α · slope                           |
| One-clue shortcut          | b₁ = ln(odds with) − ln(odds without)       |
| Odds effect of a clue      | odds × e^(points)                           |

### Decision tree and random forest

| What                       | Formula                                               |
|----------------------------|-------------------------------------------------------|
| Gini impurity              | 1 − (share left)² − (share stayed)²                   |
| Gini after a split         | (size A ÷ total)·Gini A + (size B ÷ total)·Gini B     |
| Improvement                | Gini before − Gini after                              |
| Leaf chance                | leavers in leaf ÷ customers in leaf                   |
| Share of customers per tree| 1 − (1 − 1/n)^n ≈ 63.2%                               |
| Clues per question         | √(number of clues)                                    |
| Forest chance              | (p₁ + p₂ + … + p_T) ÷ T                               |
| Forest variance            | ρσ² + (1 − ρ)σ² ÷ T                                   |

### Data preparation and evaluation

| What               | Formula                                   |
|--------------------|-------------------------------------------|
| Standard scaling   | (value − average) ÷ spread                |
| Accuracy           | (TP + TN) ÷ all                           |
| Precision          | TP ÷ (TP + FP)                            |
| Recall             | TP ÷ (TP + FN)                            |
| F1 score           | 2 × P × R ÷ (P + R)                       |

### The two models side by side

|                         | Logistic regression                   | Random forest                          |
|-------------------------|---------------------------------------|----------------------------------------|
| Main idea               | Points game                           | Many question-trees voting             |
| Measures mistakes with  | Log loss, for the whole model         | Gini impurity, one question at a time  |
| Learns by               | Walking downhill (gradient descent)   | Picking the best question, greedily    |
| Finds combinations?     | No                                    | Yes                                    |
| Needs scaling?          | Yes                                   | No                                     |
| Easy to explain?        | Yes: points and odds                  | Harder: feature importance             |
| Main risk               | Too simple for complex patterns       | Memorising if trees grow too deep      |

---

## 28. Check yourself

Try these first, then open the answers.

**Logistic regression**

1. A customer has z = 0. What is their chance of leaving?
2. Carl has no fibre and no two-year contract. Using b₀ = −1.584, what are his points and his chance of leaving?
3. A customer left, and the model said p = 0.1. Is the penalty small or big? Calculate it.
4. A customer stayed, and the model said p = 0.2. What is the penalty?
5. In round 1, the guess for a group is 50%, but the real churn rate is 30%. Is the slope positive or negative? Will the points go up or down?
6. Why did the fibre points go down in round 1, even though fibre customers leave more often?

**Random forest**

7. What is the Gini of a group with 2 red and 8 blue marbles?
8. A question splits 100 customers into 40 (Gini 0.3) and 60 (Gini 0.1). The Gini before was 0.4. What is the improvement?
9. Why did the tree never choose "Male?" as a question?
10. Five trees say 60%, 40%, 70%, 30% and 50%. What does the forest say?
11. Why does a random forest use random samples and random clues?

**Putting it together**

12. Why did scaling change logistic regression but not the random forest?
13. A model has TP = 50, FP = 50, FN = 50. What are its precision and recall?

<details>
<summary><b>Show answers</b></summary>

1. **50%.** e^0 = 1, so p = 1 ÷ (1 + 1) = 0.5.
2. z = −1.584 + 0 + 0 = **−1.584**. Then e^1.584 = 4.874; 1 + 4.874 = 5.874; p = 1 ÷ 5.874 = **0.170 → 17%**.
3. **Big.** −ln(0.1) = **2.30**. The model was confident and wrong.
4. Use the chance of staying: 1 − 0.2 = 0.8. −ln(0.8) = **0.223**, a small penalty.
5. The guess is too high, so the slope is **positive**, and the points go **down**.
6. Because the starting guess (50%) was **too high** for fibre customers too (real: 42.1%). The model only reacts to "too high" or "too low". Later, when the fibre guess dropped below 42.1%, the fibre points went up.
7. 1 − 0.2² − 0.8² = 1 − 0.04 − 0.64 = **0.32**.
8. Gini after = 0.4 × 0.3 + 0.6 × 0.1 = 0.12 + 0.06 = 0.18. Improvement = 0.4 − 0.18 = **0.22**.
9. Men and women left at almost the same rate (26.4% vs 26.6%), so the question doesn't make the groups purer. Its improvement is 0.
10. (60 + 40 + 70 + 30 + 50) ÷ 5 = **50%**, which is ≥ 50%, so **"will leave"**.
11. To make the trees **different**, so their mistakes are different and **cancel out** in the average.
12. Logistic regression **calculates** with the numbers, so their size matters. Trees only **compare** numbers in yes/no questions, and scaling keeps their order the same.
13. Precision = 50 ÷ (50 + 50) = **50%**. Recall = 50 ÷ (50 + 50) = **50%**.

</details>

---

## 29. Glossary

| Word                 | Simple meaning                                                        |
|----------------------|-----------------------------------------------------------------------|
| Clue (feature)       | A column the model learns from                                        |
| Answer (target)      | The column the model predicts (Churn)                                 |
| Coefficient          | The points a clue adds or takes away                                  |
| Intercept            | The starting points every customer gets                               |
| Sigmoid              | The machine that squeezes any number into a chance from 0 to 1        |
| Threshold            | The cut-off that turns a chance into yes/no (0.5 by default)          |
| Error                | Guess minus real answer, for one customer                             |
| Log loss             | The penalty that punishes confident wrong guesses the most            |
| Cost function        | The average penalty over all training customers                       |
| Slope (gradient)     | Which way, and how strongly, to change a number to lower the penalty  |
| Gradient descent     | Learning by taking small steps downhill on the penalty                |
| Learning rate (α)    | The size of each step                                                 |
| Iteration            | One round of guess → compare → direction → adjust                     |
| L-BFGS               | A smarter way of walking downhill, used by scikit-learn               |
| Regularisation       | An extra penalty that keeps the points from getting too big           |
| Odds                 | Chance of leaving ÷ chance of staying                                 |
| Decision tree        | A model that asks a chain of yes/no questions                         |
| Node / root / leaf   | A question point / the first question / a final group                 |
| Split                | Dividing a group into two by a yes/no question                        |
| Gini impurity        | How mixed a group is (0 = pure, 0.5 = most mixed)                     |
| Improvement          | How much a question lowers the Gini                                   |
| Greedy               | Choosing the best step for now, without planning ahead                |
| Depth                | The number of questions from the root to a leaf                       |
| min_samples_leaf     | The smallest number of customers a leaf may hold                      |
| Bootstrap sample     | A random sample picked with replacement                               |
| Random forest        | Many different trees whose answers are averaged                       |
| Variance             | How much an answer would jump around with different training data     |
| Feature importance   | How much a clue helped the forest's questions overall                 |
| Overfitting          | Memorising the training data instead of learning general patterns     |
| Standard scaling     | Rewriting numbers as "steps away from the average"                    |
| Data leakage         | Letting information from the test data into training                  |
| Precision            | When the model warns, how often it is right                           |
| Recall               | Of everyone who really left, how many the model caught                |
| F1 score             | A single score that balances precision and recall                     |
| ROC AUC              | How well the model ranks leavers above stayers (0.5 = guessing)       |