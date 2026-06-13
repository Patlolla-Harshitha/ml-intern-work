# 📈 Polynomial Regression

> **Fitting curves where straight lines fall short.**

This module teaches you how polynomial regression extends linear regression to model non-linear relationships — without leaving the familiar framework of linear algebra. You will learn how to choose the right model complexity, avoid the trap of overfitting, and understand how this technique connects to broader ML theory.

---

## What is Polynomial Regression?

Imagine you are tracking how a car's fuel efficiency changes with speed. At very low speeds, efficiency is poor (engine idling). It improves as speed rises, then drops sharply at highway speeds due to wind resistance. No straight line can capture that arc — you need a curve. Polynomial regression is the tool that draws that curve while staying mathematically close to the linear regression you already know.

At its core, polynomial regression is linear regression applied to transformed features. Instead of feeding the model just speed `x`, you also feed it `x²`, `x³`, and so on. The model is still fitting a weighted sum of inputs — it has just been given richer inputs to work with. This is why it is called "linear in the parameters" even though the relationship with the original input is curved.

The key tradeoff to internalize is this: a higher-degree polynomial can fit increasingly complex curves, but it also becomes more sensitive to noise in the training data. A degree-10 polynomial might trace every bump in your training set perfectly while being completely wrong on new data. Learning to balance expressiveness against generalization is the central skill this topic builds.

---

## Mathematical Formulation

The polynomial regression model of degree `d` is:

```
ŷ = θ₀ + θ₁x + θ₂x² + θ₃x³ + ... + θ_d · x^d
```

**Symbol breakdown:**

| Symbol | Meaning |
|--------|---------|
| `ŷ` | Predicted output value |
| `x` | Original input feature |
| `x²`, `x³`, ..., `x^d` | Polynomial feature terms (engineered from `x`) |
| `θ₀` | Bias (intercept) — shifts the curve up or down |
| `θ₁ ... θ_d` | Learned weights for each polynomial term |
| `d` | Degree of the polynomial — your key hyperparameter |

**Why this matters:** The equation looks non-linear because of `x²` and `x³`, but from the model's perspective, each term is just another input feature. This means the same ordinary least squares (OLS) solver used for linear regression can fit this model without modification. You expand the features first, then solve a standard linear problem — a powerful and elegant idea.

---

## How It Works — Step by Step

> **Worked analogy:** You are predicting plant growth (cm) from days of sunlight. Growth accelerates early, then plateaus — a curve, not a line.

1. **Start with your raw data.** You have one feature: `days_of_sunlight (x)` and a target: `growth_cm (y)`.

2. **Choose a polynomial degree `d`.** Start with `d = 2` (a parabola). This is a hyperparameter you will tune.

3. **Expand your feature matrix.** Transform `[x]` into `[1, x, x²]`. Each data point now has three features instead of one. If `x = 3`, the expanded row becomes `[1, 3, 9]`.

4. **Fit a linear regression model** on the expanded matrix. The solver finds weights `θ₀, θ₁, θ₂` that minimize the mean squared error between predictions and actual growth values.

5. **Predict on new data.** For a new input `x_new`, expand it the same way, then compute `ŷ = θ₀ + θ₁·x_new + θ₂·x_new²`.

6. **Evaluate and tune.** Plot training vs. validation error across different values of `d`. Pick the degree where validation error is lowest — not training error.

> 📊 **[Placeholder: Bias-Variance Curve — Training error vs. Validation error across degrees d = 1 to 10]**

---

## Key Assumptions

| Assumption | What happens if violated |
|---|---|
| The true relationship is polynomial (or well-approximated by one) | Model will systematically misfit the data regardless of degree chosen |
| Features are scaled / standardized | High-degree terms like `x^8` become astronomically large, causing numerical instability and poor gradient behavior |
| Training data is representative | Polynomial models extrapolate very poorly — predictions outside the training range can become wildly wrong |
| No severe multicollinearity between polynomial terms | Terms like `x³` and `x⁴` are highly correlated, making individual coefficient estimates unstable and hard to interpret |
| Residuals have roughly constant variance | Heteroscedasticity inflates error in certain regions and misleads the loss function |

---

## When to Use / When Not to Use

| ✅ Use Polynomial Regression | ❌ Avoid Polynomial Regression |
|---|---|
| The relationship visually follows a curve (residuals show a pattern after linear fit) | You have high-dimensional data — feature explosion becomes unmanageable |
| You need an interpretable, lightweight model | The true function is highly discontinuous or jagged |
| Dataset is small-to-medium and low-dimensional | You need reliable extrapolation beyond training data range |
| You want a quick non-linear baseline before trying complex models | You have many features — interaction terms multiply combinatorially |
| Domain knowledge suggests a polynomial structure (e.g., physics equations) | A tree-based or neural model already fits well enough |

---

## Implementation Overview

### a) From Scratch (NumPy)

You manually construct the polynomial feature matrix by stacking columns `[x, x², x³, ...]` using `np.column_stack` or `np.vander`. Then you solve for coefficients using `np.linalg.lstsq` (the normal equation). This approach makes the transformation completely transparent and is excellent for learning. The downside is you must handle scaling, the intercept column, and numerical issues yourself.

### b) Scikit-learn Pipeline

Scikit-learn separates the concern cleanly: `PolynomialFeatures` handles expansion, `StandardScaler` handles scaling, and `LinearRegression` (or `Ridge`) handles fitting. Wrapping these in a `Pipeline` ensures the same transformation is applied consistently to training and test data — a common source of bugs when done manually.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

model = Pipeline([
    ("poly",   PolynomialFeatures(degree=3, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge",  Ridge(alpha=1.0))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

> The `StandardScaler` step is non-optional for polynomial features — skipping it is one of the most common implementation mistakes.

> 📊 **[Placeholder: Pipeline diagram — Raw features → PolynomialFeatures → StandardScaler → Ridge → Predictions]**

---

## Top 5 Interview Questions

**Q1. Polynomial regression is called a "linear model" — what exactly is linear about it?**
- The model is linear in its *parameters* (θ₀, θ₁, ...), not in the input `x`
- The normal equation and OLS solver apply without modification
- Contrast with neural networks, where parameters enter non-linearly through activation functions

**Q2. How does polynomial degree affect bias and variance? What does the optimal degree look like empirically?**
- Low degree → high bias, low variance (underfit)
- High degree → low bias, high variance (overfit)
- Optimal degree minimizes validation error — empirically a U-shaped curve on test loss
- Mention learning curves as a complementary diagnostic

**Q3. A degree-12 polynomial fits training data perfectly but performs poorly on validation. What are all the levers you would pull?**
- Reduce degree; add L2 regularization (Ridge); increase training data
- Plot learning curves to distinguish high variance from high bias
- Consider splines or GAMs as alternatives for smoother fits

**Q4. Walk me through every decision you would make when applying polynomial regression to a new dataset end to end.**
- EDA → residual analysis from linear fit → visual curve check → feature scaling → degree selection via cross-validation → regularization → evaluation on held-out set → interpretability check

**Q5. How does polynomial regression relate to kernel methods, and when would you prefer a kernel SVM for non-linear data?**
- The polynomial kernel implicitly computes the same feature expansion without materializing the full matrix
- Kernel SVM preferred when: data is high-dimensional, explicit feature matrix is too large, or margin-based classification is the goal
- Polynomial regression scales poorly to high `d` and many features; kernel trick sidesteps this

---

## Quick Reference Table

| Property | Details |
|---|---|
| **Algorithm type** | Supervised learning — regression |
| **Model family** | Generalized linear model (linear in parameters) |
| **Training time complexity** | O(n · d²) for feature expansion; O((nd)³) for normal equation |
| **Space complexity** | O(n · d) for the expanded feature matrix |
| **Key hyperparameters** | Degree `d`; regularization strength `α` (Ridge/Lasso) |
| **Degree selection method** | k-fold cross-validation or validation curve |
| **Evaluation metrics** | MSE, RMSE, MAE, R² score |
| **Overfitting risk** | High — increases rapidly with degree |
| **Scaling required** | Yes — always standardize before fitting |
| **Interpretable?** | Partially — individual coefficients lose meaning at high degree |

---

## References & Further Reading

1. **[An Introduction to Statistical Learning (ISLR) — Chapter 7](https://www.statlearning.com/)** — The clearest treatment of polynomial and spline regression available; free PDF from the authors.

2. **[Scikit-learn: Polynomial Features User Guide](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)** — Official documentation with concise examples and pipeline guidance.

3. **[Bishop, C. (2006). Pattern Recognition and Machine Learning — Chapter 1 & 3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)** — Rigorous treatment of overfitting, model selection, and regularization through the lens of polynomial curve fitting.

4. **[Hastie, Tibshirani & Friedman — The Elements of Statistical Learning, Chapter 5](https://hastie.su.domains/ElemStatLearn/)** — Connects polynomial regression to splines, smoothing, and kernel regression; graduate-level but freely available.

5. **[Andrew Ng's ML Specialization — Coursera (Course 2)](https://www.coursera.org/specializations/machine-learning-introduction)** — Excellent visual intuition for bias-variance tradeoff with polynomial regression as the running example.

---

*Part of the ML Study Series — built for B.Tech students and working professionals alike.*
