# Stage 1 Course — ML Mental Model

Estimated time: 2–3 days  
Audience: Experienced developers who want to understand how machine learning systems behave and how to reason about experiments.

This stage builds the **mental framework required to understand all ML systems** before training models.

---

# Learning Objectives

By the end of this stage you should understand:

- What features and labels represent
- The difference between supervised and unsupervised learning
- Why datasets are split into train/validation/test
- How overfitting and underfitting occur
- How to evaluate model performance using common metrics

---

# Module 1 — What Machine Learning Actually Is

Machine learning is the process of learning patterns from data.

Instead of writing explicit rules:

```
if email contains "free money" → spam
```

A model learns patterns automatically from examples.

Typical structure of a dataset:

Example row:

```
Age | Salary | YearsExperience | BoughtProduct
45  | 70000  | 10              | Yes
```

In ML terms:

Features = inputs used for prediction  
Label = the value the model tries to predict

Features:

- Age
- Salary
- YearsExperience

Label:

- BoughtProduct

The model learns a function:

```
f(features) → label
```

---

# Exercise 1

Given the dataset below, identify features and labels.

```
SquareFeet | Bedrooms | Price
1200       | 3        | 250000
800        | 2        | 160000
1500       | 4        | 320000
```

Questions:

1. What are the features?
2. What is the label?

---

# Module 2 — Types of Machine Learning

## Supervised Learning

Supervised learning means the dataset contains the correct answers.

Example:

```
EmailText → Spam / NotSpam
HouseFeatures → HousePrice
MedicalData → DiseasePrediction
```

The model learns from labeled examples.

Used for:

- classification
- regression

Classification example:

```
Email → spam / not spam
```

Regression example:

```
House → predicted price
```

---

## Unsupervised Learning

Unsupervised learning means **there are no labels**.

The algorithm finds structure in the data.

Example tasks:

- customer segmentation
- anomaly detection
- clustering

Example:

Group customers by behavior:

```
Customer → cluster A
Customer → cluster B
Customer → cluster C
```

---

# Exercise 2

Classify the following problems:

1. Predict house price
2. Group customers by purchasing behavior
3. Detect fraudulent transactions when examples are labeled

---

# Module 3 — Train / Validation / Test Split

ML models must be evaluated on **data they have never seen before**.

Typical split:

```
Dataset
 ├── Train (70–80%)
 ├── Validation (10–15%)
 └── Test (10–15%)
```

Purpose:

Train set  
Used to learn patterns.

Validation set  
Used to tune model parameters and choose models.

Test set  
Used for final unbiased evaluation.

If you test on training data, performance will look artificially high.

---

# Exercise 3

Why is evaluating a model on training data misleading?

Write a short explanation.

---

# Module 4 — Overfitting vs Underfitting

## Underfitting

The model is **too simple** and cannot capture patterns.

Example:

Trying to predict complex relationships with a straight line.

Symptoms:

- low training accuracy
- low test accuracy

---

## Overfitting

The model memorizes the training data.

Symptoms:

- very high training accuracy
- poor test accuracy

Example:

A decision tree that perfectly memorizes training rows.

Goal in ML:

Find the balance between:

```
Model simplicity
vs
Model flexibility
```

---

# Exercise 4

A model has:

```
Training accuracy: 99%
Test accuracy: 65%
```

What problem is likely occurring?

---

# Module 5 — Evaluation Metrics

Models must be evaluated using objective metrics.

## Accuracy

Percentage of correct predictions.

```
Accuracy = correct predictions / total predictions
```

Example:

```
90 correct out of 100 → accuracy = 90%
```

Works well when classes are balanced.

---

## Precision

Precision measures **how many predicted positives are correct**.

Example:

Spam detection.

If the model marks 20 emails as spam and 18 are actually spam:

```
Precision = 18 / 20 = 0.9
```

---

## Recall

Recall measures **how many actual positives were found**.

If there were 25 spam emails but the model detected 18:

```
Recall = 18 / 25
```

---

## F1 Score

F1 balances precision and recall.

Useful when classes are imbalanced.

---

## Regression Metrics

When predicting numbers (like house prices):

Common metrics:

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

Lower values mean better predictions.

---

# Exercise 5

A spam classifier results:

```
True Positives: 40
False Positives: 10
False Negatives: 20
```

Calculate:

1. Precision
2. Recall

---

# Practical Mini Project

Use Python and scikit‑learn to run a simple ML experiment.

Steps:

1. Load a dataset (e.g. Iris dataset)
2. Split into train/test
3. Train a simple classifier
4. Evaluate accuracy

Example:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
```

Goal of this exercise:

Connect the **mental model** to a real ML workflow.

---

# Stage 1 Completion Checklist

You should now understand:

- What features and labels are
- Supervised vs unsupervised learning
- Train/validation/test split
- Overfitting vs underfitting
- Accuracy, precision, recall, F1
- Basic ML experiment workflow

If these concepts feel intuitive, move to **Stage 2 — Classical Machine Learning Algorithms**.