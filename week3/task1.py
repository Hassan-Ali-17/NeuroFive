# %% [markdown]
# # Week 3 — Task 1: Classification Metrics & Hyperparameter Tuning
# 
# In this task, we will:
# 1. Load and preprocess the Titanic dataset (following the Week 2 pipeline).
# 2. Evaluate our baseline Logistic Regression model using Precision, Recall, and F1-score.
# 3. Explain why accuracy alone is misleading for imbalanced datasets.
# 4. Use GridSearchCV to tune the hyperparameters `C` and `penalty` of our Logistic Regression model.
# 5. Compare the baseline and tuned models in a before/after table.

# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# %% [markdown]
# ## Why Accuracy Alone is Misleading for Imbalanced Datasets
# 
# Accuracy is defined as the number of correct predictions divided by the total number of predictions.
# While intuitive, it can be highly misleading when dealing with imbalanced datasets (where one class significantly outnumbers the other).
# 
# **Example:**
# Consider a dataset of 100 passengers where 95 did not survive (Class 0) and only 5 survived (Class 1).
# A naive classifier that predicts "did not survive" for *every* passenger would achieve **95% accuracy**.
# However, this model is completely useless for identifying survivors. It has:
# - **Recall** for survivors = 0% (it found 0 out of 5 survivors).
# - **Precision** for survivors = 0% (since it never predicted survival).
# 
# By looking only at accuracy, we might think we have an excellent model, when in reality it fails completely on the minority class of interest.
# Precision, Recall, and F1-score help us evaluate performance on individual classes:
# - **Precision** (Positive Predictive Value): Of all predicted survivors, how many actually survived? (Avoids false positives).
# - **Recall** (Sensitivity): Of all actual survivors, how many did we correctly identify? (Avoids false negatives).
# - **F1-score**: The harmonic mean of Precision and Recall, providing a single metric that balances both.

# %% [markdown]
# ## 1. Load and Clean the Dataset

# %%
# Load the dataset
# Ensure paths are correct relative to project root
train_path = "week3/train.csv"
if not os.path.exists(train_path):
    # Fallback for alternative execution environments
    train_path = "train.csv"

print(f"Loading dataset from: {train_path}")
df = pd.read_csv(train_path)

# Cleaning pipeline
df_clean = df.copy()

# 1. Age: Fill missing with median
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

# 2. Embarked: Fill missing with mode
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])

# 3. Cabin: Capture presence of cabin, then drop original Cabin column
df_clean["HasCabin"] = df_clean["Cabin"].notna().astype(int)
df_clean = df_clean.drop(columns=["Cabin"])

# %% [markdown]
# ## 2. Feature Selection & Encoding

# %%
# Drop identifiers and text columns
model_df = df_clean.drop(columns=["PassengerId", "Name", "Ticket"])

# One-hot encode Sex and Embarked
model_df = pd.get_dummies(model_df, columns=["Sex", "Embarked"], drop_first=True)

print("Features columns:", model_df.columns.tolist())

# %% [markdown]
# ## 3. Train/Test Split (Stratified)

# %%
X = model_df.drop(columns=["Survived"])
y = model_df["Survived"]

# Stratify on target to maintain the same class ratio in train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# %% [markdown]
# ## 4. Baseline Model Training & Evaluation

# %%
# Initialize and fit baseline model (matching Week 2)
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)

# Predict on test set
y_pred_baseline = baseline_model.predict(X_test)

# Calculate metrics
baseline_acc = accuracy_score(y_test, y_pred_baseline)
print("\n=== Baseline Model Performance ===")
print(f"Accuracy: {baseline_acc:.4f}")
print("\nClassification Report (Baseline):")
print(classification_report(y_test, y_pred_baseline, target_names=["Did Not Survive", "Survived"]))

# Extract metrics for comparison table (class 1: Survived)
b_prec, b_rec, b_f1, _ = precision_recall_fscore_support(y_test, y_pred_baseline, average="binary")

# %% [markdown]
# ## 5. Hyperparameter Tuning using GridSearchCV
# 
# We tune:
# - `C`: Inverse of regularization strength. Smaller values specify stronger regularization.
# - `penalty`: Type of regularization (L1 vs L2).
# 
# Note: We use `solver='liblinear'` because it supports both L1 and L2 penalties.

# %%
# Define hyperparameter grid
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Fit on training data
print("\nRunning GridSearchCV...")
grid_search.fit(X_train, y_train)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# %% [markdown]
# ## 6. Evaluate Tuned Model

# %%
# Predict using best model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

# Calculate metrics
tuned_acc = accuracy_score(y_test, y_pred_tuned)
print("\n=== Tuned Model Performance ===")
print(f"Accuracy: {tuned_acc:.4f}")
print("\nClassification Report (Tuned):")
print(classification_report(y_test, y_pred_tuned, target_names=["Did Not Survive", "Survived"]))

# Extract metrics for comparison table (class 1: Survived)
t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(y_test, y_pred_tuned, average="binary")

# %% [markdown]
# ## 7. Performance Comparison: Before vs After

# %%
# Create and display comparison table
comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (Survived)", "Recall (Survived)", "F1-Score (Survived)"],
    "Baseline Model": [baseline_acc, b_prec, b_rec, b_f1],
    "Tuned Model": [tuned_acc, t_prec, t_rec, t_f1]
})

# Format values as percentages
for col in ["Baseline Model", "Tuned Model"]:
    comparison_df[col] = comparison_df[col].apply(lambda x: f"{x*100:.2f}%")

print("\n=== Performance Comparison Table ===")
print(comparison_df.to_string(index=False))
