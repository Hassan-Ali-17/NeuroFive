# %% [markdown]
# # Housing Price Regression
#
# Week 2, Task 2: predicting a continuous number (house price) instead of a
# category — regression instead of classification.
#
# Using the **Ames Housing** dataset (Kaggle "House Prices - Advanced
# Regression Techniques" train set), loaded from a local CSV file.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# %% [markdown]
# ## 1. Load the dataset

# %%
df = pd.read_csv("week2/train house.csv")

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# %%
df.head()

# %%
df.info()

# %% [markdown]
# `SalePrice` is the target — the actual sale price of the house in dollars.
# Everything else is a candidate feature. This dataset has 81 columns, many
# with missing values (e.g. `PoolQC`, `Fence`, `Alley` are mostly NA because
# most houses simply don't have those features) — we'll sidestep that by
# picking features that are complete.

# %% [markdown]
# ## 2. Select features
#
# Picking 5 features that plausibly drive housing price the most (and that
# have zero missing values, so no imputation is needed):
#
# - **`OverallQual`** — overall material and finish quality, rated 1-10.
#   Consistently the single strongest predictor of sale price in this dataset.
# - **`GrLivArea`** — above-ground living area in square feet. The classic
#   "bigger house, higher price" driver.
# - **`TotalBsmtSF`** — total basement square footage. Basements add usable
#   space and value, especially finished ones.
# - **`GarageCars`** — size of the garage in car capacity. A strong proxy for
#   both garage size and overall home quality/size.
# - **`YearBuilt`** — the year the house was originally built. Newer homes
#   tend to sell for more, all else equal.

# %%
features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt"]
target = "SalePrice"

X = df[features]
y = df[target]

X.describe()

# %% [markdown]
# ## 3. Train/test split

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# %% [markdown]
# ## 4. Train a Linear Regression model

# %%
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# %%
coeffs = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
print("Feature coefficients (impact in dollars per unit increase):")
coeffs

# %% [markdown]
# ## 5. Evaluate: RMSE and R²

# %%
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:,.0f}")
print(f"R²:   {r2:.4f}")

# %% [markdown]
# ## 6. Predicted vs. actual scatter plot
#
# A perfect model would put every point exactly on the diagonal line. The
# tighter the scatter hugs that line, the better the model's predictions.

# %%
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.4, s=15)

lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, color="red", linestyle="--", label="Perfect prediction")

plt.xlabel("Actual Sale Price ($)")
plt.ylabel("Predicted Sale Price ($)")
plt.title("Predicted vs. Actual House Prices")
plt.legend()
plt.show()

# %% [markdown]
# ## 7. What does the R² score mean, in plain English?
#
# R² measures how much of the variation in house prices the model successfully
# explains, on a scale from 0 to 1 (higher is better). An R² of, say, 0.75
# means the model accounts for about 75% of why house prices differ from one
# home to another — using just five features like quality, size, and age —
# while the remaining 25% comes down to things the model doesn't see at all:
# neighborhood desirability, kitchen/bathroom finish quality, lot shape,
# recent renovations, and general market timing. It's not a grade out of 100
# in the usual sense — it's a measure of "how much of the story does this
# model capture," and a high R² here means the model has genuinely learned a
# real pricing pattern, not just noise, even though it's far from perfect.
#
# *(Swap in your actual R² value above once you run this — the interpretation*
# *paragraph should reference your real number, not a placeholder.)*