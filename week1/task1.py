# %% [markdown]
# # Titanic EDA — Exploratory Data Analysis
#
# First step in the Neurofive ML track: load the Titanic dataset and "listen" to it
# before doing any modeling.

# %%
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

# %% [markdown]
# ## 1. Load the dataset
#
# Download `train.csv` from the Kaggle "Titanic - Machine Learning from Disaster"
# competition and place it in `data/train.csv` relative to this script.

# %%
df = pd.read_csv("data/train.csv")
df.head()

# %% [markdown]
# ## 2. Basic shape

# %%
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
df.columns.tolist()

# %% [markdown]
# ## 3. Structure and dtypes — `.info()`

# %%
df.info()

# %% [markdown]
# ## 4. Summary statistics — `.describe()` (numerical columns)

# %%
df.describe()

# %% [markdown]
# ### 4b. Summary for categorical/object columns

# %%
df.describe(include="object")

# %% [markdown]
# ## 5. Missing values

# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}).sort_values(
    "missing_count", ascending=False
)

# %% [markdown]
# ## 6. Categorical vs numerical columns
#
# Note: some numeric-looking columns (`Survived`, `Pclass`) are actually categorical
# labels encoded as integers, not true continuous quantities.

# %%
numerical_cols = ["Age", "Fare", "SibSp", "Parch"]
categorical_cols = ["Survived", "Pclass", "Sex", "Embarked", "Name", "Ticket", "Cabin"]

print("Numerical:", numerical_cols)
print("Categorical:", categorical_cols)

# %% [markdown]
# ## 7. Data story — first impressions
#
# - The dataset has **891 passengers** and **12 columns**, combining numerical fields
#   (`Age`, `Fare`, `SibSp`, `Parch`) with categorical ones (`Survived`, `Pclass`,
#   `Sex`, `Embarked`).
# - Three columns have missing values: `Cabin` (~77% missing — likely too sparse to
#   use directly), `Age` (~20% missing — needs imputation), and `Embarked` (only 2
#   missing — trivial to fill or drop).
# - `Fare` and `Age` are right-skewed, with a handful of high-`Fare` outliers pulling
#   the mean well above the median.
# - `Survived` is the target column, encoded as 0/1, so this is a binary
#   classification problem.
# - Before modeling I'll need a plan for imputing `Age`, deciding whether `Cabin` is
#   worth engineering into a "deck" feature or dropping outright, and encoding the
#   categorical columns (`Sex`, `Embarked`, `Pclass`).
#
# (Numbers above are the well-known values for the standard Titanic `train.csv` —
# re-run the cells above on your own download and adjust this summary to match
# your actual output.)