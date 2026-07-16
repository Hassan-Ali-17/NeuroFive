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
df = pd.read_csv("week2/train.csv")
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
df.describe(include="str")

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

# %% [markdown]
# ## 8. Handling missing values
#
# Recall from Section 5: `Age` (~20% missing), `Cabin` (~77% missing), and
# `Embarked` (2 missing). Each gets a different treatment because the *reason*
# and *scale* of missingness is different for each:
#
# - **`Age`** → fill with the **median**, not the mean, because `Age` is right-skewed
#   and the median is less distorted by outliers. Filling with a single global value
#   is a simplification — a more refined approach would impute per `Pclass`/`Sex`
#   group — but the median is a reasonable, defensible baseline at this stage.
# - **`Embarked`** → fill with the **mode** (most frequent port). Only 2 rows are
#   missing, so whatever we pick has negligible effect on the dataset as a whole.
# - **`Cabin`** → **drop the column** entirely rather than fill it. At ~77% missing,
#   any imputation would be mostly fabricated data. Instead of filling, we extract
#   the one bit of real signal it has — whether a cabin was recorded at all — as a
#   new binary feature, then drop the original noisy column.

# %%
df_clean = df.copy()

# Age: fill with median
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

# Embarked: fill with mode
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])

# Cabin: capture signal as a new feature, then drop the sparse original column
df_clean["HasCabin"] = df_clean["Cabin"].notna().astype(int)
df_clean = df_clean.drop(columns=["Cabin"])

print("Missing values after cleaning:")
df_clean.isnull().sum()

# %% [markdown]
# ## 9. Visualizations
#
# Four visualizations to catch mistakes and surface patterns before modeling:
# a histogram, a boxplot (for outlier detection), a bar chart, and a correlation
# heatmap.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# %% [markdown]
# ### 9a. Histogram — Age distribution
#
# Shows the overall shape of `Age`: where passengers cluster and whether the
# distribution is skewed.

# %%
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["Age"], bins=30, kde=True)
plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ### 9b. Boxplot — Fare outlier detection
#
# `Fare` is the clearest outlier case in this dataset. The boxplot below shows a
# tight cluster of low fares with a long tail of extreme values (including a few
# passengers who paid 500+, more than 10x the median).

# %%
plt.figure(figsize=(8, 5))
sns.boxplot(x=df_clean["Fare"])
plt.title("Fare Distribution — Outlier Detection")
plt.xlabel("Fare")
plt.show()

# %%
# Quantify the outliers using the IQR rule
Q1 = df_clean["Fare"].quantile(0.25)
Q3 = df_clean["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean["Fare"] < lower_bound) | (df_clean["Fare"] > upper_bound)]
print(f"IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Number of Fare outliers: {len(outliers)} ({len(outliers) / len(df_clean) * 100:.1f}% of passengers)")

# %% [markdown]
# ### 9c. Bar chart — Survival rate by passenger class
#
# A quick way to compare a categorical feature (`Pclass`) against the target
# (`Survived`).

# %%
plt.figure(figsize=(8, 5))
survival_by_class = df_clean.groupby("Pclass")["Survived"].mean()
sns.barplot(x=survival_by_class.index, y=survival_by_class.values)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.show()

# %% [markdown]
# ### 9d. Correlation heatmap
#
# Shows how the numerical features relate to each other and to `Survived`.

# %%
plt.figure(figsize=(9, 7))
numeric_for_corr = df_clean[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "HasCabin"]]
corr = numeric_for_corr.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# ## 10. Which feature most affects survival, and why?
#
# **`Sex` appears to be the single strongest predictor of survival**, followed
# closely by `Pclass` and `Fare` (which are themselves correlated — higher fares
# mean higher class).
#
# From the data: women survived at a dramatically higher rate than men — a direct
# reflection of the "women and children first" evacuation policy on the Titanic.
# `Pclass` matters for a related but distinct reason: first-class cabins were
# physically closer to the lifeboats and their passengers were prioritized during
# evacuation, so survival rate drops steadily from 1st to 3rd class (visible in the
# bar chart above). `Fare` correlates with survival mostly *because* it's a proxy
# for `Pclass`, not because paying more directly saved lives.
#
# `Age` has a weaker but still real effect: very young children had somewhat better
# survival odds, consistent with the same evacuation priority. `SibSp` and `Parch`
# show only mild correlations with survival — having a small family aboard seems to
# help slightly (more people to look out for you), but very large families fare
# worse, likely because large groups were harder to keep together during evacuation.
#
# Overall: **`Sex` > `Pclass` ≈ `Fare` > `Age` > `SibSp`/`Parch`**, and this ranking
# will shape which features I prioritize when building a model later.

# %% [markdown]
# ## 11. Building a classification model
#
# Time to put the EDA to work. Goal: predict `Survived` (0/1) from the cleaned
# feature set using Logistic Regression — a good, interpretable baseline for
# binary classification.

# %% [markdown]
# ### 11a. Select features and encode categoricals
#
# `Name` and `Ticket` are dropped — they're high-cardinality free text, not
# useful in this basic model without further feature engineering. `PassengerId`
# is just a row identifier. Everything else is either already numeric or gets
# one-hot encoded via `pd.get_dummies()`.
#
# `drop_first=True` avoids the "dummy variable trap" — e.g. for `Sex`, we only
# need one column (`Sex_male`) since "not male" already implies female.

# %%
model_df = df_clean.drop(columns=["PassengerId", "Name", "Ticket"])

model_df = pd.get_dummies(model_df, columns=["Sex", "Embarked"], drop_first=True)

model_df.head()

# %% [markdown]
# ### 11b. Train/test split
#
# 80% train / 20% test, with `stratify=y` so both sets keep the same survival
# ratio as the full dataset (important since `Survived` isn't perfectly balanced).

# %%
from sklearn.model_selection import train_test_split

X = model_df.drop(columns=["Survived"])
y = model_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# %% [markdown]
# ### 11c. Train the Logistic Regression model

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# %% [markdown]
# ### 11d. Evaluate: accuracy

# %%
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# %% [markdown]
# ### 11e. Confusion matrix

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Did not survive", "Survived"])
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix — Logistic Regression")
plt.show()

# %% [markdown]
# ### 11f. What the confusion matrix tells us
#
# The confusion matrix is a 2×2 breakdown of predictions vs. actual outcomes:
#
# - **Top-left (True Negatives)** — passengers who actually did *not* survive,
#   and the model correctly predicted that.
# - **Top-right (False Positives)** — passengers who did *not* survive, but the
#   model predicted they *would*. These are the model's overly optimistic mistakes.
# - **Bottom-left (False Negatives)** — passengers who *did* survive, but the
#   model predicted they wouldn't. These are the model's overly pessimistic
#   mistakes.
# - **Bottom-right (True Positives)** — passengers who actually survived, and
#   the model correctly predicted that.
#
# Accuracy alone (the single number above) hides *which kind* of mistake the
# model tends to make. If, for example, false negatives outnumber false
# positives, the model is more likely to under-predict survival — meaning it's
# a bit too pessimistic. For this problem specifically, that asymmetry matters:
# the model correctly separates most passengers, but the errors it does make
# tend to cluster around men in lower classes who technically survived (harder
# cases, since the strongest signal — `Sex`/`Pclass` — predicts against them)
# and women/first-class passengers who didn't survive (also harder, for the
# same reason in reverse). In other words, the model leans heavily on `Sex` and
# `Pclass`, so its mistakes are concentrated exactly where those features give
# misleading signal.
#
# *(Re-run this on your own data and check your actual confusion matrix — the*
# *specific counts will confirm which type of error is more common for your run.)*