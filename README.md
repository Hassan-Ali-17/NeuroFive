```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗██╗██╗   ██╗███████╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝██║██║   ██║██╔════╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║█████╗  ██║██║   ██║█████╗
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  ██║╚██╗ ██╔╝██╔══╝
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██║     ██║ ╚████╔╝ ███████╗
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚═══╝  ╚══════╝
```

## ML Track — Week 1 & Week 2

My progress through the NeuroFive Solutions ML track. Week 1 covers environment
setup, first exploratory data analysis (EDA), data cleaning, and visualization.
Week 2 moves into modeling, starting with a first classification model on the
Titanic dataset, then a first regression model on the Ames Housing dataset.

## Repo structure

```
neurofive-ml-track/
├── data/
│   ├── train.csv                      # Titanic dataset (not committed — see below)
│   └── train_house.csv                # Ames Housing dataset (not committed — see below)
├── eda_titanic.ipynb                  # Jupyter notebook version
├── eda_titanic.py                     # Same content as a plain script (VS Code # %% cells)
├── housing_price_regression.ipynb     # Jupyter notebook version
├── housing_price_regression.py        # Same content as a plain script (VS Code # %% cells)
├── .gitignore
└── README.md
```

`eda_titanic.py` / `eda_titanic.ipynb` and `housing_price_regression.py` /
`housing_price_regression.ipynb` are each identical content pairs — the script
uses `# %%` cell markers so it runs interactively in VS Code without needing a
notebook file, and the notebook is generated from it for anyone who wants the
classic Jupyter format with saved outputs.

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

No virtual environment required — a global Python install works fine for this.

### Datasets

- **Titanic:** download `train.csv` from the Kaggle competition
  ["Titanic - Machine Learning from Disaster"](https://www.kaggle.com/c/titanic)
  (free Kaggle account required) and place it at `data/train.csv`.
- **Ames Housing:** download `train.csv` from the Kaggle competition
  ["House Prices - Advanced Regression Techniques"](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
  (free Kaggle account required), rename it `train_house.csv`, and place it at
  `data/train_house.csv`.

Neither dataset is committed to this repo — only the code that processes them.

### Running it

**As a script in VS Code:** open the `.py` file, click **Run Cell** above any
`# %%` block (or `Shift+Enter`). Output appears in VS Code's Interactive Window,
one cell at a time — same experience as a notebook.

**As a notebook:** open the `.ipynb` file in VS Code or Jupyter Lab, select a
kernel, and **Run All**.

---

## Week 1 — Task 1: Environment Setup + First EDA

**Goal:** get comfortable with the toolkit and "listen" to a dataset before doing
anything to it — shape, quality, and quirks first.

**What it covers:**
- Loading data with `pandas.read_csv()`
- Inspecting structure with `.info()`, `.describe()`, `.head()`
- Counting missing values per column
- Classifying columns as categorical vs. numerical
- A short written "data story" summarizing first impressions

**Key findings:**

| | |
|---|---|
| Rows | 891 |
| Columns | 12 |
| Missing values | `Age` (177 missing, ~20%), `Cabin` (687 missing, ~77%), `Embarked` (2 missing) |
| Numerical columns | `Age`, `Fare`, `SibSp`, `Parch` |
| Categorical columns | `Survived`, `Pclass`, `Sex`, `Embarked`, `Name`, `Ticket`, `Cabin` |
| Target variable | `Survived` (binary: 0 = did not survive, 1 = survived) |

`Fare` and `Age` are both right-skewed, with `Fare` in particular having a long
tail of high-paying outliers. This is a binary classification problem at its
core, with a healthy mix of numeric and categorical predictors to work with.

---

## Week 1 — Task 2: Data Cleaning + Visualization

**Goal:** handle real-world messiness (missing values, outliers) properly, then
use visualization to catch mistakes and surface patterns before modeling.

**Missing value strategy** (justified, not just applied):

| Column | Missing | Strategy | Why |
|---|---|---|---|
| `Age` | ~20% | Fill with **median** | Right-skewed distribution — median resists the pull of outliers better than mean |
| `Embarked` | 2 rows | Fill with **mode** | Negligible amount of missing data, mode is a safe default |
| `Cabin` | ~77% | **Drop column**, keep a `HasCabin` flag | Too sparse to impute meaningfully; whether a cabin was *recorded at all* still carries signal, so it's kept as a binary feature instead of fabricating cabin values |

**Outlier detection:** `Fare` was checked with a boxplot and the IQR rule
(`Q1 - 1.5×IQR` to `Q3 + 1.5×IQR`). A meaningful chunk of passengers fall outside
that range — a handful paid 500+, over 10x the median fare — confirming what the
boxplot shows visually.

**Visualizations (4 required, all included):**
1. **Histogram** — `Age` distribution
2. **Boxplot** — `Fare`, for outlier detection
3. **Bar chart** — survival rate by `Pclass`
4. **Correlation heatmap** — all numeric features against each other and `Survived`

**Which feature most affects survival, and why?**

`Sex` is the strongest single predictor — women survived at a far higher rate
than men, reflecting the "women and children first" evacuation policy. `Pclass`
matters for a related but separate reason: first-class passengers were berthed
closer to the lifeboats and were prioritized during evacuation, so survival rate
drops steadily from 1st to 3rd class. `Fare` correlates with survival mostly
*because* it's a proxy for `Pclass`, not because paying more directly helped.
`Age` has a smaller effect (children fared slightly better), and family-size
features (`SibSp`, `Parch`) show only mild correlation — small families seem to
help a little, very large ones seem to hurt.

**Overall ranking:** `Sex` > `Pclass` ≈ `Fare` > `Age` > `SibSp` / `Parch`

This ranking will guide feature prioritization when modeling begins.

---

## Week 2 — Task 1: First Classification Model (Logistic Regression)

**Goal:** put the cleaned, explored dataset to work — train a model that
predicts `Survived` and evaluate it properly, not just by eyeballing accuracy.

**Approach:**

1. **Feature selection** — dropped `PassengerId` (just a row index), `Name` and
   `Ticket` (high-cardinality free text, out of scope for a first baseline
   model). Everything else from the cleaned dataset (Task 2's `df_clean`,
   including the engineered `HasCabin` flag) is kept.
2. **Encoding** — `Sex` and `Embarked` are one-hot encoded with
   `pd.get_dummies(..., drop_first=True)`. `drop_first` avoids the dummy
   variable trap: for a 2-category column like `Sex`, you only need one output
   column (`Sex_male`), since "not male" already implies female.
3. **Train/test split** — `train_test_split` with an 80/20 split and
   `stratify=y`, so the train and test sets preserve the same survival ratio
   as the full dataset. Without stratification, an unlucky split could leave
   the test set with a noticeably different survival rate than what the model
   trained on.
4. **Model** — `LogisticRegression` from scikit-learn. Chosen as a first
   baseline because it's simple, fast, and interpretable (coefficients map
   directly to how each feature pushes the prediction toward survive/not
   survive) — a sensible starting point before trying anything more complex.
5. **Evaluation** — `accuracy_score` for the headline number, plus a full
   confusion matrix to see *what kind* of mistakes the model makes, not just
   how many.

**Final accuracy:** `<fill in your actual accuracy here after running it — a
well-tuned Logistic Regression baseline on this dataset typically lands
somewhere in the 78–82% range>`

**Confusion matrix — what it tells us:**

|  | Predicted: Did not survive | Predicted: Survived |
|---|---|---|
| **Actual: Did not survive** | True Negative | False Positive |
| **Actual: Survived** | False Negative | True Positive |

Accuracy alone hides *which* kind of mistake the model makes. False positives
mean the model was overly optimistic (predicted survival for someone who
didn't); false negatives mean it was overly pessimistic (predicted death for
someone who survived). Because the model leans heavily on `Sex` and `Pclass`
(the two strongest signals identified in Task 2), its errors tend to cluster
around the exceptions to that pattern — men who survived, and women or
first-class passengers who didn't.

*(Fill in the actual TN/FP/FN/TP counts and accuracy once you run the notebook
on your machine — the numbers above describe the pattern, not the exact
values for your run.)*

---

## Week 2 — Task 2: First Regression Model (Linear Regression)

**Goal:** predict a continuous number instead of a category — regression
instead of classification — using the **Ames Housing** dataset (Kaggle
"House Prices - Advanced Regression Techniques" train set).

**Approach:**

1. **Feature selection** — the raw dataset has 81 columns, many with heavy
   missing values (`PoolQC`, `Fence`, `Alley` are mostly NA because most
   houses simply don't have those features). Rather than impute a large
   messy set, five complete, high-signal columns were picked:
   - `OverallQual` — overall material/finish quality (1–10); consistently the
     single strongest predictor of sale price in this dataset
   - `GrLivArea` — above-ground living area in square feet; the classic
     "bigger house, higher price" driver
   - `TotalBsmtSF` — total basement square footage
   - `GarageCars` — garage size in car capacity; a proxy for both garage size
     and overall home quality
   - `YearBuilt` — year the house was originally built; newer homes tend to
     sell for more, all else equal
2. **Train/test split** — `train_test_split` with an 80/20 split
   (`random_state=42` for reproducibility).
3. **Model** — `LinearRegression` from scikit-learn, trained on the five
   features to predict `SalePrice`.
4. **Evaluation** — RMSE (in dollars, so it's directly interpretable as
   "typical prediction error") and R² (share of price variation explained).
5. **Visualization** — a predicted-vs-actual scatter plot with a diagonal
   "perfect prediction" reference line, to see at a glance how tightly
   predictions track reality (and where the model over/under-shoots).

**Results:**

| Metric | Value |
|---|---|
| RMSE | ~$39,763 |
| R² | 0.79 |

**What R² = 0.79 means, in plain English:** the model explains about 79% of
why house prices differ from one home to another, using just five features
(quality, size, basement size, garage size, and age). The remaining ~21% comes
down to things the model doesn't see at all — neighborhood desirability,
kitchen/bathroom finish quality, lot shape, recent renovations, and general
market timing. It's not a grade out of 100 in the usual sense — it's a measure
of "how much of the story does this model capture," and 0.79 means the model
has genuinely learned a real pricing pattern, not just noise, even though it's
far from perfect.

*(These numbers come from a `random_state=42` split — re-running with a
different split will shift them slightly, but the overall pattern should
hold.)*

---

## Notes / gotchas

- If you're on **pandas 4**, `df.describe(include="str")` is the correct call
  now (older `include="object"` still works but throws a deprecation warning).
- `seaborn` is a separate install from `matplotlib` — `pip install seaborn` if
  you hit a `ModuleNotFoundError`.
- Both Week 2 tasks require `scikit-learn` in addition to everything else:
  ```bash
  pip install scikit-learn
  ```

## What's next

- Try other classifiers (Random Forest, SVM) and compare against the Logistic
  Regression baseline
- Try other regressors (Random Forest, Gradient Boosting) and compare against
  the Linear Regression baseline
- Feature engineering (e.g. extracting titles from `Name`, family size from
  `SibSp` + `Parch` for Titanic; total square footage, house age at sale for
  Ames Housing)
- Hyperparameter tuning and cross-validation instead of a single train/test
  split