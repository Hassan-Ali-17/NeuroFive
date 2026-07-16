# NeuroFive ML Track — Week 1

My progress through the NeuroFive Solutions ML track. Week 1 covers environment
setup, first exploratory data analysis (EDA), data cleaning, and visualization —
all using the Titanic dataset.

## Repo structure

```
neurofive-ml-track/
├── data/
│   └── train.csv          # Titanic dataset (not committed — see below)
├── eda_titanic.ipynb      # Jupyter notebook version
├── eda_titanic.py         # Same content as a plain script (VS Code # %% cells)
├── .gitignore
└── README.md
```

`eda_titanic.py` and `eda_titanic.ipynb` contain identical content — the script
uses `# %%` cell markers so it runs interactively in VS Code without needing a
notebook file, and the notebook is generated from it for anyone who wants the
classic Jupyter format with saved outputs.

## Setup

```bash
pip install pandas numpy matplotlib seaborn
```

No virtual environment required — a global Python install works fine for this.

### Dataset

Download `train.csv` from the Kaggle competition
["Titanic - Machine Learning from Disaster"](https://www.kaggle.com/c/titanic)
(free Kaggle account required) and place it at `data/train.csv`. The dataset
itself isn't committed to this repo — only the code that processes it.

### Running it

**As a script in VS Code:** open `eda_titanic.py`, click **Run Cell** above any
`# %%` block (or `Shift+Enter`). Output appears in VS Code's Interactive Window,
one cell at a time — same experience as a notebook.

**As a notebook:** open `eda_titanic.ipynb` in VS Code or Jupyter Lab, select a
kernel, and **Run All**.

---

## Task 1 — Environment Setup + First EDA

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

## Task 2 — Data Cleaning + Visualization

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

## Notes / gotchas

- If you're on **pandas 4**, `df.describe(include="str")` is the correct call
  now (older `include="object"` still works but throws a deprecation warning).
- `seaborn` is a separate install from `matplotlib` — `pip install seaborn` if
  you hit a `ModuleNotFoundError`.

## What's next

- Feature engineering (e.g. extracting titles from `Name`, family size from
  `SibSp` + `Parch`)
- Encoding categorical variables for modeling
- Baseline classification model (logistic regression) as a starting benchmark