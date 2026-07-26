```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗██╗██╗   ██╗███████╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝██║██║   ██║██╔════╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║█████╗  ██║██║   ██║█████╗
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  ██║╚██╗ ██╔╝██╔══╝
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██║     ██║ ╚████╔╝ ███████╗
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚═══╝  ╚══════╝

## ML Track — Week 1 to Week 5

My progress through the NeuroFive Solutions ML track:
- **Week 1:** Environment setup, first exploratory data analysis (EDA), data cleaning, and visualization.
- **Week 2:** Classification model (Titanic) and Regression model (Ames Housing).
- **Week 3:** Advanced classification metrics, cross-validated Grid Search, and Telco Customer Churn.
- **Week 4:** Pipelines, ColumnTransformers, custom feature engineering, and ensemble comparison (RandomForest vs XGBoost).
- **Week 5:** Class imbalance handling (class weighting) on Telco Churn, metric analysis, and building/deploying a Streamlit web app for Titanic Survival Prediction.

## Repo structure

```
neurofive-ml-track/
├── week1/
│   ├── eda_titanic.ipynb              # Jupyter notebook version of Titanic EDA
│   ├── task1.py                       # Python script version
│   └── train.csv                      # Titanic dataset
├── week2/
│   ├── eda_titanic (week 2).ipynb     # Titanic modeling and analysis notebook
│   ├── housing_price_regression.ipynb # House price regression notebook
│   ├── task1.py                       # Titanic classification model script
│   ├── task2.py                       # House price regression script
│   ├── train house.csv                # Ames Housing dataset
│   └── train.csv                      # Titanic dataset
├── week3/
│   ├── task1.py                       # Grid Search and classification report script
│   ├── task2.py                       # Telco Customer Churn analysis (EDA, models, comparison)
│   ├── train.csv                      # Titanic dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Telco Customer Churn dataset
├── week4/
│   ├── feature_importances.png        # Feature importances comparison plot
│   ├── task1.py                       # Titanic classification pipeline & feature engineering
│   ├── task2.py                       # RandomForest vs XGBoost model comparison
│   ├── titanic_pipeline.joblib        # Serialized pipeline model
│   └── train.csv                      # Titanic dataset
├── week5/
│   ├── class_balance.png              # Churn class distribution plot
│   ├── task1.py                       # Churn class imbalance handling & metrics comparison
│   ├── task2.py                       # Titanic survival predictor Streamlit app
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Telco Customer Churn dataset
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

**Final accuracy:** **80.45%**

**Confusion matrix — what it tells us:**

|  | Predicted: Did not survive | Predicted: Survived |
|---|---|---|
| **Actual: Did not survive** | **96** (True Negative) | **14** (False Positive) |
| **Actual: Survived** | **21** (False Negative) | **48** (True Positive) |

Accuracy alone hides *which* kind of mistake the model makes. False positives
mean the model was overly optimistic (predicted survival for someone who
didn't); false negatives mean it was overly pessimistic (predicted death for
someone who survived). Because the model leans heavily on `Sex` and `Pclass`
(the two strongest signals identified in Task 2), its errors tend to cluster
around the exceptions to that pattern — men who survived, and women or
first-class passengers who didn't.

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

## Week 3 — Task 1: Classification Metrics & Hyperparameter Tuning

**Goal:** Evaluate a classification model using advanced metrics (Precision, Recall, F1-score) and perform hyperparameter tuning using cross-validated Grid Search.

### Why Accuracy Alone is Misleading for Imbalanced Datasets
Accuracy measures the proportion of correct predictions out of all predictions. While simple, it can be extremely misleading when classes are highly imbalanced. For example, if a dataset contains 95% non-survivors and 5% survivors, a naive classifier that predicts "did not survive" for everyone will achieve 95% accuracy. However, this model completely fails to identify survivors (Recall of 0% and Precision of 0% for survivors). 
By evaluating metrics like **Precision** (avoiding false positives), **Recall** (avoiding false negatives), and **F1-score** (harmonic mean of the two), we get a realistic picture of how the model performs on the minority class of interest.

### Tuning & Grid Search
We tuned the Logistic Regression model using `GridSearchCV` with 5-fold cross-validation. The tuned hyperparameters were:
- `C` (Inverse of regularization strength): tested `[0.001, 0.01, 0.1, 1, 10, 100]`
- `penalty` (Regularization type): tested `['l1', 'l2']` with `liblinear` solver.

The grid search identified the best parameters on the training set:
- **Best Hyperparameters:** `{'C': 0.1, 'penalty': 'l2'}`
- **Best Cross-Validation (CV) Accuracy:** 79.79%

### Performance Comparison: Before vs After
Below is the comparison of the baseline model (default parameters, `C=1.0`, `penalty='l2'`) and the tuned model (`C=0.1`, `penalty='l2'`) evaluated on the holdout test set (20% of the data):

| Metric | Baseline Model | Tuned Model |
|---|---|---|
| **Accuracy** | 80.45% | 78.77% |
| **Precision (Survived)** | 77.42% | 78.18% |
| **Recall (Survived)** | 69.57% | 62.32% |
| **F1-Score (Survived)** | 73.28% | 69.35% |

*Note on results:* Although the tuned model achieved a better and more robust cross-validation score during search, the test accuracy was slightly lower than the baseline. This can happen due to minor variance on a relatively small test set (179 samples), or because the baseline's default regularization strength happened to align slightly better with the test set sample split.

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

---

## Week 3 — Task 2: Telco Customer Churn EDA & Model Comparison

**Goal:** Perform EDA, address class imbalance, train Logistic Regression and Decision Tree models, identify top drivers of churn, and write a business summary.

### Exploratory Data Analysis & Class Imbalance
- **Dataset**: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7043 rows, 21 columns).
- **Cleaning**: Cleaned empty spaces in `TotalCharges` (dropped 11 rows with 0 tenure, leaving 7032 rows).
- **Class Imbalance**: 73.4% of customers have not churned ("No"), while 26.6% have churned ("Yes"). This significant class imbalance was addressed by utilizing `class_weight='balanced'` in model training.
- **Key EDA Findings**:
  - `tenure` has a strong negative correlation (-0.354) with churn (longer tenure = lower churn).
  - Customers on Month-to-month contracts have a much higher churn rate (**42.7%**) than those on One-year (**11.3%**) or Two-year (**2.8%**) contracts.
  - Fiber optic internet users churn at a rate of **41.9%**.

### Model Performance Comparison
Both models were trained using class balancing to account for the minority class:

| Metric | Logistic Regression (Balanced) | Decision Tree (Depth=5, Balanced) |
|---|---|---|
| **Accuracy** | 72.49% | **73.35%** |
| **Precision** | 48.93% | **49.92%** |
| **Recall (Sensitivity)** | **79.68%** | 79.41% |
| **F1-Score** | 60.63% | **61.30%** |
| **ROC-AUC** | **83.52%** | 82.76% |

- **Verdict**: Logistic Regression provided better overall class separation (higher ROC-AUC) and slightly better Recall (which is essential for proactively flagging churners).

### Top 3 Drivers of Churn (Decision Tree)
1. **`Contract_Month-to-month`** (Importance: 61.52%)
2. **`tenure`** (Importance: 11.00%)
3. **`InternetService_Fiber optic`** (Importance: 9.86%)

---

## Week 4 — Task 1: Scikit-Learn Pipeline & Custom Feature Engineering

**Goal:** Build a robust, unified Scikit-Learn pipeline using `ColumnTransformer`, engineer custom features, and serialize the final model.

### Preprocessing Pipeline Structure
- **Numerical features** (`Age`, `Fare`, `SibSp`, `Parch`): Imputed with median values and scaled using `StandardScaler`.
- **Categorical features** (`Pclass`, `Sex`, `Embarked`): Imputed with the most frequent value and encoded using `OneHotEncoder`.
- **Validation**: Confirmed that the `ColumnTransformer` Pipeline produces identical accuracy (**80.45%**) and ROC-AUC (**84.20%**) to a manual preprocessing flow.

### Custom Feature Engineering
We implemented a custom transformer `TitanicFeatureExtractor` that constructs:
1. **`FamilySize`** = `SibSp + Parch + 1` (combines sibling/spouse and parent/child features).
2. **`IsAlone`** = `1` if `FamilySize == 1` else `0`.
3. **`Title`** = Extracted titles from Name (e.g. Mr, Mrs, Miss, Master, Other) to represent passenger status.

By incorporating these features and regularizing our model (`max_depth=6` to prevent overfitting), we achieved the following performance on the hold-out test set:

| Model / Pipeline | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Baseline Pipeline** | 80.45% | 79.37% | 66.67% | 72.46% | 84.20% |
| **Engineered Pipeline** | **82.12%** | **81.36%** | **69.57%** | **75.00%** | **85.67%** |
| **Improvement** | **+1.68%** | **+1.99%** | **+2.90%** | **+2.54%** | **+1.47%** |

### Serialization
- The final feature-engineered pipeline is serialized to `week4/titanic_pipeline.joblib` using `joblib`.
- Reloading the pipeline via `joblib.load` verified that it reproduces the exact same predictions on new data.

---

## Week 4 — Task 2: RandomForest vs. XGBoost Model Comparison

**Goal:** Train RandomForest and XGBoost ensemble models, compare their performance against a single model (Logistic Regression baseline), and plot/evaluate feature importances.

### Model Performance Comparison
All models were evaluated on the stratified Titanic test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 80.45% | 79.31% | **66.67%** | **72.44%** | **84.32%** |
| **Random Forest (Ensemble)** | 79.89% | **83.67%** | 59.42% | 69.49% | 84.15% |
| **XGBoost (Ensemble)** | 80.45% | 80.36% | 65.22% | 72.00% | 81.22% |

### Feature Importances Comparison
The feature importances plot is saved at [feature_importances.png](file:///c:/Users/User/OneDrive%20-%20Higher%20Education%20Commission/Desktop/NeuroFive/week4/feature_importances.png).
- **Random Forest** placed the highest importance on `Sex_male` (approx. 43%), followed by `Fare` (approx. 18%) and `Age` (approx. 12%).
- **XGBoost** also heavily prioritized `Sex_male` (approx. 56%), followed by `Pclass_3` (approx. 17%) and `Fare` (approx. 7%).

### Ensemble Model Aggregation Comparison
- **Random Forest (Bagging):** Fits independent, fully-grown decision trees in parallel on bootstrapped subsets of the training data. The trees' final predictions are combined via voting or averaging, reducing model variance without increasing bias.
- **XGBoost (Boosting):** Fits shallow, weak decision trees sequentially rather than in parallel. Each subsequent tree is trained to predict the residual errors (gradients of the loss function) of the previous ensemble, sequentially reducing model bias.

---

## Week 5 — Task 1: Handling Class Imbalanced Data on Telco Churn

**Goal:** Check and visualize the class imbalance of the Telco Churn dataset, apply class weighting (`class_weight='balanced'`) to a Logistic Regression model, and compare performance metrics before and after.

### Class Imbalance Visualization
The class distribution is visualized in [class_balance.png](file:///c:/Users/User/OneDrive%20-%20Higher%20Education%20Commission/Desktop/NeuroFive/week5/class_balance.png).
- **No Churn (Class 0):** 5163 customers (73.42%)
- **Churn (Class 1):** 1869 customers (26.58%)

### Model Performance Before vs. After Class Balancing
Evaluating a Logistic Regression model on the test set:

| Metric | Imbalanced Baseline | Balanced Model |
| :--- | :--- | :--- |
| **Accuracy** | **80.45%** | 72.57% |
| **Precision (Churn)** | **65.05%** | 49.01% |
| **Recall (Churn)** | 57.22% | **79.41%** |
| **F1-Score (Churn)** | **60.88%** | 60.61% |

- *Observation*: While the overall Accuracy drops, the **Recall (sensitivity)** for predicting churners jumps dramatically from **57.22%** to **79.41%**, capturing more at-risk customers.

### Why Accuracy is a Misleading Metric
Accuracy is highly misleading for imbalanced datasets because a naive model that always predicts the majority class ("No Churn") will achieve **73.42%** accuracy without identifying a single actual churner. Thus, accuracy rewards models for ignoring the minority class. In churn prediction, missing a customer who is about to leave (a false negative) carries a high business cost, making **Recall** and **F1-Score** much more informative performance indicators than overall accuracy.

---

## Week 5 — Task 2: Titanic Survival Predictor Streamlit Web App

**Goal:** Build an interactive, premium web application with Streamlit to load our best-performing pipeline and predict passenger survival, then deploy it for free.

### App Description
The web app load the serialized Titanic pipeline [titanic_pipeline.joblib](file:///c:/Users/User/OneDrive%20-%20Higher%20Education%20Commission/Desktop/NeuroFive/week4/titanic_pipeline.joblib) from Week 4. It includes the custom feature engineering step (`TitanicFeatureExtractor`) to transform raw inputs dynamically.
The user enters passenger demographic details (Name, Title, Sex, Age) and travel details (Class, Port, Family, Fare), and the model displays whether they survived, including the exact probability.

* **Live Demo:** [Click here to view the app](https://neurofive-hassan.streamlit.app/) *(Replace with actual deployed URL once deployed)*

### How to Run Locally

1. Install dependencies:
   ```bash
   pip install streamlit pandas joblib scikit-learn
   ```
2. Navigate to the `week5` directory and run Streamlit:
   ```bash
   streamlit run task2.py
   ```