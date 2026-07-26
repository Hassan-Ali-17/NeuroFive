import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ----------------------------------------------------
# CUSTOM TRANSFORMER FOR FEATURE ENGINEERING
# ----------------------------------------------------
class TitanicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer features:
    1. FamilySize = SibSp + Parch + 1
    2. IsAlone = 1 if FamilySize == 1 else 0
    3. Title = Extracted and cleaned titles from passenger Name
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Work on a copy to prevent warnings
        X_out = X.copy()
        
        # Feature 1: FamilySize
        X_out['FamilySize'] = X_out['SibSp'] + X_out['Parch'] + 1
        
        # Feature 2: IsAlone
        X_out['IsAlone'] = (X_out['FamilySize'] == 1).astype(int)
        
        # Feature 3: Title
        X_out['Title'] = X_out['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Countess', 'Ms', 'Lady', 'Jonkheer', 'Don', 'Mme', 'Capt', 'Sir']
        X_out['Title'] = X_out['Title'].replace(rare_titles, 'Other')
        X_out['Title'] = X_out['Title'].fillna('Mr')
        
        return X_out

def run_pipeline_demo():
    # Resolve CSV path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "train.csv")

    print("--- 1. LOADING DATA ---")
    df = pd.read_csv(csv_path)
    print(f"Loaded Titanic dataset. Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Cast Pclass to string to make it categorical
    df['Pclass'] = df['Pclass'].astype(str)

    # Define target and features
    y = df['Survived']
    # Drop features we are not using directly or are extracting from
    X = df.drop(columns=['PassengerId', 'Survived'])

    # Train/Test Split (80% train, 20% test, stratified on Survival)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

    # Baseline feature categories
    num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    cat_cols = ['Pclass', 'Sex', 'Embarked']

    # ----------------------------------------------------
    # 2. MANUAL PREPROCESSING FLOW (BASELINE)
    # ----------------------------------------------------
    print("\n--- 2. MANUAL PREPROCESSING BASELINE ---")
    
    # Impute numerical columns
    imputer_num = SimpleImputer(strategy='median')
    X_train_num_imp = imputer_num.fit_transform(X_train[num_cols])
    X_test_num_imp = imputer_num.transform(X_test[num_cols])
    
    # Scale numerical columns
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num_imp)
    X_test_num_scaled = scaler.transform(X_test_num_imp)
    
    # Impute categorical columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train_cat_imp = imputer_cat.fit_transform(X_train[cat_cols])
    X_test_cat_imp = imputer_cat.transform(X_test[cat_cols])
    
    # One-hot encode categorical columns
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat_encoded = ohe.fit_transform(X_train_cat_imp)
    X_test_cat_encoded = ohe.transform(X_test_cat_imp)
    
    # Combine features
    X_train_manual = np.hstack([X_train_num_scaled, X_train_cat_encoded])
    X_test_manual = np.hstack([X_test_num_scaled, X_test_cat_encoded])
    
    # Train baseline Classifier (Random Forest)
    clf_manual = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf_manual.fit(X_train_manual, y_train)
    manual_preds = clf_manual.predict(X_test_manual)
    manual_probs = clf_manual.predict_proba(X_test_manual)[:, 1]
    
    manual_acc = accuracy_score(y_test, manual_preds)
    manual_auc = roc_auc_score(y_test, manual_probs)
    print(f"Manual Baseline Accuracy: {manual_acc:.4f}")
    print(f"Manual Baseline ROC-AUC:  {manual_auc:.4f}")

    # ----------------------------------------------------
    # 3. BASELINE PIPELINE WITH COLUMNTRANSFORMER
    # ----------------------------------------------------
    print("\n--- 3. BASELINE PIPELINE WITH COLUMNTRANSFORMER ---")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )
    
    baseline_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    
    # Fit & Predict
    baseline_pipeline.fit(X_train, y_train)
    pipe_preds = baseline_pipeline.predict(X_test)
    pipe_probs = baseline_pipeline.predict_proba(X_test)[:, 1]
    
    pipe_acc = accuracy_score(y_test, pipe_preds)
    pipe_auc = roc_auc_score(y_test, pipe_probs)
    print(f"Pipeline Baseline Accuracy: {pipe_acc:.4f}")
    print(f"Pipeline Baseline ROC-AUC:  {pipe_auc:.4f}")
    
    # Confirm they are exactly the same
    assert np.allclose(pipe_preds, manual_preds), "Predictions mismatch between manual and pipeline!"
    assert np.allclose(pipe_probs, manual_probs), "Prediction probabilities mismatch!"
    print("SUCCESS: Pipeline matches manual baseline performance exactly.")

    # ----------------------------------------------------
    # 4. PIPELINE WITH CUSTOM FEATURE ENGINEERING
    # ----------------------------------------------------
    print("\n--- 4. PIPELINE WITH CUSTOM FEATURE ENGINEERING ---")
    
    # Define columns after feature extractor runs
    num_cols_eng = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
    cat_cols_eng = ['Pclass', 'Sex', 'Embarked', 'Title']
    
    preprocessor_eng = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols_eng),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols_eng)
        ]
    )
    
    # Create the full engineered pipeline
    engineered_pipeline = Pipeline(steps=[
        ('feat_eng', TitanicFeatureExtractor()),
        ('preprocessor', preprocessor_eng),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    
    # Fit & Predict
    engineered_pipeline.fit(X_train, y_train)
    eng_preds = engineered_pipeline.predict(X_test)
    eng_probs = engineered_pipeline.predict_proba(X_test)[:, 1]
    
    eng_acc = accuracy_score(y_test, eng_preds)
    eng_prec = precision_score(y_test, eng_preds)
    eng_rec = recall_score(y_test, eng_preds)
    eng_f1 = f1_score(y_test, eng_preds)
    eng_auc = roc_auc_score(y_test, eng_probs)
    
    print(f"Engineered Pipeline Accuracy:  {eng_acc:.4f}")
    print(f"Engineered Pipeline Precision: {eng_prec:.4f}")
    print(f"Engineered Pipeline Recall:    {eng_rec:.4f}")
    print(f"Engineered Pipeline F1-Score:  {eng_f1:.4f}")
    print(f"Engineered Pipeline ROC-AUC:   {eng_auc:.4f}")
    
    # Check for improvement
    improvement = eng_acc - pipe_acc
    print(f"\nAccuracy Change: {improvement:+.4f} (Baseline: {pipe_acc:.4f} -> Engineered: {eng_acc:.4f})")

    # ----------------------------------------------------
    # 5. MODEL SERIALIZATION
    # ----------------------------------------------------
    print("\n--- 5. MODEL SERIALIZATION ---")
    model_filename = os.path.join(script_dir, "titanic_pipeline.joblib")
    
    # Save the pipeline
    joblib.dump(engineered_pipeline, model_filename)
    print(f"Saved final pipeline to: {model_filename}")
    
    # Verify reloading
    loaded_pipeline = joblib.load(model_filename)
    loaded_preds = loaded_pipeline.predict(X_test)
    assert np.allclose(loaded_preds, eng_preds), "Reloaded model predictions mismatch!"
    print("SUCCESS: Saved model reloads successfully and produces identical predictions.")

if __name__ == "__main__":
    run_pipeline_demo()
