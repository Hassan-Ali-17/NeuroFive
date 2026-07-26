import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_ensemble_comparison():
    # Resolve CSV path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "train.csv")

    print("--- 1. LOADING DATA ---")
    df = pd.read_csv(csv_path)
    print(f"Loaded Titanic dataset. Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Cast Pclass to string to make it categorical
    df['Pclass'] = df['Pclass'].astype(str)

    # ----------------------------------------------------
    # 2. PREPROCESSING & CLEANING
    # ----------------------------------------------------
    print("\n--- 2. PREPROCESSING DATA ---")
    
    # Drop columns that are IDs or text with high cardinality
    df_clean = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    
    # Handle missing values
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
    
    # One-hot encode categorical features
    categorical_cols = ['Pclass', 'Sex', 'Embarked']
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    
    # Define target and features
    y = df_encoded['Survived']
    X = df_encoded.drop(columns=['Survived'])
    
    # Save feature names before scaling
    feature_names = X.columns.tolist()

    # Split dataset (80% train, 20% test, stratified on Survival)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
    print(f"Features: {feature_names}")

    # Scale numerical columns for Logistic Regression (and keep them scaled for simplicity across models)
    scaler = StandardScaler()
    num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # ----------------------------------------------------
    # 3. TRAINING MODELS
    # ----------------------------------------------------
    print("\n--- 3. TRAINING MODELS ---")
    
    # 1. Single Model: Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # 2. Ensemble Model: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # 3. Ensemble Model: XGBoost
    xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    
    print("All models trained successfully.")

    # ----------------------------------------------------
    # 4. PERFORMANCE COMPARISON
    # ----------------------------------------------------
    print("\n--- 4. PERFORMANCE COMPARISON ---")
    
    models = {
        "Logistic Regression (Baseline)": lr_model,
        "Random Forest (Ensemble)": rf_model,
        "XGBoost (Ensemble)": xgb_model
    }
    
    def fnz(val):
        return f"{val:.4f}"

    comparison_data = []
    
    for name, model in models.items():
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        comparison_data.append({
            "Model": name,
            "Accuracy": fnz(acc),
            "Precision": fnz(prec),
            "Recall": fnz(rec),
            "F1-Score": fnz(f1),
            "ROC-AUC": fnz(auc)
        })
        
    df_comparison = pd.DataFrame(comparison_data)
    # Format properly
    print(df_comparison.to_string(index=False))

    # ----------------------------------------------------
    # 5. FEATURE IMPORTANCES PLOTTING
    # ----------------------------------------------------
    print("\n--- 5. PLOTTING FEATURE IMPORTANCES ---")
    
    rf_importances = rf_model.feature_importances_
    xgb_importances = xgb_model.feature_importances_
    
    df_rf = pd.DataFrame({'Feature': feature_names, 'Importance': rf_importances}).sort_values(by='Importance', ascending=False)
    df_xgb = pd.DataFrame({'Feature': feature_names, 'Importance': xgb_importances}).sort_values(by='Importance', ascending=False)
    
    # Set up matplotlib style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Random Forest Importances
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=df_rf.head(10), 
        ax=axes[0], 
        hue='Feature',
        palette='viridis',
        legend=False
    )
    axes[0].set_title('Top 10 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Importance Score')
    axes[0].set_ylabel('Features')
    
    # Plot XGBoost Importances
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=df_xgb.head(10), 
        ax=axes[1], 
        hue='Feature',
        palette='magma',
        legend=False
    )
    axes[1].set_title('Top 10 Feature Importances - XGBoost', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Importance Score')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plot_filename = os.path.join(script_dir, "feature_importances.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved feature importance comparison plot to: {plot_filename}")

    # ----------------------------------------------------
    # 6. ENSEMBLE CONCEPT EXPLANATION
    # ----------------------------------------------------
    print("\n--- 6. MODEL COMBINING EXPLANATION ---")
    explanation = (
        "Random Forest and XGBoost combine individual tree models using fundamentally different strategies. "
        "Random Forest utilizes Bagging (Bootstrap Aggregation), building many deep, fully-grown decision trees in parallel "
        "and independently on random bootstrapped subsets of the data; it then averages their predictions to reduce overall model variance. "
        "In contrast, XGBoost utilizes Boosting, building shallow, weak decision trees sequentially rather than in parallel. "
        "Each consecutive tree is specifically trained to fit and correct the residual errors (gradients of the loss function) "
        "left behind by the cumulative combination of all previous trees, thereby systematically reducing model bias."
    )
    print(explanation)

if __name__ == "__main__":
    run_ensemble_comparison()
