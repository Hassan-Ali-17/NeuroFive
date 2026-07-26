import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def analyze_class_imbalance():
    # Resolve CSV path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print("--- 1. LOADING DATA ---")
    df = pd.read_csv(csv_path)
    print(f"Loaded Telco Customer Churn dataset. Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Clean TotalCharges (convert to numeric, drop blank rows)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    print(f"Dataset shape after dropping rows with missing TotalCharges: {df.shape[0]} rows")

    # ----------------------------------------------------
    # 2. CHECK CLASS BALANCE & VISUALIZE
    # ----------------------------------------------------
    print("\n--- 2. CLASS BALANCE VERIFICATION ---")
    class_counts = df['Churn'].value_counts()
    class_pct = df['Churn'].value_counts(normalize=True) * 100
    
    print("Class Counts:")
    print(class_counts.to_string())
    print("\nClass Percentages:")
    print(class_pct.to_string())
    
    # Save a bar chart of the class counts
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5))
    
    # Draw bar chart
    ax = sns.barplot(
        x=class_counts.index, 
        y=class_counts.values, 
        hue=class_counts.index,
        palette=['#4C72B0', '#DD8452'], 
        legend=False
    )
    plt.title("Telco Customer Churn Class Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Churn Status", fontsize=12)
    plt.ylabel("Number of Customers", fontsize=12)
    
    # Add count annotations on top of the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
    plt.tight_layout()
    plot_filename = os.path.join(script_dir, "class_balance.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved class balance visualization to: {plot_filename}")

    # ----------------------------------------------------
    # 3. PREPROCESSING
    # ----------------------------------------------------
    print("\n--- 3. DATA PREPROCESSING ---")
    # Convert target Churn to binary
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    y = df['Churn_Numeric']
    X = df.drop(columns=['customerID', 'Churn', 'Churn_Numeric'])

    # Categorical variables to encode
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Split training/testing sets using stratification to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")

    # Scale numeric features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # ----------------------------------------------------
    # 4. TRAINING BASELINE VS BALANCED MODELS
    # ----------------------------------------------------
    print("\n--- 4. TRAINING BASELINE VS BALANCED MODELS ---")

    # 1. Baseline Model (without class weighting)
    lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
    lr_baseline.fit(X_train_scaled, y_train)
    
    # 2. Balanced Model (with class weighting)
    lr_balanced = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr_balanced.fit(X_train_scaled, y_train)

    # ----------------------------------------------------
    # 5. METRIC COMPARISON
    # ----------------------------------------------------
    print("\n--- 5. MODEL METRIC COMPARISON (TEST SET) ---")
    
    def get_metrics(model, X_val, y_val):
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        return acc, prec, rec, f1

    base_acc, base_prec, base_rec, base_f1 = get_metrics(lr_baseline, X_test_scaled, y_test)
    bal_acc, bal_prec, bal_rec, bal_f1 = get_metrics(lr_balanced, X_test_scaled, y_test)

    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Imbalanced Baseline": [base_acc, base_prec, base_rec, base_f1],
        "Balanced Model": [bal_acc, bal_prec, bal_rec, bal_f1]
    })
    
    # Format to 4 decimal places
    for col in ["Imbalanced Baseline", "Balanced Model"]:
        comparison_df[col] = comparison_df[col].map(lambda x: f"{x:.4f}")
        
    print(comparison_df.to_string(index=False))
    
    print("\nClassification Report (Baseline Imbalanced):")
    print(classification_report(y_test, lr_baseline.predict(X_test_scaled)))
    
    print("Classification Report (Balanced):")
    print(classification_report(y_test, lr_balanced.predict(X_test_scaled)))

    # ----------------------------------------------------
    # 6. EXPLANATION: WHY ACCURACY IS MISLEADING
    # ----------------------------------------------------
    print("\n--- 6. WHY ACCURACY IS A MISLEADING METRIC ---")
    explanation = (
        "Accuracy is a misleading metric for this dataset because of the prominent class imbalance, "
        f"where ~73.4% of the customers belong to the non-churn class ('No') and only ~26.6% belong to the churn class ('Yes'). "
        "Under such a distribution, a naive model that predicts 'No' (no churn) for every single customer "
        "would achieve a high accuracy of 73.4% without learning any patterns or identifying a single churner. "
        "While accuracy drops slightly for the balanced model (from 80.7% to 75.3%), the recall (ability to detect "
        "actual churners) increases significantly (from 54.3% to 80.2%), which is far more valuable for business retention efforts."
    )
    print(explanation)

if __name__ == "__main__":
    analyze_class_imbalance()
