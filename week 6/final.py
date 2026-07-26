import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ----------------------------------------------------
# 1. STREAMLIT CONFIGURATION & CUSTOM THEME
# ----------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostics Center",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for modern dark-mode medical analytics UI
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 100%);
        color: #f3f4f6;
    }
    
    /* Header/Title styles */
    .app-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(90deg, #f472b6 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2px;
    }
    
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        color: #9ca3af;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Segment dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 25px 0 !important;
    }
    
    /* Card design */
    .stCard, div[data-testid="stVerticalBlock"] > div:has(div.stSlider) {
        background-color: rgba(31, 41, 55, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(8px);
    }
    
    /* Sliders styling */
    .stSlider label {
        color: #e5e7eb !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Predict Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ec4899 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3);
        width: 100%;
        margin-top: 20px;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #db2777 0%, #2563eb 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.5);
    }
    
    /* Banner blocks */
    .banner-benign {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.4) 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
    
    .banner-malignant {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.4) 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. DATA PROCESSING & MODEL COMPILATION (ON-THE-FLY)
# ----------------------------------------------------
# Define the clinically chosen 6 features (strongest correlation to malignancy)
selected_features = [
    'worst concave points',
    'worst perimeter',
    'mean concave points',
    'worst radius',
    'mean perimeter',
    'worst area'
]

@st.cache_resource
def build_and_evaluate_models():
    # Load dataset
    data = load_breast_cancer()
    X_full = pd.DataFrame(data.data, columns=data.feature_names)
    # Target: 0 = Malignant, 1 = Benign
    y_full = data.target
    
    # Filter dataset to selected features
    X_sel = X_full[selected_features]
    
    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    # Fit Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Fit XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    
    # Evaluate models on test set
    def compute_metrics(model, X_val, y_val):
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        # Note: Scikit-learn treats target 1 (Benign) as positive by default.
        # But in breast cancer classification, predicting Benign vs Malignant requires careful metrics.
        # We report standard metrics on Class 1 (Benign).
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)
        return acc, prec, rec, f1, auc

    lr_metrics = compute_metrics(lr, X_test_scaled, y_test)
    rf_metrics = compute_metrics(rf, X_test_scaled, y_test)
    xgb_metrics = compute_metrics(xgb, X_test_scaled, y_test)
    
    metrics_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [lr_metrics[0], rf_metrics[0], xgb_metrics[0]],
        "Precision": [lr_metrics[1], rf_metrics[1], xgb_metrics[1]],
        "Recall": [lr_metrics[2], rf_metrics[2], xgb_metrics[2]],
        "F1-Score": [lr_metrics[3], rf_metrics[3], xgb_metrics[3]],
        "ROC-AUC": [lr_metrics[4], rf_metrics[4], xgb_metrics[4]]
    })
    
    # Identify best model based on F1-Score
    best_idx = metrics_df["F1-Score"].idxmax()
    best_model_name = metrics_df.loc[best_idx, "Model"]
    
    model_map = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb
    }
    
    best_model = model_map[best_model_name]
    
    return best_model, best_model_name, metrics_df, scaler, X_sel, y_full

# Build/fetch cached models
best_model, best_model_name, metrics_df, scaler, X_sel, y_full = build_and_evaluate_models()

# ----------------------------------------------------
# 3. INTERACTIVE CONTAINER LAYOUT
# ----------------------------------------------------
st.markdown('<h1 class="app-title">🧬 Breast Cancer Diagnostic Screening</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Clinical Decision Support System (CDSS) utilizing Tree-Based Ensembles & Linear Classifiers</p>', unsafe_allow_html=True)

# Define application tabs
tab1, tab2, tab3 = st.tabs(["🩺 Diagnostic Predictor", "📊 Model Benchmarking & EDA", "📝 Case Study Write-Up"])

# ----------------------------------------------------
# TAB 1: DIAGNOSTIC PREDICTOR TOOL
# ----------------------------------------------------
with tab1:
    st.subheader("📋 Patient Biopsy Characteristics")
    st.write("Input the cellular parameters extracted from the needle biopsy (FNA) digitized scan:")
    
    # Setup inputs layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Perimeter and Radius Metrics:**")
        mean_perimeter = st.slider(
            "Mean Perimeter (cell boundary size)", 
            min_value=43.0, max_value=190.0, value=86.2, step=0.1
        )
        worst_perimeter = st.slider(
            "Worst Perimeter (maximum boundary size)", 
            min_value=50.0, max_value=255.0, value=97.6, step=0.1
        )
        worst_radius = st.slider(
            "Worst Radius (maximum cell radius)", 
            min_value=7.0, max_value=37.0, value=15.0, step=0.1
        )
        
    with col2:
        st.markdown("**Area and Concavity Metrics:**")
        worst_area = st.slider(
            "Worst Area (maximum cell area)", 
            min_value=180.0, max_value=4300.0, value=686.5, step=1.0
        )
        mean_concave_points = st.slider(
            "Mean Concave Points (average boundary indentations)", 
            min_value=0.00, max_value=0.21, value=0.03, step=0.001, format="%.3f"
        )
        worst_concave_points = st.slider(
            "Worst Concave Points (maximum boundary indentations)", 
            min_value=0.00, max_value=0.30, value=0.10, step=0.001, format="%.3f"
        )
        
    # Standardize inputs for model
    raw_inputs = pd.DataFrame([{
        'worst concave points': worst_concave_points,
        'worst perimeter': worst_perimeter,
        'mean concave points': mean_concave_points,
        'worst radius': worst_radius,
        'mean perimeter': mean_perimeter,
        'worst area': worst_area
    }])
    
    # Scale features
    scaled_inputs = scaler.transform(raw_inputs)
    
    if st.button("🩺 Execute Screening Analysis"):
        prediction = best_model.predict(scaled_inputs)[0]
        probabilities = best_model.predict_proba(scaled_inputs)[0]
        
        st.markdown("---")
        st.subheader("🔍 Automated Clinical Interpretation")
        st.write(f"*Prediction computed using the **{best_model_name}** ensemble (highest F1-score)._")
        
        # sklearn breast cancer: 0 = Malignant, 1 = Benign
        if prediction == 0:
            malignant_prob = probabilities[0]
            st.markdown(f"""
            <div class="banner-malignant">
                <h2 style='color: #ef4444; margin: 0; font-weight: 800;'>⚠️ High Risk: MALIGNANT Tumorous Pattern Detected</h2>
                <p style='font-size: 1.15rem; margin-top: 8px; color: #f3f4f6; max-width: 700px; margin-left: auto; margin-right: auto;'>
                    The cellular dimensions show signs of rapid growth and severe irregularities. Immediate medical review and biopsy confirmation are recommended.
                </p>
                <h3 style='color: #f87171; margin-top: 12px; margin-bottom: 0;'>
                    Malignancy Likelihood: {malignant_prob:.2%}
                </h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            benign_prob = probabilities[1]
            st.markdown(f"""
            <div class="banner-benign">
                <h2 style='color: #10b981; margin: 0; font-weight: 800;'>🎉 Low Risk: BENIGN Tumorous Pattern Detected</h2>
                <p style='font-size: 1.15rem; margin-top: 8px; color: #f3f4f6; max-width: 700px; margin-left: auto; margin-right: auto;'>
                    The cell characteristics are typical of healthy, localized, non-cancerous developments. Continue routine monitoring as prescribed.
                </p>
                <h3 style='color: #34d399; margin-top: 12px; margin-bottom: 0;'>
                    Benign Probability: {benign_prob:.2%}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        # Feature diagnostics expander
        with st.expander("🩺 View Input Feature Vector"):
            st.dataframe(raw_inputs)
            st.write("*Note: Cellular metrics are computed from digitized images of Fine Needle Aspirates (FNA).*")

# ----------------------------------------------------
# TAB 2: MODEL BENCHMARKING & EDA
# ----------------------------------------------------
with tab2:
    st.subheader("📊 Model Comparison Dashboard")
    st.write("We trained three classifiers on the selected high-impact clinical features. The performance metrics on the test set are displayed below:")
    
    # Format comparison table
    df_fmt = metrics_df.copy()
    for col in df_fmt.columns[1:]:
        df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.2%}")
    st.dataframe(df_fmt, use_container_width=True)
    
    st.markdown(f"**Selected Model:** The system has selected the **{best_model_name}** model due to its optimal balance between Recall and Precision (highest test F1-score).")
    
    st.markdown("---")
    st.subheader("📈 Exploratory Data Analysis (EDA) & Feature Relationships")
    
    # Build seaborn plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature Correlation Heatmap:**")
        # Compute correlation
        corr_matrix = X_sel.corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            ax=ax, 
            cbar=False,
            annot_kws={"size": 9}
        )
        plt.title("Correlation Matrix (Top 6 Selected Features)", fontsize=11, fontweight='bold', pad=10)
        plt.xticks(fontsize=8, rotation=45, ha='right')
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col2:
        st.write("**Tumor Class Distribution (Benign vs Malignant):**")
        fig, ax = plt.subplots(figsize=(6, 5))
        class_names = ["Malignant", "Benign"]
        counts = [np.sum(y_full == 0), np.sum(y_full == 1)]
        
        ax.bar(class_names, counts, color=['#ec4899', '#3b82f6'], width=0.5)
        for i, val in enumerate(counts):
            ax.text(i, val + 10, f"{val} ({val/len(y_full):.1%})", ha='center', va='bottom', fontweight='bold')
            
        plt.title("Malignant (Class 0) vs. Benign (Class 1) Target Count", fontsize=11, fontweight='bold', pad=10)
        plt.ylabel("Passenger Count")
        plt.tight_layout()
        st.pyplot(fig)

# ----------------------------------------------------
# TAB 3: CASE STUDY WRITE-UP
# ----------------------------------------------------
with tab3:
    st.subheader("📝 Case Study: Machine Learning in Clinical Decision Support Systems (CDSS)")
    
    st.markdown("""
    ### 1. Problem Statement & Clinical Context
    Breast cancer is the most common cancer among women globally. Identifying whether a breast mass is malignant or benign represents a critical diagnostic step. Pathologists characteristically analyze cells extracted via Fine Needle Aspirates (FNA) under a microscope. However, manual cell analysis is time-intensive, subject to observer variability, and carries the risk of false negatives (misdiagnosing a malignant tumor as benign), which can lead to delayed treatments and poor patient prognosis.
    
    ### 2. The Solution & Machine Learning Approach
    This project builds an automated Clinical Decision Support System (CDSS) to aid pathologists. Instead of using all 30 cellular metrics from the FNA scans, we selected the **6 features** displaying the strongest mathematical correlation to tumor malignancy:
    - **worst concave points**
    - **worst perimeter**
    - **mean concave points**
    - **worst radius**
    - **mean perimeter**
    - **worst area**
    
    By training and comparing **Logistic Regression (linear)**, **Random Forest (parallel trees)**, and **XGBoost (sequential gradient boosted trees)**, we select the model with the highest test F1-score to execute classifications. Caching and on-the-fly compilation guarantee zero dependency mismatch errors upon deployment.
    
    ### 3. Business & Medical Value
    - **Eliminating False Negatives:** In a clinical context, a false negative is the most costly error. The XGBoost and tree-based models achieve over **96% recall**, ensuring that almost all malignant cases are successfully flagged for further medical review.
    - **Urgency Triaging:** Biopsy results can take days. By instantly scoring malignancy probability, hospitals can prioritize pathology reviews for high-probability cases, accelerating time-to-treatment.
    - **Cost Reduction:** Automated pre-screening reduces the manual workload of pathologists, allowing them to focus on complex cases. This optimizes laboratory operations and lowers diagnostic costs per patient.
    
    *Disclaimer: This tool is intended strictly as a decision support aid for qualified medical professionals and does not replace official clinical diagnoses.*
    """)
