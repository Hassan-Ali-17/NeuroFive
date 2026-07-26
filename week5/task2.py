import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------
# 1. CUSTOM CLASS DEFINITION (REQUIRED FOR JOBLIB LOAD)
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
        X_out = X.copy()
        X_out['FamilySize'] = X_out['SibSp'] + X_out['Parch'] + 1
        X_out['IsAlone'] = (X_out['FamilySize'] == 1).astype(int)
        X_out['Title'] = X_out['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Countess', 'Ms', 'Lady', 'Jonkheer', 'Don', 'Mme', 'Capt', 'Sir']
        X_out['Title'] = X_out['Title'].replace(rare_titles, 'Other')
        X_out['Title'] = X_out['Title'].fillna('Mr')
        return X_out

# ----------------------------------------------------
# 2. STREAMLIT APP CONFIGURATION & STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Custom premium CSS for glassmorphism aesthetics and premium look
st.markdown("""
<style>
    /* Gradient Background Effect */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Title styling */
    .title-text {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Container/Card styling */
    .css-1r6g72t, .stCard {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    
    /* Input field label styling */
    .stSlider label, .stSelectbox label, .stTextInput label, .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Custom Predict Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        width: 100%;
        margin-top: 15px;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #0369a1 0%, #4338ca 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
    }
    
    /* Results cards styling */
    .result-survived {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.4) 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    
    .result-died {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.4) 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. INTERACTIVE HEADER & MODEL LOADING
# ----------------------------------------------------
st.markdown('<h1 class="title-text">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Enter passenger credentials below to predict their likelihood of surviving the historic voyage.</p>', unsafe_allow_html=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_and_train_pipeline():
    # Load dataset
    csv_path = os.path.join(script_dir, "train.csv")
    df = pd.read_csv(csv_path)
    
    # Cast Pclass to string to make it categorical
    df['Pclass'] = df['Pclass'].astype(str)
    
    # Define target and features
    y = df['Survived']
    X = df.drop(columns=['PassengerId', 'Survived'])
    
    # Baseline columns
    num_cols_eng = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
    cat_cols_eng = ['Pclass', 'Sex', 'Embarked', 'Title']
    
    # Define Preprocessor
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
    
    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('feat_eng', TitanicFeatureExtractor()),
        ('preprocessor', preprocessor_eng),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    
    # Fit the pipeline on the full dataset
    pipeline.fit(X, y)
    return pipeline

try:
    pipeline = load_and_train_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"Error building model pipeline: {str(e)}")
    st.warning("Please ensure 'train.csv' is present in the week5 directory.")
    model_loaded = False

# ----------------------------------------------------
# 4. APP INTERFACE & INPUT FORM
# ----------------------------------------------------
if model_loaded:
    with st.container():
        st.subheader("📋 Passenger Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name", "John", help="Passenger's first name")
            last_name = st.text_input("Last Name", "Smith", help="Passenger's last name / surname")
            title = st.selectbox(
                "Title Prefix", 
                ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Other"],
                help="Social prefix, critical for engineered title matching"
            )
            sex = st.radio("Gender / Sex", ["Male", "Female"], horizontal=True)
            age = st.slider("Passenger Age (Years)", 0.5, 100.0, 28.0, 0.5)
            
        with col2:
            pclass = st.selectbox("Ticket Class (Pclass)", ["1st Class", "2nd Class", "3rd Class"])
            embarked_port = st.selectbox(
                "Port of Embarkation", 
                ["Southampton (UK)", "Cherbourg (France)", "Queenstown (Ireland)"]
            )
            sibsp = st.slider("Siblings & Spouses Aboard (SibSp)", 0, 8, 0)
            parch = st.slider("Parents & Children Aboard (Parch)", 0, 6, 0)
            fare = st.slider("Ticket Fare (British Pounds)", 0.0, 512.0, 32.0, 1.0)
            
        # Parse inputs into correct model format
        pclass_map = {"1st Class": "1", "2nd Class": "2", "3rd Class": "3"}
        embarked_map = {"Southampton (UK)": "S", "Cherbourg (France)": "C", "Queenstown (Ireland)": "Q"}
        
        # Format passenger Name for Title extractor regex: "LastName, Title. FirstName"
        full_name = f"{last_name}, {title}. {first_name}"
        
        # Build the input DataFrame
        input_data = pd.DataFrame([{
            'Pclass': pclass_map[pclass],
            'Name': full_name,
            'Sex': sex.lower(),
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked_map[embarked_port]
        }])

        # Trigger prediction on button click
        if st.button("🔮 Predict Survival Likelihood"):
            prediction = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0][1]
            
            st.markdown("---")
            st.subheader("📊 Prediction Analysis")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="result-survived">
                    <h2 style='color: #10b981; margin: 0;'>🎉 Passenger Survived!</h2>
                    <p style='font-size: 1.1rem; margin-top: 5px; color: #e2e8f0;'>
                        Our model predicts this passenger would have <strong>survived</strong> the shipwreck.
                    </p>
                    <h3 style='color: #34d399; margin-top: 10px; margin-bottom: 0;'>
                        Survival Probability: {probability:.2%}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="result-died">
                    <h2 style='color: #ef4444; margin: 0;'>⚠️ Passenger Did Not Survive</h2>
                    <p style='font-size: 1.1rem; margin-top: 5px; color: #e2e8f0;'>
                        Our model predicts this passenger would <strong>not have survived</strong> the disaster.
                    </p>
                    <h3 style='color: #f87171; margin-top: 10px; margin-bottom: 0;'>
                        Survival Probability: {probability:.2%}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
            # Render a detailed metrics breakdown
            with st.expander("🔍 Show Feature Diagnostics"):
                st.write("**Feature Extract Input Data:**")
                st.dataframe(input_data)
                
                family_size = sibsp + parch + 1
                is_alone = 1 if family_size == 1 else 0
                st.write(f"- **Engineered Title:** {title}")
                st.write(f"- **Engineered Family Size:** {family_size} (SibSp: {sibsp} + Parch: {parch} + 1)")
                st.write(f"- **Engineered IsAlone:** {'Yes (Alone)' if is_alone == 1 else 'No (With Family)'}")
