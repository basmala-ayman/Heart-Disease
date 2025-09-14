import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'feature_selection_heart_disease.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.pkl')

st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_mode():
    """Loads the pre-trained model pipeline."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f'Error: Model file not found at "{MODEL_PATH}". Please, check the file path.')
        return None
    
@st.cache_data
def load_data():
    """Loads the dataset for visualization."""
    try:
        dataset = pd.read_csv(DATASET_PATH)
        return dataset
    except FileNotFoundError:
        st.error(f'Error: Data file not found at "{DATASET_PATH}". Please, check the file path.')

model = load_mode()
df = load_data()

if model is None or df is None:
    st.stop()

st.title('Heart Disease Predictor')
st.markdown("""
Welcome to the Heart Disease Predictor! This tool uses a machine learning model to estimate the risk of heart disease based on several health metrics. 
Simply input your data in the sidebar and click 'Predict' to see the result.
""")

st.header("User Input Features")

def user_input_features():
    """Gathers user input for each feature."""

    col1, col2 = st.columns(2)
    with col1:
        thalach = st.slider("Maximum Heart Rate Achieved (thalach)", min_value=70, max_value=220, value=150)
        
        cp_options = {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal Pain', 4: 'Asymptomatic'}
        cp = st.selectbox("Chest Pain Type (cp)", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

        ca_options = {0: '0', 1: '1', 2: '2', 3: '3'}
        ca = st.selectbox("Number of Major Vessels (ca)", options=list(ca_options.keys()), format_func=lambda x: ca_options[x])
    with col2:
        oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)

        thal_options = {3: 'Normal (3)', 6: 'Fixed Defect (6)', 7: 'Reversible Defect (7)'}
        thal = st.selectbox("Thalassemia (thal)", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

        exang_options = {0: 'No', 1: 'Yes'}
        exang = st.selectbox("Exercise Induced Angina (exang)", options=list(exang_options.keys()), format_func=lambda x: exang_options[x])


    user_data = {
        'thalach': thalach,
        'cp': cp,
        'ca': ca,
        'oldpeak': oldpeak,
        'thal': thal,
        'exang': exang
    }

    user_features = pd.DataFrame(user_data, index=[0])
    return user_features

input_df = user_input_features()

col_predict = st.columns([2,6,2])
with col_predict[1]:
    if st.button('Predict',use_container_width=True):
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.error('SVM Model predicts **Heart Disease**.')
        else:
            st.success("SVM Model predicts **No Heart Disease**.")

st.subheader("Data Visualization")

## Heart Disease Distribution Visualization
st.markdown("#### Heart Disease Distribution in the Dataset")
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.countplot(x='num', data=df, ax=ax1, palette='viridis')
ax1.set_title("Count of Heart Disease Cases", fontsize=16)
ax1.set_xlabel("Heart Disease Status (0 = No, 1 = Yes)", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_xticklabels(['No Heart Disease', 'Heart Disease'])
st.pyplot(fig1)

## Correlation Heatmap Visualization
st.markdown('#### Feature Correlation Heatmap')
fig2, ax2 = plt.subplots(figsize=(8,4))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
ax2.set_title("Correlation Matrix of Features", fontsize=16)
st.pyplot(fig2)

## Distribution of 'thalach' by 'num'
st.markdown("#### Maximum Heart Rate Achieved by Heart Disease Status")
fig3, ax3 = plt.subplots(figsize=(8,4))
sns.violinplot(x='num', y='thalach', data=df, ax=ax3, palette='muted')
ax3.set_title("Distribution of Max Heart Rate", fontsize=16)
ax3.set_xlabel("Heart Disease Status (0 = No, 1 = Yes)", fontsize=12)
ax3.set_ylabel("Max Heart Rate (thalach)", fontsize=12)
ax3.set_xticklabels(['No Heart Disease', 'Heart Disease'])
st.pyplot(fig3)