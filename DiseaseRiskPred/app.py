"""
Streamlit web application for diabetes risk prediction.
Provides an interactive interface for making predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.preprocessor import load_data, prepare_data
from src.models import (
    train_and_evaluate_all, get_best_model, save_model, load_model,
    predict_diabetes, get_feature_importance
)

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare_data():
    """Load and prepare the diabetes dataset."""
    data_path = Path(__file__).parent / 'data' / 'diabetes.csv'
    if not data_path.exists():
        st.error("Dataset not found! Please ensure diabetes.csv is in the data folder.")
        return None, None
    
    df = load_data(str(data_path))
    prepared = prepare_data(df)
    return df, prepared


@st.cache_resource
def train_models(prepared_data):
    """Train all models and return results."""
    results = train_and_evaluate_all(
        prepared_data['X_train'], prepared_data['y_train'],
        prepared_data['X_test'], prepared_data['y_test']
    )
    return results


def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">AI-powered diabetes risk assessment using machine learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/diabetes.png", width=80)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Predict Risk", "📊 Model Performance", "📈 Data Insights"])
    
    # Load data
    df, prepared_data = load_and_prepare_data()
    
    if df is None:
        st.warning("⚠️ Please download the diabetes dataset first!")
        st.code("# Download from Kaggle and place in data/ folder")
        return
    
    # Train models
    with st.spinner("Training models..."):
        results = train_models(prepared_data)
    
    best_name, best_model, best_metrics = get_best_model(results)
    
    if page == "🏠 Home":
        show_home(df, best_name, best_metrics)
    elif page == "🔮 Predict Risk":
        show_prediction(best_model, prepared_data['scaler'], prepared_data['feature_names'])
    elif page == "📊 Model Performance":
        show_model_performance(results)
    elif page == "📈 Data Insights":
        show_data_insights(df)


def show_home(df, best_name, best_metrics):
    """Display home page with project overview."""
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Best Model", best_name.split()[0])
    with col4:
        st.metric("Accuracy", f"{best_metrics['accuracy']:.1%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 About This Project")
        st.markdown("""
        This application uses **machine learning** to predict diabetes risk based on 
        health metrics. The model analyzes factors like:
        
        - Blood glucose levels
        - BMI (Body Mass Index)
        - Age and pregnancy history
        - Blood pressure
        - Insulin levels
        
        > **Note:** This is for educational purposes only and should not replace 
        > professional medical advice.
        """)
    
    with col2:
        st.subheader("🎯 How It Works")
        st.markdown("""
        1. **Data Collection** - Uses the Pima Indians Diabetes dataset
        2. **Preprocessing** - Handles missing values and scales features
        3. **Model Training** - Trains multiple ML models
        4. **Prediction** - Enter your health metrics to get a risk assessment
        
        The system compares **Random Forest**, **XGBoost**, **Gradient Boosting**, 
        and **Logistic Regression** to find the best predictor.
        """)


def show_prediction(model, scaler, feature_names):
    """Display prediction interface."""
    st.markdown("---")
    st.subheader("Enter Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1, help="Number of times pregnant")
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120, help="Plasma glucose concentration")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70, help="Diastolic blood pressure")
    
    with col2:
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness")
        insulin = st.number_input("Insulin (μU/mL)", 0, 900, 80, help="2-Hour serum insulin")
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1, help="Body mass index")
    
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01, 
                              help="Genetic diabetes risk score")
        age = st.number_input("Age", 18, 100, 30, help="Age in years")
    
    st.markdown("---")
    
    if st.button("🔮 Predict Diabetes Risk", use_container_width=True):
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        result = predict_diabetes(model, scaler, input_data)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if result['prediction'] == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h2>⚠️ {result['risk_label']}</h2>
                    <p style="font-size: 1.5rem;">Probability: {result['probability_diabetes']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2>✅ {result['risk_label']}</h2>
                    <p style="font-size: 1.5rem;">Probability: {result['probability_no_diabetes']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['probability_diabetes'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Diabetes Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#764ba2"},
                    'steps': [
                        {'range': [0, 30], 'color': "#4facfe"},
                        {'range': [30, 70], 'color': "#ffecd2"},
                        {'range': [70, 100], 'color': "#f5576c"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("📊 Feature Importance")
        importance_df = get_feature_importance(model, feature_names)
        if importance_df is not None:
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                        color='importance', color_continuous_scale='Viridis')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)


def show_model_performance(results):
    """Display model comparison and performance metrics."""
    st.markdown("---")
    
    # Create comparison dataframe
    metrics_data = []
    for name, data in results.items():
        m = data['metrics']
        metrics_data.append({
            'Model': name,
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1 Score': m['f1_score'],
            'ROC AUC': m.get('roc_auc', 0)
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Bar chart comparison
    st.subheader("📊 Model Comparison")
    fig = px.bar(df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                 x='Model', y='Score', color='Metric', barmode='group',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics table
    st.subheader("📋 Detailed Metrics")
    st.dataframe(df_metrics.style.format({
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1 Score': '{:.2%}',
        'ROC AUC': '{:.2%}'
    }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'], 
                     color='#90EE90'), use_container_width=True)
    
    # Confusion matrices
    st.subheader("🔢 Confusion Matrices")
    cols = st.columns(len(results))
    for i, (name, data) in enumerate(results.items()):
        with cols[i]:
            cm = data['metrics']['confusion_matrix']
            fig = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual"),
                           x=['No Diabetes', 'Diabetes'],
                           y=['No Diabetes', 'Diabetes'],
                           color_continuous_scale='Blues')
            fig.update_layout(title=name.split()[0], height=300)
            st.plotly_chart(fig, use_container_width=True)


def show_data_insights(df):
    """Display exploratory data analysis visualizations."""
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Target Distribution")
        outcome_counts = df['Outcome'].value_counts()
        fig = px.pie(values=outcome_counts.values, names=['No Diabetes', 'Diabetes'],
                    color_discrete_sequence=['#4facfe', '#f5576c'],
                    hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Feature Distributions")
        feature = st.selectbox("Select Feature", df.columns[:-1])
        fig = px.histogram(df, x=feature, color='Outcome', 
                          color_discrete_map={0: '#4facfe', 1: '#f5576c'},
                          barmode='overlay', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlations")
    corr = df.corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                   aspect='auto')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plots
    st.subheader("📦 Feature Box Plots by Outcome")
    feature_box = st.selectbox("Select Feature for Box Plot", df.columns[:-1], key='box')
    fig = px.box(df, x='Outcome', y=feature_box, color='Outcome',
                color_discrete_map={0: '#4facfe', 1: '#f5576c'},
                labels={'Outcome': 'Diabetes Status'})
    fig.update_xaxes(ticktext=['No Diabetes', 'Diabetes'], tickvals=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
