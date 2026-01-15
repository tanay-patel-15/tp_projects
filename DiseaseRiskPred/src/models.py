"""
Machine learning models for diabetes prediction.
Includes training, evaluation, and prediction functions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import joblib
from pathlib import Path


def get_models() -> dict:
    """
    Get dictionary of models to train.
    
    Returns:
        Dictionary mapping model names to model instances
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
    }


def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """Train a single model and return it."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained model on test data.
    
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    return metrics


def train_and_evaluate_all(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Train and evaluate all models.
    
    Returns:
        Dictionary mapping model names to (model, metrics) tuples
    """
    models = get_models()
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        trained_model = train_model(model, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)
        results[name] = {
            'model': trained_model,
            'metrics': metrics
        }
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return results


def get_best_model(results: dict) -> tuple:
    """
    Get the best model based on F1 score.
    
    Returns:
        Tuple of (model_name, model, metrics)
    """
    best_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
    return best_name, results[best_name]['model'], results[best_name]['metrics']


def save_model(model, filepath: str):
    """Save trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load trained model from disk."""
    return joblib.load(filepath)


def predict_diabetes(model, scaler, input_data: dict) -> dict:
    """
    Make a diabetes prediction for new patient data.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        input_data: Dictionary of patient features
    
    Returns:
        Dictionary with prediction and probability
    """
    # Expected feature order
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Create feature array
    X = np.array([[input_data.get(f, 0) for f in features]])
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'risk_label': 'High Risk' if prediction == 1 else 'Low Risk',
        'probability_no_diabetes': float(probability[0]),
        'probability_diabetes': float(probability[1])
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Returns:
        DataFrame with feature names and importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df
