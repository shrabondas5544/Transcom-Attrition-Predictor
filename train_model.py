import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import joblib
import os

def load_data(filepath='transcom_hr/data/transcom_field_officer_attrition.csv'):
    df = pd.read_csv(filepath)
    # Target variable
    y = df['Previous Attrition Label'].apply(lambda x: 1 if x == 'Yes' else 0)
    X = df.drop(columns=['Previous Attrition Label'])
    return X, y

def train_and_save_model():
    X, y = load_data()
    
    # Identify categorical and numerical columns
    categorical_cols = ['Gender', 'Educational Qualification', 'Location']
    numerical_cols = ['Age', 'Tenure', 'Monthly Salary', 'Incentive Earnings', 'Attendance %', 
                      'Leave Utilization', 'Distance from Workplace', 'Number of Transfers', 
                      'Performance Rating', 'Training Hours', 'Promotion History', 
                      'Manager Effectiveness Score', 'Employee Engagement Score', 'Overtime Hours']
                      
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
        ])
        
    # We use ImbPipeline to incorporate SMOTE
    # Optimizing for Recall as requested
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("Training model with SMOTE...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
    print(f"Recall (Yes class): {recall_score(y_test, y_pred):.2f}")
    
    # Generate SHAP explainer
    # We need to transform the data to pass to SHAP since it doesn't natively handle sklearn pipelines well
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    
    # Get feature names after one-hot encoding
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(cat_feature_names)
    
    # Save model, explainer, and feature names
    os.makedirs('transcom_hr/ml_models', exist_ok=True)
    joblib.dump(pipeline, 'transcom_hr/ml_models/attrition_model.pkl')
    joblib.dump(explainer, 'transcom_hr/ml_models/shap_explainer.pkl')
    joblib.dump(all_feature_names, 'transcom_hr/ml_models/feature_names.pkl')
    print("Model and explainer saved to transcom_hr/ml_models/")

if __name__ == "__main__":
    train_and_save_model()
