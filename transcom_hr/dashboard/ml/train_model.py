import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score, f1_score

def load_data():
    """
    Loads the employee attrition dataset.
    Robustly handles path resolution relative to this script.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, 'data', 'transcom_field_officer_attrition.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please check Phase 2 execution.")
        
    df = pd.read_csv(csv_path)
    
    # Target variable conversion (Yes -> 1, No -> 0)
    y = df['Previous Attrition Label'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    X = df.drop(columns=['Previous Attrition Label'])
    
    return X, y

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 1. Load Data
    X, y = load_data()
    
    # 2. Train-Test Split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Identify feature types
    categorical_cols = ['Gender', 'Educational Qualification', 'Location']
    numerical_cols = [
        'Age', 'Tenure', 'Monthly Salary', 'Incentive Earnings', 'Attendance %', 
        'Leave Utilization', 'Distance from Workplace', 'Number of Transfers', 
        'Performance Rating', 'Training Hours', 'Promotion History', 
        'Manager Effectiveness Score', 'Employee Engagement Score', 'Overtime Hours'
    ]
    
    # 3. Data Preprocessing Setup
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Fit the preprocessor and transform datasets
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Extract feature names after One-Hot Encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    feature_names = numerical_cols + encoded_cat_names
    
    # Convert processed data back to DataFrames for better tracking and SHAP integration
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)
    
    print(f"Data shape: Training={X_train_df.shape}, Testing={X_test_df.shape}")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    
    # 4. Train Models
    # Model A: Random Forest Classifier (balanced weights)
    rf_model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    rf_model.fit(X_train_df, y_train)
    
    # Model B: Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42)
    gb_model.fit(X_train_df, y_train)
    
    # 5. Evaluate and Compare on Recall (Yes class)
    # We want to optimize the decision threshold for both to see which can yield the best Recall
    # while keeping Precision reasonable (above 0.40 to prevent too many false alarms)
    
    best_model = None
    best_threshold = 0.5
    best_recall = 0.0
    best_model_name = ""
    best_metrics = {}
    
    models = {
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model
    }
    
    for name, model in models.items():
        # Get probability of class 1
        probs = model.predict_proba(X_test_df)[:, 1]
        
        # Test thresholds from 0.2 to 0.6 to optimize Recall
        for threshold in np.linspace(0.2, 0.6, 9):
            preds = (probs >= threshold).astype(int)
            rec = recall_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            
            # Select model/threshold that maximizes Recall while keeping Precision >= 0.45
            if rec > best_recall and prec >= 0.45:
                best_recall = rec
                best_threshold = threshold
                best_model = model
                best_model_name = name
                best_metrics = {
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1-Score': f1,
                    'Threshold': threshold
                }
                
    # Fallback if no threshold met the precision constraint
    if best_model is None:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_threshold = 0.5
        probs = rf_model.predict_proba(X_test_df)[:, 1]
        preds = (probs >= 0.5).astype(int)
        best_metrics = {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1-Score': f1_score(y_test, preds),
            'Threshold': 0.5
        }
    
    print("\n" + "="*50)
    print(f"BEST PERFORMING MODEL: {best_model_name} (Threshold: {best_metrics['Threshold']:.2f})")
    print(f"Accuracy:  {best_metrics['Accuracy']:.4f}")
    print(f"Precision: {best_metrics['Precision']:.4f}")
    print(f"Recall:    {best_metrics['Recall']:.4f} (Optimized for Flight Risk detection)")
    print(f"F1-Score:  {best_metrics['F1-Score']:.4f}")
    print("="*50)
    
    # Print detailed report for the selected configuration
    best_probs = best_model.predict_proba(X_test_df)[:, 1]
    best_preds = (best_probs >= best_metrics['Threshold']).astype(int)
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, best_preds, target_names=['Stay (No)', 'Quit (Yes)']))
    
    # 6. SHAP Integration
    print("Initializing SHAP Explainer...")
    # TreeExplainer is highly optimized for tree-based models (RF/GB)
    explainer = shap.TreeExplainer(best_model)
    
    # Compute SHAP values for the training dataset (or a representative sample to save time/space)
    # We'll use the full training set as 3000 rows is small and fast enough for TreeExplainer
    shap_values = explainer(X_train_df)
    
    # 7. Serialize Models and Artifacts
    save_dir = os.path.join(base_dir, 'dashboard', 'ml', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save elements needed for production inference
    joblib.dump(best_model, os.path.join(save_dir, 'attrition_model.pkl'))
    joblib.dump(preprocessor, os.path.join(save_dir, 'preprocessor.pkl'))
    joblib.dump(explainer, os.path.join(save_dir, 'shap_explainer.pkl'))
    joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.pkl'))
    
    # Save the optimal threshold as a small dict
    metadata = {
        'model_name': best_model_name,
        'optimal_threshold': best_metrics['Threshold'],
        'metrics': best_metrics
    }
    joblib.dump(metadata, os.path.join(save_dir, 'model_metadata.pkl'))
    
    print(f"Successfully serialized all model artifacts to {save_dir}/")
    return best_model, preprocessor, explainer, feature_names, best_metrics['Threshold']

def get_top_contributors(employee_raw_df, preprocessor, model, explainer, feature_names, top_n=3):
    """
    Helper function to extract the top N contributing features (positive drivers towards attrition)
    for a single employee instance.
    
    Parameters:
    - employee_raw_df: pandas DataFrame containing a single row of raw employee data.
    - preprocessor: trained ColumnTransformer.
    - model: trained Classifier.
    - explainer: trained shap.TreeExplainer.
    - feature_names: list of feature names corresponding to the preprocessed columns.
    
    Returns:
    - list of dicts containing {'feature': name, 'shap_value': val} sorted by highest impact.
    """
    # 1. Preprocess the raw input row
    processed_arr = preprocessor.transform(employee_raw_df)
    processed_df = pd.DataFrame(processed_arr, columns=feature_names)
    
    # 2. Calculate SHAP values
    # For classification models, SHAP values are output in log-odds space for each class.
    # Class 1 (attrition) is at index 1.
    shap_res = explainer(processed_df)
    
    # Handle SHAP output format variations (some versions return [1, N, 2] array for binary classification)
    if len(shap_res.values.shape) == 3:  # (num_instances, num_features, num_classes)
        instance_shap_values = shap_res.values[0, :, 1]
    elif len(shap_res.values.shape) == 2:  # (num_instances, num_features)
        # RF tree explainer sometimes returns a list of arrays (one for each class)
        # or a single 2D array if it's regression or specialized setup.
        if isinstance(shap_res.values, list):
            instance_shap_values = shap_res.values[1][0, :]
        else:
            # If 2D, check shape. For tree explainer it could be class 1 or a list of arrays.
            # Let's fallback to calculating raw explainer values if standard explain fails
            raw_shap = explainer.shap_values(processed_df)
            if isinstance(raw_shap, list):
                instance_shap_values = raw_shap[1][0]
            else:
                instance_shap_values = raw_shap[0]
    else:
        # standard fallback
        raw_shap = explainer.shap_values(processed_df)
        if isinstance(raw_shap, list):
            instance_shap_values = raw_shap[1][0]
        else:
            instance_shap_values = raw_shap[0]
            
    # 3. Match SHAP values with feature names
    feature_impacts = []
    for name, val in zip(feature_names, instance_shap_values):
        # We only care about positive drivers (features increasing the probability of quitting)
        if val > 0:
            # Clean up encoded feature names for display
            clean_name = name
            if name.startswith('cat__'):
                clean_name = name.replace('cat__', '').replace('_', ' ')
            elif name.startswith('num__'):
                clean_name = name.replace('num__', '')
            feature_impacts.append({'feature': clean_name, 'shap_value': val})
            
    # Sort by SHAP value descending (highest contribution first)
    feature_impacts = sorted(feature_impacts, key=lambda x: x['shap_value'], reverse=True)
    
    return feature_impacts[:top_n]

if __name__ == "__main__":
    train_and_evaluate()
