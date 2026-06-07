import os
import time
import joblib
import pandas as pd
import numpy as np
from django.db import transaction
from dashboard.models import Employee

def run_system_wide_inference():
    """
    Loads serialized ML models, processes all Employee database records into a DataFrame,
    runs inference, calculates SHAP feature importance for each, and updates fields in bulk.
    """
    start_time = time.time()
    
    # 1. Load ML artifacts
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    saved_models_dir = os.path.join(base_dir, 'dashboard', 'ml', 'saved_models')
    
    model_path = os.path.join(saved_models_dir, 'attrition_model.pkl')
    preprocessor_path = os.path.join(saved_models_dir, 'preprocessor.pkl')
    shap_explainer_path = os.path.join(saved_models_dir, 'shap_explainer.pkl')
    feature_names_path = os.path.join(saved_models_dir, 'feature_names.pkl')
    
    if not all(os.path.exists(p) for p in [model_path, preprocessor_path, shap_explainer_path, feature_names_path]):
        raise FileNotFoundError("One or more model artifacts are missing. Run Phase 3 training first.")
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    explainer = joblib.load(shap_explainer_path)
    feature_names = joblib.load(feature_names_path)
    
    # 2. Query all employees
    employees = list(Employee.objects.all())
    if not employees:
        return {
            'success': True,
            'processed_count': 0,
            'execution_time': time.time() - start_time,
            'message': 'No employees found in the database.'
        }
        
    # 3. Build DataFrame matching preprocessor features
    data = {
        'Age': [emp.age for emp in employees],
        'Gender': [emp.gender for emp in employees],
        'Educational Qualification': [emp.educational_qualification for emp in employees],
        'Location': [emp.location for emp in employees],
        'Tenure': [emp.tenure for emp in employees],
        'Monthly Salary': [emp.monthly_salary for emp in employees],
        'Incentive Earnings': [emp.incentive_earnings for emp in employees],
        'Attendance %': [emp.attendance_pct for emp in employees],
        'Leave Utilization': [emp.leave_utilization for emp in employees],
        'Distance from Workplace': [emp.distance_from_workplace for emp in employees],
        'Number of Transfers': [emp.num_transfers for emp in employees],
        'Performance Rating': [emp.performance_rating for emp in employees],
        'Training Hours': [emp.training_hours for emp in employees],
        'Promotion History': [emp.promotion_history for emp in employees],
        'Manager Effectiveness Score': [emp.manager_effectiveness_score for emp in employees],
        'Employee Engagement Score': [emp.employee_engagement_score for emp in employees],
        'Overtime Hours': [emp.overtime_hours for emp in employees]
    }
    df = pd.DataFrame(data)
    
    # 4. Run Preprocessing & Inference
    X_proc = preprocessor.transform(df)
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
    
    probabilities = model.predict_proba(X_proc_df)[:, 1]
    
    # 5. Compute SHAP values in batch
    shap_res = explainer(X_proc_df)
    
    # Extract shape-agnostic SHAP values for class 1 (attrition)
    if len(shap_res.values.shape) == 3:  # (instances, features, classes)
        shap_values_class1 = shap_res.values[:, :, 1]
    elif len(shap_res.values.shape) == 2:
        # Check if list (older RF explainer outputs)
        if isinstance(shap_res.values, list):
            shap_values_class1 = shap_res.values[1]
        else:
            raw_shap = explainer.shap_values(X_proc_df)
            if isinstance(raw_shap, list):
                shap_values_class1 = raw_shap[1]
            else:
                shap_values_class1 = raw_shap
    else:
        raw_shap = explainer.shap_values(X_proc_df)
        if isinstance(raw_shap, list):
            shap_values_class1 = raw_shap[1]
        else:
            shap_values_class1 = raw_shap
            
    # Convert list SHAP output to numpy array if necessary
    if isinstance(shap_values_class1, list):
        shap_values_class1 = np.array(shap_values_class1)
        
    # 6. Process individual records
    updated_employees = []
    
    for i, emp in enumerate(employees):
        prob = float(probabilities[i])
        emp.attrition_probability = prob
        
        # Risk Category classification
        if prob >= 0.50:
            emp.risk_category = 'High'
        elif prob >= 0.20:
            emp.risk_category = 'Medium'
        else:
            emp.risk_category = 'Low'
            
        # Extract Top Attrition Driver from positive SHAP values
        row_shap = shap_values_class1[i]
        
        # Find feature with maximum positive SHAP value
        positive_indices = np.where(row_shap > 0)[0]
        if len(positive_indices) > 0:
            max_idx = positive_indices[np.argmax(row_shap[positive_indices])]
            raw_driver_name = feature_names[max_idx]
            
            # Clean feature names for UI display
            clean_name = raw_driver_name
            if raw_driver_name.startswith('cat__'):
                clean_name = raw_driver_name.replace('cat__', '').replace('_', ' ')
            elif raw_driver_name.startswith('num__'):
                clean_name = raw_driver_name.replace('num__', '')
            
            emp.primary_driver = clean_name
        else:
            emp.primary_driver = "None (Stable)"
            
        updated_employees.append(emp)
        
    # 7. Batch Update Employee records in database
    with transaction.atomic():
        Employee.objects.bulk_update(
            updated_employees, 
            ['attrition_probability', 'risk_category', 'primary_driver'],
            batch_size=500
        )
        
    execution_time = time.time() - start_time
    
    # Calculate and log category distribution
    categories = [emp.risk_category for emp in updated_employees]
    cat_counts = {cat: categories.count(cat) for cat in ['Low', 'Medium', 'High']}
    print(f"Risk Category Distribution: {cat_counts}")
    print(f"System-wide inference completed in {execution_time:.2f}s for {len(updated_employees)} records.")
    
    return {
        'success': True,
        'processed_count': len(updated_employees),
        'execution_time': execution_time
    }
