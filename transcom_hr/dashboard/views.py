import os
import json
import joblib
import shap
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Employee
from .services.inference import run_system_wide_inference

def dashboard_home(request):
    """
    Renders the predictive HR dashboard home page with metrics, chart data, and employee list.
    """
    total_employees = Employee.objects.count()
    high_risk_count = Employee.objects.filter(risk_category='High').count()
    medium_risk_count = Employee.objects.filter(risk_category='Medium').count()
    low_risk_count = Employee.objects.filter(risk_category='Low').count()
    
    current_attrition_rate = 0
    if total_employees > 0:
        current_attrition_rate = round(((high_risk_count + medium_risk_count) / total_employees) * 100, 2)
        
    # Risk category distribution
    risk_distribution = {
        'Low': low_risk_count,
        'Medium': medium_risk_count,
        'High': high_risk_count
    }
    
    # Attrition by location
    locations = ['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna', 'Barisal']
    location_labels = []
    location_counts = []
    for loc in locations:
        risk_count = Employee.objects.filter(location=loc, risk_category__in=['High', 'Medium']).count()
        location_labels.append(loc)
        location_counts.append(risk_count)
        
    context = {
        'total_employees': total_employees,
        'high_risk_count': high_risk_count,
        'medium_risk_count': medium_risk_count,
        'target_attrition_rate': 32.00,
        'current_attrition_rate': current_attrition_rate,
        'risk_distribution': risk_distribution,
        'location_labels': location_labels,
        'location_counts': location_counts,
    }
    return render(request, 'dashboard/index.html', context)

@csrf_exempt
def TriggerInferenceView(request):
    """
    API View to run batch inference on all employees.
    Supports both GET and POST requests.
    """
    if request.method not in ['GET', 'POST']:
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
        
    try:
        results = run_system_wide_inference()
        return JsonResponse({
            'success': True,
            'processed_count': results['processed_count'],
            'execution_time_seconds': round(results['execution_time'], 3),
            'message': 'System-wide predictions completed successfully.'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def EmployeeListAPIView(request):
    """
    API View returning a list of all employees ordered by attrition probability descending.
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
        
    try:
        employees = Employee.objects.all().order_by('-attrition_probability')
        
        data_list = []
        for emp in employees:
            data_list.append({
                'id': emp.id,
                'age': emp.age,
                'gender': emp.gender,
                'educational_qualification': emp.educational_qualification,
                'location': emp.location,
                'tenure': emp.tenure,
                'monthly_salary': emp.monthly_salary,
                'incentive_earnings': emp.incentive_earnings,
                'attendance_pct': emp.attendance_pct,
                'leave_utilization': emp.leave_utilization,
                'distance_from_workplace': emp.distance_from_workplace,
                'num_transfers': emp.num_transfers,
                'performance_rating': emp.performance_rating,
                'training_hours': emp.training_hours,
                'promotion_history': emp.promotion_history,
                'manager_effectiveness_score': emp.manager_effectiveness_score,
                'employee_engagement_score': emp.employee_engagement_score,
                'overtime_hours': emp.overtime_hours,
                'previous_attrition_label': emp.previous_attrition_label,
                'attrition_probability': round(emp.attrition_probability, 4) if emp.attrition_probability is not None else None,
                'risk_category': emp.risk_category,
                'primary_driver': emp.primary_driver
            })
            
        return JsonResponse({
            'success': True,
            'count': len(data_list),
            'employees': data_list
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
def UploadCSVView(request):
    """
    Endpoint to upload a CSV file, validate column headers, clear DB, seed, and predict.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
    if 'file' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No file uploaded'}, status=400)
        
    uploaded_file = request.FILES['file']
    filename = uploaded_file.name.lower()
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            try:
                df = pd.read_excel(uploaded_file)
            except ImportError:
                return JsonResponse({
                    'success': False, 
                    'error': "Excel file support (openpyxl) is not configured. Please convert your file to CSV and try again."
                }, status=400)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported file format. Please upload .csv or .xlsx.'}, status=400)
            
        required_cols = [
            'Age', 'Gender', 'Educational Qualification', 'Location', 'Tenure', 
            'Monthly Salary', 'Incentive Earnings', 'Attendance %', 'Leave Utilization', 
            'Distance from Workplace', 'Number of Transfers', 'Performance Rating', 
            'Training Hours', 'Promotion History', 'Manager Effectiveness Score', 
            'Employee Engagement Score', 'Overtime Hours', 'Previous Attrition Label'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return JsonResponse({
                'success': False,
                'error': f"Validation failed. Missing column(s): {', '.join(missing_cols)}"
            }, status=400)
            
        employees_to_create = []
        for _, row in df.iterrows():
            emp = Employee(
                age=int(row['Age']),
                gender=row['Gender'],
                educational_qualification=row['Educational Qualification'],
                location=row['Location'],
                tenure=int(row['Tenure']),
                monthly_salary=int(row['Monthly Salary']),
                incentive_earnings=float(row['Incentive Earnings']),
                attendance_pct=float(row['Attendance %']),
                leave_utilization=int(row['Leave Utilization']),
                distance_from_workplace=int(row['Distance from Workplace']),
                num_transfers=int(row['Number of Transfers']),
                performance_rating=int(row['Performance Rating']),
                training_hours=int(row['Training Hours']),
                promotion_history=int(row['Promotion History']),
                manager_effectiveness_score=int(row['Manager Effectiveness Score']),
                employee_engagement_score=int(row['Employee Engagement Score']),
                overtime_hours=int(row['Overtime Hours']),
                previous_attrition_label=row['Previous Attrition Label'],
            )
            employees_to_create.append(emp)
            
        from django.db import transaction
        with transaction.atomic():
            Employee.objects.all().delete()
            Employee.objects.bulk_create(employees_to_create)
            
        results = run_system_wide_inference()
        
        return JsonResponse({
            'success': True,
            'message': f"Successfully processed {len(employees_to_create)} records.",
            'processed_count': len(employees_to_create),
            'execution_time_seconds': round(results['execution_time'], 3)
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
def PredictSingleView(request):
    """
    In-memory inference view to predict flight risk and SHAP driver for a single manual input.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
    try:
        data = json.loads(request.body)
        
        # Map parameters
        raw_data = {
            'Age': int(data.get('age')),
            'Gender': data.get('gender'),
            'Educational Qualification': data.get('educational_qualification'),
            'Location': data.get('location'),
            'Tenure': int(data.get('tenure')),
            'Monthly Salary': int(data.get('monthly_salary')),
            'Incentive Earnings': float(data.get('incentive_earnings')),
            'Attendance %': float(data.get('attendance_pct')),
            'Leave Utilization': int(data.get('leave_utilization')),
            'Distance from Workplace': int(data.get('distance_from_workplace')),
            'Number of Transfers': int(data.get('num_transfers')),
            'Performance Rating': int(data.get('performance_rating')),
            'Training Hours': int(data.get('training_hours')),
            'Promotion History': int(data.get('promotion_history')),
            'Manager Effectiveness Score': int(data.get('manager_effectiveness_score')),
            'Employee Engagement Score': int(data.get('employee_engagement_score')),
            'Overtime Hours': int(data.get('overtime_hours'))
        }
        
        from django.conf import settings
        saved_models_dir = os.path.join(settings.BASE_DIR, 'dashboard', 'ml', 'saved_models')
        
        model = joblib.load(os.path.join(saved_models_dir, 'attrition_model.pkl'))
        preprocessor = joblib.load(os.path.join(saved_models_dir, 'preprocessor.pkl'))
        explainer = joblib.load(os.path.join(saved_models_dir, 'shap_explainer.pkl'))
        feature_names = joblib.load(os.path.join(saved_models_dir, 'feature_names.pkl'))
        
        df_single = pd.DataFrame([raw_data])
        X_proc = preprocessor.transform(df_single)
        X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
        
        prob = float(model.predict_proba(X_proc_df)[0, 1])
        
        if prob >= 0.50:
            risk_category = 'High'
        elif prob >= 0.20:
            risk_category = 'Medium'
        else:
            risk_category = 'Low'
            
        # Compute SHAP
        shap_res = explainer(X_proc_df)
        if len(shap_res.values.shape) == 3:
            instance_shap = shap_res.values[0, :, 1]
        elif len(shap_res.values.shape) == 2:
            if isinstance(shap_res.values, list):
                instance_shap = shap_res.values[1][0]
            else:
                raw_shap = explainer.shap_values(X_proc_df)
                if isinstance(raw_shap, list):
                    instance_shap = raw_shap[1][0]
                else:
                    instance_shap = raw_shap[0]
        else:
            raw_shap = explainer.shap_values(X_proc_df)
            if isinstance(raw_shap, list):
                instance_shap = raw_shap[1][0]
            else:
                instance_shap = raw_shap[0]
                
        positive_indices = np.where(instance_shap > 0)[0]
        if len(positive_indices) > 0:
            max_idx = positive_indices[np.argmax(instance_shap[positive_indices])]
            raw_driver_name = feature_names[max_idx]
            
            clean_name = raw_driver_name
            if raw_driver_name.startswith('cat__'):
                clean_name = raw_driver_name.replace('cat__', '').replace('_', ' ')
            elif raw_driver_name.startswith('num__'):
                clean_name = raw_driver_name.replace('num__', '')
            primary_driver = clean_name
        else:
            primary_driver = "None (Stable)"
            
        return JsonResponse({
            'success': True,
            'probability': prob,
            'risk_category': risk_category,
            'primary_driver': primary_driver
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
def ChatbotAPIView(request):
    """
    API endpoint that receives chatbot queries and retrieves policy-informed responses.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
        
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({'success': False, 'error': 'Message cannot be empty'}, status=400)
            
        from chatbot.services import generate_retention_response
        response_text = generate_retention_response(user_message)
        
        return JsonResponse({
            'success': True,
            'response': response_text
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
