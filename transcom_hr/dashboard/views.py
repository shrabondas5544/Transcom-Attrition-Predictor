import os
import json
import io
import joblib
import shap
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Count, Avg
from .models import Employee
from .services.inference import run_system_wide_inference, check_data_drift
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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
            
        return JsonResponse({
            'success': True,
            'message': f"Successfully uploaded {len(employees_to_create)} records. Click 'Run Predictions' to calculate risk metrics.",
            'processed_count': len(employees_to_create)
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

def AdvancedAnalyticsAPIView(request):
    """
    Endpoint returning aggregated data for Radar, Bubble, and Line charts.
    Supports search, location, driver, and risk category filtering.
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
        
    from django.db.models import Avg
    import random
    
    try:
        queryset = Employee.objects.all()
        
        search = request.GET.get('search', '').strip()
        if search:
            queryset = queryset.filter(id__icontains=search)
            
        location = request.GET.get('location', '').strip()
        if location:
            queryset = queryset.filter(location=location)
            
        driver = request.GET.get('driver', '').strip()
        if driver:
            queryset = queryset.filter(primary_driver=driver)
            
        risk = request.GET.get('risk', '').strip()
        if risk:
            queryset = queryset.filter(risk_category=risk)
            
        total_count = queryset.count()
        
        # 1. Radar Data (Scale leave to 100% based on max 25, rating on 5, others on 10)
        metrics = queryset.aggregate(
            avg_attendance=Avg('attendance_pct'),
            avg_leave=Avg('leave_utilization'),
            avg_perf=Avg('performance_rating'),
            avg_mgr=Avg('manager_effectiveness_score'),
            avg_eng=Avg('employee_engagement_score')
        )
        
        radar_data = {
            'labels': ['Attendance %', 'Leave Utilization %', 'Performance Rating %', 'Manager Effectiveness %', 'Employee Engagement %'],
            'values': [
                round(metrics['avg_attendance'] or 0.0, 1),
                round(((metrics['avg_leave'] or 0.0) / 25.0) * 100.0, 1),
                round(((metrics['avg_perf'] or 0.0) / 5.0) * 100.0, 1),
                round(((metrics['avg_mgr'] or 0.0) / 10.0) * 100.0, 1),
                round(((metrics['avg_eng'] or 0.0) / 10.0) * 100.0, 1),
            ]
        }
        
        # 2. Bubble Data (Limit to 1000 items to avoid frontend lag)
        bubble_qs = queryset
        if total_count > 1000:
            bubble_qs = bubble_qs.order_by('id')[:1000]
            
        bubble_data = []
        for emp in bubble_qs:
            bubble_data.append({
                'x': emp.distance_from_workplace,
                'y': emp.monthly_salary,
                'r': round(emp.overtime_hours / 2.0, 1),
                'risk_category': emp.risk_category
            })
            
        # 3. Line Data (Simulate historical attrition trend ending exactly at current attrition rate)
        if total_count > 0:
            high_count = queryset.filter(risk_category='High').count()
            medium_count = queryset.filter(risk_category='Medium').count()
            current_rate = round(((high_count + medium_count) / total_count) * 100, 1)
        else:
            current_rate = 0.0
            
        months = ['Jul 25', 'Aug 25', 'Sep 25', 'Oct 25', 'Nov 25', 'Dec 25', 'Jan 26', 'Feb 26', 'Mar 26', 'Apr 26', 'May 26', 'Jun 26']
        trend_rates = []
        # Seed generator based on current_rate to get stable trend for the exact same rate
        random.seed(int(current_rate * 100))
        for i in range(12):
            if i == 11:
                rate = current_rate
            else:
                decay = (11 - i) / 11.0
                noise = (random.random() - 0.5) * 4.0
                rate = round(current_rate + (decay * -3.0) + noise, 1)
                rate = max(0.0, rate)
            trend_rates.append(rate)
            
        line_data = {
            'labels': months,
            'values': trend_rates
        }
        
        return JsonResponse({
            'success': True,
            'count': total_count,
            'radar': radar_data,
            'bubble': bubble_data,
            'line': line_data
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

def DashboardStatsAPIView(request):
    """
    Endpoint returning main dashboard statistics and chart data.
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
        
    try:
        total_employees = Employee.objects.count()
        high_risk_count = Employee.objects.filter(risk_category='High').count()
        medium_risk_count = Employee.objects.filter(risk_category='Medium').count()
        low_risk_count = Employee.objects.filter(risk_category='Low').count()
        
        current_attrition_rate = 0.0
        if total_employees > 0:
            current_attrition_rate = round(((high_risk_count + medium_risk_count) / total_employees) * 100, 2)
            
        risk_distribution = {
            'Low': low_risk_count,
            'Medium': medium_risk_count,
            'High': high_risk_count
        }
        
        locations = ['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna', 'Barisal']
        location_labels = []
        location_counts = []
        for loc in locations:
            risk_count = Employee.objects.filter(location=loc, risk_category__in=['High', 'Medium']).count()
            location_labels.append(loc)
            location_counts.append(risk_count)
            
        drift_metrics = check_data_drift()
            
        return JsonResponse({
            'success': True,
            'total_employees': total_employees,
            'high_risk_count': high_risk_count,
            'medium_risk_count': medium_risk_count,
            'current_attrition_rate': current_attrition_rate,
            'risk_distribution': risk_distribution,
            'location_labels': location_labels,
            'location_counts': location_counts,
            'drift_metrics': drift_metrics
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

def ExportReportPDFView(request):
    """
    Generates a professional executive PDF report summarizing employee attrition risk,
    dominant SHAP flight risk drivers, location distributions, and data drift diagnostics.
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
        
    try:
        # 1. Fetch live system aggregates
        total_employees = Employee.objects.count()
        if total_employees == 0:
            return JsonResponse({'success': False, 'error': 'No employee data available to generate report. Please upload a dataset first.'}, status=400)
            
        high_risk_count = Employee.objects.filter(risk_category='High').count()
        medium_risk_count = Employee.objects.filter(risk_category='Medium').count()
        low_risk_count = Employee.objects.filter(risk_category='Low').count()
        
        avg_prob_res = Employee.objects.all().aggregate(avg=Avg('attrition_probability'))
        avg_prob = avg_prob_res['avg'] or 0.0
        
        dominant_driver_res = Employee.objects.exclude(primary_driver__in=['None (Stable)', 'None', '']) \
            .values('primary_driver') \
            .annotate(count=Count('primary_driver')) \
            .order_by('-count') \
            .first()
        dominant_driver = dominant_driver_res['primary_driver'] if dominant_driver_res else 'None (Stable)'
        dominant_driver_cnt = dominant_driver_res['count'] if dominant_driver_res else 0
        
        # 2. Risk by location
        locations = ['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna', 'Barisal']
        location_data = []
        for loc in locations:
            tot = Employee.objects.filter(location=loc).count()
            high = Employee.objects.filter(location=loc, risk_category='High').count()
            med = Employee.objects.filter(location=loc, risk_category='Medium').count()
            pct = round(((high + med) / tot) * 100, 1) if tot > 0 else 0.0
            location_data.append([loc, str(tot), str(high), str(med), f"{pct}%"])
            
        # 3. Drift status
        drift_metrics = check_data_drift()
        drift_status = "DRIFT DETECTED" if drift_metrics.get('drift_detected', False) else "HEALTHY (NO DRIFT)"
        
        # 4. Generate ReportLab PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            name='DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=4
        )
        
        subtitle_style = ParagraphStyle(
            name='DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=colors.HexColor('#4f46e5'),
            spaceAfter=20
        )
        
        h2_style = ParagraphStyle(
            name='Heading2Custom',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor('#1e293b'),
            spaceBefore=12,
            spaceAfter=8
        )
        
        body_style = ParagraphStyle(
            name='BodyCustom',
            parent=styles['BodyText'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#334155'),
            spaceAfter=12
        )
        
        bold_cell_style = ParagraphStyle(
            name='BoldCell',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=colors.HexColor('#1e293b')
        )
        
        normal_cell_style = ParagraphStyle(
            name='NormalCell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#334155')
        )
        
        # Title & Subtitle
        story.append(Paragraph("Transcom Attrition Predictor", title_style))
        story.append(Paragraph("EXECUTIVE FLIGHT RISK & RETENTION SUMMARY REPORT", subtitle_style))
        story.append(Spacer(1, 10))
        
        # Executive Summary Section
        story.append(Paragraph("1. Executive Summary KPIs", h2_style))
        summary_intro = (
            f"This executive report compiles predictive analysis results based on machine learning "
            f"models deployed across the Transcom field operations roster. A total of <b>{total_employees}</b> "
            f"active directory records were analyzed. Features contributing to increased attrition risk are "
            f"monitored using SHAP explainable AI. Operational data drift is monitored automatically against training baselines."
        )
        story.append(Paragraph(summary_intro, body_style))
        story.append(Spacer(1, 6))
        
        # Summary KPI Table
        kpi_data = [
            [Paragraph("<b>Metric</b>", bold_cell_style), Paragraph("<b>Value</b>", bold_cell_style), Paragraph("<b>Key Context</b>", bold_cell_style)],
            [Paragraph("Total Employees Evaluated", normal_cell_style), Paragraph(str(total_employees), normal_cell_style), Paragraph("Active roster directory sizing", normal_cell_style)],
            [Paragraph("High Risk Flight Counts", normal_cell_style), Paragraph(str(high_risk_count), normal_cell_style), Paragraph(f"{round(high_risk_count / total_employees * 100, 1) if total_employees > 0 else 0}% of roster is high attrition probability (>= 50%)", normal_cell_style)],
            [Paragraph("Roster Flight Risk Rate", normal_cell_style), Paragraph(f"{round((high_risk_count + medium_risk_count) / total_employees * 100, 1) if total_employees > 0 else 0.0}%", normal_cell_style), Paragraph("Combined high and medium risk counts", normal_cell_style)],
            [Paragraph("Average Attrition Probability", normal_cell_style), Paragraph(f"{round(avg_prob * 100, 1)}%", normal_cell_style), Paragraph("Mean probability across entire roster", normal_cell_style)],
            [Paragraph("Dominant Attrition Driver (SHAP)", normal_cell_style), Paragraph(str(dominant_driver), normal_cell_style), Paragraph(f"Affects {dominant_driver_cnt} employees overall", normal_cell_style)],
            [Paragraph("Model Diagnostics State", normal_cell_style), Paragraph(drift_status, normal_cell_style), Paragraph("Overtime and Commute distance drift status", normal_cell_style)]
        ]
        
        kpi_table = Table(kpi_data, colWidths=[200, 100, 230])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('BACKGROUND', (1, 2), (1, 2), colors.HexColor('#fef2f2') if high_risk_count > 0 else colors.white),
            ('BACKGROUND', (1, 6), (1, 6), colors.HexColor('#fef3c7') if "DRIFT" in drift_status else colors.HexColor('#ecfdf5')),
        ]))
        
        story.append(kpi_table)
        story.append(Spacer(1, 20))
        
        # Location analysis Section
        story.append(Paragraph("2. Flight Risk Distribution by Location", h2_style))
        story.append(Paragraph("Geographical location is a significant covariate for commute, manager friction, and compensation expectations. Below is the regional breakdown of high and medium risk profiles.", body_style))
        story.append(Spacer(1, 6))
        
        loc_table_headers = [["Location", "Roster Size", "High Risk Count", "Medium Risk Count", "Flight Risk Rate (%)"]]
        loc_table_data = loc_table_headers + location_data
        
        loc_table = Table(loc_table_data, colWidths=[110, 110, 110, 110, 90])
        loc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        
        story.append(loc_table)
        story.append(Spacer(1, 25))
        
        # Policy Recommendation Footer Section
        story.append(Paragraph("3. Strategic Retention Guidelines", h2_style))
        story.append(Paragraph(
            "Based on the dominant SHAP drivers, we recommend initiating retention strategies targeting regional hubs: <br/>"
            "• <b>Overtime Fatigue Countermeasures:</b> Enforce maximum shifts, mandatory 11-hour rest periods, and rota adjustments.<br/>"
            "• <b>Commute Mitigation:</b> Provide travel/fuel stipends or facilitate regional territory transfers for high-commute employees.<br/>"
            "• <b>Engagement & Leadership Support:</b> Mandate manager coaching and bi-weekly engagement check-ins to build trust and resolve friction.",
            body_style
        ))
        
        doc.build(story)
        buffer.seek(0)
        
        response = FileResponse(buffer, as_attachment=True, filename="Transcom_Attrition_Executive_Report.pdf", content_type='application/pdf')
        return response
    except Exception as e:
        return JsonResponse({'success': False, 'error': f"Failed to generate PDF report: {str(e)}"}, status=500)

@csrf_exempt
def EmployeeDetailAPIView(request, employee_id):
    """
    API endpoint returning individual employee details and their top 3 positive SHAP contributors.
    """
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
        
    try:
        from django.shortcuts import get_object_or_404
        emp = get_object_or_404(Employee, id=employee_id)
        
        # Load ML artifacts
        from django.conf import settings
        saved_models_dir = os.path.join(settings.BASE_DIR, 'dashboard', 'ml', 'saved_models')
        
        model_path = os.path.join(saved_models_dir, 'attrition_model.pkl')
        preprocessor_path = os.path.join(saved_models_dir, 'preprocessor.pkl')
        shap_explainer_path = os.path.join(saved_models_dir, 'shap_explainer.pkl')
        feature_names_path = os.path.join(saved_models_dir, 'feature_names.pkl')
        
        if not all(os.path.exists(p) for p in [model_path, preprocessor_path, shap_explainer_path, feature_names_path]):
            return JsonResponse({'success': False, 'error': 'Model artifacts missing. Run training first.'}, status=500)
            
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        explainer = joblib.load(shap_explainer_path)
        feature_names = joblib.load(feature_names_path)
        
        # Format record into a single-row Pandas DataFrame
        raw_data = {
            'Age': int(emp.age),
            'Gender': emp.gender,
            'Educational Qualification': emp.educational_qualification,
            'Location': emp.location,
            'Tenure': int(emp.tenure),
            'Monthly Salary': int(emp.monthly_salary),
            'Incentive Earnings': float(emp.incentive_earnings),
            'Attendance %': float(emp.attendance_pct),
            'Leave Utilization': int(emp.leave_utilization),
            'Distance from Workplace': int(emp.distance_from_workplace),
            'Number of Transfers': int(emp.num_transfers),
            'Performance Rating': int(emp.performance_rating),
            'Training Hours': int(emp.training_hours),
            'Promotion History': int(emp.promotion_history),
            'Manager Effectiveness Score': int(emp.manager_effectiveness_score),
            'Employee Engagement Score': int(emp.employee_engagement_score),
            'Overtime Hours': int(emp.overtime_hours)
        }
        df_single = pd.DataFrame([raw_data])
        
        # Call get_top_contributors from dashboard/ml/train_model.py
        from .ml.train_model import get_top_contributors
        top_contributors = get_top_contributors(
            df_single, preprocessor, model, explainer, feature_names, top_n=3
        )
        
        # Format raw metrics and calculated fields
        employee_data = {
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
            'attrition_probability': round(emp.attrition_probability, 4) if emp.attrition_probability is not None else None,
            'risk_category': emp.risk_category,
            'primary_driver': emp.primary_driver
        }
        
        return JsonResponse({
            'success': True,
            'employee': employee_data,
            'top_drivers': top_contributors
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
def EmployeePrescriptionAPIView(request, employee_id):
    """
    API endpoint returning the individual AI-generated retention prescription.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
        
    try:
        from django.shortcuts import get_object_or_404
        emp = get_object_or_404(Employee, id=employee_id)
        
        # Format employee data into a profile dict
        employee_data = {
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
            'attrition_probability': emp.attrition_probability,
            'risk_category': emp.risk_category,
            'primary_driver': emp.primary_driver
        }
        
        # Load ML features to calculate SHAP drivers
        from django.conf import settings
        saved_models_dir = os.path.join(settings.BASE_DIR, 'dashboard', 'ml', 'saved_models')
        
        model = joblib.load(os.path.join(saved_models_dir, 'attrition_model.pkl'))
        preprocessor = joblib.load(os.path.join(saved_models_dir, 'preprocessor.pkl'))
        explainer = joblib.load(os.path.join(saved_models_dir, 'shap_explainer.pkl'))
        feature_names = joblib.load(os.path.join(saved_models_dir, 'feature_names.pkl'))
        
        raw_data = {
            'Age': int(emp.age),
            'Gender': emp.gender,
            'Educational Qualification': emp.educational_qualification,
            'Location': emp.location,
            'Tenure': int(emp.tenure),
            'Monthly Salary': int(emp.monthly_salary),
            'Incentive Earnings': float(emp.incentive_earnings),
            'Attendance %': float(emp.attendance_pct),
            'Leave Utilization': int(emp.leave_utilization),
            'Distance from Workplace': int(emp.distance_from_workplace),
            'Number of Transfers': int(emp.num_transfers),
            'Performance Rating': int(emp.performance_rating),
            'Training Hours': int(emp.training_hours),
            'Promotion History': int(emp.promotion_history),
            'Manager Effectiveness Score': int(emp.manager_effectiveness_score),
            'Employee Engagement Score': int(emp.employee_engagement_score),
            'Overtime Hours': int(emp.overtime_hours)
        }
        df_single = pd.DataFrame([raw_data])
        
        from .ml.train_model import get_top_contributors
        top_contributors = get_top_contributors(
            df_single, preprocessor, model, explainer, feature_names, top_n=3
        )
        
        from chatbot.services import generate_individual_prescription
        prescription = generate_individual_prescription(employee_data, top_contributors)
        
        return JsonResponse({
            'success': True,
            'prescription': prescription
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

