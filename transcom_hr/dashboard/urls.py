from django.urls import path
from .views import (
    TriggerInferenceView, EmployeeListAPIView, dashboard_home, 
    UploadCSVView, PredictSingleView, ChatbotAPIView, 
    AdvancedAnalyticsAPIView, DashboardStatsAPIView, ExportReportPDFView,
    EmployeeDetailAPIView, EmployeePrescriptionAPIView, PredictScenarioView
)

urlpatterns = [
    path('', dashboard_home, name='dashboard_home'),
    path('run-predictions/', TriggerInferenceView, name='run_predictions'),
    path('employees/', EmployeeListAPIView, name='employee_list_api'),
    path('upload-csv/', UploadCSVView, name='upload_csv'),
    path('predict-single/', PredictSingleView, name='predict_single'),
    path('chatbot/', ChatbotAPIView, name='chatbot_api'),
    path('advanced-analytics/', AdvancedAnalyticsAPIView, name='advanced_analytics_api'),
    path('dashboard-stats/', DashboardStatsAPIView, name='dashboard_stats_api'),
    path('export-report/', ExportReportPDFView, name='export_report_pdf'),
    path('employees/<int:employee_id>/insights/', EmployeeDetailAPIView, name='employee_insights_api'),
    path('employees/<int:employee_id>/prescription/', EmployeePrescriptionAPIView, name='employee_prescription_api'),
    path('predict-scenario/', PredictScenarioView, name='predict_scenario'),
]



