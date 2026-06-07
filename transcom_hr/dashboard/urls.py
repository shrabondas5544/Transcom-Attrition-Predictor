from django.urls import path
from .views import TriggerInferenceView, EmployeeListAPIView, dashboard_home, UploadCSVView, PredictSingleView

urlpatterns = [
    path('', dashboard_home, name='dashboard_home'),
    path('run-predictions/', TriggerInferenceView, name='run_predictions'),
    path('employees/', EmployeeListAPIView, name='employee_list_api'),
    path('upload-csv/', UploadCSVView, name='upload_csv'),
    path('predict-single/', PredictSingleView, name='predict_single'),
]
