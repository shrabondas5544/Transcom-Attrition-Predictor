from django.contrib import admin
from .models import Employee

@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('id', 'age', 'gender', 'location', 'risk_category', 'attrition_probability')
    list_filter = ('risk_category', 'location', 'gender', 'previous_attrition_label')
    search_fields = ('location', 'primary_driver')
