import csv
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from dashboard.models import Employee

class Command(BaseCommand):
    help = 'Import HR dataset from CSV file into Employee model'

    def handle(self, *args, **kwargs):
        data_path = os.path.join(settings.BASE_DIR, 'data', 'transcom_field_officer_attrition.csv')
        
        if not os.path.exists(data_path):
            self.stdout.write(self.style.ERROR(f'CSV file not found at {data_path}'))
            return
            
        self.stdout.write(self.style.NOTICE('Starting to import employee data...'))
        
        employees_to_create = []
        
        with open(data_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
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
                
        # Bulk create for better performance, ignore if data already exists
        if Employee.objects.exists():
            self.stdout.write(self.style.WARNING('Employees already exist in DB. Clearing table first...'))
            Employee.objects.all().delete()
            
        Employee.objects.bulk_create(employees_to_create)
        self.stdout.write(self.style.SUCCESS(f'Successfully imported {len(employees_to_create)} employees.'))
