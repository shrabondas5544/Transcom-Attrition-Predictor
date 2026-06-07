import os
import random
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Generate 10 distinct sample CSV & Excel files for attrition predictions'

    def handle(self, *args, **kwargs):
        # Create output directory
        samples_dir = os.path.join(settings.BASE_DIR, 'data', 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        # Define sizes and formats
        csv_sizes = [5, 50, 300, 1000, 5000]
        xlsx_sizes = [10, 100, 500, 3000, 10000]
        
        self.stdout.write(self.style.NOTICE(f'Generating sample datasets in {samples_dir}...'))
        
        # 1. Generate CSV files
        for size in csv_sizes:
            df = self.generate_data(size, seed=42 + size)
            file_path = os.path.join(samples_dir, f'sample_{size}.csv')
            df.to_csv(file_path, index=False)
            self.stdout.write(self.style.SUCCESS(f'Successfully generated CSV: {file_path} ({size} rows)'))
            
        # 2. Generate Excel files
        for size in xlsx_sizes:
            df = self.generate_data(size, seed=100 + size)
            file_path = os.path.join(samples_dir, f'sample_{size}.xlsx')
            df.to_excel(file_path, index=False, engine='openpyxl')
            self.stdout.write(self.style.SUCCESS(f'Successfully generated Excel: {file_path} ({size} rows)'))
            
        self.stdout.write(self.style.SUCCESS('All 10 sample datasets generated successfully.'))
        
    def generate_data(self, num_employees, seed):
        # Set seeds for reproducibility of this specific size
        np.random.seed(seed)
        random.seed(seed)
        
        genders = ['Male', 'Female', 'Other']
        qualifications = ['High School', 'Bachelors', 'Masters', 'PhD']
        locations = ['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna', 'Barisal']
        
        # Generate features
        age = np.random.randint(22, 60, size=num_employees)
        gender = np.random.choice(genders, size=num_employees, p=[0.55, 0.40, 0.05])
        education = np.random.choice(qualifications, size=num_employees, p=[0.2, 0.5, 0.25, 0.05])
        location = np.random.choice(locations, size=num_employees)
        tenure = np.random.randint(1, 15, size=num_employees)
        monthly_salary = np.random.randint(15000, 150000, size=num_employees)
        incentive_earnings = monthly_salary * np.random.uniform(0.05, 0.2, size=num_employees)
        attendance_pct = np.random.uniform(75, 100, size=num_employees)
        leave_utilization = np.random.randint(5, 25, size=num_employees)
        distance_from_workplace = np.random.randint(1, 30, size=num_employees)
        num_transfers = np.random.randint(0, 5, size=num_employees)
        performance_rating = np.random.randint(1, 6, size=num_employees)
        training_hours = np.random.randint(10, 100, size=num_employees)
        promotion_history = np.random.randint(0, 4, size=num_employees)
        manager_effectiveness_score = np.random.randint(1, 11, size=num_employees)
        employee_engagement_score = np.random.randint(1, 11, size=num_employees)
        overtime_hours = np.random.randint(0, 50, size=num_employees)
        
        # Calculate Attrition Label based on Phase 3 logit-weighting system
        logits = np.zeros(num_employees)
        for i in range(num_employees):
            logit = -1.5 # baseline logit
            
            if overtime_hours[i] > 40:
                logit += 2.2
            if distance_from_workplace[i] > 25:
                logit += 1.8
            if manager_effectiveness_score[i] < 2.5:
                logit += 2.0
            if monthly_salary[i] < 22000:
                logit += 1.5
            if performance_rating[i] in [1, 2]:
                logit += 1.2
                
            logits[i] = logit
            
        # Sigmoid function
        attrition_prob = 1 / (1 + np.exp(-logits))
        
        # Assign 'Yes' to the top 32% of employees with the highest probabilities
        num_attrite = int(num_employees * 0.32)
        top_indices = np.argsort(attrition_prob)[-num_attrite:] if num_attrite > 0 else []
        
        previous_attrition_label = ['No'] * num_employees
        for idx in top_indices:
            previous_attrition_label[idx] = 'Yes'
            
        # Create DataFrame matching exactly the 18 mandatory columns
        data = {
            'Age': age,
            'Gender': gender,
            'Educational Qualification': education,
            'Location': location,
            'Tenure': tenure,
            'Monthly Salary': monthly_salary,
            'Incentive Earnings': np.round(incentive_earnings, 2),
            'Attendance %': np.round(attendance_pct, 2),
            'Leave Utilization': leave_utilization,
            'Distance from Workplace': distance_from_workplace,
            'Number of Transfers': num_transfers,
            'Performance Rating': performance_rating,
            'Training Hours': training_hours,
            'Promotion History': promotion_history,
            'Manager Effectiveness Score': manager_effectiveness_score,
            'Employee Engagement Score': employee_engagement_score,
            'Overtime Hours': overtime_hours,
            'Previous Attrition Label': previous_attrition_label
        }
        
        return pd.DataFrame(data)
