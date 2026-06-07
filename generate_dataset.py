import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_dataset(num_employees=3000):
    # Variables definition
    genders = ['Male', 'Female', 'Other']
    qualifications = ['High School', 'Bachelors', 'Masters', 'PhD']
    locations = ['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna', 'Barisal']
    
    # Generate features
    age = np.random.randint(22, 60, size=num_employees)
    gender = np.random.choice(genders, size=num_employees, p=[0.55, 0.40, 0.05])
    education = np.random.choice(qualifications, size=num_employees, p=[0.2, 0.5, 0.25, 0.05])
    location = np.random.choice(locations, size=num_employees)
    tenure = np.random.randint(1, 15, size=num_employees) # in years
    monthly_salary = np.random.randint(15000, 150000, size=num_employees) # BDT roughly
    incentive_earnings = monthly_salary * np.random.uniform(0.05, 0.2, size=num_employees)
    attendance_pct = np.random.uniform(75, 100, size=num_employees)
    leave_utilization = np.random.randint(5, 25, size=num_employees) # days
    distance_from_workplace = np.random.randint(1, 30, size=num_employees) # km
    num_transfers = np.random.randint(0, 5, size=num_employees)
    performance_rating = np.random.randint(1, 6, size=num_employees) # 1 to 5
    training_hours = np.random.randint(10, 100, size=num_employees)
    promotion_history = np.random.randint(0, 4, size=num_employees)
    manager_effectiveness_score = np.random.randint(1, 11, size=num_employees) # 1 to 10
    employee_engagement_score = np.random.randint(1, 11, size=num_employees) # 1 to 10
    overtime_hours = np.random.randint(0, 50, size=num_employees) # per month
    
    # Calculate Attrition Label based on logit-weighting system
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
    top_indices = np.argsort(attrition_prob)[-num_attrite:]
    
    previous_attrition_label = ['No'] * num_employees
    for idx in top_indices:
        previous_attrition_label[idx] = 'Yes'
    
    # Create DataFrame
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
    
    df = pd.DataFrame(data)
    
    # Check current attrition rate
    actual_rate = (df['Previous Attrition Label'] == 'Yes').mean()
    print(f"Generated dataset with {num_employees} rows. Actual Attrition Rate: {actual_rate:.2%}")
    
    # Ensure directory exists
    os.makedirs('transcom_hr/data', exist_ok=True)
    
    # Save to CSV
    csv_path = 'transcom_hr/data/transcom_field_officer_attrition.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset to {csv_path}")

if __name__ == "__main__":
    generate_dataset()
