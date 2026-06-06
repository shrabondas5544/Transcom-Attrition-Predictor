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
    monthly_salary = np.random.randint(25000, 150000, size=num_employees) # BDT roughly
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
    
    # Calculate Attrition Label based on some logical rules to make the dataset somewhat realistic
    # Higher probability if: low engagement, high overtime, high distance, low manager score
    attrition_prob = np.zeros(num_employees)
    
    for i in range(num_employees):
        prob = 0.1 # base probability
        
        if employee_engagement_score[i] < 4: prob += 0.2
        if overtime_hours[i] > 30: prob += 0.15
        if distance_from_workplace[i] > 20: prob += 0.1
        if manager_effectiveness_score[i] < 4: prob += 0.15
        if performance_rating[i] <= 2: prob += 0.1
        if monthly_salary[i] < 40000: prob += 0.1
        if tenure[i] < 2: prob += 0.1
        
        # Cap probability at 0.95
        prob = min(prob, 0.95)
        attrition_prob[i] = prob
        
    # Scale probabilities slightly to match the ~30% target
    target_attrition_rate = 0.32
    current_mean_prob = np.mean(attrition_prob)
    scaling_factor = target_attrition_rate / current_mean_prob
    
    attrition_prob = np.clip(attrition_prob * scaling_factor, 0, 1)
    
    previous_attrition_label = np.random.binomial(1, attrition_prob)
    previous_attrition_label = ['Yes' if val == 1 else 'No' for val in previous_attrition_label]
    
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
