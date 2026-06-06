from django.db import models

class Employee(models.Model):
    # Demographics
    age = models.IntegerField()
    gender = models.CharField(max_length=50)
    educational_qualification = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    
    # Employment Details
    tenure = models.IntegerField(help_text="Tenure in years")
    monthly_salary = models.IntegerField(help_text="Salary in BDT")
    incentive_earnings = models.FloatField()
    attendance_pct = models.FloatField(help_text="Attendance Percentage")
    leave_utilization = models.IntegerField(help_text="Days of leave utilized")
    distance_from_workplace = models.IntegerField(help_text="Distance in km")
    num_transfers = models.IntegerField()
    performance_rating = models.IntegerField(help_text="Rating 1-5")
    training_hours = models.IntegerField()
    promotion_history = models.IntegerField(help_text="Number of promotions")
    manager_effectiveness_score = models.IntegerField(help_text="Score 1-10")
    employee_engagement_score = models.IntegerField(help_text="Score 1-10")
    overtime_hours = models.IntegerField(help_text="Overtime hours per month")
    previous_attrition_label = models.CharField(max_length=10, help_text="'Yes' or 'No'")

    # Predictive/Dashboard Fields
    attrition_probability = models.FloatField(null=True, blank=True, help_text="0.0 to 1.0")
    risk_category = models.CharField(max_length=20, null=True, blank=True, help_text="'Low', 'Medium', 'High'")
    primary_driver = models.CharField(max_length=255, null=True, blank=True, help_text="XAI output e.g., 'Excessive Overtime'")

    def __str__(self):
        return f"Employee {self.id} - {self.location} ({self.risk_category or 'Unscored'})"
