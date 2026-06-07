from django.test import TestCase, Client
from django.urls import reverse
from dashboard.models import Employee

class AdvancedAnalyticsTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('advanced_analytics_api')
        
        # Create a few test employees
        Employee.objects.create(
            age=30, gender='Male', educational_qualification='Bachelors', location='Dhaka',
            tenure=3, monthly_salary=30000, incentive_earnings=3000, attendance_pct=95.0,
            leave_utilization=10, distance_from_workplace=15, num_transfers=0, performance_rating=4,
            training_hours=20, promotion_history=1, manager_effectiveness_score=8,
            employee_engagement_score=7, overtime_hours=12, previous_attrition_label='No',
            attrition_probability=0.15, risk_category='Low', primary_driver='None (Stable)'
        )
        Employee.objects.create(
            age=28, gender='Female', educational_qualification='Masters', location='Chittagong',
            tenure=5, monthly_salary=18000, incentive_earnings=1500, attendance_pct=82.0,
            leave_utilization=22, distance_from_workplace=28, num_transfers=1, performance_rating=2,
            training_hours=15, promotion_history=0, manager_effectiveness_score=2,
            employee_engagement_score=3, overtime_hours=45, previous_attrition_label='Yes',
            attrition_probability=0.78, risk_category='High', primary_driver='Overtime Hours'
        )

    def test_endpoint_resolves_and_responds(self):
        """Verify the advanced-analytics API endpoint responds with success and expected keys."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data.get('success'))
        self.assertEqual(data.get('count'), 2)
        self.assertIn('radar', data)
        self.assertIn('bubble', data)
        self.assertIn('line', data)
        
        # Verify radar format
        radar = data['radar']
        self.assertEqual(len(radar['labels']), 5)
        self.assertEqual(len(radar['values']), 5)
        
        # Verify bubble format
        bubble = data['bubble']
        self.assertEqual(len(bubble), 2)
        self.assertEqual(bubble[0]['x'], 15)
        self.assertEqual(bubble[0]['risk_category'], 'Low')
        self.assertEqual(bubble[1]['risk_category'], 'High')
        
        # Verify line format
        line = data['line']
        self.assertEqual(len(line['labels']), 12)
        self.assertEqual(len(line['values']), 12)

    def test_endpoint_filtering(self):
        """Verify that filters like location and risk_category successfully isolate subsets."""
        # Test location filter
        response_dhaka = self.client.get(self.url + '?location=Dhaka')
        data_dhaka = response_dhaka.json()
        self.assertEqual(data_dhaka['count'], 1)
        self.assertEqual(data_dhaka['bubble'][0]['x'], 15)
        
        # Test risk category filter
        response_high = self.client.get(self.url + '?risk=High')
        data_high = response_high.json()
        self.assertEqual(data_high['count'], 1)
        self.assertEqual(data_high['bubble'][0]['risk_category'], 'High')
        
        # Test no results match
        response_none = self.client.get(self.url + '?location=Sylhet')
        data_none = response_none.json()
        self.assertEqual(data_none['count'], 0)
        self.assertEqual(data_none['radar']['values'], [0.0, 0.0, 0.0, 0.0, 0.0])

class DashboardStatsTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.stats_url = reverse('dashboard_stats_api')
        self.upload_url = reverse('upload_csv')
        self.run_predictions_url = reverse('run_predictions')
        
        # Seed test database with a few employees who do not have predictions yet
        Employee.objects.create(
            age=30, gender='Male', educational_qualification='Bachelors', location='Dhaka',
            tenure=3, monthly_salary=30000, incentive_earnings=3000, attendance_pct=95.0,
            leave_utilization=10, distance_from_workplace=15, num_transfers=0, performance_rating=4,
            training_hours=20, promotion_history=1, manager_effectiveness_score=8,
            employee_engagement_score=7, overtime_hours=12, previous_attrition_label='No'
            # risk_category, attrition_probability, primary_driver are NULL by default
        )
        
    def test_dashboard_stats_empty_predictions(self):
        """Verify that dashboard stats can handle records with no predictions."""
        response = self.client.get(self.stats_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['total_employees'], 1)
        self.assertEqual(data['high_risk_count'], 0)
        self.assertEqual(data['medium_risk_count'], 0)
        self.assertEqual(data['current_attrition_rate'], 0.0)
        self.assertEqual(data['risk_distribution']['Low'], 0)
        self.assertEqual(data['risk_distribution']['Medium'], 0)
        self.assertEqual(data['risk_distribution']['High'], 0)
        
    def test_run_predictions_updates_stats(self):
        """Verify that running predictions updates the employees and stats in the database."""
        # Initial stats
        data_before = self.client.get(self.stats_url).json()
        self.assertEqual(data_before['high_risk_count'], 0)
        
        # Run predictions
        response = self.client.post(self.run_predictions_url)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
        
        # Verify employee is predicted
        emp = Employee.objects.first()
        self.assertIsNotNone(emp.attrition_probability)
        self.assertIsNotNone(emp.risk_category)
        self.assertIsNotNone(emp.primary_driver)
        
        # Updated stats
        data_after = self.client.get(self.stats_url).json()
        self.assertEqual(data_after['total_employees'], 1)
        # Check if the stats count reflects the computed risk category
        category = emp.risk_category
        self.assertEqual(data_after['risk_distribution'][category], 1)

