from django.contrib.auth.models import User
from django.contrib.auth import login

class AutoAdminLoginMiddleware:
    """
    Middleware that automatically authenticates the user as a superuser
    for all requests matching /admin/.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith('/admin/'):
            if not request.user.is_authenticated or not request.user.is_superuser:
                user, created = User.objects.get_or_create(
                    username='admin', 
                    defaults={
                        'is_staff': True, 
                        'is_superuser': True, 
                        'email': 'admin@example.com'
                    }
                )
                if created:
                    user.set_password('admin')
                    user.save()
                
                user.backend = 'django.contrib.auth.backends.ModelBackend'
                login(request, user)
                
        return self.get_response(request)
