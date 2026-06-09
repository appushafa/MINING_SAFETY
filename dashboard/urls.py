from django.urls import path
from django.views.generic import RedirectView
from .views import dashboard, landing

urlpatterns = [
    path('', landing, name='landing'),
    path('dashboard/', dashboard, name='dashboard'),
    path(
        'dashboard//',
        RedirectView.as_view(url='/dashboard/', permanent=False),
        name='dashboard_double_slash_redirect',
    ),
]
