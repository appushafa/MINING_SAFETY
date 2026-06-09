from django.urls import path
from .views import live_detection, detect_frame_api

urlpatterns = [
    path('', live_detection, name='live_detection'),
    path('detect_api/', detect_frame_api, name='detect_frame_api'),
]