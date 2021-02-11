from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='inference-home'),
    path('about', views.about, name='inference=about')
]