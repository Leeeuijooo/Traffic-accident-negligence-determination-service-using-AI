from django.urls import path
from . import views
from django.urls import path, include


urlpatterns = [

    path('', views.home),
    path('check/', views.check),
    path('result/', views.detectobj),
    path('factor/', views.factor),

]
