from django.urls import path
from car import views

urlpatterns = [
    path('', views.index, name='index'),
    path('img_load.html', views.img_load, name='img_load'),
    path('index.html', views.index, name='index2'),
    path('pre', views.pre, name='result'),
    path('pre2', views.pre2, name='result2'),
]