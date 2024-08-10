from django.urls import path
from .views import predict, index

urlpatterns = [
    path('predict/', predict, name='predict'),
    path('', index, name='index'),
]