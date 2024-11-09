from django.urls import path
from core.views import FindSimmilar

urlpatterns = [
    path('search', FindSimmilar.as_view(), name='find_simmilar'),
]
