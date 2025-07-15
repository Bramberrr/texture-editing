from django.urls import path
from . import views

urlpatterns = [
    path('', views.select_dataset, name='select_dataset'),
    path('skins/', views.select_skin_type, name='select_skin_type'),
    path('browse/<str:domain>/', views.home, name='browse_textures'),
    path('edit/<str:domain>/<int:index>/', views.edit_texture, name='edit_texture'),
    path('api/update_image/', views.update_image, name='update_image'),
    path('api/update_image_rough/', views.update_image_rough, name='update_image_rough'),
    path("api/submit_answer/", views.submit_answer),

]
