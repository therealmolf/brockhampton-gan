from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.home, name='inference-home'),
    path('about', views.about, name='inference-about'),
    path('song', views.generate_image, name='inference-generate'),
    path('progan', views.progan_image, name='inference-progan')
]

# urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)