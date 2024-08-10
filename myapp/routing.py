# from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import re_path
from .consumers import GenerateTextConsumer

websocket_urlpatterns = [
    re_path(r'ws/generate_text/', GenerateTextConsumer.as_asgi()),
]