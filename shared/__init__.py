"""
shared/__init__.py

A collection of Python modules for assisting in the automatic detection of and reporting speeders

"""
# shared/__init__.py
from .mqtt_client_wrapper import MqttClientWrapper
from .config_loader import load_config
from .home_assistant_api_helper import HomeAssistantRest, HomeAssistantWebSocket
from .unifi_protect_helpers import Protect, ProtectMediaNotAvailable
from . import opencv_detection_helpers
from . import opencv_contours_clustering
from .opencv_classifier import Classifier, Detection
from .video_processor import VideoProcessor
from .summary_generator import SummaryGenerator
from .openai_license_plate_reader import OpenAILicensePlateReader

