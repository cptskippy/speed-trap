"""
task_object_classifier.py

Subscribes to an MQTT topic and classifies objects in 
video clips from the NVR.

TODO::
* Load videos from MQTT payload
* Use OpenCV to pull stills
* Classify Objects using Model
* Update MQTT with images & meta data
"""
import logging
import cv2
from datetime import datetime
import imutils
import json
import os

from shared import MqttClientWrapper, load_config
from shared import VideoProcessor, load_config

logger = logging.getLogger(__name__)

# Load configuration from yaml
config = load_config()
mqtt_config = config["servers"]["mqtt"]
task_config = config["task"]["object_classifier"]

camera_details = config["cameras"]
video_extension = config["media"]["video_extension"]
image_extension = config["media"]["image_extension"]
open_cv_settings = task_config["open_cv_settings"]
dnn = open_cv_settings["deep_neural_network"]

PROTOTXT = os.getcwd() + dnn["prototxt_path"]
MODEL = os.getcwd() + dnn["model_path"]
CLASSES = os.getcwd() + dnn["classes_path"]
CLASSES_TO_TRACK = dnn["classes_to_track"] #Bicycle,Bus,Car,Motorbike
CONFIDENCE_THRESHOLD = dnn["confidence_threshold"]
LEARNING_RATE = open_cv_settings["learning_rate"]


# Populate configuration variables
MQTT_URI = mqtt_config["uri"]
MQTT_USER = mqtt_config["username"]
MQTT_PASSWORD = mqtt_config["password"]
MQTT_QOS = mqtt_config["qos"]

MQTT_CLIENT_ID = task_config["mqtt"]["client_id"]
MQTT_SUBSCRIBE_TOPIC = task_config["mqtt"]["topics"]["subscribe"]
MQTT_PUBLISH_TOPIC = task_config["mqtt"]["topics"]["publish"]
MQTT_ERROR_TOPIC = task_config["mqtt"]["topics"]["error"]


def on_connect(client, userdata, flags, reason_code, properties):
    """Subscribe to topic on successful connection."""
    if reason_code == 0:
        client.subscribe(MQTT_SUBSCRIBE_TOPIC, MQTT_QOS)
        logger.info("Subscribed to topic: %s", MQTT_SUBSCRIBE_TOPIC)


def on_message(client, userdata, message):
    """Processes the message from MQTT Broker"""
    try:
        payload = message.payload.decode('utf-8')
        data = json.loads(payload)
        logger.info("Classifier Event Received:")
        logger.info("  Timestamp: %s", data.get('timestamp'))
        logger.info("  Sensor ID: %s", data.get('sensor_id'))
        logger.info("  Speed: %s %s", data.get('speed'), data.get('uom'))
        logger.info("  Folder: %s", data.get('folder'))
        logger.info("  Data File: %s", data.get('data_file'))
        logger.info("  Video Files: %s", data.get('videos'))
        logger.debug("  Payload: %s", payload)

        handle_event(client, data)

    except json.JSONDecodeError as e:
        logger.error("Received invalid JSON: %s", e)
    except Exception as e:
        logger.exception("Error processing message: %s", e)


def handle_event(client, data):
    # {
    #     "timestamp": "2025-06-10T22:47:44", 
    #     "speed": 28.6, 
    #     "uom": "mph", 
    #     "sensor_id": "sensor.speedometer_speed", 
    #     "folder": "./media/20250610224744", 
    #     "data_file": "./media/20250610224744/data.json", 
    #     "videos": [
    #         "./media/20250610224744/globalshutter.mpg", 
    #         "./media/20250610224744/street.mpg", 
    #         "./media/20250610224744/driveway.mpg"
    #     ]
    # }

    # Convert timestamp to string
    timestamp = data.get("timestamp")
    occurred = datetime.fromisoformat(timestamp)

    try:
        logger.info("Fetching media...")
        videos = data.get("videos")

        # VideoProcessor
        vp = VideoProcessor(PROTOTXT, MODEL, CLASSES, CLASSES_TO_TRACK, CONFIDENCE_THRESHOLD, LEARNING_RATE)

        images, thumbs = vp.process_videos(videos, camera_details, video_extension, image_extension)

        logger.info("  Images: %s", images)
        logger.info("  Thumbs: %s", thumbs)

        # Update Payload
        data["images"] = images
        data["thumbnails"] = thumbs

        logger.info("  Appending to payload...")

        payload = json.dumps(data)
        logger.debug("  New Payload: %s", payload)

        # Publish message for next task
        logger.info("  Publishing message...")
        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        logger.info("  Message Published: %s", MQTT_PUBLISH_TOPIC)

    except Exception as e:
        logger.exception("Exception occurred: %s", e)

        # Update Payload
        data["error"] = str(e)
        payload = json.dumps(data)

        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        logger.error("  Error message published to: %s", MQTT_ERROR_TOPIC)


# Configure MQTT and wait...
client = MqttClientWrapper(MQTT_URI, MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)
client.connect(on_connect, on_message)
