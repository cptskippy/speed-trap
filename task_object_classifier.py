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
import cv2
from datetime import datetime
import imutils
import json
import logging
import os

from shared import MqttClientWrapper, load_config
from shared import VideoProcessor, load_config


# logging.basicConfig(
#     level=logging.DEBUG,  # or INFO, WARNING
#     format='[%(levelname)s] %(name)s: %(message)s'
# )

logger = logging.getLogger(__name__)
# logger.debug("Debug message")
# logger.info("Info message")
# logger.warning("Warning message")

# Load configuration from yaml
config = load_config()
mqtt_config = config["servers"]["mqtt"]
task_config = config["task"]["object_classifier"]

video_clip_details = task_config["video_clip_details"]
open_cv_settings = task_config["open_cv_settings"]
dnn = open_cv_settings["deep_neural_network"]

PROTOTXT = os.getcwd() + dnn["prototxt_path"]
MODEL = os.getcwd() + dnn["model_path"]
CLASSES = os.getcwd() + dnn["classes_path"]
CLASSES_TO_TRACK = dnn["classes_to_track"] #Bicycle,Bus,Car,Motorbike
CONFIDENCE_THRESHOLD = dnn["confidence_threshold"]


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
        print(f"Subscribed to topic: {MQTT_SUBSCRIBE_TOPIC}")

def on_message(client, userdata, message):
    """"Processes the message from MQTT Broker"""
    try:
        payload = message.payload.decode('utf-8')
        data = json.loads(payload)
        print("\nClassifier Event Received:")
        print(f"  Timestamp: {data.get('timestamp')}")
        print(f"  Sensor ID: {data.get('sensor_id')}")
        print(f"  Speed: {data.get('speed')} {data.get('uom')}")
        print(f"  Folder: {data.get('folder')}")
        print(f"  Data File: {data.get('data_file')}")
        print(f"  Video Files: {data.get('videos')}")
        print(f"  Payload: {payload}")

        handle_event(data)

    except json.JSONDecodeError:
        print("Received invalid JSON: {message}")
    except Exception as e:
        print(f"Error processing message: {e}")

def handle_event(data):
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
        print("Fetching media...")
        videos = data.get("videos")

        # VideoProcessor
        vp = VideoProcessor(PROTOTXT, MODEL, CLASSES, CLASSES_TO_TRACK, CONFIDENCE_THRESHOLD)

        images, thumbs = vp.process_videos(videos, video_clip_details)

        print(f"  Images: {images}")
        print(f"  Thumbs: {thumbs}")

        # Update Payload
        data["images"] = images
        data["thumbnails"] = thumbs

        print(f"  Appending Payload...")
        
        payload = json.dumps(data)
        print(f"  New Payload: {payload}")

        # Publish message for next task
        print(f"  Publishing Message...")
        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        print(f"  Message Published: {MQTT_PUBLISH_TOPIC}")

    except Exception as e:
        print(f"Exception Occurred: {e}")

        # Update Payload
        data["error"] = e
        payload = json.dumps(data)

        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        print(f"  Error Message Published: {MQTT_ERROR_TOPIC}")


# Configure MQTT and wait...
client = MqttClientWrapper(MQTT_URI, MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)
client.connect(on_connect, on_message)
