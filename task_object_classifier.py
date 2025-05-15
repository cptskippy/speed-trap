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
import asyncio
import json
from datetime import datetime
from shared import MqttClientWrapper, Protect, ProtectMediaNotAvailable, load_config


VERBOSE = False
DEBUG = False
LOGGING = False

# Load configuration from yaml
config = load_config()
mqtt_config = config["servers"]["mqtt"]
protect_config = config["servers"]["unifi_protect"]
task_config = config["task"]["object_classifier"]

# Populate configuration variables
MQTT_URI = mqtt_config["uri"]
MQTT_USER = mqtt_config["username"]
MQTT_PASSWORD = mqtt_config["password"]
MQTT_QOS = mqtt_config["qos"]

MQTT_CLIENT_ID = task_config["mqtt"]["client_id"]
MQTT_SUBSCRIBE_TOPIC = task_config["mqtt"]["topics"]["subscribe"]
MQTT_PUBLISH_TOPIC = task_config["mqtt"]["topics"]["publish"]
MQTT_ERROR_TOPIC = task_config["mqtt"]["topics"]["error"]

UI_URI = protect_config["uri"]
UI_USERNAME = protect_config["username"]
UI_PASSWORD = protect_config["password"]
CAMERA_NAMES = protect_config["camera_names"]

# Configure NVR client
NVR_CLIENT = Protect(UI_URI, UI_USERNAME, UI_PASSWORD)


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

async def save_media(nvr_client, cams: list[dict[str, any]], dt, path: str):
    """Retrieves images from specified cameras"""
    output_files = []

    for camera in cams:

        filename = path  + "/" + camera["filename"]
        cam_id = camera["id"]
        print(f"  For camera: {cam_id}")

        still_name = await nvr_client.save_still(cam_id, dt, filename)

        output_files.append(still_name)
        print(f"  Saved image: {still_name}\n")

    return output_files

def handle_event(data):
    """Creates a folder based on the timestamp of the event"""

    # Convert timestamp to string
    timestamp = data.get("timestamp")
    occurred = datetime.fromisoformat(timestamp)

    try:
        print("Saving media...")
        folder = data.get("folder")
        images = asyncio.run(save_media(NVR_CLIENT, cameras, occurred, folder))

        # Update Payload
        data["images"] = images
        payload = json.dumps(data)
        print(f"  New Payload: {payload}")

        # Publish message for next task
        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        print(f"  Message Published: {MQTT_PUBLISH_TOPIC}")

    except ProtectMediaNotAvailable as e:
        print("Media Not Available.")
        print(e)

        # Update Payload
        data["error"] = e
        payload = json.dumps(data)

        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        print(f"  Error Message Published: {MQTT_ERROR_TOPIC}")


# Retrieve a list of cameras from the NVR
cameras = asyncio.run(NVR_CLIENT.get_cameras(CAMERA_NAMES))

# Configure MQTT and wait...
client = MqttClientWrapper(MQTT_URI, MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)
client.connect(on_connect, on_message)
