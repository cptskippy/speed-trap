"""
task_video_clip_fetcher.py

Subscribes to an MQTT topic and exports video clips 
from the NVR based on timestamps in published messages
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
mqtt_config = config["mqtt"]
rest_config = config["rest"]
task_config = config["task"]["video_clip_fetcher"]
sensor_config = config["sensor"]

# Populate configuration variables
MQTT_URI = mqtt_config["uri"]
MQTT_USER = mqtt_config["username"]
MQTT_PASSWORD = mqtt_config["password"]
MQTT_QOS = mqtt_config["qos"]

MQTT_CLIENT_ID = task_config["mqtt"]["client_id"]
MQTT_SUBSCRIBE_TOPIC = task_config["mqtt"]["topics"]["subscribe"]
MQTT_PUBLISH_TOPIC = task_config["mqtt"]["topics"]["publish"]
MQTT_ERROR_TOPIC = task_config["mqtt"]["topics"]["error"]

DELTA_OFFSET = task_config["delta_offset"]
WAIT_PERIOD = task_config["wait_period"]

UI_HOST = task_config["nvr"]["host"]
UI_PORT = task_config["nvr"]["port"]
UI_USERNAME = task_config["nvr"]["username"]
UI_PASSWORD = task_config["nvr"]["password"]
CAMERA_NAMES = task_config["nvr"]["camera_names"]

# Configure NVR client
NVR_CLIENT = Protect(UI_HOST, UI_PORT, UI_USERNAME, UI_PASSWORD)


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
        print("\nVideo Event Received:")
        print(f"  Timestamp: {data.get('timestamp')}")
        print(f"  Sensor ID: {data.get('sensor_id')}")
        print(f"  Speed: {data.get('speed')} {data.get('uom')}")
        print(f"  Folder: {data.get('folder')}")
        print(f"  Data File: {data.get('data_file')}")
        print(f"  Payload: {payload}")

        handle_event(data)

    except json.JSONDecodeError:
        print("Received invalid JSON: {message}")
    except Exception as e:
        print(f"Error processing message: {e}")

async def save_media(nvr_client, cams: list[dict[str, any]], dt, path: str):
    """Retrieves videos from specified cameras and the specified times of the length defined"""
    output_files = []

    for camera in cams:

        filename = path  + "/" + camera["filename"]
        cam_id = camera["id"]
        print(f"  For camera: {cam_id}")

        video_name = await nvr_client.save_video(cam_id, dt, filename, DELTA_OFFSET)
        
        output_files.append(video_name)
        print(f"  Saved video: {video_name}\n")

    return output_files

def handle_event(data):
    """Creates a folder based on the timestamp of the event"""

    # Convert timestamp to string
    timestamp = data.get("timestamp")
    occurred = datetime.fromisoformat(timestamp)

    try:
        print("Saving media...")
        folder = data.get("folder")
        videos = asyncio.run(save_media(NVR_CLIENT, cameras, occurred, folder))

        # Update Payload
        data["videos"] = videos
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
