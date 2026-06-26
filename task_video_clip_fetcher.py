"""
task_video_clip_fetcher.py

Subscribes to an MQTT topic and exports video clips
from the NVR based on timestamps in published messages
"""
import json
import signal
import sys
from datetime import datetime
from shared import MqttClientWrapper, Protect, ProtectMediaNotAvailable, ProtectCredentialError, retry_with_backoff, load_config


VERBOSE = False
DEBUG = False
LOGGING = False

# Load configuration from yaml
config = load_config()
mqtt_config = config["servers"]["mqtt"]
protect_config = config["servers"]["unifi_protect"]
task_config = config["task"]["video_clip_fetcher"]
cameras_config = config["cameras"]

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

UI_URI = protect_config["uri"]
UI_USERNAME = protect_config["username"]
UI_PASSWORD = protect_config["password"]
CAMERA_NAMES = [cam["camera_id"] for cam in cameras_config]

# Configure NVR client
NVR_CLIENT = Protect(UI_URI, UI_USERNAME, UI_PASSWORD)
CAMERAS = NVR_CLIENT.get_cameras(CAMERA_NAMES)

def on_connect(client, userdata, flags, reason_code, properties):
    """Subscribe to topic on successful connection."""

    if reason_code == 0:
        client.subscribe(MQTT_SUBSCRIBE_TOPIC, MQTT_QOS)
        print(f"Subscribed to topic: {MQTT_SUBSCRIBE_TOPIC}")

def on_message(client, userdata, message):
    """Processes the message from MQTT Broker"""
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
        print(f"Received invalid JSON: {message}")
    except Exception as e:
        print(f"Error processing message: {e}")


def handle_event(data):
    """Creates a folder based on the timestamp of the event"""

    # Convert timestamp to string
    timestamp = data.get("timestamp")
    occurred = datetime.fromisoformat(timestamp)

    def fetch_videos():
        nonlocal occurred
        nonlocal data
        folder = data.get("folder")

        videos = []
        for camera in CAMERAS:
            filename = folder + "/" + camera["filename"]
            cam_id = camera["id"]
            print(f"  For camera: {cam_id}")

            try:
                video_name = retry_with_backoff(
                    NVR_CLIENT.save_video,
                    cam_id, occurred, filename, DELTA_OFFSET
                )
                videos.append(video_name)
                print(f"  Saved video: {video_name}\n")

            except ProtectCredentialError as e:
                print(f"Credential error for Cam: {cam_id}")
                print(f"  {e}")
                raise

            except Exception as e:
                print(f"Error Saving Cam: {cam_id}")
                print(f"  Exception Details: {e}")

        if not videos:
            raise ProtectMediaNotAvailable("All cameras returned no video clips", 500)

        return videos

    try:
        print("Retrieving cameras and saving media...")
        videos = fetch_videos()

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
        data["error"] = str(e)
        payload = json.dumps(data)

        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        print(f"  Error Message Published: {MQTT_ERROR_TOPIC}")


def shutdown(*_args):
    """Gracefully closes the NVR client's connection before exiting."""
    print("\nShutting down, closing NVR client...")
    NVR_CLIENT.close()
    sys.exit(0)


# Configure MQTT and wait...
client = MqttClientWrapper(MQTT_URI, MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)
client.connect(on_connect, on_message)

# Ensure the long-lived NVR client connection is closed deliberately on
# Ctrl+C / SIGTERM, rather than relying on Protect.__del__ (which now only
# logs a warning instead of trying to close anything itself).
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)