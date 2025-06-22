"""
task_report_publisher.py

Subscribes to an MQTT topic and exports video clips
from the NVR based on timestamps in published messages

TODO::
* Pull metadata into report
"""
import json
from datetime import datetime
import shutil
from shared import MqttClientWrapper, load_config


VERBOSE = False
DEBUG = False
LOGGING = False

# Load configuration from yaml
config = load_config()
mqtt_config = config["servers"]["mqtt"]
task_config = config["task"]["report_publisher"]
media_config = config["media"]

# Populate configuration variables
MQTT_URI = mqtt_config["uri"]
MQTT_USER = mqtt_config["username"]
MQTT_PASSWORD = mqtt_config["password"]
MQTT_QOS = mqtt_config["qos"]

MQTT_CLIENT_ID = task_config["mqtt"]["client_id"]
MQTT_SUBSCRIBE_TOPIC = task_config["mqtt"]["topics"]["subscribe"]
MQTT_PUBLISH_TOPIC = task_config["mqtt"]["topics"]["publish"]
MQTT_ERROR_TOPIC = task_config["mqtt"]["topics"]["error"]

FOLDER_FORMAT = media_config["output_folder_format"]
PUBLISH_PATH = task_config["publish_path"]
PUBLISH_URL_TEMPLATE = task_config["publish_url_template"]
PUBLISH_HTML_FILE_NAME = task_config["html_file_name"]
PUBLISH_HTML_FILE_CONTENTS = task_config["html_file_contents"]

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
        print("\nPublishing Event Received:")
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

def generate_html_page(folder_name):
    """Generates a static HTML Page in the folder"""
    html_page = f"{folder_name}/{PUBLISH_HTML_FILE_NAME}"

    # Hardcoded page...
    html_page_output = PUBLISH_HTML_FILE_CONTENTS

    # Save data to disk
    with open(html_page, "w", encoding="utf-8") as f:
        f.write(html_page_output)

    return html_page

def handle_event(data):
    """Creates a folder based on the timestamp of the event"""

    # Convert timestamp to string
    timestamp = data.get("timestamp")
    occurred = datetime.fromisoformat(timestamp)
    local = occurred.astimezone()
    dts = local.strftime(FOLDER_FORMAT)

    # Create HTML
    print("Saving page...")
    folder = data.get("folder")
    page = generate_html_page(folder)
    print("  File saved")

    # Publish Files
    print("Publishing files...")
    publish_folder = PUBLISH_PATH + dts
    print(f"  Source: {folder}")
    print(f"  Target: {publish_folder}")
    shutil.copytree(src=folder, dst=publish_folder, dirs_exist_ok=True)
    print("  Files published")

    # Update Payload
    data["url"] = PUBLISH_URL_TEMPLATE.format(dts)
    data["page"] = page
    payload = json.dumps(data)
    print(f"  New Payload: {payload}")

    # Publish message for next task
    client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
    print(f"  Message Published: {MQTT_PUBLISH_TOPIC}")

# Configure MQTT and wait...
client = MqttClientWrapper(MQTT_URI, MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)
client.connect(on_connect, on_message)
