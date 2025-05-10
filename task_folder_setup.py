"""Subscribes to an MQTT topic and creates a folder based on a timestamp in published messages"""
import os
from datetime import datetime
from types import SimpleNamespace
import json
from shared import MqttClientWrapper, load_config


VERBOSE = False
DEBUG = False
LOGGING = False

# Load configuration from yaml
config = load_config()
mqtt_config = config["mqtt"]
task_config = config["task"]["folder_setup"]
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

FOLDER_PATH = media_config["output_path"]
FOLDER_FORMAT = media_config["output_folder_format"]

state = SimpleNamespace(last_processed_time=datetime.now())

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
        print("\nSensor Event Received:")
        print(f"  Timestamp: {data.get('timestamp')}")
        print(f"  Sensor ID: {data.get('sensor_id')}")
        print(f"  Speed: {data.get('speed')} {data.get('uom')}")
        print(f"  Payload: {payload}")

        setup_folder(data)

    except json.JSONDecodeError:
        print("Received invalid JSON")
        client.publish(MQTT_PUBLISH_TOPIC, message.payload, MQTT_QOS)
        print(f"  Error Message Published: {MQTT_ERROR_TOPIC}")

    except Exception as e:
        print(f"Error processing message: {e}")
        # Update Payload
        data["error"] = e
        payload = json.dumps(data)

        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        print(f"  Error Message Published: {MQTT_ERROR_TOPIC}")

def setup_folder(data):
    """Creates a folder based on the timestamp of the event"""

    # Convert timestamp to string
    timestamp = data.get("timestamp")
    occurred = datetime.fromisoformat(timestamp)
    local = occurred.astimezone()
    dts = local.strftime(FOLDER_FORMAT)

    # Create folder
    folder = FOLDER_PATH + dts
    os.makedirs(folder, exist_ok=True)
    print(f"  Folder Created: {folder}")

    # Update Payload
    data["folder"] = folder
    payload = json.dumps(data)
    print(f"  New Payload: {payload}")

    # Publish message for next task
    client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
    print(f"  Message Published: {MQTT_PUBLISH_TOPIC}")


client = MqttClientWrapper(MQTT_URI, MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)

client.connect(on_connect, on_message)
