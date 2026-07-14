"""Subscribes to an MQTT topic and creates a folder based on a timestamp in published messages"""
import logging
import os
from datetime import datetime
import json
from shared import MqttClientWrapper, load_config

logger = logging.getLogger(__name__)

# Load configuration from yaml
config = load_config()
mqtt_config = config["servers"]["mqtt"]
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


def on_connect(client, userdata, flags, reason_code, properties):
    """Subscribe to topic on successful connection."""
    if reason_code == 0:
        client.subscribe(MQTT_SUBSCRIBE_TOPIC, MQTT_QOS)
        logger.info("Subscribed to topic: %s", MQTT_SUBSCRIBE_TOPIC)


def on_message(client, userdata, message):
    """Processes the message from MQTT Broker"""
    data = {}
    try:
        payload = message.payload.decode('utf-8')
        data = json.loads(payload)
        logger.info("Sensor Event Received:")
        logger.info("  Timestamp: %s", data.get('timestamp'))
        logger.info("  Sensor ID: %s", data.get('sensor_id'))
        logger.info("  Speed: %s %s", data.get('speed'), data.get('uom'))
        logger.debug("  Payload: %s", payload)

        setup_folder(client, data)

    except json.JSONDecodeError as e:
        logger.error("Received invalid JSON: %s", e)
        data = {"error": str(e)}
        payload = json.dumps(data)
        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        logger.error("  Error message published to: %s", MQTT_ERROR_TOPIC)

    except Exception as e:
        logger.exception("Error processing message: %s", e)
        data["error"] = str(e)
        payload = json.dumps(data)
        client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
        logger.error("  Error message published to: %s", MQTT_ERROR_TOPIC)


def setup_folder(client, data):
    """Creates a folder based on the timestamp of the event"""
    # Convert timestamp to string
    timestamp = data.get("timestamp")
    occurred = datetime.fromisoformat(timestamp)
    local = occurred.astimezone()
    dts = local.strftime(FOLDER_FORMAT)

    # Create folder
    folder = FOLDER_PATH + dts
    os.makedirs(folder, exist_ok=True)
    logger.info("  Folder Created: %s", folder)

    # Update Payload
    data["folder"] = folder
    payload = json.dumps(data)
    logger.debug("  New Payload: %s", payload)

    # Publish message for next task
    client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
    logger.info("  Message Published: %s", MQTT_PUBLISH_TOPIC)


client = MqttClientWrapper(MQTT_URI, MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)
client.connect(on_connect, on_message)
