"""
task_sensor_log_handler.py

Subscribes to an MQTT topic and exports sensor 
logs based on timestamp in published messages
"""
import logging
import os
import time
import urllib.parse
import json
from datetime import datetime, timedelta
from shared import MqttClientWrapper, HomeAssistantRest, SummaryGenerator, retry_with_backoff, load_config

logger = logging.getLogger(__name__)

# Load configuration from yaml
config = load_config()
mqtt_config = config["servers"]["mqtt"]
ha_config = config["servers"]["homeassistant"]
task_config = config["task"]["sensor_log_handler"]
sensor_config = config["sensor"]
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

HA_TOKEN = ha_config["token"]
HA_REST_URL = ha_config["rest_uri"]
HA_REST_QUERY_TEMPLATE = ha_config["query_template"]
HA_REST_CLIENT = HomeAssistantRest(
    url=HA_REST_URL,
    bearer_token=HA_TOKEN,
)

HA_SENSOR = sensor_config["id"]

SPEED_THRESHOLD = sensor_config["trigger_threshold"]
ERRONEOUS_DATA_THRESHOLD = sensor_config["error_threshold"]

DELTA_OFFSET = task_config["delta_offset"]
WAIT_PERIOD = task_config["wait_period"]
DATA_FILE_NAME = task_config["data_file_name"]
SUMMARY_FILE_NAME = task_config["summary_file_name"]

SUMMARY_GENERATOR = SummaryGenerator(FOLDER_FORMAT, SPEED_THRESHOLD)


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
        logger.info("Logging Event Received:")
        logger.info("  Timestamp: %s", data.get('timestamp'))
        logger.info("  Sensor ID: %s", data.get('sensor_id'))
        logger.info("  Speed: %s %s", data.get('speed'), data.get('uom'))
        logger.info("  Folder: %s", data.get('folder'))
        logger.debug("  Payload: %s", payload)

        handle_event(client, data)

    except json.JSONDecodeError as e:
        logger.error("Received invalid JSON: %s", e)
    except Exception as e:
        logger.exception("Error processing message: %s", e)


def parse_json(data):
    """Parses JSON from Home Assistant api to a list object"""
    approaching = []
    retreating = []

    for inner_list in data:
        for item in inner_list:
            speed = float(item.get('state'))
            last_changed = datetime.fromisoformat(item.get('last_changed'))
            zoned = last_changed

            if speed < 0.0:
                retreating.append([abs(speed), zoned])
            if speed > 0.0:
                approaching.append([speed, zoned])

    return approaching, retreating


def format_json(data_list):
    return [{"speed": speed, "occurred": dt.isoformat()} for speed, dt in data_list]


def clean_sensor_data(sensor_data):
    """Cleans sensor data to remove erroneous data points"""
    cleaned_data = sensor_data.copy()

    for i in range(1, len(sensor_data) - 1):
        prev_speed = sensor_data[i - 1]["speed"]
        curr_speed = sensor_data[i]["speed"]
        next_speed = sensor_data[i + 1]["speed"]

        expected_speed = (prev_speed + next_speed) / 2
        if (curr_speed < expected_speed) and (expected_speed - curr_speed) > ERRONEOUS_DATA_THRESHOLD:
            logger.debug("Outlier at index %d: %s -> replacing with %.2f", i, curr_speed, expected_speed)
            cleaned_data[i]["speed"] = expected_speed

    return cleaned_data


def get_sensor_data(dt):
    """Uses a timestamp to query Home Assistant for sensor data"""

    before = dt - timedelta(seconds=DELTA_OFFSET)
    after = dt + timedelta(seconds=DELTA_OFFSET)

    # Wait until the post-event window has elapsed so we don't miss data
    now = datetime.now(dt.tzinfo or datetime.now().astimezone().tzinfo)
    if now < after:
        wait_seconds = (after - now).total_seconds()
        logger.info("  Waiting %.1fs for post-event window to elapse", wait_seconds)
        time.sleep(wait_seconds)

    start_time = urllib.parse.quote(before.isoformat())
    end_time = urllib.parse.quote(after.isoformat())

    uri = HA_REST_QUERY_TEMPLATE.format(start_time, end_time, HA_SENSOR)

    data = HA_REST_CLIENT.get_data(uri)

    # Split into approaching and retreating
    a, r = parse_json(data)

    # Format data
    af = format_json(a)
    rf = format_json(r)

    # Clean up erroneous data
    ac = clean_sensor_data(af)
    rc = clean_sensor_data(rf)

    return {"approaching": ac, "retreating": rc}


def handle_event(client, data):
    """Creates a folder based on the timestamp of the event"""

    # Convert timestamp to string
    timestamp = data.get("timestamp")
    occurred = datetime.fromisoformat(timestamp)
    #local = occurred.astimezone()

    logger.info("Saving data:")

    # Pull data from Home Assistant with retry
    try:
        sensor_data = retry_with_backoff(
            get_sensor_data,
            occurred,
            on_empty=lambda r: not r.get("approaching") and not r.get("retreating"),
        )
    except Exception as e:
        logger.error("  Failed to retrieve sensor data: %s", e)
        sensor_data = {"approaching": [], "retreating": []}

    logger.debug("  Sensor data: %s", sensor_data)

    # Save data to disk
    folder = data.get("folder")
    data_file = f"{folder}/{DATA_FILE_NAME}"
    logger.info("  Save file: %s", data_file)

    has_data = sensor_data.get("approaching") or sensor_data.get("retreating")

    if not has_data and os.path.exists(data_file):
        logger.warning("  No sensor data from HA — preserving existing %s", data_file)
    else:
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sensor_data, f, indent=4)
        logger.info("  File saved")

    # Summarize Data
    summary_file = f"{folder}/{SUMMARY_FILE_NAME}"
    SUMMARY_GENERATOR.generate_summary_file(data_file, summary_file)

    # Update Payload
    data["data_file"] = data_file
    data["summary_file"] = summary_file
    payload = json.dumps(data)
    logger.debug("  New Payload: %s", payload)

    # Publish message for next task
    client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
    logger.info("  Message Published: %s", MQTT_PUBLISH_TOPIC)


# Configure MQTT and wait...
client = MqttClientWrapper(MQTT_URI, MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)
client.connect(on_connect, on_message)
