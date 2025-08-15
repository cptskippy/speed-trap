"""
task_license_plate_reader.py

Subscribes to an MQTT topic and performs LPR activities
against specified images.
"""
import json
from datetime import datetime
import logging
import shutil
from shared import MqttClientWrapper, load_config, OpenAILicensePlateReader


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
task_config = config["task"]["license_plate_reader"]

# Populate configuration variables
MQTT_URI = mqtt_config["uri"]
MQTT_USER = mqtt_config["username"]
MQTT_PASSWORD = mqtt_config["password"]
MQTT_QOS = mqtt_config["qos"]

MQTT_CLIENT_ID = task_config["mqtt"]["client_id"]
MQTT_SUBSCRIBE_TOPIC = task_config["mqtt"]["topics"]["subscribe"]
MQTT_PUBLISH_TOPIC = task_config["mqtt"]["topics"]["publish"]
MQTT_ERROR_TOPIC = task_config["mqtt"]["topics"]["error"]

AI_API_KEY = task_config["openai_api_key"]
AI_MODEL = task_config["openai_model"]
AI_PROMPT = task_config["openai_prompt"]
IMAGE_TO_LPR = task_config["image_to_lpr"]

# Configure LPR client
LPR_CLIENT = OpenAILicensePlateReader(AI_API_KEY, AI_MODEL, AI_PROMPT)


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
        print("\nLPR Event Received:")
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


def update_summary(summary_path, license_plate):
    """Opens the summary file and updates the license_plate"""        

    logger.debug(f"  Opening summary data: {summary_path}")

    with open(summary_path, "r") as f:
      summary_data = json.load(f)

    logger.debug(f"  Updating summary data...")
    summary_data["license_plate"] = license_plate

    logger.debug(f"  Saving summary data...")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
        print("  File saved")


def handle_event(data):
    """Performs a license plate read."""

    # {
    #   "timestamp": "2025-08-15T19:54:57.406713+00:00", 
    #   "speed": 25.041258573096, 
    #   "uom": "mph", 
    #   "sensor_id": "sensor.speedometer_speed", 
    #   "folder": "./media/20250815125457", 
    #   "data_file": "./media/20250815125457/data.json", 
    #   "summary_file": "./media/20250815125457/summary.json", 
    #   "videos": [
    #     "./media/20250815125457/street.mpg", 
    #     "./media/20250815125457/driveway.mpg", 
    #     "./media/20250815125457/globalshutter.mpg"
    #   ], 
    #   "images": [
    #     "./media/20250815125457/street.png", 
    #     "./media/20250815125457/driveway.png", 
    #     "./media/20250815125457/globalshutter.png"
    #   ], 
    #   "thumbnails": [
    #     "./media/20250815125457/street_thumb.png", 
    #     "./media/20250815125457/driveway_thumb.png", 
    #     "./media/20250815125457/globalshutter_thumb.png"
    #   ]
    # }
    
    # Generate Thumbnail Path
    folder = data.get("folder")
    image_path = folder + "/" + IMAGE_TO_LPR

    # Perform LPR
    result = LPR_CLIENT.get_license_plate_read(image_path)
    status = result.status
    error = result.error_message
    plate = result.license_plate

    if status == "success":
      # Update Summary file
      summary_path = data.get("summary_file")

      update_summary(summary_path, plate)
      print(f"  License plate read: {plate}")
    else:
      print(f"  No plate data: {error}")


    # Update Payload
    payload = json.dumps(data)
    print(f"  New Payload: {payload}")

    # Publish message for next task
    client.publish(MQTT_PUBLISH_TOPIC, payload, MQTT_QOS)
    print(f"  Message Published: {MQTT_PUBLISH_TOPIC}")

# Configure MQTT and wait...
client = MqttClientWrapper(MQTT_URI, MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)
client.connect(on_connect, on_message)