"""
mqtt_client_wrapper.py

Implements wrapper around the Paho MQTT client to 
simplify configuration and message handling.
"""
import logging
from urllib.parse import urlparse
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# This is the class responsible for pushing 
# notifications.  It utilizes MQTT to push 
# to an instance of Home Assistant.
class MqttClientWrapper:
    """Paho MQTT Wrapper Class"""
    def __init__(self, mqtt_uri, client_id, username, password, timeout = 60):

        # MQTT Config
        self.mqtt_uri = mqtt_uri
        self.client_id = client_id
        self.username = username
        self.password = password
        self.timeout = timeout

        uri = urlparse(mqtt_uri)

        self.hostname = uri.hostname
        self.port = uri.port
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id, clean_session=False)

        # Custom callbacks
        self.custom_on_connect = None
        self.custom_on_message = None
        self.custom_on_disconnect = None

    def connect(self, on_connect = None, on_message = None, on_disconnect = None):
        """Connect to defined MQTT Server"""
        print("Connecting to MQTT:", self.hostname, self.port)

        self.custom_on_connect = on_connect
        self.custom_on_message = on_message
        self.custom_on_disconnect = on_disconnect

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

        self.client.username_pw_set(self.username, self.password)
        self.client.connect(self.hostname, self.port, self.timeout)

        try:
            print("Starting MQTT client loop...")
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            self.client.disconnect()

    def on_connect(self, client, userdata, flags, reason_code, properties):
        """Callback when connected to MQTT Broker"""

        if reason_code == 0:
            print("Connected to MQTT Broker.")

            if flags.session_present:
                print("Session present.")

        if reason_code > 0:
            print(f"Failed to connect, return code {reason_code}")

        if self.custom_on_connect is not None:
            print("Calling user defined on_connect...")
            self.custom_on_connect(client, userdata, flags, reason_code, properties)

    def on_message(self, client, userdata, message):
        """"Receive the message from MQTT Broker"""

        try:
            payload = message.payload.decode('utf-8')
            print("\nMessage Received:")
            logger.debug(f"  payload: {payload}")

        except Exception as e:
            print(f"Error processing message: {e}")

        if self.custom_on_message is not None:
            print("  Calling user defined on_message...")
            self.custom_on_message(client, userdata, message)

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        """Callback when disconnected to MQTT Broker"""

        if reason_code == 0:
            print("Disconnected from MQTT broker.")
        if reason_code > 0:
            print(f"Failed to disconnect, return code {reason_code}")

        if self.custom_on_disconnect is not None:
            print("Calling user defined on_disconnect...")
            self.custom_on_disconnect(client, userdata, flags, reason_code, properties)

    def publish(self, topic, payload, qos=1, retain=False):
        """Publish message to MQTT Broker"""
        return self.client.publish(topic, payload, qos, retain)

    def subscribe(self, topic, qos=1):
        """Publish message to MQTT Broker"""
        return self.client.subscribe(topic,qos)

    def unsubscribe(self, topic):
        """Publish message to MQTT Broker"""
        return self.client.unsubscribe(topic)

    def disconnect(self):
        """Disconnects client and stops loop"""
        self.client.disconnect()
        self.client.loop_stop()
