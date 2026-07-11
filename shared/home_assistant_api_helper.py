"""
home_assistant_api_helpers.py

Implements wrapper classes for Websocket and Rest request 
for extracting data from a Home Assistant Server
"""
import json
import logging
import requests
import websockets

logger = logging.getLogger(__name__)


class HomeAssistantWebSocket:
    """Helper class with functions around the websocket api"""
    def __init__(self, url, access_token):
        self.url = url
        self.access_token = access_token

    async def subscribe_entity(self, subscription_id: int, entity_id: str, callback: callable):
        """Connect to the defined Home Assistant Server via Websocket"""
        async with websockets.connect(self.url) as websocket:
            response = await websocket.recv()
            logger.debug("Connection Response: %s", response)

            # Authenticate with Home Assistant
            auth_message = {
                "type": "auth",
                "access_token": self.access_token
            }
            await websocket.send(json.dumps(auth_message))
            response = await websocket.recv()
            logger.debug("Authentication Response: %s", response)

            subscribe_message = {
                "id": subscription_id,
                "type": "subscribe_entities",
                "entity_ids": [entity_id]
            }

            await websocket.send(json.dumps(subscribe_message))
            response = await websocket.recv()
            logger.debug("Subscription Response: %s", response)

            data = json.loads(response)

            if data["success"]:
                # Listen for events and trigger the callback
                while True:
                    event = await websocket.recv()
                    logger.debug("Received Event: %s", event)
                    await callback(event)
            else:
                logger.error("Subscription failed: %s", response)


class HomeAssistantRest:
    """Helper class with functions around the rest api"""
    def __init__(self, url, bearer_token):
        self.url = url
        self.bearer_token = bearer_token

    def get_data(self, query, timeout=10):
        """Queries a REST endpoint and returns JSON"""

        # Set up the headers with the Authorization token
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json"
        }

        try:
            # Send the GET request
            response = requests.get(self.url + query, headers=headers, timeout=timeout)

            # Check if request was successful (status code 200-299)
            response.raise_for_status()

            # Try parsing JSON content
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error("Request failed: %s", e)
            return {}  # Return empty object on failure

        except ValueError:
            logger.error("Failed to parse JSON.")
            return {}  # JSON parsing failed
