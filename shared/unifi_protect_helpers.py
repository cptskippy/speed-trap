"""
unifi_protect_helpers.py

Helper classes for extracting media from a UniFi Protect Server 
using uiprotect module.
"""
import asyncio
import logging
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse
from uiprotect import ProtectApiClient
from uiprotect.data import Camera, SmartDetectObjectType
from shared import LicensePlateRead, BoundingBox

logger = logging.getLogger(__name__)

class Protect:
    """Helper class with functions around the uiprotect library"""
    def __init__(self, nvr_uri: str, user_name: str, password: str):
        uri = urlparse(nvr_uri)

        self.host = uri.hostname
        self.port = uri.port

        self.user_name = user_name
        self.password = password
        self.client = None

    def __del__(self):
        asyncio.run(self.destroy_client())

    async def get_client(self) -> ProtectApiClient:
        """Retrieves a Unifi Protect Client"""
        if self.client is None:
            self.client = ProtectApiClient(host=self.host, 
                                           port=self.port,
                                           username=self.user_name, 
                                           password=self.password,
                                           verify_ssl=False)

            # this will initialize the protect .bootstrap and
            # open a Websocket connection for updates
            await self.client.update()

        return self.client

    async def destroy_client(self):
        """Destroys a Unifi Protect Client"""
        #print("Destroying Client")
        if self.client is not None:
            await self.client.close_session()
            self.client = None

    async def save_still(self, camera_id: str, dt: datetime, filename: str):
        """Saves Still from camera using specified time."""
        fq_filename = filename + ".jpg"

        client = await self.get_client()
        pic = await client.get_camera_snapshot(camera_id, dt=dt)

        await self.destroy_client()

        if pic is not None:
            binary_file = open(fq_filename, "wb")
            binary_file.write(pic)
            binary_file.close()
        else:
            raise ProtectMediaNotAvailable("The call to get_camera_snapshot return a NoneType, media not yet available.", 500)

        return fq_filename

    async def save_video(self, camera_id: str, dt: datetime, filename: str, offset: int):
        """Saves Clip from camera using delta offset around specified time."""
        
        logger.info("  Fetching client...")
        client = await self.get_client()

        delta = timedelta(seconds = offset)
        start = dt-delta
        end = dt+delta
        fq_filename = filename + ".mpg"

        logger.debug(f"Video File: {fq_filename}")

        await client.get_camera_video(camera_id, start, end,
                                      channel_index = 0,
                                      validate_channel_id = True,
                                      output_file = Path(fq_filename),
                                      chunk_size = 65536)

        await self.destroy_client()

        return fq_filename

    async def get_cameras(self, camera_filter: list[str]=[]):
        """Connects to UniFi Protect and enumerates cameras"""
        client = await self.get_client()

        # get names of your cameras
        cameras = [
            {
                "name": camera.name,
                "id": camera.id,
                "filename": f"{self.get_camera_filename(camera)}"
            }
            for camera in client.bootstrap.cameras.values()
        ]

        await self.destroy_client()

        if camera_filter is None:
            return cameras

        return [camera for camera in cameras if camera['name'] in camera_filter]

    async def get_license_plate_reads(self, source: str, dt: datetime, offset: int) -> list[LicensePlateRead]:
        """Returns LPR reads using delta offset around specified time."""
        plates = []

        logger.info("  Fetching client...")
        client = await self.get_client()

        delta = timedelta(seconds = offset)
        start = dt-delta
        end = dt+delta

        events = await client.get_events(start=start, 
                                         end=end, 
                                         smart_detect_types=[SmartDetectObjectType.LICENSE_PLATE])

        for event in events:
            if source == event.camera_id:
                thumb = event.get_detected_thumbnail()

                if thumb != None:
                    if 'vehicle' in (thumb.type or []):
                        try:
                            plate = thumb.name
                        except AttributeError:
                            plate = None
                            
                        try:
                            type = thumb.attributes.vehicleType.val
                        except AttributeError:
                            type = None
                            
                        try:
                            color = thumb.attributes.color.val
                        except AttributeError:
                            color = None

                        try:
                            confidence = thumb.confidence
                        except AttributeError:
                            confidence = None

                        try:
                            coord = thumb.coord
                            bbox = BoundingBox(x=coord[0], y=coord[1], w=coord[2], h=coord[3])
                        except AttributeError:
                            bbox = None

                        #thumb.

                        lpr = LicensePlateRead(license_plate=plate, 
                                               vehicle_type=type, 
                                               vehicle_color=color, 
                                               confidence=confidence,
                                               bounding_box=bbox,
                                               status="success", 
                                               diagnostic_messages="")
                        plates.append(lpr)    
                        
                        #plates.append({
                        #    "plate": plate,
                        #    "type": type,
                        #    "color": color,
                        #    #"thumbnail": event.get_thumbnail(width=480, height=480)
                        #})

        return plates
    
    def get_camera_filename(self, camera: Camera):
        return camera.name.lower().replace(" ", "")

class ProtectMediaNotAvailable(Exception):
    """Exception raised when request media isn't available from UI Protect"""

    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code

    def __str__(self):
        return f"Protect Media Not Available(Error Code: {self.error_code})"
