"""
unifi_protect_helpers.py

Helper classes for extracting media from a UniFi Protect Server 
using uiprotect module.
"""
import asyncio
from datetime import datetime
from datetime import timedelta
from uiprotect import ProtectApiClient

class Protect:
    """Helper class with functions around the uiprotect library"""
    def __init__(self, host: str, port: str,
                 user_name: str, password: str):
        self.host = host
        self.port = port
        self.user_name = user_name
        self.password = password
        self.client = None

    def __del__(self):
        asyncio.run(self.destroy_client())

    async def get_client(self) -> ProtectApiClient:
        """Retrieves a Unifi Protect Client"""
        if self.client is None:
            self.client = ProtectApiClient(self.host, self.port,
                                           self.user_name, self.password,
                                           False)

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
            binary_file = open(fq_filename + ".jpg", "wb")
            binary_file.write(pic)
            binary_file.close()
        else:
            raise ProtectMediaNotAvailable("The call to get_camera_snapshot return a NoneType, media not yet available.", 500)

        return fq_filename

    async def save_video(self, camera_id: str, dt: datetime, filename: str, offset: int):
        """Saves Clip from camera using delta offset around specified time."""
        client = await self.get_client()

        delta = timedelta(seconds = offset)
        start = dt-delta
        end = dt+delta
        fq_filename = filename + ".mpg"

        await client.get_camera_video(camera_id, start, end,
                                      channel_index = 0,
                                      validate_channel_id = True,
                                      output_file = fq_filename,
                                      chunk_size = 65536)

        await self.destroy_client()

        return fq_filename

    async def get_cameras(self, camera_filter: list[str]=None):
        """Connects to UniFi Protect and enumerates cameras"""
        client = await self.get_client()

        # get names of your cameras
        cameras = [
            {
                "name": camera.name,
                "id": camera.id,
                "filename": f"{camera.name.lower().replace(" ", "")}"
            }
            for camera in client.bootstrap.cameras.values()
        ]

        await self.destroy_client()

        if camera_filter is None:
            return cameras

        return [camera for camera in cameras if camera['name'] in camera_filter]



class ProtectMediaNotAvailable(Exception):
    """Exception raised when request media isn't available from UI Protect"""

    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code

    def __str__(self):
        return f"Protect Media Not Available(Error Code: {self.error_code})"
