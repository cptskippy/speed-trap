"""
unifi_protect_helpers.py

Helper classes for extracting media from a UniFi Protect Server
using uiprotect module.  This class is designed for long running
processes processes so it creates and manages it's on event loop 
and exposes the async methods via synchronous methods.
"""
import asyncio
import logging
import threading
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
        self.client: ProtectApiClient | None = None

        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ProtectEventLoop",
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()
        self._loop.close()

    def _run_coroutine(self, coro, timeout: float | None = None):
        """Submits a coroutine to the background loop and blocks for the result """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self):
        """Closes the underlying client (if open) and stops the background loop."""
        if self.client is not None:
            try:
                self._run_coroutine(self.destroy_client(), timeout=10)
            except Exception:
                logger.exception("Error closing Protect client during shutdown")

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)

    def __del__(self):
        if getattr(self, "_thread", None) is not None and self._thread.is_alive():
            logger.warning("Protect instance destroyed without calling close();")

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
        if self.client is not None:
            await self.client.close_session()
            self.client = None

    @staticmethod
    def _is_auth_error(e: Exception) -> bool:
        """Check if an exception is a 401 Unauthorized / session expiry error."""
        err_str = str(e).lower()
        return "401" in err_str or "unauthorized" in err_str

    async def _run_with_auth_recovery(self, coro_factory):
        """Run a coroutine with automatic 401 session recovery.

        Calls coro_factory() to get a coroutine, runs it. If it fails with a
        401 Unauthorized, destroys the stale client, recreates it via
        get_client(), and retries once. If the retry also gets 401, raises
        ProtectCredentialError so the caller knows credentials are bad.

        Args:
            coro_factory: Zero-arg callable that returns a new coroutine.
                          Must be callable again after client recreation.

        Returns:
            Result of the coroutine.

        Raises:
            ProtectCredentialError: If 401 persists after re-authentication.
            The original exception: For any non-401 errors.
        """
        try:
            return await coro_factory()
        except Exception as first_error:
            if not self._is_auth_error(first_error):
                raise

            logger.warning(
                "Got 401 Unauthorized — destroying stale client and re-authenticating"
            )
            await self.destroy_client()
            try:
                return await coro_factory()
            except Exception as second_error:
                if self._is_auth_error(second_error):
                    raise ProtectCredentialError(
                        f"Re-authentication failed with 401 — check credentials. "
                        f"Original error: {first_error}. Retry error: {second_error}"
                    ) from second_error
                raise

    async def _save_still(self, camera_id: str, dt: datetime, filename: str):
        """Saves Still from camera using specified time."""
        fq_filename = filename + ".jpg"

        async def do_save():
            client = await self.get_client()
            pic = await client.get_camera_snapshot(camera_id, dt=dt)
            if pic is not None:
                with open(fq_filename, "wb") as binary_file:
                    binary_file.write(pic)
            else:
                raise ProtectMediaNotAvailable("The call to get_camera_snapshot return a NoneType, media not yet available.", 500)

        await self._run_with_auth_recovery(do_save)
        return fq_filename

    def save_still(self, camera_id: str, dt: datetime, filename: str):
        return self._run_coroutine(self._save_still(camera_id, dt, filename))

    async def _save_video(self, camera_id: str, dt: datetime, filename: str, offset: int):
        """Saves Clip from camera using delta offset around specified time."""

        delta = timedelta(seconds = offset)
        start = dt-delta
        end = dt+delta
        fq_filename = filename + ".mpg"

        logger.debug(f"Video File: {fq_filename}")

        async def do_save():
            logger.info("  Fetching client...")
            client = await self.get_client()
            await client.get_camera_video(camera_id, start, end,
                                          channel_index = 0,
                                          validate_channel_id = True,
                                          output_file = Path(fq_filename),
                                          chunk_size = 65536)

        await self._run_with_auth_recovery(do_save)
        return fq_filename

    def save_video(self, camera_id: str, dt: datetime, filename: str, offset: int):
        return self._run_coroutine(self._save_video(camera_id, dt, filename, offset))

    async def _get_cameras(self, camera_filter: list[str] = None):
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

        if camera_filter is None:
            return cameras

        return [camera for camera in cameras if camera['name'] in camera_filter]

    def get_cameras(self, camera_filter: list[str] = None):
        return self._run_coroutine(self._get_cameras(camera_filter))

    async def _get_license_plate_reads(self, source: str, dt: datetime, offset: int) -> list[LicensePlateRead]:
        """Returns LPR reads using delta offset around specified time."""
        plates = []

        delta = timedelta(seconds = offset)
        start = dt-delta
        end = dt+delta

        async def do_fetch():
            logger.info("  Fetching client...")
            client = await self.get_client()
            events = await client.get_events(start=start,
                                             end=end,
                                             smart_detect_types=[SmartDetectObjectType.LICENSE_PLATE])

            for event in events:
                if source == event.camera_id:
                    thumb = event.get_detected_thumbnail()

                    if thumb is not None:
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

                            lpr = LicensePlateRead(license_plate=plate,
                                                   vehicle_type=type,
                                                   vehicle_color=color,
                                                   confidence=confidence,
                                                   bounding_box=bbox,
                                                   status="success",
                                                   diagnostic_messages="")
                            plates.append(lpr)

        await self._run_with_auth_recovery(do_fetch)
        return plates
    
    def get_license_plate_reads(self, source: str, dt: datetime, offset: int) -> list[LicensePlateRead]:
        return self._run_coroutine(self._get_license_plate_reads(source, dt, offset))
    
    def get_camera_filename(self, camera: Camera):
        return camera.name.lower().replace(" ", "")

class ProtectCredentialError(Exception):
    """Raised when Protect re-authentication fails (bad credentials, account disabled, etc.)"""


class ProtectMediaNotAvailable(Exception):
    """Exception raised when request media isn't available from UI Protect"""

    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code

    def __str__(self):
        return f"Protect Media Not Available(Error Code: {self.error_code})"
