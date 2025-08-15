import base64
import logging
import json
import os
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Literal

logger = logging.getLogger(__name__)
# logger.debug("Debug message")
# logger.info("Info message")
# logger.warning("Warning message")

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class LicensePlateRead(BaseModel):
    license_plate: Optional[str] = None
    confidence: Optional[float] = None
    bounding_box: Optional[BoundingBox] = None
    diagnostic_messages: str
    status: Literal["success", "no_plate_found", "error"]
    error_message: Optional[str] = None


class OpenAILicensePlateReader:
    """
    Helper class that reads a license plate using OpenAI.

    Functions:
        get_license_plate_read: Submits the image to OpenAI to perform an LPR and 
                                returns the result as JSON.
    """
    def __init__(self,
                 api_key: str,
                 model: str,
                 prompt: str ):
        """
        Initialize LicensePlateReader.

        Args:
            api_key (str): OpenAI API Key
            model (str): OpenAI Model
            prompt (str): Prompt for model to read license plate.
        """

        self.api_key = api_key
        self.model = model
        self.prompt = prompt

    def __repr__(self):

        return (f"LicensePlateReader("
                f"api_key='{self.api_key}', "
                f"model='{self.model}', "
                f"prompt='{self.prompt})'")

    def get_license_plate_read(self, image_path):
        """Passes the prompt and image to OpenAI and receives a response."""

        result = LicensePlateRead(
            status="error",
            diagnostic_messages="Default return when image path is invalid.",
            error_message="Image not found."
        )

        # Initialize the client (requires `OPENAI_API_KEY` in your environment)
        client = OpenAI(api_key=self.api_key)

        # Encode the image to base64
        logger.info(f"  Opening file: {image_path}")
        if os.path.isfile(image_path):
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            if image_base64: # Check if list is not empty
                response = client.responses.parse(
                    model=self.model,
                    store=False,
                    input=[
                        {"role": "system", "content": self.prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                },
                            ],
                        }
                    ],
                    text_format=LicensePlateRead,
                )

                result = response.output_parsed

        return result
