"""
lpr_helpers.py

Helper classes for license plate reader functions.
"""
from pydantic import BaseModel
from typing import Optional, Literal

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class LicensePlateRead(BaseModel):
    license_plate: Optional[str] = None
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    confidence: Optional[float] = None
    bounding_box: Optional[BoundingBox] = None
    diagnostic_messages: str
    status: Literal["success", "no_plate_found", "error"]
    error_message: Optional[str] = None
