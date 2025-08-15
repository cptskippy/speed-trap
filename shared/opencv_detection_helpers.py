"""
opencv_detection_helpers.py

Functions to help with motion detection.
"""
#!/usr/bin/env python3

import cv2
import imutils
import logging
import numpy as np
from typing import Optional
import re

logger = logging.getLogger(__name__)
# logger.debug("Debug message")
# logger.info("Info message")
# logger.warning("Warning message")


def get_crop_contour(frame, 
                     contour, 
                     min_contour_area, 
                     padding = 20) -> tuple[Optional[np.ndarray], 
                                            Optional[tuple[int, int, int, int]]]:
    """
    Crops a region of interest from the frame based on a contour and 
    returns the cropped image and bounding box.

    This function filters out small contours, calculates a padded bounding
    box around the remaining contour, ensures it fits within the frame 
    bounds, and returns the cropped region along with the bounding box.

    Args:
        frame (np.ndarray): The input image from which to crop.
        contour (np.ndarray): A single contour (as returned by cv2.findContours).
        min_contour_area (float): Minimum bounding box area to consider the 
                                  contour valid.
        padding (int, optional): Amount of padding (in pixels) to add 
                                 around the bounding box. Default is 20.

    Returns:
        Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
            - The cropped region of the frame containing the contour, or 
              None if contour is too small.
            - A tuple (x, y, w, h) representing the bounding box in the 
              original frame, or None if ignored.
    """
    # Compute the bounding box for the contour
    (x, y, w, h) = cv2.boundingRect(contour)

    # if the contour is too small, ignore it
    if (w * h) < min_contour_area:
        return None, (0,0,0,0)   

    if padding > 0:
        # Get the dimensions of the frame
        image_height, image_width = frame.shape[:2]

        # We want padding on both sides
        padding2x = padding*2

        # Min/Max to ensure contour doesn't clip frame dimensions
        x = max(x-padding, 0)
        y = max(y-padding, 0)
        w = min(w+padding2x,image_width-x)
        h = min(h+padding2x,image_height-y)

    # Crop the countor area from the original frame
    contour_crop = frame[y:y+h, x:x+w]
    bounding_box = (x, y, w, h)

    logger.debug(f"Contour Bounding Box: {(x, y, w, h)}")
    logger.debug(f"  Box Area: {(w * h)}")
    logger.debug(f"  Min Area: {min_contour_area}")

    return contour_crop, bounding_box

def oriented_ccw(a, b, c):
    """Determines if the orientation of 3 ordered points are counterclockwise"""
    return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

def intersect(p1, q1, p2, q2):
    """Return true if line segments [p1,q1] and [p2,q2] intersect"""
    return oriented_ccw(p1,p2,q2) != oriented_ccw(q1,p2,q2) and \
            oriented_ccw(p1,q1,p2) != oriented_ccw(p1,q1,q2)

def detect_threshold_crossings(zone, detection):
    """Determines if a detection crosses one of two thresholds"""
    # Only check if we haven't detected a crossing
    if detection.threshold_crop is not None:
        return detection

    (z_p, z_q) = zone

    # Gets the latest detection event box
    le = detection.events[-1]

    (x, y, w, h) = le.box

    l_p = [x, y]
    l_q = [x, y+h]

    r_p = [x+w, y]
    r_q = [x+w, y+h]


    # Check if the left edge is in zone
    if intersect(l_p, l_q, z_p, z_q):
        detection.threshold_crossed = le.seen
        detection.threshold_crop = le.crop
        detection.threshold_box = le.box
        print("Threshold crossed")
        cv2.line(detection.threshold_crop, l_p, l_q, (128,128,0), 3)
        cv2.line(detection.threshold_crop, z_p, z_q, (0,128,128), 3)

    # Check if the right edge is in zone
    if intersect(r_p, r_q, z_p, z_q):
        detection.threshold_crossed = le.seen
        detection.threshold_crop = le.crop
        detection.threshold_box = le.box
        print("Threshold crossed")
        cv2.line(detection.threshold_crop, r_p, r_q, (128,128,0), 3)
        cv2.line(detection.threshold_crop, z_p, z_q, (0,128,128), 3)

    return detection

def threshold_crossed(zone, box):
    """Determines if a detection crossed a threshold"""

    (z_p, z_q) = zone

    (x, y, w, h) = box

    l_p = [x, y]
    l_q = [x, y+h]

    r_p = [x+w, y]
    r_q = [x+w, y+h]


    # Check if the left edge is in zone
    if intersect(l_p, l_q, z_p, z_q):
        return True

    # Check if the right edge is in zone
    if intersect(r_p, r_q, z_p, z_q):
        return True

    return False

def point_in_polygon(point: tuple[int, int], polygon: np.ndarray):
    result = cv2.pointPolygonTest(polygon, point, measureDist=False)

    return result >= 0

def draw_line(img: np.ndarray, line: tuple[int, int], 
              color: tuple[int, int, int] = (0,255,0)) -> np.ndarray:
    """Draws a line on the image and returns the marked up image"""
    p = np.int32(np.array(line[0], np.int32))
    q = np.int32(np.array(line[1], np.int32))

    cv2.line(img, p, q, color, 3)

    return img

def draw_lines(img: np.ndarray, lines) -> np.ndarray:
    """Draws lines on the image and returns the marked up image"""
    output = img.copy()

    for line in lines:
        output = draw_line(output, line)

    return output


def draw_polygon(img, polygon, color = (0,0,255)) -> np.ndarray:
    """Draws a polygon over an area of a image"""

    pts = np.array(polygon, np.int32)
    cv2.fillPoly(img=img, pts=np.int32([pts]), color=color)

    return img

def draw_polygons(img, polygons, color = (0,0,255)) -> np.ndarray:
    """Draws polygons over areas of a image"""
    output = img.copy()

    for polygon in polygons or []:
        logger.debug(f"Drawing Polygon: {polygon}")
        output = draw_polygon(output, polygon, color)

    return output


def draw_box(frame, box, color = (0,255,0)):
    """Draws a box on a frame"""
    (x, y, w, h) = box

    # Draw box around detection and label
    cv2.rectangle(frame, (x+2, y+2), (x+w+2, y+h+2), (0,0,0), thickness=2)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=2)


def label_object(frame, class_name, class_color, confidence, box):
    """Labels the object in a frame"""
    (x, y, w, h) = box

    # Draw box around detection and label
    draw_box(frame, box, class_color)

    y2 = y - 15 if y - 15 > 15 else y + 15

    class_label = f"{class_name}: {confidence*100:.2f}%".capitalize()
    cv2.putText(frame, class_label, (x+2, y2+2), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    cv2.putText(frame, class_label, (x, y2), cv2.FONT_HERSHEY_COMPLEX, 1, class_color, 2)

    # boxY = y + 15
    # box_label = f"Box Area: {(w*h)}".capitalize()
    # cv2.putText(frame, box_label, (x+2, boxY+2), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    # cv2.putText(frame, box_label, (x, boxY), cv2.FONT_HERSHEY_COMPLEX, 1, class_color, 2)

def frame_to_byte_array(frame):
    """Converts a frame to a PNG encoded byte array"""
    buffer = cv2.imencode('.png',frame)
    return buffer[1].tobytes()


def display_frame(frame, excluded):
    """Displays a frame on the desktop with exclusions and FPS, helpful for debugging"""
    # Get the dimensions of the frame
    image_height, image_width = frame.shape[:2]

    # So we can check our Exclusion Zones
    combined = cv2.addWeighted(frame, .75, excluded, .25, 0.0)

    # show the frame and log key presses
    cv2.imshow("Debug stream", imutils.resize(combined, width=480))


def calculate_box_distance(box1, box2):
    """Calculates the distance between the centers of two 
       bounding boxes, accounting for their sizes."""
    (x1, y1, w1, h1) = box1
    b_x1 = x1 + w1/2
    b_y1 = y1 + h1/2

    (x2, y2, w2, h2) = box2
    b_x2 = x2 + w2/2
    b_y2 = y2 + h2/2

    return max(abs(b_x1 - b_x2) - (w1 + w2)/2, abs(b_y1 - b_y2) - (h1 + h2)/2)

def merge_detections(d1, d2):
    """Helper function to merge two detections"""

    if d2.confidence > d1.confidence:
        d1.confidence = d2.confidence

    d1.events = [*d1.events, *d2.events]

    if d1.threshold_crossed is None:
        d1.threshold_crossed = d2.threshold_crossed
        d1.threshold_crop = d2.threshold_crop
        d1.threshold_box = d2.threshold_box

    return d1

def agglomerative_detections(detections, threshold_distance=40.0):
    """Helper function to merge multiple detections"""
    
    while len(detections) > 1:
        min_distance = 0
        min_coordinate = (0, 0)

        for x in range(len(detections)-1):
            for y in range(x+1, len(detections)):
                if detections[x].class_id == detections[y].class_id:
                    distance = calculate_box_distance(detections[x].events[-1].box, detections[y].events[-1].box)
                    if min_distance == 0:
                        min_distance = distance
                        min_coordinate = (x, y)
                    elif distance < min_distance:
                        min_distance = distance
                        min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            detections[index1] = merge_detections(detections[index1], detections[index2])
            del detections[index2]
        else:
            print("Threshold too great: " + str(min_distance))
            break

    return detections
