"""
opencv_contours_clustering.py

Functions to help with motion detection.
Adapted from https://github.com/CullenSUN/fish_vision/blob/master/playground/contours_clustering.py
"""
#!/usr/bin/env python3

import os
import cv2
import imutils
import numpy as np

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image to improve performance"""

    #output = normalize_frame(img)
    output = img.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    output = cv2.GaussianBlur(output, (21, 21), 0)

    return output


def calculate_threshold(img: np.ndarray, preprocess_image_first: bool = False) -> np.ndarray:
    """
    Convert an image to a binary (black and white) thresholded image, 
    optionally applying preprocessing first.

    Parameters:
        img (np.ndarray): The input image, expected to be a grayscale or single-channel image.
        preprocess_image_first (bool): If True, applies a preprocessing step before thresholding.

    Returns:
        np.ndarray: The binary image after thresholding and dilation. Pixel values will be 0 or 255.
    """
    thresh_img = img

    if preprocess_image_first:
        thresh_img = preprocess_image(thresh_img)

    # Apply binary thresholding
    _, thresh = cv2.threshold(thresh_img, 25, 255, cv2.THRESH_BINARY)

    # Dilate to fill small holes and connect regions
    thresh = cv2.dilate(thresh, None, iterations=2)

    return thresh


def calculate_box_distance(box1: tuple[int, int, int, int],
                           box2: tuple[int, int, int, int]) -> float:
    """
    Calculate the distance between the centers of two bounding boxes, 
    adjusted for their sizes.

    This returns the separation between the boxes along the X or Y axis, 
    accounting for width and height. If boxes overlap, the result may be 0 or negative.

    Parameters:
        box1 (Tuple[int, int, int, int]): The first bounding box as (x, y, w, h).
        box2 (Tuple[int, int, int, int]): The second bounding box as (x, y, w, h).

    Returns:
        float: The adjusted distance between the boxes along the dominant axis.
    """
    x1, y1, w1, h1 = box1
    b1_x = x1 + w1 / 2
    b1_y = y1 + h1 / 2

    x2, y2, w2, h2 = box2
    b2_x = x2 + w2 / 2
    b2_y = y2 + h2 / 2

    dx = abs(b1_x - b2_x) - (w1 + w2) / 2
    dy = abs(b1_y - b2_y) - (h1 + h2) / 2

    return max(dx, dy)


def calculate_contour_distance(contour1, contour2):
    """Calculates the distance between the centers of two 
       contours, accounting for their sizes."""
    box1 = cv2.boundingRect(contour1)
    box2 = cv2.boundingRect(contour2)

    return calculate_box_distance(box1, box2)


def merge_contours(contour1, contour2):
    """Merges two contours into a single contour"""
    return np.concatenate((contour1, contour2), axis=0)


def agglomerative_cluster(contours, threshold_distance=40.0):
    """Merges countours that are within a certain
       distance from one another."""
    current_contours = list(contours)

    while len(current_contours) > 1:
        min_distance = 0
        min_coordinate = (0, 0)

        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance == 0:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1],
                                                      current_contours[index2])
            del current_contours[index2]
        else:
            break

    return tuple(current_contours)
