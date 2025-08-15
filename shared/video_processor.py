"""Scans video files and extracts thumbnails of specified class in defined regions"""
import cv2
import imutils
import numpy as np
import logging
import os

from shared import Classifier, Detection, opencv_contours_clustering as cc, opencv_detection_helpers as dh, load_config

logger = logging.getLogger(__name__)
# logger.debug("Debug message")
# logger.info("Info message")
# logger.warning("Warning message")


class AreaState:
    """Helper Class to track event metadata"""
    def __init__(self, detection_area):
        self.area = detection_area
        self.area_entered_index = -1
        self.area_entered_frame = None

class VideoProcessor:
    """
    Helper class that processes video objects to extract images.

    Functions:
        process_video: Processes a video to extract an image.
        process_videos: Processes videos to extract an images.

    """
    def __init__(self, model_prototxt_path: str,
                       model_weights_path: str,
                       model_classes_path: str,
                       relevant_classes: list[int] = [],
                       confidence_threshold: float = 0.7):
        """
        Initialize a VideoProcessor object.

        Args:
            model_prototxt_path (str): Path to the model's prototxt configuration file.
            model_weights_path (str): Path to the model's weights file.
            model_classes_path (str): Path to a file containing class labels, one per line.
            relevant_classes (list[int]): Optional list of class IDs to retain in detection results.
            confidence_threshold (float): Minimum confidence required to accept a detection.
        """

        self._prototxt_path = model_prototxt_path
        self._model_path = model_weights_path
        self._classes_path = model_classes_path
        self._relevant_classes = relevant_classes
        self._confidence_threshold = confidence_threshold
        
        cl = Classifier(model_prototxt_path, 
                        model_weights_path, 
                        model_classes_path, 
                        relevant_classes, 
                        confidence_threshold)

        self._cc_preprocess_image = cc.preprocess_image
        self._cc_calculate_threshold = cc.calculate_threshold
        self._cc_calculate_box_distance = cc.calculate_box_distance
        self._cc_agglomerative_cluster = cc.agglomerative_cluster

        self._dh_draw_polygons = dh.draw_polygons
        self._dh_get_crop_contour = dh.get_crop_contour
        self._dh_detect_threshold_crossings = dh.detect_threshold_crossings
        self._dh_point_in_polygon = dh.point_in_polygon
        self._dh_threshold_crossed = dh.threshold_crossed
        self._dh_label_object = dh.label_object
        self._dh_frame_to_byte_array = dh.frame_to_byte_array

        self._cl_classify_image = cl.classify_image

    def __repr__(self):
        
        return (f"VideoProcessor("
                    f"prototxt_path={self._prototxt_path}, "
                    f"model_path={self._model_path}, "
                    f"classes_path={self._classes_path}, "
                    f"relevant_classes={self._relevant_classes}, "
                    f"confidence_threshold={self._confidence_threshold:.2f}, ")

    def _prep_frame(self, frame, zones):
        """"""
        # Draw over areas we want to exclude
        redacted = self._dh_draw_polygons(frame, zones)

        # Create a preprocessed image for motion detection
        preprocessed = self._cc_preprocess_image(redacted)

        return preprocessed

    def _detect_motion(self, frame, keyFrame, zones):
        """Detects differences between frame and keyframe, creates contours, and agglomerates them."""

        # Create a preprocessed image for motion detection
        preprocessed = self._prep_frame(frame, zones)

        # We need a key frame to conpare to the current frame
        if keyFrame is None:
            keyFrame = preprocessed
            return None, keyFrame

        # Compute the absolute difference between our keyframe and the current frame
        frame_delta = cv2.absdiff(keyFrame, preprocessed)

        # Blend the frame onto our keyFrame
        newkey = cv2.addWeighted(keyFrame, .95, preprocessed, .05, 0.0)
        #newkey = cv2.addWeighted(keyFrame, .80, preprocessed, .20, 0.0)

        # Compute threshold
        thresh = self._cc_calculate_threshold(frame_delta)

        # Find contours on thresholded image
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

        if cnts is not None:
            # Consolidate clusters of contours
            cnts = self._cc_agglomerative_cluster(cnts)
        else:
            cnts = []

        return cnts, newkey

    def _get_point(self, bounding_box: tuple[int, int, int, int]) -> tuple[int, int]:
        (x, y, w, h) = bounding_box

        point = (int(x+w), int(y+h))

        return point

    def _get_polygon(self, area: list[list[int]]) -> np.ndarray:
        """Convery list of points to numpy array"""
        polygon_np = np.array(area)

        return polygon_np

    def _lower_corner_crossings(self, frame, detections: list[Detection], state: AreaState, min_contour_area):
        """
        This function will detect when the lower left corner of a detection is inside a box
        made up of two thresholds.
        """
        for d in detections:


            polygon_np = self._get_polygon(state.area)

            result = False

            # Compute the bounding box for the class
            (x, y, w, h) = d.class_bounding_box

            # if the class box is too small, ignore it
            if (w * h) >= min_contour_area:
                point = self._get_point(d.class_bounding_box)
                result = self._dh_point_in_polygon(point, polygon_np)

            # If the class was too small check the contour's bounding box
            if result is True:
                # dh.label_object(frame, d.class_name, d.class_color, d.confidence, d.class_bounding_box)
                d.bounding_box = d.class_bounding_box
            else:
                # dh.label_object(frame, d.class_name, d.class_color, d.confidence, d.bounding_box)
                point = self._get_point(d.bounding_box)
                # Check if the point is in the polygon defined by the thresholds            
                result = self._dh_point_in_polygon(point, polygon_np)


            if result == True:
                state.area_entered_index = 0
                state.area_entered_frame = frame
                logger.debug("Inside Polygon.")

                # Reshape to the format OpenCV expects: (n_points, 1, 2)

                # pts = [polygon_np.reshape((-1, 1, 2))]
                # print(point, polygon_np)
                # print(pts)

                # dh.draw_box(frame, d.bounding_box)
                # dh.draw_box(frame, d.class_bounding_box)
                # cv2.polylines(frame, pts, True, (0,0,255))
                # cv2.circle(frame, point, 3, (255, 0, 0), -1)
                # cv2.imshow("Frame",imutils.resize(frame, width=960))
                # cv2.waitKey(0)

                # At this point we're done and can exit
                return True
        
        return False

    def _contours_to_detections(self, frame, contours, min_detection_area) -> list[Detection]:
        """Returns a list of Detections for each contour"""

        detections: list[Detection] = []

        for contour in contours or []:
            # Crops the frame based on the contour
            c_crop, c_box = self._dh_get_crop_contour(frame, contour, min_detection_area)

            if c_crop is None:
                continue

            # Create a detection for the contour
            detection = Detection(c_crop)
            detection.bounding_box = c_box

            # Get the detections from the classifier of the contour
            ds = self._cl_classify_image(c_crop)

            # Check classified detections
            for d in ds or []:
                # Check if the classified detection is the better
                if (detection.class_id < 0 and d.class_id >= 0) or ( detection.class_id >= 0 and d.confidence > detection.confidence):
                    # Make it the new baseline
                    detection.confidence = d.confidence
                    detection.class_color = d.class_color
                    detection.class_id = d.class_id
                    detection.class_name = d.class_name

                    # Classified detection bounding_boxes are relative to the crop
                    # Use the c_box to make them relative to the frame.
                    cx, cy, cw, ch = c_box or [0,0,0,0]
                    d_x, d_y, d_w, d_h = d.bounding_box or [0,0,0,0]
                    d.bounding_box = d_x +cx, d_y+cy, d_w, d_h
                    detection.class_bounding_box = d.bounding_box

                    #detection.shape = d.shape
                    
            # if len(ds) == 0:
            #     # Since the DNN was unable to classfy detections
            #     # from the contour crop, just create a detection
            #     # of the countour as a whole.
            #     d = Detection(-1, "unknown", (0,0,255), 0, np.ndarray(0))
            #     d.bounding_box = c_box
            #     ds.append(d)
            detections.append(detection)

        return detections
                
    def _check_frame(self, frame, keyFrame, state: AreaState, min_detection_area, zones):
        # Find differences between images as contours
        contours, keyFrame = self._detect_motion(frame, keyFrame, zones)

        # Filter contours to detections
        detections = self._contours_to_detections(frame, contours, min_detection_area)

        # Check for crossing of two thresholds
        #if self._threshold_crossings(frame, detections, state):
        #    return None

        # Check for point inside polygon
        if self._lower_corner_crossings(frame, detections, state, min_detection_area):
            return None

        zoned = dh.draw_polygons(frame, zones)
        frame = cv2.addWeighted(frame, .90, zoned, .10, 0.0)
        frame = dh.draw_lines(frame, state.area)
        cv2.imshow("Frame",imutils.resize(frame, width=960))
        cv2.waitKey(1)

        return keyFrame


    def process_video(self, image_name, video_name, min_detection_area, detection_area, ex_zones):
        vs = cv2.VideoCapture()
        logger.info("  Opening stream...")
        vs.open(video_name)
        frame = None
        keyFrame = None
        lastFrame = None
        state = AreaState(detection_area)
        frame_cnt = 0

        # loop through the video
        while vs.isOpened():
            # grab the next frame
            if vs.grab():
                # decode it
                _, frame = vs.retrieve()

            # we have reached the end of the video
            if frame is None:
                if lastFrame is not None:
                    #cv2.imwrite(image_name, lastFrame)
                    logger.info("  No more frames")
                    cv2.destroyAllWindows()
                    vs.release()
                    logger.info("  Stream closed")
                    logger.info(f"    Frames processed: {frame_cnt}")
                    return lastFrame
            
            frame_cnt += 1
            logger.debug(f"Frame {frame_cnt}")

            # Get the dimensions of the frame
            # image_height, image_width = frame.shape[:2]

            keyFrame = self._check_frame(frame, keyFrame, state, min_detection_area, ex_zones)

            if keyFrame is None:
                if state.area_entered_frame is not None:
                    #cv2.imwrite(image_name, state.first_crossed_frame)
                    logger.info("  Area entered")
                    cv2.destroyAllWindows()
                    vs.release()
                    logger.info("  Stream closed")
                    logger.info(f"    Frames processed: {frame_cnt}")
                    return state.area_entered_frame
                else:
                    # cv2.imwrite(image_name, frame)
                    logger.info("  No keyframe")
                    cv2.destroyAllWindows()
                    vs.release()
                    logger.info("  Stream closed")
                    logger.info(f"    Frames processed: {frame_cnt}")
                    return frame

            # Blank out the frame so that if capture fails the script ends
            lastFrame = frame
            frame = None

        cv2.destroyAllWindows()
        vs.release()
        logger.info("  Stream closed")
        logger.info(f"    Frames processed: {frame_cnt}")
        return None

    def process_videos(self, videos, video_clip_details):
        images = []
        thumbs = []

        # Build mapping from the filename to extraction details
        video_details_map = {d.get('file_name'): d for d in video_clip_details}

        logger.debug(f"Videos: {videos}")
        logger.debug(f"Map: {video_details_map}")

        total = len(videos)
        count = 0

        for video_file_path in videos:
            count += 1
            logger.info(f"Processing video {count} of {total}")
            logger.info(f"Video file path: {video_file_path}")

            video_filename = os.path.basename(video_file_path)

            if video_filename in video_details_map:
                details = video_details_map[video_filename]

                camera_name = details["camera_name"]
                image_name = video_file_path.replace(".mpg","") + ".png"
                video_name = video_file_path
                thumb_size = int(details["thumbnail_max_height"])
                thumb_name = video_file_path.replace(".mpg","") + "_thumb.png"
                lpr = details["perform_lpr"]
                min_detection_area = details["minimum_detection_area"]
                detection_area = details["detection_polygon"]
                ex_zones = details["exclusion_zones"]

                logger.info(f"Processing video for camera: {camera_name}")
                image = self.process_video(image_name, video_name, min_detection_area, detection_area, ex_zones)

                if image is not None:
                    logger.info(f"Saving image: {image_name}")
                    cv2.imwrite(filename=image_name, img=image)
                    cv2.imwrite(filename=thumb_name, img=imutils.resize(image, height=thumb_size))
                    images.append(image_name)
                    thumbs.append(thumb_name)
        
        return [images, thumbs]