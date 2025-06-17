"""Scans video files and extracts thumbnails of specified class in defined regions"""
import cv2
import imutils
import logging
import os

from shared import Classifier, Detection, opencv_contours_clustering as cc, opencv_detection_helpers as dh, load_config

logger = logging.getLogger(__name__)
# logger.debug("Debug message")
# logger.info("Info message")
# logger.warning("Warning message")


class ThresholdState:
    """Helper Class to track event metadata"""
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.first_crossed_index = -1
        self.first_crossed_frame = None
        self.second_crossed_index = -1
        self.second_crossed_frame = None

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

    def _threshold_crossings(self, frame, detections: list[Detection], state: ThresholdState):
        """"""
        for d in detections:

            dh.label_object(frame, d.class_name, d.class_color, d.confidence, d.bounding_box)

            # Check for threshold crossings
            for index, threshold in enumerate(state.thresholds):
                if dh.threshold_crossed(threshold, d.bounding_box):
                    # If a threshold is crossed...

                    if state.first_crossed_index < 0:
                        # cv2.imshow("First Crossed",dh.draw_line(frame.copy(),threshold, (255,0,0)))
                        # cv2.waitKey(0)

                        logger.debug("First line crossed.")
                        state.first_crossed_index = index
                        state.first_crossed_frame = frame

                    elif index != state.first_crossed_index and \
                            state.second_crossed_index < 0:
                        # cv2.imshow("Second Crossed",dh.draw_line(frame.copy(),threshold, (255,0,0)))
                        # cv2.waitKey(0)

                        state.second_crossed_index = index
                        state.second_crossed_frame = frame
                        logger.debug("Second line crossed.")

                        # At this point we're done and can exit
                        return True
        
        return False

    def _contours_to_detections(self, frame, contours, min_contour_area) -> list[Detection]:
        """"""

        detections: list[Detection] = []

        for contour in contours or []:
            c_crop, c_box = self._dh_get_crop_contour(frame, contour, min_contour_area)

            if c_crop is None:
                continue

            ds = self._cl_classify_image(c_crop)

            # Detection bounding_boxes are relative to the crop
            # Use the c_box to make them relative to the frame.
            for d in ds or []:
                cx, cy, cw, ch = c_box or [0,0,0,0]
                d_x, d_y, d_w, d_h = d.bounding_box or [0,0,0,0]

                d.bounding_box = d_x +cx, d_y+cy, d_w, d_h

            # if len(ds) == 0:
            #     # Since the DNN was unable to classfy detections
            #     # from the contour crop, just create a detection
            #     # of the countour as a whole.
            #     d = Detection(-1, "unknown", (0,0,255), 0, np.ndarray(0))
            #     d.bounding_box = c_box
            #     ds.append(d)
                
            detections.extend(ds)

        return detections
                
    def _check_frame(self, frame, keyFrame, state: ThresholdState, min_contour_area, zones):
        # Find differences between images as contours
        contours, keyFrame = self._detect_motion(frame, keyFrame, zones)

        # Filter contours to detections
        detections = self._contours_to_detections(frame, contours, min_contour_area)

        # Check for crossing of two thresholds
        if self._threshold_crossings(frame, detections, state):
            return None

        # zoned = dh.draw_polygons(frame, zones)
        # frame = cv2.addWeighted(frame, .90, zoned, .10, 0.0)
        # frame = dh.draw_lines(frame, state.thresholds)
        # cv2.imshow("Frame",imutils.resize(frame, width=960))
        # cv2.waitKey(1)

        return keyFrame


    def process_video(self, image_name, video_name, min_contour_area, thresholds, ex_zones):
        vs = cv2.VideoCapture()
        print("  Stream opened.")
        vs.open(video_name)
        frame = None
        keyFrame = None
        lastFrame = None
        state = ThresholdState(thresholds)
        frame_cnt = 0

        while vs.isOpened():
            # grab the next frame
            if vs.grab():
                # decode it
                _, frame = vs.retrieve()

            # we have reached the end of the video
            if frame is None:
                if state.first_crossed_frame is not None:
                    #cv2.imwrite(image_name, state.first_crossed_frame)
                    logger.debug("  One threshold crossed")
                    return state.first_crossed_frame
                elif lastFrame is not None:
                    #cv2.imwrite(image_name, lastFrame)
                    logger.debug("  No more frames")
                    return lastFrame
            
            frame_cnt += 1
            logger.debug(f"Frame {frame_cnt}")

            # Get the dimensions of the frame
            # image_height, image_width = frame.shape[:2]

            keyFrame = self._check_frame(frame, keyFrame, state, min_contour_area, ex_zones)

            if keyFrame is None:
                if state.second_crossed_frame is not None:
                    # cv2.imwrite(image_name, state.second_crossed_frame)
                    logger.debug("  Two threshold crossed")
                    return state.second_crossed_frame

                else:
                    # cv2.imwrite(image_name, frame)
                    logger.debug("  No keyframe")
                    return frame

            # Blank out the frame so that if capture fails the script ends
            lastFrame = frame
            frame = None

        cv2.destroyAllWindows()
        vs.release()
        print("  Stream closed.")
        return None


    def process_videos(self, videos, video_clip_details):
        result = []

        # Build mapping from the  to full paths
        video_file_map = {os.path.basename(p): p for p in videos}

        logger.debug(f"Videos: {videos}")
        logger.debug(f"Map: {video_file_map}")

        for details in video_clip_details:
            video_filename = details.get('file_name')
            logger.debug(f"Video Filename: {video_filename}")
            if video_filename in video_file_map:
                video_file_path = video_file_map[video_filename]

                camera_name = details["camera_name"]
                image_name = video_file_path.replace(".mpg","") + ".png"
                video_name = video_file_path
                thumb_size = int(details["thumbnail_max_height"])
                thumb_name = video_file_path.replace(".mpg","") + "_thumb.png"
                lpr = details["perform_lpr"]
                min_contour_area = details["minimum_contour_area"]
                thresholds = details["detection_thresholds"]
                ex_zones = details["exclusion_zones"]

                print(f"Saving image for camera: {camera_name}")
                image = self.process_video(image_name, video_name, min_contour_area, thresholds, ex_zones)

                if result is not None:
                    cv2.imwrite(filename=image_name, img=image)
                    cv2.imwrite(filename=thumb_name, img=imutils.resize(image, height=thumb_size))
                    result.append(image_name)
        
        return result
