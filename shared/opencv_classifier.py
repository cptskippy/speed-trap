import cv2
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)
# logger.debug("Debug message")
# logger.info("Info message")
# logger.warning("Warning message")

class Detection:
    """
    Represents a single object detected in an image.

    Attributes:
        class_id (int): The numeric ID of the predicted class.
        class_name (str): The human-readable name of the class.
        class_color (Tuple[int, int, int]): A color (e.g., BGR tuple) associated with the class for display.
        confidence (float): Confidence score of the detection (typically 0.0 to 1.0).
        shape (np.ndarray): Normalized bounding box coordinates (x_min, y_min, x_max, y_max), as floats.
        bounding_box (Optional[Tuple[int, int, int, int]]): Absolute pixel coordinates
            of the bounding box, computed relative to the original image in XYWH format.
    """
    def __init__(self,
                 class_id: int,
                 class_name: str,
                 class_color: tuple[int, int, int],
                 confidence: float,
                 shape: np.ndarray):
        """
        Initialize a Detection object.

        Args:
            class_id (int): The class ID predicted by the model.
            class_name (str): The name of the detected class.
            class_color: A color used for visualizing this class.
            confidence (float): Confidence level of the prediction.
            shape (np.ndarray): Normalized bounding box coordinates (x_min, y_min, x_max, y_max).
        """

        self.class_id = class_id
        self.class_name = class_name
        self.class_color = class_color
        self.confidence = confidence
        self.shape = shape
        self.bounding_box: Optional[tuple[int, int, int, int]] = None

    def __repr__(self):

        return (f"Detection("
                f"class_id={self.class_id}, "
                f"class_name='{self.class_name}', "
                f"class_color='{self.class_color}', "
                f"confidence={self.confidence:.2f}, "
                f"shape={self.shape.tolist()})")


class Classifier:
    """
    Helper class that classifies objects in images using a pretrained neural network.

    This class loads a Caffe model and provides configuration for preprocessing, 
    confidence filtering, class filtering, and color labeling of detected classes.

    Functions:
        classify_image: Classifies objects in an image and returns detections with bounding boxes.
        
    """
    def __init__(self, model_prototxt_path: str,
                       model_weights_path: str,
                       model_classes_path: str,
                       relevant_classes: list[int] = [],
                       confidence_threshold: float = 0.7,
                       mean_scalar : float = 127.5,
                       mean_subtraction: float = (1/127.5),
                       spatial_size: tuple[int, int] = (300, 300),
                       swap_red_blue = False):
        """
        Initialize a Detection object.

        Args:
            model_prototxt_path (str): Path to the model's prototxt configuration file.
            model_weights_path (str): Path to the model's weights file.
            model_classes_path (str): Path to a file containing class labels, one per line.
            relevant_classes (list[int]): Optional list of class IDs to retain in detection results.
            confidence_threshold (float): Minimum confidence required to accept a detection.
            mean_scalar (float): Scalar used to normalize input image pixel values.
            mean_subtraction (float): Factor for mean subtraction normalization.
            spatial_size (tuple[int, int]): Desired input size (width, height) for the neural network.
            swap_red_blue (bool): Whether to swap red and blue channels (used for BGR/RGB adjustment).
        """

        self._prototxt_path = model_prototxt_path
        self._model_path = model_weights_path
        self._classes_path = model_classes_path
        self._relevant_classes = relevant_classes
        self._confidence_threshold = confidence_threshold
        self._mean_scalar = mean_scalar
        self._mean_subtraction = mean_subtraction
        self._spatial_size = spatial_size
        self._swap_red_blue = swap_red_blue

        if self._classes_path is not None:
            # Create a list of class names
            self._classes = tuple(open(self._classes_path).read().strip().split("\n"))

            # Create a set of colors for the classes.
            np.random.seed(42)
            colors = np.random.randint(0, 256, size=(len(self._classes), 3), dtype=np.uint8)
            #self._colors = [tuple(int(c) for c in color) for color in colors]
            self._colors = tuple((int(c[0]),int(c[1]),int(c[2])) for c in colors)
        else:
            logger.warning("No classes_path, class names and colors will be undefined.")

        # Load NN from model
        self._dnn=cv2.dnn.readNetFromCaffe(self._prototxt_path, self._model_path)

    def __repr__(self):
        
        return (f"Classifier("
                    f"prototxt_path={self._prototxt_path}, "
                    f"model_path={self._model_path}, "
                    f"classes_path={self._classes_path}, "
                    f"relevant_classes={self._relevant_classes}, "
                    f"confidence_threshold={self._confidence_threshold:.2f}, "
                    f"mean_scalar={self._mean_scalar:.2f}, "
                    f"mean_subtraction={self._mean_subtraction:.2f}, "
                    f"spatial_size={self._spatial_size:.2f}, "
                    f"swap_red_blue={self._swap_red_blue})")

    def _get_detections(self, img: np.ndarray) -> list[Detection]:
        """
        Detects and classifies objects in an image using a neural network.

        This method preprocesses the input image, passes it through a pre-trained 
        neural network, and returns the detection results.

        Args:
            img (np.ndarray): The input image in BGR format as a NumPy array.

        Returns:
            Detection: Represents a single object detected in an image with
                       information such as:
                           [class_id, class_name, class_color, confidence, bounding_box]
        """

        # Preprocess the image for the NN
        blob = cv2.dnn.blobFromImage(img, self._mean_subtraction, self._spatial_size,
                                     self._mean_scalar, swapRB = self._swap_red_blue)

        # Pass the image to the NN for classification
        self._dnn.setInput(blob)
        output = self._dnn.forward()

        # Output is 4D but only the last two dimensions have data
        results = output[0, 0, :, :]

        # We need to parse the results into Detections
        detections = list[Detection]()

        for r in results:
            d = self._parse_and_filter(r)

            if d is None:
                continue

            detections.append(d)

        return detections

    def _parse_and_filter(self, dnn_result: np.ndarray) -> Optional[Detection]:
        """
        Parses and filters a single DNN detection result.

        This method extracts the class ID, confidence score, and bounding box
        from a raw DNN output row and filters it based on the confidence threshold
        and optionally a list of relevant class IDs.

        It also attaches the class name and class specific color for labeling output.

        Args:
            dnn_result (np.ndarray): A 1D array from the DNN output representing one detection.
                Format is typically [image_id, class_id, confidence, x_min, y_min, x_max, y_max].

        Returns:
            Optional[Detection]: A Detection object if the detection is valid and passes
            all filters; otherwise, None.
        """
        
        # Extract the Confidence
        class_id = int(dnn_result[1])
        confidence = dnn_result[2]

        # Evaluate relevancy
        if self._relevant_classes and class_id not in self._relevant_classes:
            return None
        
        # Evaluate Confidence
        if confidence < self._confidence_threshold:
            return None

        # Extract the Class
        class_name = self._classes[class_id]

        # Associate Class to a Color
        class_color = self._colors[class_id]

        return Detection(class_id, class_name, class_color, confidence, dnn_result[3:7])

    def classify_image(self, img: np.ndarray) -> list[Detection]:
        """
        Classifies objects in an image and returns detections with bounding boxes.

        This method processes the input image, performs object detection using the neural
        network, and converts normalized detection coordinates to absolute pixel positions.
        It then converts those coordinates to a XYWH bounding box.

        Args:
            img (np.ndarray): The input image (in BGR format) to classify.

        Returns:
            list[Detection]: A list of Detection objects for all valid detections in the image.
        """

        # Get the dimensions of the source image
        h, w = img.shape[:2]

        detections = self._get_detections(img)

        for d in detections:
            # Extract the corners around the detection.
            # These are percentages of the height and width.
            # So we multiply by height and width
            corners = d.shape * np.array([w, h, w, h])

            # Then round to integers because pixels
            x_min, y_min, x_max, y_max = corners.round().astype(int)

            # Crop the dection from the image
            # crop = img[y_min:y_max, x_min:x_max]
            # cv2.imshow("Crop",crop)

            x = x_min  # x position of crop in the img
            y = y_min  # y position of crop in the img
            w = x_max - x_min  # width of the crop
            h = y_max - y_min  # height of the crop

            # Relative box in a format OpenCV expects
            d.bounding_box = x, y, w, h

        return detections
