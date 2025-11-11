from __future__ import annotations

import cv2
import numpy as np
import yaml
import supervision as sv
import time
import threading
from collections import deque
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


class YOLOv8TFLite:
    """
    A YOLOv8 object detection class using TensorFlow Lite for efficient inference.

    This class handles model loading, preprocessing, inference, and visualization of detection results for YOLOv8
    models converted to TensorFlow Lite format.

    Attributes:
        model (Interpreter): TensorFlow Lite interpreter for the YOLOv8 model.
        conf (float): Confidence threshold for filtering detections.
        iou (float): Intersection over Union threshold for non-maximum suppression.
        classes (dict): Dictionary mapping class IDs to class names.
        color_palette (np.ndarray): Random color palette for visualization with shape (num_classes, 3).
        in_width (int): Input width required by the model.
        in_height (int): Input height required by the model.
        in_index (int): Input tensor index in the model.
        in_scale (float): Input quantization scale factor.
        in_zero_point (int): Input quantization zero point.
        int8 (bool): Whether the model uses int8 quantization.
        out_index (int): Output tensor index in the model.
        out_scale (float): Output quantization scale factor.
        out_zero_point (int): Output quantization zero point.

    Methods:
        letterbox: Resize and pad image while maintaining aspect ratio.
        draw_detections: Draw bounding boxes and labels on the input image.
        preprocess: Preprocess the input image before inference.
        postprocess: Process model outputs to extract and visualize detections.
        detect: Perform object detection on an input image.
    """

    def __init__(
        self,
        model: str,
        conf: float = 0.25,
        iou: float = 0.45,
        metadata: str | None = None,
    ):
        """
        Initialize the YOLOv8TFLite detector.

        Args:
            model (str): Path to the TFLite model file.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            metadata (str | None): Path to the metadata file containing class names.
        """
        self.conf = conf
        self.iou = iou
        if metadata is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(metadata) as f:
                self.classes = yaml.safe_load(f)["names"]
        np.random.seed(42)  # Set seed for reproducible colors
        self.color_palette = np.random.uniform(128, 255, size=(len(self.classes), 3))

        # Initialize the TFLite interpreter
        edgetpu_delegate = load_delegate("libedgetpu.so.1", {"device": "usb"})
        self.model = Interpreter(
            model_path=model, experimental_delegates=[edgetpu_delegate]
        )
        self.model.allocate_tensors()

        # Get input details
        input_details = self.model.get_input_details()[0]
        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]
        self.in_scale, self.in_zero_point = input_details["quantization"]
        self.int8 = input_details["dtype"] == np.int8

        # Get output details
        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]
        self.out_scale, self.out_zero_point = output_details["quantization"]

    def letterbox(
        self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image with shape (H, W, C).
            new_shape (tuple[int, int]): Target shape (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
            (tuple[float, float]): Padding ratios (top/height, left/width) for coordinate adjustment.
        """
        shape = img.shape[:2]  # Current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            (new_shape[1] - new_unpad[0]) / 2,
            (new_shape[0] - new_unpad[1]) / 2,
        )  # wh padding

        if shape[::-1] != new_unpad:  # Resize if needed
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, (top / img.shape[0], left / img.shape[1])

    def draw_detections(
        self, img: np.ndarray, box: np.ndarray, score: np.float32, class_id: int
    ) -> None:
        """
        Draw bounding boxes and labels on the input image based on detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (np.ndarray): Detected bounding box in the format [x1, y1, width, height].
            score (np.float32): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        x1, y1, w, h = box
        color = self.color_palette[class_id]

        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create label with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Get text size for background rectangle
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Position label above or below box depending on space
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw label background
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw text
        cv2.putText(
            img,
            label,
            (int(label_x), int(label_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """
        Preprocess the input image before performing inference.

        Args:
            img (np.ndarray): The input image to be preprocessed with shape (H, W, C).

        Returns:
            (np.ndarray): Preprocessed image ready for model input.
            (tuple[float, float]): Padding ratios for coordinate adjustment.
        """
        t0 = time.time()

        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[..., ::-1][
            None
        ]  # BGR to RGB and add batch dimension (N, H, W, C) for TFLite
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        t1 = time.time()
        print(f"{t1 - t0} preprocess\n")
        return img / 255, pad  # Normalize to [0, 1]

    def postprocess(
        self,
        img: np.ndarray,
        outputs: np.ndarray,
        pad: tuple[float, float],
        tracker,
        box_annotator,
        label_annotator,
    ) -> np.ndarray:
        """
        Process model outputs to extract and visualize detections.

        Args:
            img (np.ndarray): The original input image.
            outputs (np.ndarray): Raw model outputs.
            pad (tuple[float, float]): Padding ratios from preprocessing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        t0 = time.time()

        # Adjust coordinates based on padding and scale to original image size
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(img.shape)

        # Transform outputs to [x, y, w, h] format
        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0] -= outputs[..., 2] / 2  # x center to top-left x
        outputs[..., 1] -= outputs[..., 3] / 2  # y center to top-left y

        detections_list = []
        for out in outputs:
            # Get scores and apply confidence threshold
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)

            # Apply non-maximum suppression
            nms_result = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)

            if len(nms_result) > 0:
                indices = nms_result.flatten()
            else:
                indices = np.array([])

            if len(indices) == 0:
                continue

            detections = sv.Detections(
                xyxy=np.array(
                    [
                        [
                            boxes[i][0],
                            boxes[i][1],
                            boxes[i][0] + boxes[i][2],
                            boxes[i][1] + boxes[i][3],
                        ]
                        for i in indices
                    ]
                ),
                confidence=np.array([scores[i] for i in indices]),
                class_id=np.array([class_ids[i] for i in indices]),
            )
            detections_list.append(detections)

        if len(detections_list) == 0:
            return img

        detections = sv.Detections.merge(detections_list)
        detections = tracker.update_with_detections(detections)

        labels = [
            f"COW {confidence:0.2f}"
            for xyxy, mask, confidence, class_id, tracker_id, dtype in detections
        ]

        img = box_annotator.annotate(img, detections)
        img = label_annotator.annotate(img, detections, labels)

        t1 = time.time()
        print(f"{t1 - t0} postprocess\n")
        return img

    def detect(self, img, tracker, box_annotator, label_annotator) -> np.ndarray:
        """
        Perform object detection on an input image.

        Args:
            img (np.ndarray): Input image as numpy array.
            tracker: Object tracker instance.
            box_annotator: Annotation tool for bounding boxes.
            label_annotator: Annotation tool for labels.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        x, pad = self.preprocess(img)

        # Apply quantization if model is int8
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # Set input tensor and run inference
        t0 = time.time()
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()

        # Get output and dequantize if necessary
        y = self.model.get_tensor(self.out_index)
        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale
        t1 = time.time()
        # Process detections and return result
        print(f"{t1 - t0} inference\n")
        return self.postprocess(img, y, pad, tracker, box_annotator, label_annotator)


# Параметры детектора
model = "yolo_demo_edgetpu.tflite"
conf = 0.85
iou = 0.35
metadata = "metadata.yaml"

# Инициализация ROS-ноды
rospy.init_node("yolo_object_detector", anonymous=True)
bridge = CvBridge()
frame_buffer = deque(maxlen=1)
buffer_lock = threading.Lock()

# Инициализация трекера и аннотаторов
fps = 30  # Предполагаемая частота кадров
tracker = sv.ByteTrack(
    track_activation_threshold=0.3,
    minimum_matching_threshold=0.8,
    lost_track_buffer=30,
    frame_rate=fps,
)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.3, text_padding=3)

# Создание детектора
detector = YOLOv8TFLite(model, conf, iou, metadata)


def image_callback(msg):
    """Callback для обработки изображений из ROS-топика /camera/image_color"""
    try:
        # Конвертация ROS Image message в OpenCV BGR формат
        # Топик /camera/image_color содержит RGB изображение, конвертируем в BGR для OpenCV
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        with buffer_lock:
            frame_buffer.append(cv_image)
    except CvBridgeError as e:
        rospy.logerr(f"Ошибка конвертации изображения: {e}")


# Подписка на ROS-топик /camera/image_color
rospy.Subscriber(
    "/camera/image_color", Image, image_callback, queue_size=1, buff_size=2**24
)

# Запуск ROS в отдельном потоке
spin_thread = threading.Thread(target=lambda: rospy.spin())
spin_thread.daemon = True
spin_thread.start()

try:
    while not rospy.is_shutdown():
        t0 = time.time()

        # Получение последнего кадра из буфера
        with buffer_lock:
            if not frame_buffer:
                time.sleep(0.01)  # Избегаем высокой загрузки CPU при отсутствии кадров
                continue
            frame_bgr = frame_buffer[-1].copy()

        # Обработка кадра детектором
        result = detector.detect(frame_bgr, tracker, box_annotator, label_annotator)

        # Отображение результата
        cv2.imshow("Object Detection", result)

        # Расчет и вывод FPS
        t1 = time.time()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
        print(f"FPS: {fps:.1f}")

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    rospy.signal_shutdown("Завершение работы")
