# inference.py
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
from clusterization import find_clusters, get_cluster_parameters

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


class YOLOv8TFLite:
    def __init__(
        self,
        model: str,
        conf: float = 0.25,
        iou: float = 0.45,
        metadata: str | None = None,
    ):
        self.conf = conf
        self.iou = iou
        if metadata is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(metadata) as f:
                self.classes = yaml.safe_load(f)["names"]
        np.random.seed(42)
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
        shape = img.shape[:2]

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            (new_shape[1] - new_unpad[0]) / 2,
            (new_shape[0] - new_unpad[1]) / 2,
        )

        if shape[::-1] != new_unpad:
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
        x1, y1, w, h = box
        color = self.color_palette[class_id]

        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        label = f"{self.classes[class_id]}: {score:.2f}"

        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

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
        t0 = time.time()

        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[..., ::-1][None]
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        t1 = time.time()
        print(f"{t1 - t0} preprocess\n")
        return img / 255, pad

    def postprocess(
        self,
        img: np.ndarray,
        outputs: np.ndarray,
        pad: tuple[float, float],
        tracker,
        box_annotator,
        label_annotator,
    ) -> np.ndarray:
        t0 = time.time()

        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(img.shape)

        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0] -= outputs[..., 2] / 2
        outputs[..., 1] -= outputs[..., 3] / 2

        detections_list = []
        for out in outputs:
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)

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

        # Подготовка данных для кластеризации
        data_for_clustering = []
        for i, (xyxy, confidence, class_id, tracker_id) in enumerate(
            zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
                detections.tracker_id,
            )
        ):
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]

            data_for_clustering.append(
                {
                    "id": str(tracker_id) if tracker_id is not None else str(i),
                    "x": str(x_center),
                    "y": str(y_center),
                    "width": str(width),
                    "height": str(height),
                }
            )

        # Кластеризация
        clusters, av_cow_size, av_area, cluster_labels = find_clusters(
            data_for_clustering
        )

        # Отрисовка альфа-форм для кластеров
        for cluster in clusters:
            contour_points, area, density = get_cluster_parameters(cluster, av_cow_size)
            if contour_points is not None and len(contour_points) > 2:
                # Рисуем альфа-форму зеленым цветом
                cv2.fillPoly(img, [contour_points], (0, 255, 0))
                cv2.polylines(img, [contour_points], True, (0, 200, 0), 2)

        # Создаем маску для объектов не входящих в кластеры
        non_cluster_mask = np.array(cluster_labels) == -1

        # Фильтруем детекции для объектов вне кластеров
        non_cluster_detections = sv.Detections(
            xyxy=detections.xyxy[non_cluster_mask],
            confidence=detections.confidence[non_cluster_mask],
            class_id=detections.class_id[non_cluster_mask],
            tracker_id=detections.tracker_id[non_cluster_mask]
            if detections.tracker_id is not None
            else None,
        )

        # Отрисовываем только объекты вне кластеров синим цветом
        labels = [
            f"COW {confidence:0.2f}" for confidence in non_cluster_detections.confidence
        ]

        # Создаем аннотаторы с синим цветом для некластеризованных объектов
        blue_box_annotator = sv.BoxAnnotator(color=sv.Color(255, 0, 0), thickness=2)
        blue_label_annotator = sv.LabelAnnotator(
            text_color=sv.Color(255, 255, 255),
            color=sv.Color(255, 0, 0),
            text_thickness=1,
            text_scale=0.3,
            text_padding=3,
        )

        img = blue_box_annotator.annotate(img, non_cluster_detections)
        img = blue_label_annotator.annotate(img, non_cluster_detections, labels)

        t1 = time.time()
        print(f"{t1 - t0} postprocess\n")
        return img

    def detect(self, img, tracker, box_annotator, label_annotator) -> np.ndarray:
        x, pad = self.preprocess(img)

        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        t0 = time.time()
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()

        y = self.model.get_tensor(self.out_index)
        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale
        t1 = time.time()
        print(f"{t1 - t0} inference\n")
        return self.postprocess(img, y, pad, tracker, box_annotator, label_annotator)


# Параметры детектора
model = "Yolo/yolo_demo_edgetpu.tflite"
conf = 0.85
iou = 0.35
metadata = "Yolo/metadata.yaml"

# Инициализация ROS-ноды
rospy.init_node("yolo_object_detector", anonymous=True)
bridge = CvBridge()
frame_buffer = deque(maxlen=1)
buffer_lock = threading.Lock()

# Инициализация трекера и аннотаторов
fps = 30
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
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        with buffer_lock:
            frame_buffer.append(cv_image)
    except CvBridgeError as e:
        rospy.logerr(f"Ошибка конвертации изображения: {e}")


rospy.Subscriber(
    "/camera/image_color", Image, image_callback, queue_size=1, buff_size=2**24
)

spin_thread = threading.Thread(target=lambda: rospy.spin())
spin_thread.daemon = True
spin_thread.start()

try:
    while not rospy.is_shutdown():
        t0 = time.time()

        with buffer_lock:
            if not frame_buffer:
                time.sleep(0.01)
                continue
            frame_bgr = frame_buffer[-1].copy()

        result = detector.detect(frame_bgr, tracker, box_annotator, label_annotator)

        cv2.imshow("Object Detection", result)

        t1 = time.time()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
        print(f"FPS: {fps:.1f}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    rospy.signal_shutdown("Завершение работы")
