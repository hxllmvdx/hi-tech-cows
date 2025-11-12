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
        conf: float = 0.4,  # Повышенный порог уверенности
        iou: float = 0.45,
        metadata: str | None = None,
    ):
        self.conf = conf
        self.iou = iou

        # For custom cow detection model
        self.classes = {0: "cow"}

        if metadata is not None:
            try:
                with open(metadata) as f:
                    loaded_classes = yaml.safe_load(f)["names"]
                    self.classes = {int(k): v for k, v in loaded_classes.items()}
                    print(f"Loaded classes from metadata: {self.classes}")
            except Exception as e:
                print(f"Error loading metadata: {e}, using default cow class")

        print(f"Model classes: {self.classes}")

        # Initialize the TFLite interpreter
        try:
            # Try EdgeTPU first, then CPU
            try:
                edgetpu_delegate = load_delegate("libedgetpu.so.1", {"device": "usb"})
                self.model = Interpreter(
                    model_path=model, experimental_delegates=[edgetpu_delegate]
                )
                print("✅ Using EdgeTPU")
            except:
                self.model = Interpreter(model_path=model)
                print("✅ Using CPU")

            self.model.allocate_tensors()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        # Get input details
        input_details = self.model.get_input_details()[0]
        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]

        # Handle quantization safely
        quantization = input_details.get("quantization", (1.0, 0))
        self.in_scale, self.in_zero_point = quantization
        self.int8 = input_details["dtype"] == np.int8

        # Get output details
        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]

        output_quantization = output_details.get("quantization", (1.0, 0))
        self.out_scale, self.out_zero_point = output_quantization

        print(f"📐 Model input: {self.in_width}x{self.in_height}")

    def letterbox(
        self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)
    ) -> tuple[
        np.ndarray, tuple[float, float], tuple[float, float], tuple[float, float]
    ]:
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        # Divide padding by 2 for each side
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, (left, top), (r, r), (dw, dh)

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple, tuple, tuple]:
        img, pad, ratio, pad_detail = self.letterbox(
            img, (self.in_width, self.in_height)
        )
        img = img[..., ::-1][None]  # BGR to RGB and add batch dimension
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        return img / 255, pad, ratio, pad_detail

    def postprocess(
        self,
        img: np.ndarray,
        outputs: np.ndarray,
        pad: tuple[float, float],
        ratio: tuple[float, float],
        pad_detail: tuple[float, float],
        tracker,
    ) -> tuple[np.ndarray, list]:
        try:
            orig_h, orig_w = img.shape[:2]
            outputs = outputs[0].T

            # Extract parameters
            boxes = outputs[
                :, :4
            ]  # [x_center, y_center, width, height] normalized to 0-1
            scores = outputs[:, 4]  # Confidence scores

            # Filter by confidence
            keep = scores > self.conf
            boxes = boxes[keep]
            scores = scores[keep]

            if len(scores) == 0:
                return img, []

            # Convert from normalized [x_center, y_center, width, height] to pixel coordinates
            x_center = boxes[:, 0] * self.in_width
            y_center = boxes[:, 1] * self.in_height
            width = boxes[:, 2] * self.in_width
            height = boxes[:, 3] * self.in_height

            # Convert to corner coordinates [x1, y1, x2, y2] on padded image
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Remove padding and scale back to original image coordinates
            pad_left, pad_top = pad
            ratio_w, ratio_h = ratio

            # Adjust coordinates to original image scale
            x1 = (x1 - pad_left) / ratio_w
            y1 = (y1 - pad_top) / ratio_h
            x2 = (x2 - pad_left) / ratio_w
            y2 = (y2 - pad_top) / ratio_h

            # Clip to image boundaries
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)

            boxes_xyxy = np.column_stack([x1, y1, x2, y2])

            # Apply NMS
            if len(boxes_xyxy) > 0:
                nms_result = cv2.dnn.NMSBoxes(
                    boxes_xyxy.tolist(), scores.tolist(), self.conf, self.iou
                )

                if len(nms_result) == 0:
                    return img, []

                indices = nms_result.flatten()

                # Create detections
                class_ids = np.zeros(len(indices), dtype=int)
                detections = sv.Detections(
                    xyxy=boxes_xyxy[indices],
                    confidence=scores[indices],
                    class_id=class_ids,
                )

                # Update with tracker
                detections = tracker.update_with_detections(detections)

                # Prepare data for clustering
                data_for_clustering = []
                tracker_ids = getattr(
                    detections, "tracker_id", [None] * len(detections.xyxy)
                )

                for i, (xyxy, confidence, class_id, tracker_id) in enumerate(
                    zip(
                        detections.xyxy,
                        detections.confidence,
                        detections.class_id,
                        tracker_ids,
                    )
                ):
                    if confidence < 0.4:  # Higher threshold for clustering
                        continue

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

                return img, data_for_clustering

            return img, []

        except Exception as e:
            print(f"❌ Postprocessing error: {e}")
            import traceback

            traceback.print_exc()
            return img, []

    def detect(self, img, tracker) -> tuple[np.ndarray, list]:
        x, pad, ratio, pad_detail = self.preprocess(img)

        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        self.model.set_tensor(self.in_index, x)
        self.model.invoke()

        y = self.model.get_tensor(self.out_index)

        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

        return self.postprocess(img, y, pad, ratio, pad_detail, tracker)


def visualize_results(frame, detections_data):
    """Визуализация результатов с кластерами и одиночными коровами"""
    if not detections_data:
        return frame

    frame_vis = frame.copy()

    # Кластеризация
    if len(detections_data) >= 2:
        try:
            clusters, av_cow_size, av_area, cluster_labels = find_clusters(
                detections_data,
                min_samples=2,
                n_cows=3,  # Расстояние 3 линейных размера коровы
            )

            # Рисуем кластеры
            for cluster_idx, cluster in enumerate(clusters):
                contour_points, area, density = get_cluster_parameters(
                    cluster, av_cow_size
                )
                if contour_points is not None and len(contour_points) > 2:
                    # Альфа-форма кластера
                    overlay = frame_vis.copy()
                    cv2.fillPoly(
                        overlay, [contour_points], (0, 255, 0), lineType=cv2.LINE_AA
                    )
                    cv2.polylines(
                        overlay,
                        [contour_points],
                        True,
                        (0, 200, 0),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.addWeighted(overlay, 0.3, frame_vis, 0.7, 0, frame_vis)

                    # Информация о кластере
                    centroid = np.mean(contour_points, axis=0)
                    cv2.putText(
                        frame_vis,
                        f"Cluster {cluster_idx + 1}: {len(cluster)} cows",
                        (int(centroid[0]), int(centroid[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 0),
                        2,
                    )

            # Одиночные коровы (синие bbox)
            if len(cluster_labels) > 0:
                single_indices = [
                    i for i, label in enumerate(cluster_labels) if label == -1
                ]

                for idx in single_indices:
                    if idx < len(detections_data):
                        cow_data = detections_data[idx]
                        x = float(cow_data["x"])
                        y = float(cow_data["y"])
                        width = float(cow_data["width"])
                        height = float(cow_data["height"])

                        # Рисуем синий bbox
                        x1 = int(x - width / 2)
                        y1 = int(y - height / 2)
                        x2 = int(x + width / 2)
                        y2 = int(y + height / 2)

                        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            frame_vis,
                            "COW",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                        )
        except Exception as e:
            print(f"❌ Clustering error: {e}")
    else:
        # Если нет кластеризации, показываем все детекции как одиночные коровы
        for cow_data in detections_data:
            x = float(cow_data["x"])
            y = float(cow_data["y"])
            width = float(cow_data["width"])
            height = float(cow_data["height"])

            # Рисуем синий bbox
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame_vis,
                "COW",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    return frame_vis


# Инициализация ROS
rospy.init_node("cow_detection_node", anonymous=True)
bridge = CvBridge()

# Буфер для кадров и блокировка для потокобезопасности
frame_buffer = deque(maxlen=1)
buffer_lock = threading.Lock()

# Параметры детектора
model_path = "../Yolo/yolo_demo.tflite"
metadata_path = "../Yolo/metadata.yaml"
conf = 0.4  # Повышенный порог уверенности
iou = 0.4

# Инициализация детектора
try:
    detector = YOLOv8TFLite(model_path, conf, iou, metadata_path)
    print("✅ Detector initialized successfully")
except Exception as e:
    print(f"❌ Error initializing detector: {e}")
    exit(1)

# Инициализация трекера
tracker = sv.ByteTrack(
    track_activation_threshold=0.4,
    minimum_matching_threshold=0.6,
    lost_track_buffer=15,
    frame_rate=30,  # Примерная частота кадров
)

# Публикатор для результатов детекции
result_pub = rospy.Publisher("/cow_detection_result", Image, queue_size=1)

# Счетчики производительности
frame_count = 0
start_time = time.time()
fps_history = []


def image_callback(msg):
    """Callback для получения изображений из ROS топика"""
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        with buffer_lock:
            frame_buffer.append(cv_image)
    except CvBridgeError as e:
        rospy.logerr(f"Ошибка конвертации изображения: {e}")


# Подписка на топик с изображениями
rospy.Subscriber(
    "/main_camera/color", Image, image_callback, queue_size=1, buff_size=2**24
)

# Запуск ROS в отдельном потоке
spin_thread = threading.Thread(target=lambda: rospy.spin())
spin_thread.daemon = True
spin_thread.start()

print("🚀 Cow Detection Node started")
print("📡 Subscribed to: /main_camera/color")
print("📤 Publishing to: /cow_detection_result")
print("🎨 Green: Cow clusters | Blue: Single cows")

try:
    while not rospy.is_shutdown():
        # Получаем кадр из буфера
        with buffer_lock:
            if not frame_buffer:
                time.sleep(0.01)
                continue
            frame = frame_buffer[-1].copy()

        frame_count += 1

        # Детекция
        try:
            result, detections_data = detector.detect(frame, tracker)

            # Визуализация результатов
            if detections_data:
                result = visualize_results(result, detections_data)

            # Добавляем информацию о FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_history.append(current_fps)
            if len(fps_history) > 10:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)

            cv2.putText(
                result,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                result,
                f"Cows: {len(detections_data)}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Легенда
            h, w = result.shape[:2]
            cv2.putText(
                result,
                "Green: Clusters | Blue: Single cows",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # Публикуем результат в ROS топик
            try:
                result_msg = bridge.cv2_to_imgmsg(result, "bgr8")
                result_msg.header.stamp = rospy.Time.now()
                result_pub.publish(result_msg)
            except CvBridgeError as e:
                rospy.logerr(f"Ошибка публикации результата: {e}")

            # Показываем результат в окне (опционально)
            cv2.imshow("Cow Detection", result)

        except Exception as e:
            print(f"❌ Detection error: {e}")
            # В случае ошибки публикуем оригинальный кадр
            try:
                result_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
                result_msg.header.stamp = rospy.Time.now()
                result_pub.publish(result_msg)
            except CvBridgeError as bridge_error:
                rospy.logerr(f"Ошибка публикации оригинального кадра: {bridge_error}")

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("⏹️ Interrupted by user")
finally:
    cv2.destroyAllWindows()
    rospy.signal_shutdown("Завершение работы")
    total_time = time.time() - start_time
    print(f"\n📊 Processed {frame_count} frames in {total_time:.2f}s")
    print(f"📈 Average FPS: {frame_count / total_time:.2f}")
