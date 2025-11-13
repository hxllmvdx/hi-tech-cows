from __future__ import annotations

import cv2
import numpy as np
import yaml
import supervision as sv
import time
import os
from pathlib import Path
from clustering import find_clusters, get_cluster_parameters

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


class StableYOLOv8TFLite:
    def __init__(
        self,
        model: str,
        conf: float = 0.4,  # Немного снизим для лучшего покрытия
        iou: float = 0.4,
        metadata: str | None = None,
    ):
        self.conf = conf
        self.iou = iou
        self.classes = {0: "cow"}

        if metadata is not None and os.path.exists(metadata):
            try:
                with open(metadata) as f:
                    loaded_classes = yaml.safe_load(f)["names"]
                    self.classes = {int(k): v for k, v in loaded_classes.items()}
            except Exception as e:
                print(f"Error loading metadata: {e}")

        # Initialize the TFLite interpreter
        try:
            if not os.path.exists(model):
                raise FileNotFoundError(f"Model file not found: {model}")

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

        # Handle quantization
        quantization = input_details.get("quantization", (1.0, 0))
        self.in_scale, self.in_zero_point = quantization
        self.int8 = input_details["dtype"] == np.int8

        # Get output details
        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]

        output_quantization = output_details.get("quantization", (1.0, 0))
        self.out_scale, self.out_zero_point = output_quantization

        print(f"📊 Model quantization: int8={self.int8}")

        # Кэш для стабильности
        self._last_shape = None
        self._cached_params = None

    def letterbox_fast(self, img: np.ndarray) -> tuple:
        """Оптимизированный letterbox с кэшированием"""
        h, w = img.shape[:2]

        # Используем кэш если размер не изменился
        if self._last_shape == (h, w) and self._cached_params is not None:
            return self._cached_params

        # Вычисляем scaling factor
        scale = min(self.in_width / w, self.in_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Вычисляем паддинг
        dw, dh = self.in_width - new_w, self.in_height - new_h
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        # Кэшируем параметры
        self._last_shape = (h, w)
        self._cached_params = (new_w, new_h, left, top, right, bottom, scale)

        return self._cached_params

    def preprocess_fast(self, img: np.ndarray) -> tuple:
        """Оптимизированный препроцессинг"""
        new_w, new_h, left, top, right, bottom, scale = self.letterbox_fast(img)

        # Resize + pad за одну операцию
        if (new_w, new_h) != (img.shape[1], img.shape[0]):
            img_resized = cv2.resize(
                img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            img_resized = img

        # Добавляем паддинг
        img_padded = cv2.copyMakeBorder(
            img_resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # BGR to RGB и нормализация
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # Добавляем batch dimension
        return np.expand_dims(img_normalized, axis=0), (left, top, scale)

    def postprocess_stable(
        self,
        img: np.ndarray,
        outputs: np.ndarray,
        pad_params: tuple,
        tracker,
    ) -> tuple:
        """Стабильный постпроцессинг с улучшенной кластеризацией"""
        try:
            orig_h, orig_w = img.shape[:2]
            left, top, scale = pad_params

            # Обработка вывода YOLO
            outputs = outputs[0].T
            boxes = outputs[:, :4]
            scores = outputs[:, 4]

            # Более либеральная фильтрация для лучшего покрытия
            mask = scores > self.conf
            boxes = boxes[mask]
            scores = scores[mask]

            if len(scores) == 0:
                return img, []

            # Преобразование координат
            x_center = boxes[:, 0] * self.in_width
            y_center = boxes[:, 1] * self.in_height
            width = boxes[:, 2] * self.in_width
            height = boxes[:, 3] * self.in_height

            # Конвертация в xyxy и обратное масштабирование
            x1 = (x_center - width / 2 - left) / scale
            y1 = (y_center - height / 2 - top) / scale
            x2 = (x_center + width / 2 - left) / scale
            y2 = (y_center + height / 2 - top) / scale

            # Clipping
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)

            boxes_xyxy = np.column_stack([x1, y1, x2, y2])

            # Менее агрессивная фильтрация по размеру
            widths = x2 - x1
            heights = y2 - y1
            areas = widths * heights

            # Более либеральные ограничения по размеру
            size_mask = (
                (widths > 25)
                & (heights > 25)
                & (areas > 400)
                & (widths < 600)
                & (heights < 600)
            )
            boxes_xyxy = boxes_xyxy[size_mask]
            scores = scores[size_mask]

            if len(boxes_xyxy) == 0:
                return img, []

            # NMS
            nms_result = cv2.dnn.NMSBoxes(
                boxes_xyxy.tolist(), scores.tolist(), self.conf, self.iou
            )

            if len(nms_result) == 0:
                return img, []

            indices = nms_result.flatten()

            # Создаем детекции
            class_ids = np.zeros(len(indices), dtype=int)
            detections = sv.Detections(
                xyxy=boxes_xyxy[indices],
                confidence=scores[indices],
                class_id=class_ids,
            )

            # Трекинг
            detections = tracker.update_with_detections(detections)

            # Подготовка данных для кластеризации - ВКЛЮЧАЕМ БОЛЬШЕ ДЕТЕКЦИЙ
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
                # СНИЖЕН порог для кластеризации - включаем больше детекций
                if confidence < 0.25:  # Было 0.4
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

            print(
                f"📊 For clustering: {len(data_for_clustering)} detections (conf > 0.25)"
            )
            return img, data_for_clustering

        except Exception as e:
            print(f"❌ Postprocessing error: {e}")
            return img, []

    def detect_stable(self, img, tracker) -> tuple:
        """Стабильная детекция"""
        x, pad_params = self.preprocess_fast(img)

        # Конвертация для квантованной модели
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # Инференс
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()
        y = self.model.get_tensor(self.out_index)

        # Конвертация выхода для квантованной модели
        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

        return self.postprocess_stable(img, y, pad_params, tracker)


def visualize_stable(frame, detections_data):
    """Улучшенная визуализация с лучшей кластеризацией"""
    if not detections_data:
        return frame

    frame_vis = frame.copy()
    h, w = frame_vis.shape[:2]

    # Кластеризация - используем более агрессивные параметры
    if len(detections_data) >= 2:
        try:
            # УВЕЛИЧИВАЕМ параметры для лучшего поиска кластеров
            clusters, av_cow_size, av_area, cluster_labels = find_clusters(
                detections_data,
                min_samples=2,  # Минимум 2 коровы для кластера
                n_cows=4,  # УВЕЛИЧЕНО: расстояние 4 размера коровы для лучшего объединения
            )

            print(
                f"🔍 Found {len(clusters)} clusters from {len(detections_data)} detections"
            )

            # Рисуем кластеры
            for cluster_idx, cluster in enumerate(clusters):
                contour_points, area, density = get_cluster_parameters(
                    cluster, av_cow_size
                )
                if contour_points is not None and len(contour_points) > 2:
                    # Яркие и заметные кластеры
                    overlay = frame_vis.copy()
                    # Заливка кластера
                    cv2.fillPoly(
                        overlay, [contour_points], (0, 255, 0), lineType=cv2.LINE_AA
                    )
                    # Толстая граница
                    cv2.polylines(
                        overlay,
                        [contour_points],
                        True,
                        (0, 200, 0),
                        4,
                        lineType=cv2.LINE_AA,
                    )
                    # Меньше прозрачности - кластеры более заметны
                    cv2.addWeighted(overlay, 0.4, frame_vis, 0.6, 0, frame_vis)

                    # Яркая информация о кластере
                    centroid = np.mean(contour_points, axis=0)
                    cluster_text = f"Cluster {cluster_idx + 1}: {len(cluster)} cows"
                    cv2.putText(
                        frame_vis,
                        cluster_text,
                        (int(centroid[0]), int(centroid[1]) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        3,
                    )

            # Одиночные коровы
            single_indices = [
                i for i, label in enumerate(cluster_labels) if label == -1
            ]
            print(f"🐄 Single cows: {len(single_indices)}")

            for idx in single_indices:
                if idx < len(detections_data):
                    cow_data = detections_data[idx]
                    x, y = float(cow_data["x"]), float(cow_data["y"])
                    width, height = float(cow_data["width"]), float(cow_data["height"])

                    x1, y1 = int(x - width / 2), int(y - height / 2)
                    x2, y2 = int(x + width / 2), int(y + height / 2)

                    # Толстые синие bbox
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(
                        frame_vis,
                        "COW",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
        except Exception as e:
            print(f"❌ Clustering error: {e}")

    # Одиночные коровы если нет кластеризации
    else:
        print(f"🐄 Only single cows: {len(detections_data)}")
        for cow_data in detections_data:
            x, y = float(cow_data["x"]), float(cow_data["y"])
            width, height = float(cow_data["width"]), float(cow_data["height"])

            x1, y1 = int(x - width / 2), int(y - height / 2)
            x2, y2 = int(x + width / 2), int(y + height / 2)

            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(
                frame_vis,
                "COW",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

    return frame_vis


def main():
    script_dir = Path(__file__).parent.absolute()

    # Define paths
    model_path = script_dir / "yolo_demo.tflite"
    video_path = script_dir.parent / "resources" / "test_input.mp4"
    metadata_path = script_dir / "metadata.yaml"

    # Create metadata if it doesn't exist
    if not metadata_path.exists():
        with open(metadata_path, "w") as f:
            f.write("names:\n  0: cow\n")

    # Проверка файлов
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎥 Video: {frame_width}x{frame_height} at {original_fps:.2f} FPS")

    # Правильная задержка для оригинальной скорости видео
    frame_delay = int(1000 / original_fps) if original_fps > 0 else 33

    # Стабильный трекер
    tracker = sv.ByteTrack(
        track_activation_threshold=0.35,
        minimum_matching_threshold=0.6,
        lost_track_buffer=25,
        frame_rate=original_fps,
    )

    # Инициализация детектора
    try:
        detector = StableYOLOv8TFLite(
            str(model_path), conf=0.4, iou=0.4, metadata=str(metadata_path)
        )
        print("✅ Stable detector initialized successfully")
        print("🎯 Improved clustering with lower confidence threshold (0.25)")
        print("🔍 Aggressive cluster search (n_cows=4)")
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        cap.release()
        return

    # Создание окна
    window_name = "Cow Detection - Improved Clustering"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    # Переменные для производительности
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    fps_history = []

    # Ручное управление скоростью
    skip_frames = 0

    print("🎬 Starting detection with IMPROVED CLUSTERING...")
    print("⚡ Controls: '+'=skip more, '-'=skip less, SPACE=pause, 'q'=quit")
    print("🎨 Green: Big clusters | Blue: Single cows")

    paused = False

    try:
        while True:
            current_time = time.time()

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("✅ End of video")
                    break

                frame_count += 1

                # Ручной пропуск кадров
                if frame_count % (skip_frames + 1) == 0:
                    processed_count += 1

                    # Детекция
                    result, detections_data = detector.detect_stable(frame, tracker)

                    # Визуализация
                    if detections_data:
                        result = visualize_stable(result, detections_data)
                    display_frame = result
                else:
                    # Пропускаем обработку, но показываем оригинальный кадр
                    display_frame = frame
                    detections_data = []

                # Расчет FPS
                elapsed_time = current_time - start_time
                current_fps = processed_count / elapsed_time if elapsed_time > 0 else 0
                fps_history.append(current_fps)
                if len(fps_history) > 10:
                    fps_history.pop(0)
                avg_fps = np.mean(fps_history)

                # Информация на кадре
                info_text = [
                    f"Video FPS: {original_fps:.1f}",
                    f"Processing FPS: {avg_fps:.1f}",
                    f"Frame: {frame_count}/{total_frames}",
                    f"Skip: {skip_frames}",
                    f"Cows: {len(detections_data)}",
                    "ClustConf: 0.25",
                ]

                for i, text in enumerate(info_text):
                    cv2.putText(
                        display_frame,
                        text,
                        (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # Легенда
                cv2.putText(
                    display_frame,
                    "GREEN: Big clusters | BLUE: Single cows | Improved clustering",
                    (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                # Показ результата с правильной скоростью
                cv2.imshow(window_name, display_frame)

            # Правильная синхронизация с оригинальной скоростью видео
            if not paused:
                key = cv2.waitKey(frame_delay) & 0xFF
            else:
                key = cv2.waitKey(0) & 0xFF

            # Обработка клавиш
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord("+"):
                skip_frames = min(10, skip_frames + 1)
                print(f"🔧 Frame skip increased to: {skip_frames}")
            elif key == ord("-"):
                skip_frames = max(0, skip_frames - 1)
                print(f"🔧 Frame skip decreased to: {skip_frames}")
            elif key == ord("0"):
                skip_frames = 0
                print("🔧 No frame skipping")

    except KeyboardInterrupt:
        print("⏹️ Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        print("\n📊 Processing complete:")
        print(f"   Total frames: {frame_count}")
        print(f"   Processed frames: {processed_count}")
        print(f"   Processing time: {total_time:.2f}s")
        print(f"   Average FPS: {processed_count / total_time:.2f}")


if __name__ == "__main__":
    main()
