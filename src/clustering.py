from sklearn.cluster import DBSCAN
import numpy as np
import alphashape as sh
from shapely.geometry import MultiPoint
import warnings

warnings.filterwarnings("ignore")  # Отключаем предупреждения для скорости


def find_clusters(data, min_samples=2, n_cows=3):
    """Оптимизированная кластеризация"""
    if len(data) < 2:
        return [], 0, 0, []

    # Быстрое преобразование данных
    points = np.array([[float(b["x"]), float(b["y"])] for b in data])

    # Быстрый расчет средних размеров
    widths = np.array([float(b["width"]) for b in data])
    heights = np.array([float(b["height"]) for b in data])
    av_cow_size = np.maximum(np.mean(widths), np.mean(heights))

    # Параметры DBSCAN
    eps = av_cow_size * n_cows

    # Быстрая кластеризация
    dbscan = DBSCAN(
        eps=eps, min_samples=min_samples, n_jobs=1
    )  # Один поток для стабильности
    labels = dbscan.fit_predict(points)

    # Быстрое создание кластеров
    clusters = []
    for label in set(labels):
        if label != -1:
            cluster_points = points[labels == label]
            clusters.append([(x, y) for x, y in cluster_points])

    return clusters, av_cow_size, np.mean(widths * heights), labels


def get_cluster_parameters(cluster, cow_size):
    """Оптимизированная обработка кластеров"""
    try:
        if len(cluster) < 3:
            # Для малых кластеров используем выпуклую оболочку
            multipoint = MultiPoint(cluster)
            convex_hull = multipoint.convex_hull

            if convex_hull.geom_type == "Polygon":
                x, y = convex_hull.exterior.xy
                contour_points = np.array(
                    [(int(xi), int(yi)) for xi, yi in zip(x, y)], dtype=np.int32
                )
                area = convex_hull.area
            else:
                return None, 0, 0
        else:
            # Для больших кластеров используем alphashape
            alpha = 0.5  # Фиксированный alpha для скорости
            alpha_shape = sh.alphashape(cluster, alpha)

            if alpha_shape.geom_type == "Polygon":
                x, y = alpha_shape.exterior.xy
                contour_points = np.array(
                    [(int(xi), int(yi)) for xi, yi in zip(x, y)], dtype=np.int32
                )
                area = alpha_shape.area
            else:
                return None, 0, 0

        density = len(cluster) / (area + 1e-6)
        return contour_points, area, density

    except Exception:
        return None, 0, 0
