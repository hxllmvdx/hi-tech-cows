from sklearn.cluster import DBSCAN
import numpy as np
import alphashape as sh
from time import localtime, strftime
import csv


def find_clusters(data, min_samples=5, n_cows=3):
    points = np.array([[float(b["x"]), float(b["y"])] for b in data])

    av_width = np.mean([float(b["width"]) for b in data])
    av_height = np.mean([float(b["height"]) for b in data])
    av_cow_size = max(av_width, av_height)

    eps = av_cow_size * n_cows

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:
            clusters[int(label)] = clusters.get(int(label), []) + [
                (float(data[i]["x"]), float(data[i]["y"]))
            ]

    return list(clusters.values()), av_cow_size, av_width * av_height, labels


def get_cluster_parameters(cluster, cow_size):
    alpha = sh.optimizealpha(cluster)
    alpha_shape = sh.alphashape(cluster, alpha)

    # Получаем координаты контура альфа-формы
    if alpha_shape.geom_type == "Polygon":
        x, y = alpha_shape.exterior.xy
        contour_points = np.array(
            [(int(xi), int(yi)) for xi, yi in zip(x, y)], dtype=np.int32
        )
    else:
        # Для других типов геометрии (LineString и т.д.)
        x, y = alpha_shape.xy
        contour_points = np.array(
            [(int(xi), int(yi)) for xi, yi in zip(x, y)], dtype=np.int32
        )

    area = alpha_shape.area / cow_size * 1.5

    return contour_points, area, len(cluster) / area


def main(data: list[dict], csv_file=True):
    clusters, av_cow_size, av_area, labels = find_clusters(data)
    densities = []
    total = len(data)

    for cluster in clusters:
        print(cluster)
        res = get_cluster_parameters(cluster, av_area)
        densities.append(res[2])

    density = np.mean(densities)
    now = strftime("%H:%M:%S %d-%m-%Y", localtime())

    if csv_file:
        with open("result.csv", "r") as f:
            if not f.read():
                with open("result.csv", "w") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Time", "Number of cows", "Density in cow/m^2"])

        with open("result.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([now, total, f"{density:0.2f}"])

    return clusters, labels

