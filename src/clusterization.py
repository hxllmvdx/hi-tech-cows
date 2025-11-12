from sklearn.cluster import DBSCAN
import numpy as np
import alphashape as sh
from time import localtime, strftime
import csv
from shapely.geometry import MultiPoint


def find_clusters(data, min_samples=2, n_cows=3):
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
    try:
        alpha = sh.optimizealpha(cluster)
        alpha_shape = sh.alphashape(cluster, alpha)

        contour_points = None
        area = 0

        # Обработка MultiPolygon
        if alpha_shape.geom_type == "MultiPolygon":
            if hasattr(alpha_shape, "geoms"):
                first_polygon = list(alpha_shape.geoms)[0]
                x, y = first_polygon.exterior.xy
            else:
                first_polygon = alpha_shape[0]
                x, y = first_polygon.exterior.xy
            contour_points = np.array(
                [(int(xi), int(yi)) for xi, yi in zip(x, y)], dtype=np.int32
            )
            area = first_polygon.area

        # Обработка обычного Polygon
        elif alpha_shape.geom_type == "Polygon":
            if hasattr(alpha_shape.exterior, "xy"):
                x, y = alpha_shape.exterior.xy
            else:
                coords = list(alpha_shape.exterior.coords)
                x = [coord[0] for coord in coords]
                y = [coord[1] for coord in coords]
            contour_points = np.array(
                [(int(xi), int(yi)) for xi, yi in zip(x, y)], dtype=np.int32
            )
            area = alpha_shape.area

        else:
            # Резервный вариант - выпуклая оболочка
            multipoint = MultiPoint(cluster)
            convex_hull = multipoint.convex_hull
            if convex_hull.geom_type == "Polygon":
                x, y = convex_hull.exterior.xy
                contour_points = np.array(
                    [(int(xi), int(yi)) for xi, yi in zip(x, y)], dtype=np.int32
                )
                area = convex_hull.area

        if contour_points is None or len(contour_points) < 3:
            return None, 0, 0

        density = len(cluster) / (area + 1e-6)
        return contour_points, area, density

    except Exception as e:
        print(f"❌ Error in get_cluster_parameters: {e}")
        return None, 0, 0


def main(data: list[dict], csv_file=True):
    clusters, av_cow_size, av_area, labels = find_clusters(data)
    densities = []
    total = len(data)

    for cluster in clusters:
        res = get_cluster_parameters(cluster, av_area)
        if res[0] is not None:
            densities.append(res[2])

    density = np.mean(densities) if densities else 0
    now = strftime("%H:%M:%S %d-%m-%Y", localtime())

    if csv_file:
        with open("result.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([now, total, f"{density:0.2f}"])

    return clusters, labels
