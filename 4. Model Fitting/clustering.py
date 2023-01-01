# Functions:
#   1. Load the point cloud data
#   2. distinguish and them remove background
#   3. cluster the remaining points

import numpy as np
import open3d as o3d
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ransac import RANSAC


def read_velodyne_bin(path):
    """
    Load point cloud data of a ‘.bin’ file
    Args:
        path: file path
    Returns:
        homograph matrix of the point cloud, N*3
    """
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def visualize_point_cloud(arr, ground=False):
    """
    Args:
        arr: numpy array of point cloud data
        ground: whether arr is a ground cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    if ground:
        pcd.paint_uniform_color(np.asarray([65, 105, 225]) / 255)
    o3d.visualization.draw_geometries([pcd])


def ground_segmentation(data):
    """
    Distinguish and remove the background points.
    Args:
        data: the complete point cloud of a frame.
    Returns:
        ground_cloud: the ground points.
        foreground_cloud: the foreground points.
    """
    ransac_model = RANSAC(max_iter=50, dist_threshold=0.12)
    ground_cloud, foreground_cloud = ransac_model.detect_ground(data)

    print('origin data points num:', data.shape[0])
    print('ground segmented data points num:', ground_cloud.shape[0])
    print('foreground segmented data points num:', foreground_cloud.shape[0])
    return ground_cloud, foreground_cloud


def clustering(data):
    """
    Cluster a point cloud.
    Args:
        data: a point cloud.
    Returns:
        clusters_index: the clustering results.
    """
    # spectral = cluster.SpectralClustering(n_clusters=9, eigen_solver='arpack', affinity="nearest_neighbors")
    # clusters_index = spectral.fit_predict(data)

    dbscan = cluster.DBSCAN(eps=1.0, min_samples=10)
    clusters_index = dbscan.fit_predict(data)

    # kmeans = cluster.MiniBatchKMeans(n_clusters=9)
    # clusters_index = kmeans.fit_predict(data)

    return clusters_index


def plot_clusters(data, cluster_index):
    """
    Visualize the clustering of a foreground cloud.
    Args:
        data: a foreground cloud.
        cluster_index: the cluster index of each point.
    """
    ax = plt.figure().add_subplot(111, projection='3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()


def main():
    np.random.seed(42)

    root_dir = 'data_samples/'
    cat = os.listdir(root_dir)
    for i in range(len(cat)):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        # visualize the origin point cloud
        visualize_point_cloud(origin_points)

        ground_cloud, foreground_cloud = ground_segmentation(origin_points)
        # visualize the ground
        visualize_point_cloud(ground_cloud, ground=True)

        cluster_index = clustering(foreground_cloud)
        # visualize the clustering
        plot_clusters(foreground_cloud, cluster_index)

    # load the point cloud data
    # file_path = "data_samples/000000.bin"
    # origin_points = read_velodyne_bin(file_path)
    # visualize the point cloud
    # visualize_point_cloud(origin_points)

    # ground_cloud, foreground_cloud = ground_segmentation(origin_points)
    # visualize the ground
    # visualize_point_cloud(ground_cloud, ground=True)

    # clusters_index = clustering(foreground_cloud)
    # plot_clusters(foreground_cloud, clusters_index)


if __name__ == '__main__':
    main()
