import random
import math
import numpy as np
import time
import os
import struct
import open3d as o3d
import scipy.spatial

import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet


def read_velodyne_bin(path):
    """
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


def visualize_point_cloud(arr):
    """
    Args:
        arr: numpy array of point cloud data
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    o3d.visualization.draw_geometries([pcd])


def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    # load the point cloud data
    file_path = "000000.bin"
    db = read_velodyne_bin(file_path)

    # visualize the point cloud
    # visualize_point_cloud(db)

    # query = db[0, :]
    np.random.seed(42)
    np.set_printoptions(suppress=True, precision=3)
    # print(f"query = {query}")

    brute_force_search_time_sum = 0
    kdtree_construction_time_sum = 0
    kdtree_knn_search_time_sum = 0
    kdtree_radius_search_time_sum = 0
    octree_construction_time_sum = 0
    octree_knn_search_time_sum = 0
    octree_radius_search_time_sum = 0
    octree_radius_fast_search_time_sum = 0
    iteration_num = 100
    for _ in range(iteration_num):
        query = np.random.rand(3)

        # numpy brute-force
        # print("\nnumpy brute-force ---------")
        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, axis=0) - db, axis=1)
        nn_idx = np.argsort(diff)
        k_neighbors = nn_idx[:k]
        brute_force_time = time.time() - begin_t
        brute_force_search_time_sum += brute_force_time
        # print(f"k nearest neighbors are {k_neighbors}")
        # print(f"search time = {brute_force_time * 1000:.3f} ms")

        # scipy.spatial.KDTree
        # print("\nscipy.spatial.KDTree ---------")
        # begin_t = time.time()
        # kd_tree = scipy.spatial.KDTree(db)
        # scipy_kdtree_construct_time = time.time() - begin_t
        # print(f"construction time = {scipy_kdtree_construct_time * 1000:.3f} ms")

        # begin_t = time.time()
        # k_nearest_dist, k_nearest_neighbors = kd_tree.query(query, k=8)
        # scipy_kdtree_search_time = time.time() - begin_t
        # print(f"k nearest neighbors are {k_nearest_neighbors}")
        # print(f"search time = {scipy_kdtree_search_time * 1000:.3f} ms")

        # KDTree
        # print("\nKDTree ---------")
        begin_t = time.time()
        root = kdtree.kdtree_construction(db, leaf_size)
        kdtree_construct_time = time.time() - begin_t
        kdtree_construction_time_sum += kdtree_construct_time
        # print(f"construction time = {kdtree_construct_time * 1000:.3f} ms")
        # depth, max_depth = [0], [0]
        # kdtree.traverse_kdtree(root, depth, max_depth)
        # print(f"tree max depth: {max_depth[0]}")

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db, result_set, query)
        kdtree_knn_search_time = time.time() - begin_t
        kdtree_knn_search_time_sum += kdtree_knn_search_time
        # print(f"k nearest neighbors are: {result_set.knn_indexes()}")
        # print(f"knn serch time = {kdtree_knn_search_time * 1000:.3f} ms")

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db, result_set, query)
        kdtree_radius_search_time = time.time() - begin_t
        kdtree_radius_search_time_sum += kdtree_radius_search_time
        # print(f"neighbors within {radius} are {result_set.sorted_neighbors_indexes()}")
        # print(f"radius search time = {kdtree_radius_search_time * 1000:.3f} ms")

        # OcTree
        # print("\nOcTree ---------")
        begin_t = time.time()
        root = octree.octree_construction(db, leaf_size, min_extent)
        octree_construction_time = time.time() - begin_t
        octree_construction_time_sum += octree_construction_time
        # print(f"construction time = {octree_construction_time * 1000:.3f} ms")

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        octree.octree_knn_search(root, db, result_set, query)
        octree_knn_search_time = time.time() - begin_t
        octree_knn_search_time_sum += octree_knn_search_time
        # print(f"k nearest neighbors are: {result_set.knn_indexes()}")
        # print(f"knn serch time = {octree_knn_search_time * 1000:.3f} ms")

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search(root, db, result_set, query)
        octree_radius_search_time = time.time() - begin_t
        octree_radius_search_time_sum += octree_radius_search_time
        # print(f"neighbors within {radius} are {result_set.sorted_neighbors_indexes()}")
        # print(f"radius search time = {octree_radius_search_time * 1000:.3f} ms")

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search_fast(root, db, result_set, query)
        octree_radius_fast_search_time = time.time() - begin_t
        octree_radius_fast_search_time_sum += octree_radius_fast_search_time
        # print(f"neighbors within {radius} are {result_set.sorted_neighbors_indexes()}")
        # print(f"radius fast search time = {octree_radius_fast_search_time * 1000:.3f} ms")

    print(f"Numpy brute-force:\n search = {brute_force_search_time_sum * 1000 / iteration_num: .3f}\n")
    print(f"K-d tree:\n build = {kdtree_construction_time_sum * 1000 / iteration_num: .3f}, "
          f"knn search = {kdtree_knn_search_time_sum * 1000 / iteration_num: .3f}, "
          f"radius search = {kdtree_radius_search_time_sum * 1000 / iteration_num: .3f}\n")
    print(f"Octree:\n build = {octree_construction_time_sum * 1000 / iteration_num: .3f}, "
          f"knn search = {octree_knn_search_time_sum * 1000 / iteration_num: .3f},"
          f"radius search = {octree_radius_search_time_sum * 1000 / iteration_num: .3f},"
          f"radius fast search = {octree_radius_fast_search_time_sum * 1000 / iteration_num: .3f}")


if __name__ == '__main__':
    main()
