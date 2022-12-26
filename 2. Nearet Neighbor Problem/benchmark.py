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

    query = db[0, :]

    # numpy brute-force
    print("numpy brute-force ---------")
    begin_t = time.time()
    diff = np.linalg.norm(np.expand_dims(query, axis=0) - db, axis=1)
    nn_idx = np.argsort(diff)
    k_neighbors = nn_idx[:k]
    brute_force_time = time.time() - begin_t
    print(f"k nearest neighbors are {k_neighbors}")
    print(f"search time = {brute_force_time * 1000:.3f} ms")

    # scipy.spatial.KDTree
    print("\nscipy.spatial.KDTree ---------")
    begin_t = time.time()
    kd_tree = scipy.spatial.KDTree(db)
    scipy_kdtree_construct_time = time.time() - begin_t
    print(f"construction time = {scipy_kdtree_construct_time * 1000:.3f} ms")

    begin_t = time.time()
    k_nearest_dist, k_nearest_neighbors = kd_tree.query(query, k=8)
    scipy_kdtree_search_time = time.time() - begin_t
    print(f"k nearest neighbors are {k_nearest_neighbors}")
    print(f"search time = {scipy_kdtree_search_time * 1000:.3f} ms")

    # KDTree
    print("\nKDTree ---------")
    begin_t = time.time()
    root = kdtree.kdtree_construction(db, leaf_size)
    kdtree_construct_time = time.time() - begin_t
    print(f"construction time = {kdtree_construct_time * 1000:.3f} ms")
    # depth, max_depth = [0], [0]
    # kdtree.traverse_kdtree(root, depth, max_depth)
    # print(f"tree max depth: {max_depth[0]}")

    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    kdtree.kdtree_knn_search(root, db, result_set, query)
    knn_search_time = time.time() - begin_t
    print(f"k nearest neighbors are: {result_set.knn_indexes()}")
    print(f"knn serch time = {knn_search_time * 1000:.3f} ms")

    begin_t = time.time()
    result_set = RadiusNNResultSet(radius=radius)
    kdtree.kdtree_radius_search(root, db, result_set, query)
    radius_search_time = time.time() - begin_t
    print(f"neighbors within {radius} are {result_set.sorted_neighbors_indexes()}")
    print(f"radius search time = {radius_search_time * 1000:.3f} ms")

    # OcTree
    print("\nOcTree ---------")
    begin_t = time.time()
    root = octree.octree_construction(db, leaf_size, min_extent)
    octree_construction_time = time.time() - begin_t
    print(f"construction time = {octree_construction_time * 1000:.3f} ms")

    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    octree.octree_knn_search(root, db, result_set, query)
    knn_search_time = time.time() - begin_t
    print(f"k nearest neighbors are: {result_set.knn_indexes()}")
    print(f"knn serch time = {knn_search_time * 1000:.3f} ms")

    begin_t = time.time()
    result_set = RadiusNNResultSet(radius=radius)
    octree.octree_radius_search(root, db, result_set, query)
    radius_search_time = time.time() - begin_t
    print(f"neighbors within {radius} are {result_set.sorted_neighbors_indexes()}")
    print(f"radius search time = {radius_search_time * 1000:.3f} ms")

    begin_t = time.time()
    result_set = RadiusNNResultSet(radius=radius)
    octree.octree_radius_search_fast(root, db, result_set, query)
    radius_fast_search_time = time.time() - begin_t
    print(f"neighbors within {radius} are {result_set.sorted_neighbors_indexes()}")
    print(f"radius fast search time = {radius_fast_search_time * 1000:.3f} ms")

    # print("octree --------------")
    # construction_time_sum = 0
    # knn_time_sum = 0
    # radius_time_sum = 0
    # brute_time_sum = 0
    # for i in range(iteration_num):
    #     filename = os.path.join(root_dir, cat[i])
    #     db_np = read_velodyne_bin(filename)

    #     begin_t = time.time()
    #     root = octree.octree_construction(db_np, leaf_size, min_extent)
    #     construction_time_sum += time.time() - begin_t

    #     query = db_np[0,:]

    #     begin_t = time.time()
    #     result_set = KNNResultSet(capacity=k)
    #     octree.octree_knn_search(root, db_np, result_set, query)
    #     knn_time_sum += time.time() - begin_t

    #     begin_t = time.time()
    #     result_set = RadiusNNResultSet(radius=radius)
    #     octree.octree_radius_search_fast(root, db_np, result_set, query)
    #     radius_time_sum += time.time() - begin_t

    #     begin_t = time.time()
    #     diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    #     nn_idx = np.argsort(diff)
    #     nn_dist = diff[nn_idx]
    #     brute_time_sum += time.time() - begin_t
    # print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
    #                                                                  knn_time_sum*1000/iteration_num,
    #                                                                  radius_time_sum*1000/iteration_num,
    #                                                                  brute_time_sum*1000/iteration_num))


if __name__ == '__main__':
    main()
