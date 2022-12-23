# Implement Voxel Grid Downsampling

import open3d as o3d 
import os
import argparse
import numpy as np
from pyntcloud import PyntCloud


def voxel_filter(data, r, centroid=True):
    """
    Apply voxel grid downsampling to point cloud
    Args:
        data: point cloud, matrix of Nx3.
        r:    the voxel grid size.
        centroid: compute the mean of points within a voxel if True, otherwise random pick a point if False.
    Returns:
        filtered_points: point cloud after downsampling
    """

    filtered_points = []

    # 1. compute the min or max of the point set
    x_min, y_min, z_min = np.min(data, axis=0)
    x_max, y_max, z_max = np.max(data, axis=0)

    # 2. compute the dimension of the voxel grid
    Dx = np.int_((x_max - x_min) // r) + 1
    Dy = np.int_((y_max - y_min) // r) + 1
    Dz = np.int_((z_max - z_min) // r) + 1
    print(f"The three dimensions of the voxel grid are {Dx}, {Dy}, and {Dz}.")

    # 3. store the index of points in the corresponding voxel grid
    voxel_dict = {}
    for idx, point in enumerate(data):
        # computer voxel index for each point
        hx = np.int_((point[0] - x_min) // r)
        hy = np.int_((point[1] - y_min) // r)
        hz = np.int_((point[2] - z_min) // r)
        h = hx + hy * Dx + hz * Dx * Dy
        if h in voxel_dict:
            voxel_dict[h].append(idx)
        else:
            voxel_dict[h] = [idx]

    # 4. downsampling
    for key, value in voxel_dict.items():
        if centroid:
            filtered_points.append(np.mean(data[value], axis=0))
        else:
            idx = np.random.choice(value)
            filtered_points.append(data[idx])

    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    opt = parse_args()

    # the names of 40 categories
    with open('../data/modelnet40_normal_resampled/modelnet40_shape_names.txt') as f:
        cates = f.readlines()

    cates = ["plant"]

    for cate in cates:
        point_cloud_pynt = PyntCloud.from_file(
            '../data/modelnet40_normal_resampled/{}/{}_0001.txt'.format(cate.strip(), cate.strip()), sep=",",
            names=["x", "y", "z", "nx", "ny", "nz"])

        # convert PyntCloud instance to open3d
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
        # visualize the original point cloud
        o3d.visualization.draw_geometries([point_cloud_o3d])

        # Apply voxel grid downsampling to point cloud
        filtered_cloud = voxel_filter(point_cloud_pynt.xyz, opt.r, opt.centroid)
        point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
        # visualize the point cloud after downsampling
        o3d.visualization.draw_geometries([point_cloud_o3d])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", default=0.05, type=float, help="voxel grid size")
    parser.add_argument("--centroid", default=True, type=bool, help="mean or random voxel grid downsampling")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    main()
