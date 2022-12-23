# Implement PCA and Surface Normal, and validate them by the data of ModelNet40

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud


# def standardize_data(arr):
#     """
#     This function standardize an array.
#     Each column subtracts its mean value, and then divide its standard devision.
#
#     param arr: array of point cloud
#     return: standardized array
#     """
#     mean_arr, std_arr = np.mean(arr, axis=0), np.std(arr, axis=0)
#     # print(f"before standardization, mean is {mean_arr} and std is {std_arr}")
#     arr = (arr - mean_arr) / std_arr
#     # print(f"after standardization, mean is {np.mean(arr, axis=0)} and std is {np.std(arr, axis=0)}")
#     return arr


def PCA(data, correlation=False, sort=True):
    """
    Apply PCA to point cloud
    Args:
        data: point cloud, matrix of Nx3
        correlation: use np.cov if False, otherwise np.corrcoef if True. default: False
        sort: whether sort according to eigenvalues. default: True.
    Returns:
        eigenvalues
        eigenvectors
    """
    # 1. compute the covariance matrix
    if correlation:
        # each row represents a variable, and each column a single observation of all those variables.
        cov_mat = np.corrcoef(data.T)
    else:
        cov_mat = np.cov(data.T)

    # 2. apply svd to the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # the names of 40 categories
    with open('../data/modelnet40_normal_resampled/modelnet40_shape_names.txt') as f:
        cates = f.readlines()

    cates = ["guitar"]

    for cate in cates:
        point_cloud_pynt = PyntCloud.from_file(
            '../data/modelnet40_normal_resampled/{}/{}_0001.txt'.format(cate.strip(), cate.strip()), sep=",",
            names=["x", "y", "z", "nx", "ny", "nz"])

        # convert PyntCloud instance to open3d
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
        # visualize the original point cloud
        # o3d.visualization.draw_geometries([point_cloud_o3d])
        # visualize the original point cloud with normal
        # o3d.visualization.draw_geometries([point_cloud_o3d], point_show_normal=True)

        # extract points from the PyntCloud object
        points = point_cloud_pynt.xyz    # {ndarray:(10000,3)}
        print('total points number is:', points.shape[0])

        # Apply PCA to point cloud
        w, v = PCA(points)
        point_cloud_vector = v[:, 0]    # vectors of the principal component
        print('the main orientation of this point cloud is: ', point_cloud_vector)
        print('the second significant orientation of this point cloud is: ', v[:, 1])

        # visualize three principal component axis
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([np.mean(points, axis=0),
                                                      np.mean(points, axis=0) + v[:, 0],
                                                      np.mean(points, axis=0) + v[:, 1],
                                                      np.mean(points, axis=0) + v[:, 2]])
        line_set.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
        line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0], [255, 0, 0], [0, 255, 0]]) # black, red, green
        o3d.visualization.draw_geometries([point_cloud_o3d, line_set])

        # Projection of point cloud on the plane of the first two principal components
        # calculate the projection of point cloud on the least significant vector of PCA
        least_pc = v[:, -1]
        scalar_arr = np.dot(points, least_pc) / np.linalg.norm(least_pc, axis=0)**2
        proj_on_least_pc = scalar_arr.reshape(scalar_arr.size, 1) * least_pc
        # subtract the projection on the least pc from original points, to obtain the projection on the plane
        projected_points = points - proj_on_least_pc
        # visualize the projection
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(projected_points)
        o3d.visualization.draw_geometries([pcd])

        # Surface Normal Estimation
        # calculate the KDTree from the point cloud
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        normals = []
        for point in points:
            # for each point, find its 50-nearest neighbors
            [_, idx, _] = pcd_tree.search_knn_vector_3d(point, knn=50)
            w, v = PCA(points[idx])
            # normal -> the least significant vector of PCA
            normals.append(v[:, 2])

        # visualize point cloud with surface normal
        normals = np.array(normals, dtype=np.float64)
        point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([point_cloud_o3d], point_show_normal=True)


if __name__ == '__main__':
    main()
