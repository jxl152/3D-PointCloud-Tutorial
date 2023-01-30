import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud
import copy


def normalize(feature):
    feature = feature - np.mean(feature, axis=0, keepdims=True) # center; (N, 3)
    max_dist = np.max(np.linalg.norm(feature, axis=1))
    feature /= max_dist # scale
    return feature


def rotate(feature, rotation_matrix):
    rotated_feature = copy.deepcopy(feature)
    rotated_feature = rotated_feature.dot(rotation_matrix)
    # rotated_feature[:, [0, 2]] = rotated_feature[:, [0, 2]].dot(rotation_matrix)  # rotation
    return rotated_feature


def visualize_with_coordinate_frame(pcd):
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    # test normalization
    feature = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [9, 5, 1], [8, 6, 4]])
    normalized_feature = normalize(feature)
    print("Before normalization, feature = \n{}\n".format(feature))
    print("After normalization, feature = \n{}".format(normalized_feature))

    # test rotation
    point_cloud_pynt = PyntCloud.from_file(
        '../../data/modelnet40_normal_resampled/bottle/bottle_0001.txt', sep=",",
        names=["x", "y", "z", "nx", "ny", "nz"])
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    visualize_with_coordinate_frame(point_cloud_o3d)

    print("####################################\n")
    points = point_cloud_pynt.xyz

    theta = np.pi / 2
    print("Rotate over the x-axis\n")
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(theta), np.sin(theta)],
                                [0, -np.sin(theta), np.cos(theta)]])
    rotated_points = rotate(points, rotation_matrix)
    print(points[:3])
    print(rotated_points[:3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rotated_points)
    visualize_with_coordinate_frame(pcd)

    print("\nRotate over the y-axis\n")
    rotation_matrix = np.array([[np.cos(theta), 0, -np.sin(theta)],
                                [0, 1, 0],
                                [np.sin(theta), 0, np.cos(theta)]])
    rotated_points = rotate(points, rotation_matrix)
    print(points[:3])
    print(rotated_points[:3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rotated_points)
    visualize_with_coordinate_frame(pcd)

    print("\nRotate over the z-axis\n")
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    rotated_points = rotate(points, rotation_matrix)
    print(points[:3])
    print(rotated_points[:3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rotated_points)
    visualize_with_coordinate_frame(pcd)
