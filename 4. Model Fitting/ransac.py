# 3D RANSAC implementation in Python
import math
import numpy as np
from ordered_set import OrderedSet


def check_colinear_of_three_points(p1, p2, p3):
    """
    A, B and C are colinear
    if and only if the largest of the lenghts of AB, AC and BC is equal to the sum of the other two.
    """
    lengths = [np.linalg.norm(p1-p2), np.linalg.norm(p1-p3), np.linalg.norm(p2-p3)]
    largest_idx = np.argmax(lengths)
    largest_length = lengths[largest_idx]
    total_length = np.sum(lengths)
    return math.isclose(total_length, 2 * largest_length)


class RANSAC:

    def __init__(self, max_iter, dist_threshold):
        self.max_iter = max_iter
        self.dist_threshold = dist_threshold
        self.pc = None

    def detect_ground(self, pc):
        """
        The interface for detecting the ground from a point cloud
        Args:
            pc: the origin point cloud
        Returns:
            ground: the point cloud of the detected ground
            foreground: the point cloud of the foreground
        """
        self.pc = pc
        ground_point_index, foreground_point_index = self._ransac_algorithm()
        ground_cloud = self.pc[ground_point_index]
        foreground_cloud = self.pc[foreground_point_index]
        return ground_cloud, foreground_cloud

    def _ransac_algorithm(self):
        """
        Implementation of the 3D RANSAC algorithm.
        Returns:
            ground_point_index: the index of the points of the detected plane
        """
        inliers_final_result = set()
        outliers_result = set()

        for _ in range(self.max_iter):

            inliers = OrderedSet()
            # randomly select three non-colinear points to determine the ground plane
            while len(inliers) < 3:
                idx = np.random.randint(low=0, high=len(self.pc)-1)
                if idx not in inliers:
                    inliers.append(idx)
                if len(inliers) == 3 and check_colinear_of_three_points(self.pc[inliers[0]], self.pc[inliers[1]],
                                                                        self.pc[inliers[2]]):
                    inliers = inliers[:-1]

            x1, y1, z1 = self.pc[inliers[0]]
            x2, y2, z2 = self.pc[inliers[1]]
            x3, y3, z3 = self.pc[inliers[2]]
            # calculate the parameters required for the plane equation (ax + by + cz + d = 0) in Cartesian form
            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            d = -(a * x1 + b * y1 + c * z1)

            for idx, point in enumerate(self.pc):
                if idx in inliers:
                    continue
                xi, yi, zi = point[0], point[1], point[2]
                dist = math.fabs(a * xi + b * yi + c * zi + d) / math.sqrt(a * a + b * b + c * c)
                if dist < self.dist_threshold:
                    inliers.add(idx)

            # inliers_final_result always contains the most inliers
            if len(inliers_final_result) < len(inliers):
                inliers_final_result.clear()
                inliers_final_result = inliers

        outliers_result = set(np.arange(0, len(self.pc))).difference(inliers_final_result)
        return list(inliers_final_result), list(outliers_result)
