import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet


class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output


def sort_key_by_vale(key, value):
    """
    Before splitting, it must sort keys and values according to values.
    Args:
        key: indexes of points.
        value: points' values along with an axis.
    Returns:
        key_sorted: sorted keys according to value_sorted.
        value_sorted: sorted values.
    :param key:
    :param value:
    :return:
    """
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted


def axis_round_robin(axis, dim):
    """
    Args:
        axis: the axis used for splitting the current tree.
        dim: the dimensionality of the data.
    Returns:
        the axis to be used for the sub-tree of the current tree.
    """
    if axis == dim - 1:
        return 0
    else:
        return axis + 1


def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    """
    Build a KDTree recursively.
    Args:
        root: the root node of the current tree (a sub-tree of the KDTree).
        db: the point cloud data.
        point_indices: the indexes of the points belonging to the current tree.
        axis: the axis used to split the current tree.
        leaf_size: the maximum number of points that a leaf node can contain.
    Returns:
        the root node of the current sub-tree.
    """
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # get the split position, i.e., median position along with axis
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])

        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        middle_right_idx = middle_left_idx + 1

        # calculate root.value
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        middle_left_point_value = db[middle_left_point_idx, axis]
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]
        root.value = (middle_left_point_value + middle_right_point_value) / 2

        # build the left sub-kdtree of the current node
        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[:middle_right_idx],
                                           axis_round_robin(axis, db.shape[1]),
                                           leaf_size)
        # build the right sub-kdtree of the current node
        root.right = kdtree_recursive_build(root.right,
                                            db,
                                            point_indices_sorted[middle_right_idx:],
                                            axis_round_robin(axis, db.shape[1]),
                                            leaf_size)
    return root


def kdtree_construction(db, leaf_size):
    """
    The interface for building a KDTree. It will invoke the method kdtree_recursive_build(...).
    Args:
        dp: the point cloud data, N*3.
        leaf_size: the maximum number of points that a leaf node can contain.
    Returns:
        root: the root node of the KDTree.
    """
    N, dim = db.shape[0], db.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


def traverse_kdtree(root: Node, depth, max_depth):
    """
    Traverse a kdtree, to know the maximum depth of a kdtree.
    Args:
        root: the root node of the current kdtree.
        depth: the depth of the root node.
        max_depth: to record the maximum depth of the whole ketree.
    Returns:
        None
    """
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    # if root.is_leaf():
        # print(root)
    if not root.is_leaf():
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1


def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    """
    Find k nearest neighbors by a KDTree.
    Args:
        root: the root node of the current kdtree.
        db: the point cloud data.
        result_set: a data structure which stores the results.
        query: the point to be searched for its k nearest neighbors.
    Returns:
        False as failure or end.
    """
    if root is None:
        return False

    if root.is_leaf():
        # compare query to every point inside the leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, axis=0) - leaf_points, axis=1)
        # put all into the result set
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value:
        # search the left sub-kdtree first
        kdtree_knn_search(root.left, db, result_set, query)
        # go to right if the distance between query and root is less than the current worst distance
        if math.fabs(query[root.axis] - root.value) < result_set.worst_distance():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worst_distance():
            kdtree_knn_search(root.left, db, result_set, query)

    return False


def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    """
    Find nearest neighbors within a radius defined by result_set.
    Args:
        root: the root node of the current kdtree.
        db: the point cloud data.
        result_set: a data structure which stores the results.
        query: the point to be searched for its nearest neighbors.
    Returns:
        False as failure or end.
    """
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worst_distance():
            kdtree_radius_search(root.right, db, result_set, query)
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worst_distance():
            kdtree_radius_search(root.left, db, result_set, query)

    return False


# def main():
#     # configuration
#     db_size = 64
#     dim = 3
#     leaf_size = 4
#     k = 1
#
#     db_np = np.random.rand(db_size, dim)
#
#     root = kdtree_construction(db_np, leaf_size=leaf_size)
#
#     depth = [0]
#     max_depth = [0]
#     traverse_kdtree(root, depth, max_depth)
#     print("tree max depth: %d" % max_depth[0])


# if __name__ == '__main__':
#     main()
