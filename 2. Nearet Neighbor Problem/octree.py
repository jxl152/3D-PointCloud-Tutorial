import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet
import benchmark

class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children
        self.center = center    # center of the cube
        self.extent = extent    # 0.5 * length
        self.point_indices = point_indices  # points inside the octant
        self.is_leaf = is_leaf

    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output


def traverse_octree(root: Octant, depth, max_depth):
    """
    Traverse an Octree, to know the maximum depth of an Octree.
    Args:
        root: the root Octant of the current Octree.
        depth: the depth of the current root Octant.
        max_depth: to record the maximum depth of the whole Octree.
    """
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        for child in root.children:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1


def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    """
    Build an OcTree recursively.
    Args:
        root: the root list of eight Octant of the OcTree.
        db: the point cloud data, N*3.
        center: the center of the current octree.
        extent: the extent of the current octree
        point_indices: the indexes of the points belonging to the current tree.
        leaf_size: the maximum number of points that a leaf node can contain.
        min_extent: the minimum extent that a leaf node can have.
    Return:
        the root list of eight Octant of the OcTree.
    """
    if len(point_indices) == 0:
        return None

    if root is None:
        root = Octant([None for _ in range(8)], center, extent, point_indices, is_leaf=True)

    # determine whether to split this octant
    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:
        root.is_leaf = False
        children_point_indices = [[] for _ in range(8)]
        # determine which child octant a point belongs to
        for point_idx in point_indices:
            point = db[point_idx]
            morton_code = 0
            if point[0] > center[0]:
                morton_code = morton_code | 1
            if point[1] > center[1]:
                morton_code = morton_code | 2
            if point[2] > center[2]:
                morton_code = morton_code | 4
            children_point_indices[morton_code].append(point_idx)
        # determine the center and extent of eight child octant
        factor = [-0.5, 0.5]
        for i in range(8):
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            child_center = np.asarray([child_center_x, child_center_y, child_center_z])
            child_extent = 0.5 * extent
            root.children[i] = octree_recursive_build(root.children[i],
                                                      db,
                                                      child_center,
                                                      child_extent,
                                                      children_point_indices[i],
                                                      leaf_size,
                                                      min_extent)
    return root


def octree_construction(db, leaf_size, min_extent):
    """
    The interface for building an OcTree. It will invoke the method octree_recursive_build(...).
    Args:
        db: the point cloud data, N*3.
        leaf_size: the maximum number of points that a leaf node can contain.
        min_extent: the minimum extent that a leaf node can have.
    Return:
        root: the root list of eight Octant of the OcTree.
    """
    N, dim = db.shape[0], db.shape[1]
    db_min = np.amin(db, axis=0)
    db_max = np.amax(db, axis=0)
    db_extent = np.max(db_max - db_min) * 0.5
    db_center = np.mean(db, axis=0)

    root = None
    root = octree_recursive_build(root, db, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent)

    return root


def inside(query: np.ndarray, radius: float, octant: Octant):
    """
    Determines if the query ball is inside the octant.
    Args:
        query: the point to be searched for its k nearest neighbors.
        radius: the radius of the query ball, i.e., the current worst distance of the result_set.
        octant: the current octant. 
    Return:
        True if the query ball is completely within the octant; otherwise False.
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)


def overlaps(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball overlaps with the octant.
    Args:
        query: the point to be searched for its k nearest neighbors.
        radius: the radius of the query ball, i.e., the current worst distance of the result_set.
        octant: the current octant.
    Return:
        True if the query ball overlaps with the octant; otherwise False.
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    if np.sum((query_offset_abs < octant.extent).astype(np.int_)) >= 2:
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    # More specifically, corner case: x_diff = query_offset_abs[0] - octant.extent;
    # edge case: x_diff = max(query_offset_abs[0] - octant.extent, 0)
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


def determine_most_relevant_child(root: Octant, query: np.ndarray):
    """
    Determine the most relevant child where query is
    Args:
        root: the current octant.
        query: the point to be searched for its k nearest neighbors.
    Returns:
        morton_code: the index of the child octant.
    """
    assert not root.is_leaf
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4
    return morton_code


def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    """
    Find k nearest neighbors by a OcTree.
    Args:
        root: OcTree.
        db: the point cloud data.
        result_set: a data structure which stores the results.
        query: the point to be searched for its k nearest neighbors.
    Return:
        True if it can stop here and does not need to search the children of the octant.
    """
    if root is None:
        return False

    # the current octant is a leaf
    if root.is_leaf and len(root.point_indices) > 0:
        # compare query to every point inside the leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worst_distance(), root)

    # determine and search the most relevant child first
    morton_code = determine_most_relevant_child(root, query)
    if octree_knn_search(root.children[morton_code], db, result_set, query):
        return True

    # check other children if necessary
    for idx, child in enumerate(root.children):
        if idx == morton_code or child is None:
            continue
        # if child does not overlap with query ball, then skip
        if not overlaps(query, result_set.worst_distance(), child):
            continue
        if octree_knn_search(child, db, result_set, query):
            return True

    # if query ball is inside the current octant, then it can stop
    return inside(query, result_set.worst_distance(), root)


def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    """
    Find nearest neighbors within a radius defined by result_set.
    Args:
        root: OcTree.
        db: the point cloud data.
        result_set: a data structure which stores the results.
        query: the point to be searched for its k nearest neighbors.
    Return:
        True if it can stop here and does not need to search the children of the octant.
    """
    if root is None:
        return False

    # the current octant is a leaf
    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worst_distance(), root)

    # determine and search the most relevant child first
    morton_code = determine_most_relevant_child(root, query)
    if octree_radius_search(root.children[morton_code], db, result_set, query):
        return True

    # check other children if necessary
    for idx, child in enumerate(root.children):
        if idx == morton_code or child is None:
            continue
        if not overlaps(query, result_set.worst_distance(), child):
            continue
        if octree_radius_search(child, db, result_set, query):
            return True

    # if query ball is inside the current octant, then it can stop
    return inside(query, result_set.worst_distance(), root)


def contains(query: np.ndarray, radius: float, octant:Octant):
    """
    Determine if the query ball contains the octant
    Args:
        query: the point to be searched for its k nearest neighbors.
        radius: the radius of the query ball, i.e., the current worst distance of the result_set.
        octant: the current octant.
    Returns:
        True if the query ball contains the octant; otherwise False.
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    return np.linalg.norm(query_offset_to_farthest_corner) < radius


# 功能：在octree中查找信息
# 输入：
#    root: octree
#    db：原始数据
#    result_set: 索引结果
#    query：索引信息
def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    """
    Faster version of octree_radius_search(...).
    Args:
        root: OcTree.
        db: the point cloud data.
        result_set: a data structure which stores the results.
        query: the point to be searched for its k nearest neighbors.
    Return:
        True if it can stop here and does not need to search the children of the octant.
    """
    if root is None:
        return False

    # the current octant is a leaf or the query ball contains the octant
    if (root.is_leaf and len(root.point_indices) > 0) or contains(query, result_set.worst_distance(), root):
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worst_distance(), root)

    # determine and search the most relevant child first
    morton_code = determine_most_relevant_child(root, query)
    if octree_radius_search_fast(root.children[morton_code], db, result_set, query):
        return True

    # check other children if necessary
    for idx, child in enumerate(root.children):
        if idx == morton_code or child is None:
            continue
        if not overlaps(query, result_set.worst_distance(), child):
            continue
        if octree_radius_search_fast(child, db, result_set, query):
            return True

    return inside(query, result_set.worst_distance(), root)


def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 5  # affect the comparison result between radius_search(...) and radius_search_fast(...)

    # load the point cloud data
    file_path = "000000.bin"
    db = benchmark.read_velodyne_bin(file_path)

    np.random.seed(42)
    np.set_printoptions(suppress=True, precision=3)


    octree_construction_time_sum = 0
    octree_knn_search_time_sum = 0
    octree_radius_search_time_sum = 0
    octree_radius_fast_search_time_sum = 0
    iteration_num = 1
    for _ in range(iteration_num):
        query = np.random.rand(3)

        # OcTree
        # print("\nOcTree ---------")
        begin_t = time.time()
        root = octree_construction(db, leaf_size, min_extent)
        octree_construction_time = time.time() - begin_t
        octree_construction_time_sum += octree_construction_time
        # print(f"construction time = {octree_construction_time * 1000:.3f} ms")

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        octree_knn_search(root, db, result_set, query)
        octree_knn_search_time = time.time() - begin_t
        octree_knn_search_time_sum += octree_knn_search_time
        # print(f"k nearest neighbors are: {result_set.knn_indexes()}")
        # print(f"knn serch time = {octree_knn_search_time * 1000:.3f} ms")

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree_radius_search(root, db, result_set, query)
        octree_radius_search_time = time.time() - begin_t
        octree_radius_search_time_sum += octree_radius_search_time
        # print(f"neighbors within {radius} are {result_set.sorted_neighbors_indexes()}")
        # print(f"radius search time = {octree_radius_search_time * 1000:.3f} ms")

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree_radius_search_fast(root, db, result_set, query)
        octree_radius_fast_search_time = time.time() - begin_t
        octree_radius_fast_search_time_sum += octree_radius_fast_search_time
        # print(f"neighbors within {radius} are {result_set.sorted_neighbors_indexes()}")
        # print(f"radius fast search time = {octree_radius_fast_search_time * 1000:.3f} ms")

    print(f"Octree:\n build = {octree_construction_time_sum * 1000 / iteration_num:.3f}, "
          f"knn search = {octree_knn_search_time_sum * 1000 / iteration_num:.3f},"
          f"radius search = {octree_radius_search_time_sum * 1000 / iteration_num:.3f},"
          f"radius fast search = {octree_radius_fast_search_time_sum * 1000 / iteration_num:.3f}")


if __name__ == '__main__':
    main()