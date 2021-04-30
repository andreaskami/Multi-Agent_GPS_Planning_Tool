from utility.misc import single_path_distance


def two_opt_path(path_to_evaluate, start_node, distances):
    """
    The two-opt is a TSP solving method that iteratively performs swaps on a provided initial path.
    The method's complexity is O(n*n) - quite expensive.
    A Method for Solving Traveling-Salesman Problems - G.A. Croes, 1958

    :param path_to_evaluate: Initial TSP path to improve
    :param start_node: Depot/start node of the agents
    :param distances: Distance lookup table for the environment
    :return an optimized 2-opt path and its total length
    """
    path = list(path_to_evaluate)
    best_dist = single_path_distance(path, distances)
    improved = True
    while improved:
        improved, path, best_dist = _try_to_improve(path, start_node, distances, best_dist)
    return path, best_dist


def _try_to_improve(path, start_node, distances, best_dist):
    """Do one cycle of two opt swaps to look for improvements."""
    for i in range(1, len(path) - 1):
        for k in range(i + 1, len(path)):
            new_path = _two_opt_swap(path, i, k)  # perform swap
            dist = single_path_distance([start_node] + new_path + [start_node], distances)
            if dist < best_dist:
                return True, new_path, dist
    return False, path, best_dist


def _two_opt_swap(path, i, k):
    """Perform a two-opt move."""
    return path[0:i] + path[i:k+1][::-1] + path[k+1:]

